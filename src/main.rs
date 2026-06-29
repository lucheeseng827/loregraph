//! loregraph (`lore`) — single-binary CLI entrypoint.
//!
//! `index` crunches chat transcripts + a repo into the memory graph; `serve` runs the API +
//! embedded canvas; `ask` is the headless recall (the MCP tool's CLI twin); `mcp` is the
//! agent-facing memory server; `doctor` shows what connectors discover. Headless-first,
//! loopback-by-default.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};

use lore::ingest::{self, SessionSource};
use lore::store::Store;

#[derive(Parser)]
#[command(
    name = "lore",
    version,
    about = "loregraph — memory knowledge-graph for AI coding-agent sessions + repos (single binary)."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Crunch chat transcripts + a repo into the memory graph.
    Index {
        /// A transcript file, a directory of transcripts, or omit to auto-discover this
        /// source's transcripts on the machine (e.g. ~/.claude/projects).
        #[arg(long)]
        sessions: Option<PathBuf>,
        /// Repo to crunch (code graph + frameworks). Optional.
        #[arg(long)]
        repo: Option<PathBuf>,
        /// Connector id (PoC: `claude_code`).
        #[arg(long, default_value = "claude_code")]
        source: String,
        /// Where the graph snapshot is stored.
        #[arg(long, default_value = ".lore")]
        data_dir: PathBuf,
        /// Parse + report only; do not write the graph.
        #[arg(long)]
        dry_run: bool,
    },
    /// Run the canvas + API (loopback by default).
    Serve {
        #[arg(long, default_value = "127.0.0.1:7700")]
        addr: String,
        #[arg(long, default_value = ".lore")]
        data_dir: PathBuf,
    },
    /// Recall memories for a query (the headless twin of the MCP `memory.search` tool).
    Ask {
        /// What to recall, e.g. "what did we decide about the storage engine".
        query: String,
        #[arg(long, default_value = ".lore")]
        data_dir: PathBuf,
        #[arg(short, long, default_value_t = 8)]
        k: usize,
        /// Emit the full `RecallResult` as JSON (for scripts / agents), instead of the
        /// human-readable listing.
        #[arg(long)]
        json: bool,
    },
    /// Run the MCP memory server over stdio (build with `--features mcp`); without the
    /// feature, prints the tool contract.
    Mcp {
        #[arg(long, default_value = ".lore")]
        data_dir: PathBuf,
    },
    /// Show what each connector can discover on this machine.
    Doctor {
        #[arg(long, default_value = "claude_code")]
        source: String,
    },
    /// Print the version.
    Version,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Index {
            sessions,
            repo,
            source,
            data_dir,
            dry_run,
        } => cmd_index(sessions, repo, &source, &data_dir, dry_run),
        Cmd::Serve { addr, data_dir } => cmd_serve(&addr, &data_dir),
        Cmd::Ask { query, data_dir, k, json } => cmd_ask(&query, &data_dir, k, json),
        Cmd::Mcp { data_dir } => cmd_mcp(&data_dir),
        Cmd::Doctor { source } => cmd_doctor(&source),
        Cmd::Version => {
            println!("lore {}", lore::version());
            Ok(())
        }
    }
}

/// Resolve the transcript files to ingest: an explicit file, a dir of `*.jsonl`, or the
/// connector's auto-discovery.
fn resolve_sessions(
    connector: &dyn SessionSource,
    sessions: Option<PathBuf>,
) -> anyhow::Result<Vec<PathBuf>> {
    match sessions {
        Some(p) if p.is_file() => Ok(vec![p]),
        Some(p) if p.is_dir() => {
            let mut out = Vec::new();
            for entry in walkdir::WalkDir::new(&p).into_iter().filter_map(Result::ok) {
                let path = entry.path();
                if path.is_file() && path.extension().is_some_and(|e| e == "jsonl") {
                    out.push(path.to_path_buf());
                }
            }
            out.sort();
            Ok(out)
        }
        Some(p) => anyhow::bail!("--sessions path not found: {}", p.display()),
        None => connector.discover(),
    }
}

/// Ingest transcripts and (optionally) a repo into the durable graph, then save.
fn cmd_index(
    sessions: Option<PathBuf>,
    repo: Option<PathBuf>,
    source: &str,
    data_dir: &Path,
    dry_run: bool,
) -> anyhow::Result<()> {
    lore::init_tracing();
    let connector = ingest::by_id(source)
        .with_context(|| format!("unknown source `{source}` (PoC supports: claude_code)"))?;

    let files = resolve_sessions(connector.as_ref(), sessions)?;
    println!("source {source}: {} transcript file(s)", files.len());

    // The commit axis of the provenance wedge: when indexing against a repo, anchor each
    // session to the commit that was HEAD when the chat happened (cheap, diff-free walk).
    let timeline = repo.as_deref().and_then(|r| lore::git::timeline(r, 1000));
    // Canonical repo path used to match `sess.repo_root` — prevents stamping sessions from
    // an unrelated repo with this timeline when multiple repos share a transcript dir.
    let repo_canon: Option<std::path::PathBuf> =
        repo.as_deref().and_then(|r| r.canonicalize().ok());
    if let Some(t) = &timeline {
        if let Some(head) = t.head_short() {
            println!("repo at HEAD {head}: {} commit(s) for provenance anchoring", t.commits.len());
        }
    }

    let mut store = Store::open(data_dir)?;
    let mut sessions_ok = 0usize;
    let mut turns_total = 0usize;
    for f in &files {
        match connector.read_session(f) {
            Ok(mut sess) => {
                // Stamp the contemporaneous commit only when the connector didn't already
                // carry one (a native session_id+commit always wins).
                if sess.commit.is_none() {
                    if let Some(t) = &timeline {
                        // Only stamp when the session's working dir is the indexed repo OR a
                        // subdirectory of it in the *same* git tree. Sessions with no repo_root,
                        // or rooted outside this repo, are skipped — stamping them would be false
                        // provenance. Nested repos (submodules / worktrees with their own .git)
                        // are also excluded so they don't inherit the outer repo's commit.
                        let repo_matches = (|| -> bool {
                            let (Some(sr), Some(rc)) = (&sess.repo_root, &repo_canon) else {
                                return false;
                            };
                            let Some(p) = std::path::Path::new(sr).canonicalize().ok() else {
                                return false;
                            };
                            same_git_tree(&p, rc)
                        })();
                        if repo_matches {
                            let ts = sess
                                .started_at_ms
                                .or_else(|| sess.turns.iter().find_map(|tn| tn.ts_ms));
                            // Store full OIDs so sess.commit keys match Commit node IDs.
                            sess.commit = match ts {
                                Some(ts) => t.commit_at(ts).map(|c| c.oid.clone()),
                                None => t.head.clone(),
                            };
                        }
                    }
                }
                turns_total += sess.turns.len();
                if !dry_run {
                    store.ingest_session(&sess)?;
                }
                sessions_ok += 1;
            }
            Err(e) => eprintln!("  ! skipped {}: {e}", f.display()),
        }
    }
    println!("ingested {sessions_ok} session(s), {turns_total} turn(s)");

    if let Some(repo) = repo {
        if dry_run {
            println!("(dry-run) would crunch repo {}", repo.display());
        } else {
            store.ingest_repo(&repo)?;
            println!("crunched repo {}", repo.display());
        }
    }

    if dry_run {
        println!("(dry-run: nothing written)");
    } else {
        // Drift detection: link explicitly-superseded decisions (R4) over the full graph.
        let superseded = store.detect_supersessions()?;
        if superseded > 0 {
            println!("supersedes: linked {superseded} superseded decision(s)");
        }
        let index_backend = store.index.backend();
        store.save()?;
        println!(
            "graph: {} nodes, {} edges, {} embeddings ({}) → {}",
            store.graph.node_count(),
            store.graph.edge_count(),
            store.index.len(),
            index_backend,
            data_dir.display()
        );
    }
    Ok(())
}

/// Load the graph and start the axum HTTP canvas + API on `addr`.
fn cmd_serve(addr: &str, data_dir: &Path) -> anyhow::Result<()> {
    lore::init_tracing();
    let store = Store::open(data_dir)?;
    if store.graph.node_count() == 0 {
        eprintln!("note: graph is empty — run `lore index` first.");
    }
    let addr: SocketAddr = addr
        .parse()
        .with_context(|| format!("invalid --addr {addr:?}"))?;
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(lore::api::serve(addr, store))
}

/// Run a recall query against the graph; `--json` emits `RecallResult` for scripts/agents.
fn cmd_ask(query: &str, data_dir: &Path, k: usize, json: bool) -> anyhow::Result<()> {
    let store = Store::open(data_dir)?;
    let result = lore::ask::recall(&store, query, k);
    if json {
        // Machine-readable: the whole RecallResult (query, degraded, ranked evidence +
        // provenance). stdout only, so it pipes cleanly.
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }
    if result.degraded {
        eprintln!("(no semantic index — lexical fallback; set LORE_EMBED_BACKEND=static + LORE_EMBED_MODEL for real embeddings)");
    }
    if result.results.is_empty() {
        println!("nothing recalled for: {query}");
        return Ok(());
    }
    println!("recalled {} memor(ies) for: {query}\n", result.results.len());
    for (i, e) in result.results.iter().enumerate() {
        println!("{}. [{}] {} (score {:.3}, via {})", i + 1, e.kind, e.label, e.score, e.why);
        if !e.snippet.is_empty() && e.snippet != e.label {
            println!("   {}", e.snippet);
        }
        if let Some(p) = &e.provenance {
            let sid = p.native_session_id.as_deref().unwrap_or("—");
            let commit = p.commit.as_deref().unwrap_or("—");
            println!("   ↳ source {} · session {} · commit {} · conf {:.2}", p.source_tool, sid, commit, p.confidence);
        }
    }
    Ok(())
}

/// Start the MCP stdio memory server (feature `mcp`), or print the tool contract when absent.
#[allow(unused_variables)]
fn cmd_mcp(data_dir: &Path) -> anyhow::Result<()> {
    #[cfg(feature = "mcp")]
    {
        lore::mcp::serve_stdio(data_dir)
    }
    #[cfg(not(feature = "mcp"))]
    {
        lore::mcp::explain();
        Ok(())
    }
}

/// Report what transcripts the named connector discovers on this machine.
fn cmd_doctor(source: &str) -> anyhow::Result<()> {
    let connector = ingest::by_id(source)
        .with_context(|| format!("unknown source `{source}`"))?;
    let files = connector.discover()?;
    println!("connector `{}`: discovered {} transcript file(s)", connector.id(), files.len());
    for f in files.iter().take(10) {
        println!("  {}", f.display());
    }
    if files.len() > 10 {
        println!("  … and {} more", files.len() - 10);
    }
    if files.is_empty() {
        println!("  (none found — is the agent installed, and have you run a session?)");
    }
    Ok(())
}

/// Return `true` when `session_cwd` belongs to the same git tree as `repo_root` — i.e. it
/// is the root itself or a subdirectory with no intervening `.git` between them. Returns
/// `false` for a nested repo (submodule / inner worktree with its own `.git`) so those
/// sessions don't inherit the outer repo's commit as false provenance.
fn same_git_tree(session_cwd: &Path, repo_root: &Path) -> bool {
    let mut cursor = session_cwd;
    loop {
        if cursor == repo_root {
            return true;
        }
        if !cursor.starts_with(repo_root) {
            return false;
        }
        if cursor.join(".git").exists() {
            return false;
        }
        match cursor.parent() {
            Some(parent) => cursor = parent,
            None => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_git_tree_exact_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        assert!(same_git_tree(&root, &root));
    }

    #[test]
    fn same_git_tree_plain_subdir() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let sub = root.join("crate/src");
        std::fs::create_dir_all(&sub).unwrap();
        assert!(same_git_tree(&sub, &root));
    }

    #[test]
    fn same_git_tree_rejects_nested_git_dir() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let nested = root.join("vendor/inner");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir(nested.join(".git")).unwrap();
        assert!(!same_git_tree(&nested, &root));
    }

    #[test]
    fn same_git_tree_rejects_nested_git_file() {
        // git worktrees use a `.git` file (not dir) pointing back to the main worktree.
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let nested = root.join("worktrees/feat");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join(".git"), "gitdir: ../../.git/worktrees/feat\n").unwrap();
        assert!(!same_git_tree(&nested, &root));
    }

    #[test]
    fn same_git_tree_rejects_outside_path() {
        let root_dir = tempfile::tempdir().unwrap();
        let other_dir = tempfile::tempdir().unwrap();
        let root = root_dir.path().canonicalize().unwrap();
        let other = other_dir.path().canonicalize().unwrap();
        assert!(!same_git_tree(&other, &root));
    }
}
