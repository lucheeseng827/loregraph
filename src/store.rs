//! The store — fuses chat + repo into the memory graph and persists it durably.
//!
//! Durability is the redb + write-ahead-log (WAL) crash-safe commit protocol from
//! `ARCHITECTURE.md`:
//!
//! - **Ingest** builds a batch of mutations ([`WalRecord`]s), appends them to the WAL, and
//!   `fsync`s — that fsync is the **ACK boundary** — *then* applies them to the in-memory
//!   [`MemGraph`] + [`BruteForceIndex`]. A crash after the fsync loses nothing: the records
//!   are replayed on next open.
//! - **Checkpoint** ([`Store::save`]) folds the in-memory state into **redb** in one ACID
//!   write transaction (redb's own fsync), and *only then* truncates the WAL.
//! - **Recovery** ([`Store::open`]) rebuilds the graph from redb (the committed base), then
//!   replays the WAL tail. Replaying records that a crash left in the WAL *after* a redb
//!   commit is harmless because every upsert is **content-addressed and idempotent** — the
//!   same `(kind, identity)` always yields the same id, so a double-apply is a no-op.
//!
//! redb is pure-Rust ACID (no C deps), keeping the air-gap / static-musl property.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::coderepo;
use crate::decision;
use crate::embed::{DynEmbedder, Embedder};
use crate::git;
use crate::graph::MemGraph;
use crate::index::BruteForceIndex;
use crate::model::{
    summarize, CanonicalSession, Edge, EdgeKind, MessageKind, Node, NodeId, NodeKind, Provenance,
};

// redb tables: content-addressed key → JSON-serialized value.
const T_NODES: TableDefinition<u128, &[u8]> = TableDefinition::new("nodes");
const T_EDGES: TableDefinition<u128, &[u8]> = TableDefinition::new("edges");
const T_VECS: TableDefinition<u128, &[u8]> = TableDefinition::new("vectors");
// Small key→value store for index metadata (e.g. which embedder built the vectors).
const T_META: TableDefinition<&str, &[u8]> = TableDefinition::new("meta");

/// Which embedder produced the stored vectors. Persisted so a later run can't silently mix
/// vectors of a different model/dimension into the index (cosine across mismatched dims is
/// meaningless and would quietly break recall).
#[derive(Serialize, Deserialize)]
struct EmbedderMeta {
    id: String,
    dim: usize,
}

/// One durable mutation. Ordered so a node always precedes edges that reference it.
#[derive(Serialize, Deserialize)]
enum WalRecord {
    Node(Node),
    Edge(Edge),
    Vector(NodeId, Vec<f32>),
}

/// CRC-framed write-ahead log: `[u32 len][u32 crc32][payload]` per record.
struct Wal {
    file: File,
    path: PathBuf,
}

impl Wal {
    fn open(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Wal {
            file,
            path: path.to_path_buf(),
        })
    }

    /// Append a batch and `fsync` once — the ACK boundary.
    fn append_batch(&mut self, recs: &[WalRecord]) -> std::io::Result<()> {
        if recs.is_empty() {
            return Ok(());
        }
        let mut buf = Vec::new();
        for r in recs {
            let payload = serde_json::to_vec(r).expect("serialize wal record");
            buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            buf.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
            buf.extend_from_slice(&payload);
        }
        self.file.write_all(&buf)?;
        self.file.flush()?;
        self.file.sync_data()?; // fsync = durable ACK
        Ok(())
    }

    /// Empty the WAL after a successful checkpoint.
    ///
    /// Reopen with `truncate(true)` rather than `set_len(0)` on the live handle: on Windows
    /// the WAL handle is append-only (`FILE_APPEND_DATA` without `FILE_WRITE_DATA`), and
    /// `SetEndOfFile` on such a handle fails with `Access denied (os error 5)`. Recreating the
    /// file zeroes it portably, then we swap back to an append handle for subsequent records.
    fn truncate(&mut self) -> std::io::Result<()> {
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        f.sync_data()?;
        self.file = OpenOptions::new().create(true).append(true).open(&self.path)?;
        Ok(())
    }

    /// Read every intact record; stop at the first torn/corrupt frame (a partial tail from a
    /// crash mid-append) rather than erroring. A missing WAL is an empty replay, but any
    /// other read error propagates — silently dropping an unreadable-but-present WAL would
    /// lose ACKed-but-uncheckpointed mutations and break the durability contract.
    fn replay(path: &Path) -> std::io::Result<Vec<WalRecord>> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut recs = Vec::new();
        let mut i = 0usize;
        while i + 8 <= bytes.len() {
            let len = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
            let crc = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap());
            let start = i + 8;
            let Some(end) = start.checked_add(len) else { break };
            if end > bytes.len() {
                break; // torn tail
            }
            let payload = &bytes[start..end];
            if crc32fast::hash(payload) != crc {
                break; // corrupt tail
            }
            match serde_json::from_slice::<WalRecord>(payload) {
                Ok(r) => recs.push(r),
                Err(_) => break,
            }
            i = end;
        }
        Ok(recs)
    }
}

pub struct Store {
    dir: PathBuf,
    pub graph: MemGraph,
    pub index: BruteForceIndex,
    embedder: DynEmbedder,
    db: Database,
    wal: Wal,
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn canon(p: &str) -> String {
    Path::new(p)
        .canonicalize()
        .ok()
        .map(|x| x.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string())
}

fn basename(p: &str) -> String {
    p.rsplit(['/', '\\']).next().unwrap_or(p).to_string()
}

/// Encode an embedding as raw little-endian `f32` bytes for the redb vectors table. JSON-text
/// encoding of a 256-float vector means formatting 256 floats to decimal *per vector* — at
/// ~23k vectors that dominated `save()`. Raw bytes make the encode a memcpy and shrink the
/// stored size ~4×.
fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Decode a raw little-endian `f32` byte blob (see [`vec_to_bytes`]). A length that isn't a
/// multiple of 4 means the blob predates the raw-bytes format (was JSON) — surface it rather
/// than return a corrupt vector.
fn bytes_to_vec(b: &[u8]) -> anyhow::Result<Vec<f32>> {
    if b.len() % 4 != 0 {
        anyhow::bail!(
            "vector blob is {} bytes (not a multiple of 4) — this store predates the raw-f32 \
             vector format; re-run `lore index` into a fresh --data-dir",
            b.len()
        );
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Content-addressed edge key (so re-checkpointing the same edge is an idempotent upsert).
fn edge_key(e: &Edge) -> u128 {
    let mut h = blake3::Hasher::new();
    h.update(&e.from.to_le_bytes());
    h.update(&e.to.to_le_bytes());
    h.update(e.kind.tag().as_bytes());
    let mut a = [0u8; 16];
    a.copy_from_slice(&h.finalize().as_bytes()[..16]);
    u128::from_le_bytes(a)
}

/// Repo-scoped `File` identity: the same relative path in two different repos must map to
/// two distinct nodes, or `Touches`/`Contains`/`Defines`/`Produced` edges leak across repos.
/// The display `path` attr stays the bare relative path; only the identity is namespaced.
fn file_identity(repo_ns: &str, rel: &str) -> String {
    format!("{repo_ns}\u{1}{rel}")
}

fn apply(graph: &mut MemGraph, index: &mut BruteForceIndex, rec: WalRecord) {
    match rec {
        WalRecord::Node(n) => graph.upsert_node(n),
        WalRecord::Edge(e) => graph.upsert_edge(e),
        WalRecord::Vector(id, v) => index.upsert(id, v),
    }
}

/// Decision candidates for a session: the deterministic cue extractor by default. With
/// `--features byo-llm` AND `LORE_LLM_*` configured, the candidate-bearing turns are
/// re-distilled by the LLM into richer `{decision, rationale, rejected}` (stamped
/// `byo-llm-v1`); any failure falls back to the deterministic result for that turn, so a
/// missing key / network blip never drops decisions.
fn extract_decisions(s: &CanonicalSession) -> Vec<decision::Candidate> {
    let deterministic = decision::extract(&s.turns);
    #[cfg(feature = "byo-llm")]
    {
        if let Some(cfg) = crate::llm::LlmConfig::from_env() {
            return llm_augment(&cfg, s, deterministic);
        }
    }
    deterministic
}

/// Re-distill each deterministic-candidate turn through the LLM (bounded to those turns to
/// cap API cost). Falls back to the deterministic candidate(s) for a turn on any error/empty.
#[cfg(feature = "byo-llm")]
fn llm_augment(
    cfg: &crate::llm::LlmConfig,
    s: &CanonicalSession,
    deterministic: Vec<decision::Candidate>,
) -> Vec<decision::Candidate> {
    use std::collections::BTreeSet;
    let turn_idxs: BTreeSet<usize> = deterministic.iter().map(|c| c.turn_index).collect();
    let mut out = Vec::new();
    for ti in turn_idxs {
        let det_for_turn = || deterministic.iter().filter(|c| c.turn_index == ti).cloned();
        let Some(turn) = s.turns.get(ti) else {
            out.extend(det_for_turn());
            continue;
        };
        match crate::llm::extract(cfg, &turn.text) {
            Ok(decs) if !decs.is_empty() => {
                for d in decs {
                    out.push(decision::Candidate {
                        turn_index: ti,
                        text: d.decision,
                        confidence: 0.9,
                        rationale: d.rationale,
                        rejected: d.rejected,
                        extractor: "byo-llm-v1",
                    });
                }
            }
            Ok(_) => out.extend(det_for_turn()), // LLM found none → keep the heuristic hit
            Err(e) => {
                tracing::warn!("byo-llm extract failed on turn {ti}: {e}; using deterministic");
                out.extend(det_for_turn());
            }
        }
    }
    out
}

impl Store {
    /// Open a store at `dir`: rebuild from redb, then replay the WAL tail (crash recovery).
    pub fn open(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let db = Database::create(dir.join("graph.redb"))?;

        // The active embedder, selected from the environment (hash default; static = R2).
        let embedder = DynEmbedder::from_env()?;

        // Ensure the tables exist (so the read path below never hits TableDoesNotExist).
        {
            let wtx = db.begin_write()?;
            wtx.open_table(T_NODES)?;
            wtx.open_table(T_EDGES)?;
            wtx.open_table(T_VECS)?;
            wtx.open_table(T_META)?;
            wtx.commit()?;
        }

        let mut graph = MemGraph::new();
        let mut index = BruteForceIndex::new();

        // Load the committed base from redb (nodes before edges, so endpoints exist).
        // Deserialize persisted embedder metadata here but defer the mismatch check until
        // after WAL replay — otherwise WAL-only vectors (not yet in redb) bypass the guard.
        let stored_meta: Option<EmbedderMeta>;
        {
            let rtx = db.begin_read()?;
            let tn = rtx.open_table(T_NODES)?;
            for item in tn.iter()? {
                let (_k, v) = item?;
                let node: Node = serde_json::from_slice(v.value())?;
                graph.upsert_node(node);
            }
            let te = rtx.open_table(T_EDGES)?;
            for item in te.iter()? {
                let (_k, v) = item?;
                let edge: Edge = serde_json::from_slice(v.value())?;
                graph.upsert_edge(edge);
            }
            let tv = rtx.open_table(T_VECS)?;
            for item in tv.iter()? {
                let (k, v) = item?;
                let id: u128 = k.value();
                let vec = bytes_to_vec(v.value())?;
                index.upsert(id, vec);
            }
            let tm = rtx.open_table(T_META)?;
            stored_meta = tm
                .get("embedder")?
                .map(|raw| serde_json::from_slice::<EmbedderMeta>(raw.value()))
                .transpose()?;
        }

        // Replay the WAL tail (records not yet folded into redb). Idempotent.
        let wal_path = dir.join("wal.log");
        for rec in Wal::replay(&wal_path)? {
            apply(&mut graph, &mut index, rec);
        }

        // Guard: refuse to continue with an embedder that disagrees with the one that built
        // the persisted vectors — the query vector would be a different model/dim and recall
        // would silently return garbage. Checked after WAL replay so WAL-only vectors (not yet
        // checkpointed into redb) are also covered. A store without metadata yet has no
        // checkpoint — all its vectors came from the current session's embedder, so it's fine.
        if !index.is_empty() {
            if let Some(ref meta) = stored_meta {
                if meta.id != embedder.id() || meta.dim != embedder.dim() {
                    anyhow::bail!(
                        "this store was indexed with embedder `{}` (dim {}), but the active \
                         embedder is `{}` (dim {}). Re-run `lore index` into a fresh --data-dir, \
                         or set LORE_EMBED_BACKEND/LORE_EMBED_MODEL to match.",
                        meta.id,
                        meta.dim,
                        embedder.id(),
                        embedder.dim()
                    );
                }
            }
        }

        let wal = Wal::open(&wal_path)?;

        Ok(Store {
            dir,
            graph,
            index,
            embedder,
            db,
            wal,
        })
    }

    /// Checkpoint: fold the in-memory state into redb in one ACID txn, then truncate the WAL.
    pub fn save(&mut self) -> anyhow::Result<()> {
        let wtx = self.db.begin_write()?;
        {
            let mut tn = wtx.open_table(T_NODES)?;
            for n in self.graph.nodes.values() {
                tn.insert(n.id, serde_json::to_vec(n)?.as_slice())?;
            }
            let mut te = wtx.open_table(T_EDGES)?;
            for e in &self.graph.edges {
                te.insert(edge_key(e), serde_json::to_vec(e)?.as_slice())?;
            }
            let mut tv = wtx.open_table(T_VECS)?;
            for entry in self.index.entries() {
                tv.insert(entry.id, vec_to_bytes(&entry.vec).as_slice())?;
            }
            // Stamp which embedder built these vectors so a future open can detect a mismatch.
            let mut tmeta = wtx.open_table(T_META)?;
            let meta = EmbedderMeta { id: self.embedder.id(), dim: self.embedder.dim() };
            tmeta.insert("embedder", serde_json::to_vec(&meta)?.as_slice())?;
        }
        wtx.commit()?; // redb fsync — the data is now durably in the cold store
        self.wal.truncate()?; // safe only after the commit above succeeded
        let _ = &self.dir; // dir retained for future per-store artifacts
        Ok(())
    }

    /// Persist a batch durably (WAL append + fsync = ACK), then apply it to memory.
    fn commit_mutations(&mut self, recs: Vec<WalRecord>) -> anyhow::Result<()> {
        self.wal.append_batch(&recs)?;
        for rec in recs {
            apply(&mut self.graph, &mut self.index, rec);
        }
        Ok(())
    }

    /// Embed an arbitrary query string with the store's embedder (used by `ask`/search).
    pub fn embed_query(&self, q: &str) -> Vec<f32> {
        self.embedder.embed(q)
    }

    /// Append-only "note": record a manual `Decision` node (the MCP `memory.note` write path)
    /// and persist it durably. Never deletes or overwrites existing memory — re-noting the
    /// same text under the same topic is an idempotent upsert (content-addressed identity).
    pub fn add_note(&mut self, topic: Option<&str>, text: &str) -> anyhow::Result<NodeId> {
        if text.trim().is_empty() {
            anyhow::bail!("note text must not be empty");
        }
        // Redact secrets BEFORE anything is persisted, embedded, or hashed — the same
        // contract every other ingest path honors (Node.body is always redacted).
        let text = crate::redact::redact(text);
        let now = now_ms();
        let scope = topic.unwrap_or("note");
        let identity = format!("manual-note::{}::{}", scope, blake3::hash(text.as_bytes()).to_hex());
        let label: String = text.chars().take(80).collect();
        let node = Node::new(NodeKind::Decision, &identity, label, now)
            .with_body(text.clone())
            .with_summary(summarize(&text))
            .with_provenance(Provenance {
                source_tool: "mcp".into(),
                extractor: "manual".into(),
                confidence: 1.0,
                source_priority: 3,
                ..Default::default()
            });
        let id = node.id;
        let recs = vec![
            WalRecord::Node(node),
            WalRecord::Vector(id, self.embedder.embed(&text)),
        ];
        // commit_mutations fsyncs the WAL (durable). No full-store checkpoint on the hot
        // note path — that would rewrite redb + truncate the WAL under the shared lock,
        // making note latency scale with total graph size.
        self.commit_mutations(recs)?;
        Ok(id)
    }

    /// Ingest one normalized session: Session/Turn/Decision nodes + the fusion edges.
    pub fn ingest_session(&mut self, s: &CanonicalSession) -> anyhow::Result<()> {
        let now = now_ms();
        let started = s.started_at_ms.unwrap_or(now);
        let repo_canon = s.repo_root.as_deref().map(canon);
        let emb = &self.embedder;
        let mut recs: Vec<WalRecord> = Vec::new();
        // push a node + its embedding together
        let push_node = |recs: &mut Vec<WalRecord>, node: Node, embed_text: &str| {
            let id = node.id;
            recs.push(WalRecord::Node(node));
            if !embed_text.trim().is_empty() {
                recs.push(WalRecord::Vector(id, emb.embed(embed_text)));
            }
            id
        };

        // Session node
        let session_identity = format!("{}:{}", s.agent.tag(), s.native_session_id);
        let intro: String = s
            .turns
            .iter()
            .find(|t| {
                matches!(t.kind, MessageKind::User) && !decision::is_low_value_turn(&t.text)
            })
            .map(|t| t.text.chars().take(200).collect())
            .unwrap_or_default();
        let prov = Provenance {
            source_tool: s.agent.tag().to_string(),
            native_session_id: Some(s.native_session_id.clone()),
            native_path: s.source_path.clone(),
            connector_version: s.connector_version.clone(),
            commit: s.commit.clone(),
            span: None,
            source_priority: 2,
            confidence: 1.0,
            extractor: "deterministic-v1".into(),
        };
        let session_node = Node::new(
            NodeKind::Session,
            &session_identity,
            format!("session {}", s.native_session_id.chars().take(8).collect::<String>()),
            started,
        )
        .with_body(intro.clone())
        .with_summary(summarize(&intro))
        .with_provenance(prov.clone());
        let session_id = push_node(&mut recs, session_node, &intro);

        // Repo node + Session-derived-from-Repo edge
        if let Some(rc) = &repo_canon {
            let repo_node = Node::new(NodeKind::Repo, rc, basename(rc), now).attr("path", rc.clone());
            let repo_id = repo_node.id;
            recs.push(WalRecord::Node(repo_node));
            recs.push(WalRecord::Edge(Edge::new(session_id, repo_id, EdgeKind::DerivedFrom)));
        }

        // Turns
        for t in &s.turns {
            // Skip non-chat turns, empties, and control/command noise (slash-commands,
            // local-command output, interrupts) so they never become recallable Turn nodes.
            if !matches!(t.kind, MessageKind::User | MessageKind::Assistant)
                || decision::is_low_value_turn(&t.text)
            {
                continue;
            }
            let content_hash = blake3::hash(t.text.as_bytes());
            let turn_identity = format!("{}:{}:{}", session_identity, t.index, content_hash.to_hex());
            let label: String = t.text.chars().take(48).collect();
            let turn_node = Node::new(NodeKind::Turn, &turn_identity, label, t.ts_ms.unwrap_or(started))
                .with_body(t.text.clone())
                .with_summary(summarize(&t.text))
                .with_provenance(Provenance {
                    span: Some((t.index, t.index)),
                    ..prov.clone()
                });
            let turn_id = push_node(&mut recs, turn_node, &t.text);
            recs.push(WalRecord::Edge(Edge::new(session_id, turn_id, EdgeKind::Contains)));

            for tp in &t.touched_files {
                let rel = rel_path(repo_canon.as_deref(), tp);
                let fid = file_identity(repo_canon.as_deref().unwrap_or(""), &rel);
                let file_node = Node::new(NodeKind::File, &fid, basename(&rel), now).attr("path", rel.clone());
                let file_id = file_node.id;
                recs.push(WalRecord::Node(file_node));
                recs.push(WalRecord::Edge(Edge::new(turn_id, file_id, EdgeKind::Touches)));
            }
        }

        // Decisions: deterministic cue extractor by default; the byo-llm extractor distills
        // richer {decision, rationale, rejected} when configured (T3, feature-gated).
        for cand in extract_decisions(s) {
            // Redact BEFORE persist/hash — the byo-llm extractor's output is not covered by
            // the ingest-time redaction (only turn text is), so a model could echo a token.
            // Idempotent for the deterministic path (its text is already-redacted turn text).
            let text = crate::redact::redact(&cand.text);
            let rationale = crate::redact::redact(&cand.rationale);
            let rejected: Vec<String> = cand.rejected.iter().map(|r| crate::redact::redact(r)).collect();

            let scope = repo_canon.clone().unwrap_or_else(|| session_identity.clone());
            let dec_identity = format!("{}::{}", scope, text.to_lowercase());
            // Keep the rationale in the body (drill-down); the summary stays the decision itself.
            let body = if rationale.is_empty() {
                text.clone()
            } else {
                format!("{text}\n\nRationale: {rationale}")
            };
            let mut dec_node = Node::new(NodeKind::Decision, &dec_identity, text.clone(), started)
                .with_body(body)
                .with_summary(summarize(&text))
                .with_provenance(Provenance {
                    confidence: cand.confidence,
                    extractor: cand.extractor.to_string(),
                    ..prov.clone()
                });
            if !rejected.is_empty() {
                dec_node = dec_node.attr("rejected", rejected.join("; "));
            }
            let dec_id = push_node(&mut recs, dec_node, &text);
            recs.push(WalRecord::Edge(Edge::new(dec_id, session_id, EdgeKind::DecidedIn)));

            if let Some(turn) = s.turns.get(cand.turn_index) {
                for tp in &turn.touched_files {
                    let rel = rel_path(repo_canon.as_deref(), tp);
                    // re-assert the File node so the Produced edge always has both endpoints
                    let fid = file_identity(repo_canon.as_deref().unwrap_or(""), &rel);
                    let file_node = Node::new(NodeKind::File, &fid, basename(&rel), now).attr("path", rel.clone());
                    let file_id = file_node.id;
                    recs.push(WalRecord::Node(file_node));
                    recs.push(WalRecord::Edge(Edge::new(dec_id, file_id, EdgeKind::Produced)));
                }
            }
        }

        self.commit_mutations(recs)
    }

    /// Crunch a repo into Repo/File/Symbol/Framework nodes + edges.
    pub fn ingest_repo(&mut self, root: &Path) -> anyhow::Result<()> {
        let now = now_ms();
        let scan = coderepo::scan(root)?;
        let root_canon = canon(&root.to_string_lossy());
        let emb = &self.embedder;
        let mut recs: Vec<WalRecord> = Vec::new();

        // Git history → the commit axis of the provenance wedge (best-effort; None if no repo).
        let git = git::mine(root, 400);
        // The complete on-disk file set (not just symbol-bearing scan.files) so commits
        // touching non-symbol files (e.g. .md, .toml, .json) still generate ChangedBy edges.
        // Reuses scan's single skip-filtered walk — a second raw `walkdir` here re-descended
        // `target/`/`.git/`/`node_modules/` and dominated repo-crunch time. Paths are
        // root-relative, forward-slashed, matching scan.files keys and the git paths below.
        let scanned: std::collections::HashSet<String> = scan.all_files.iter().cloned().collect();
        // Prefix to strip from git repo-root-relative paths when `root` is a sub-directory
        // of the repo (e.g. `--repo ./crates/foo` → strip "crates/foo/" from git paths).
        let git_path_prefix: String = git
            .as_ref()
            .and_then(|g| g.workdir.as_ref())
            .and_then(|wd| root.canonicalize().ok().map(|cr| (wd.clone(), cr)))
            .and_then(|(wd, cr)| cr.strip_prefix(&wd).ok().map(|p| p.to_path_buf()))
            .map(|rel| {
                let s = rel.to_string_lossy().replace('\\', "/");
                if s.is_empty() { s } else { format!("{s}/") }
            })
            .unwrap_or_default();

        let mut repo_node =
            Node::new(NodeKind::Repo, &root_canon, scan.repo_label.clone(), now).attr("path", root_canon.clone());
        if let Some(head) = git.as_ref().and_then(|g| g.head_short()) {
            repo_node = repo_node.attr("commit", head);
        }
        let repo_id = repo_node.id;
        recs.push(WalRecord::Node(repo_node));

        for fw in &scan.frameworks {
            let fw_node = Node::new(NodeKind::Framework, fw, fw.clone(), now);
            let fw_id = fw_node.id;
            recs.push(WalRecord::Node(fw_node));
            recs.push(WalRecord::Edge(Edge::new(repo_id, fw_id, EdgeKind::DependsOn)));
        }

        for (rel, syms) in &scan.files {
            let fid = file_identity(&root_canon, rel);
            let file_node = Node::new(NodeKind::File, &fid, basename(rel), now).attr("path", rel.clone());
            let file_id = file_node.id;
            recs.push(WalRecord::Node(file_node));
            recs.push(WalRecord::Edge(Edge::new(repo_id, file_id, EdgeKind::Contains)));
            let sym_names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
            recs.push(WalRecord::Vector(file_id, emb.embed(&format!("{} {}", rel, sym_names.join(" ")))));

            for sym in syms {
                let sym_identity = format!("{}\u{1}{}\u{1}{}", fid, sym.kind, sym.name);
                let sym_node = Node::new(NodeKind::Symbol, &sym_identity, sym.name.clone(), now)
                    .attr("kind", sym.kind)
                    .attr("file", rel.clone());
                let sym_id = sym_node.id;
                recs.push(WalRecord::Node(sym_node));
                recs.push(WalRecord::Edge(Edge::new(file_id, sym_id, EdgeKind::Defines)));
                recs.push(WalRecord::Vector(sym_id, emb.embed(&sym.name)));
            }
        }

        // Commit nodes (citable provenance) + File-ChangedBy-Commit fusion edges. Each commit
        // carries its own oid as provenance so a recalled commit is itself citable.
        if let Some(g) = &git {
            for c in &g.commits {
                let summary = crate::redact::redact(&c.summary);
                let author = crate::redact::redact(&c.author);
                let commit_node = Node::new(NodeKind::Commit, &c.oid, c.short.clone(), c.time_ms)
                    .with_body(summary)
                    .attr("oid", c.oid.clone())
                    .attr("author", author)
                    .with_provenance(Provenance {
                        source_tool: "git".into(),
                        commit: Some(c.oid.clone()),
                        source_priority: 3,
                        confidence: 1.0,
                        extractor: "repo-git-v1".into(),
                        ..Default::default()
                    });
                let commit_id = commit_node.id;
                recs.push(WalRecord::Node(commit_node));
                recs.push(WalRecord::Edge(Edge::new(repo_id, commit_id, EdgeKind::Contains)));
                // Commits are provenance anchors, not recallable "memories" — recall ranks
                // over Turn/Decision/Implementation (PLAN.md §5.1), so they are NOT embedded
                // into the search index (else a busy history floods every query).

                for rel in c.changed.iter().filter_map(|git_path| {
                    let p = git_path.strip_prefix(&git_path_prefix).unwrap_or(git_path);
                    scanned.contains(p).then(|| p.to_string())
                }) {
                    let fid = file_identity(&root_canon, &rel);
                    // Re-assert the File node so the ChangedBy edge always has both endpoints
                    // (a commit may touch a file with no extracted symbols, hence no scan node).
                    let file_node = Node::new(NodeKind::File, &fid, basename(&rel), now).attr("path", rel.clone());
                    let file_id = file_node.id;
                    recs.push(WalRecord::Node(file_node));
                    recs.push(WalRecord::Edge(Edge::new(file_id, commit_id, EdgeKind::ChangedBy)));
                }
            }
        }

        self.commit_mutations(recs)
    }
}

/// Make a touched-file path repo-relative against the session's repo root when possible, so
/// it shares identity with the repo-crunch File nodes (enabling the fusion edges).
fn rel_path(repo_canon: Option<&str>, touched: &str) -> String {
    let tc = canon(touched);
    if let Some(rr) = repo_canon {
        if let Some(stripped) = tc.strip_prefix(rr) {
            return stripped.trim_start_matches(['/', '\\']).to_string();
        }
        if let Some(stripped) = touched.strip_prefix(rr) {
            return stripped.trim_start_matches(['/', '\\']).to_string();
        }
    }
    touched.replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AgentKind, CanonicalSession, MessageKind, Turn};

    #[test]
    fn vec_codec_roundtrips_and_rejects_legacy() {
        let v = vec![0.0f32, 1.5, -2.25, f32::MIN_POSITIVE, 3.14159];
        let bytes = vec_to_bytes(&v);
        assert_eq!(bytes.len(), v.len() * 4);
        assert_eq!(bytes_to_vec(&bytes).unwrap(), v);
        // A legacy JSON blob ("[0.0,1.5]") has a length not divisible by 4 here → rejected,
        // not silently mis-decoded into garbage floats.
        assert!(bytes_to_vec(b"[0.0,1.5]").is_err());
    }

    fn turn(i: usize, k: MessageKind, text: &str, files: &[&str]) -> Turn {
        Turn {
            index: i,
            kind: k,
            text: text.to_string(),
            ts_ms: Some(1000 + i as i64),
            touched_files: files.iter().map(|s| s.to_string()).collect(),
            raw: serde_json::Value::Null,
        }
    }

    fn demo_session() -> CanonicalSession {
        CanonicalSession {
            native_session_id: "s-123".into(),
            agent: AgentKind::ClaudeCode,
            repo_root: None,
            commit: Some("abc123".into()),
            started_at_ms: Some(1000),
            source_path: Some("/x/s-123.jsonl".into()),
            connector_version: Some("2.1.195".into()),
            turns: vec![
                turn(0, MessageKind::User, "what store should we use", &[]),
                turn(1, MessageKind::Assistant, "Let's use redb for the durable store.", &["src/store.rs"]),
            ],
        }
    }

    #[test]
    fn checkpoint_then_reopen_from_redb() {
        let dir = tempfile::tempdir().unwrap();
        let count;
        {
            let mut store = Store::open(dir.path()).unwrap();
            store.ingest_session(&demo_session()).unwrap();
            store.save().unwrap(); // checkpoint → redb, WAL truncated
            count = store.graph.node_count();
            assert!(count >= 5);
            // WAL is empty after checkpoint
            assert_eq!(std::fs::metadata(dir.path().join("wal.log")).unwrap().len(), 0);
        }
        let reopened = Store::open(dir.path()).unwrap();
        assert_eq!(reopened.graph.node_count(), count, "loads from redb");
        assert!(!reopened.index.is_empty());
    }

    #[test]
    fn wal_recovers_ingest_without_checkpoint() {
        // Simulates a crash: data is ingested (WAL fsynced) but save() is never called.
        let dir = tempfile::tempdir().unwrap();
        let count;
        {
            let mut store = Store::open(dir.path()).unwrap();
            store.ingest_session(&demo_session()).unwrap();
            count = store.graph.node_count();
            assert!(std::fs::metadata(dir.path().join("wal.log")).unwrap().len() > 0);
            // no save() — drop as if the process died
        }
        let recovered = Store::open(dir.path()).unwrap();
        assert_eq!(recovered.graph.node_count(), count, "WAL replay recovers everything");
        let hits = recovered
            .index
            .search(&recovered.embed_query("durable store redb"), 3);
        assert!(!hits.is_empty(), "embeddings recovered too");
    }

    #[test]
    fn torn_wal_tail_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut store = Store::open(dir.path()).unwrap();
            store.ingest_session(&demo_session()).unwrap();
        }
        // append a garbage partial frame (a crash mid-append) to the WAL
        {
            use std::io::Write;
            let mut f = OpenOptions::new().append(true).open(dir.path().join("wal.log")).unwrap();
            f.write_all(&999u32.to_le_bytes()).unwrap(); // claims 999 bytes that aren't there
            f.write_all(&0u32.to_le_bytes()).unwrap();
            f.write_all(b"partial").unwrap();
        }
        // open still succeeds; the intact records are recovered, the torn tail dropped.
        let store = Store::open(dir.path()).unwrap();
        assert!(store.graph.node_count() >= 5);
    }
}
