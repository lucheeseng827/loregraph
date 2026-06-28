//! Git history mining (pure-Rust via `gix`) — the **commit** half of the provenance wedge.
//!
//! loregraph's headline is hard provenance: every value node cites a `session_id` **and**
//! the repo `commit` it was contemporaneous with (PLAN.md §2/§4). This module turns a
//! repo's git history into that commit axis:
//!
//! - the HEAD oid, so the repo node and freshly-ingested sessions can be stamped with the
//!   commit they were indexed against;
//! - a bounded, newest-first history (`oid` / `summary` / `author` / commit-time, plus the
//!   files each commit changed vs its first parent) → `Commit` nodes + `File`-`ChangedBy`
//!   edges in the graph;
//! - [`History::commit_at`], which maps a chat timestamp to the commit that was HEAD when
//!   the chat happened — the *contemporaneous* commit a `Decision`/`Turn` is anchored to.
//!
//! Everything here is **best-effort**: a missing/bare/corrupt repo yields `None`, never an
//! error. The air-gap default must survive being pointed at a plain directory with no
//! `.git`. gix is `default-features = false` (no network, no C) — the line matches
//! the merge gate.

use std::ops::ControlFlow;
use std::path::Path;

use gix::bstr::ByteSlice;

/// One commit, flattened to the fields the graph cites.
#[derive(Debug, Clone)]
pub struct CommitMeta {
    /// Full hex oid (the citable commit).
    pub oid: String,
    /// First 8 hex chars (display / node label).
    pub short: String,
    /// First line of the commit message.
    pub summary: String,
    /// Author name (best-effort; empty if unreadable).
    pub author: String,
    /// Commit time in epoch millis (drives the time axis + contemporaneous lookup).
    pub time_ms: i64,
    /// Repo-relative paths changed vs the first parent, forward-slashed. Empty when mined
    /// without diffs (see [`timeline`]) or for an unreadable diff.
    pub changed: Vec<String>,
}

/// A repo's mined git history, **sorted newest commit first** by commit time.
pub struct History {
    /// Full HEAD oid, if the repo has one.
    pub head: Option<String>,
    /// Absolute path to the working-tree root (None only when construction is bypassed in
    /// tests; production paths always set this from the discovered repo).
    pub workdir: Option<std::path::PathBuf>,
    pub commits: Vec<CommitMeta>,
}

impl History {
    pub fn is_empty(&self) -> bool {
        self.commits.is_empty()
    }

    /// The commit that was HEAD when `ts_ms` happened: the newest commit whose time is at or
    /// before `ts_ms`. Falls back to the oldest known commit when the chat predates all of
    /// them (so an early session still cites *a* real commit rather than nothing).
    pub fn commit_at(&self, ts_ms: i64) -> Option<&CommitMeta> {
        self.commits
            .iter()
            .find(|c| c.time_ms <= ts_ms)
            .or_else(|| self.commits.last())
    }

    /// Short HEAD oid (8 chars), if any — the "indexed against" commit.
    pub fn head_short(&self) -> Option<String> {
        self.head.as_ref().map(|h| h.chars().take(8).collect())
    }
}

/// Mine up to `cap` recent commits **with** their changed-file lists (diffs). Use for the
/// repo graph (`Commit` nodes + `ChangedBy` edges). `None` if there is no readable git repo.
pub fn mine(root: &Path, cap: usize) -> Option<History> {
    run(root, cap, true)
}

/// Mine up to `cap` recent commits **without** diffs — just `oid` + time, for the cheap
/// contemporaneous-commit lookup when stamping sessions. `None` if no readable git repo.
pub fn timeline(root: &Path, cap: usize) -> Option<History> {
    run(root, cap, false)
}

fn run(root: &Path, cap: usize, with_diffs: bool) -> Option<History> {
    match mine_inner(root, cap, with_diffs) {
        Ok(h) => Some(h),
        Err(e) => {
            tracing::debug!("git mining skipped for {}: {e}", root.display());
            None
        }
    }
}

fn mine_inner(root: &Path, cap: usize, with_diffs: bool) -> anyhow::Result<History> {
    // `discover` walks up to the enclosing repo, so `lore index --repo .` works from any
    // subdirectory (the common case), not only the repo root.
    let repo = gix::discover(root)?;

    // Bare repos have no working tree — nothing to anchor to a scanned file set.
    if repo.work_dir().is_none() {
        anyhow::bail!("bare repository at {}; skipping git mining", root.display());
    }
    let workdir = repo.work_dir().map(|p| p.to_path_buf());

    let head_id = repo.head_id().ok().map(|id| id.detach());
    let head = head_id.map(|o| o.to_hex().to_string());

    let mut commits = Vec::new();
    if let Some(start) = head_id {
        for info in repo.rev_walk([start]).all()?.take(cap) {
            let info = info?;
            let commit = repo.find_commit(info.id)?;

            let oid = info.id.to_hex().to_string();
            let short: String = oid.chars().take(8).collect();
            let summary = commit
                .message_raw()
                .ok()
                .and_then(|m| m.lines().next().map(|l| String::from_utf8_lossy(l).trim().to_string()))
                .unwrap_or_default();
            let author = commit
                .author()
                .map(|a| a.name.to_str_lossy().into_owned())
                .unwrap_or_default();
            let time_ms = commit.time().map(|t| t.seconds * 1000).unwrap_or(0);
            let changed = if with_diffs {
                changed_files(&repo, &commit).unwrap_or_default()
            } else {
                Vec::new()
            };

            commits.push(CommitMeta { oid, short, summary, author, time_ms, changed });
        }
    }

    // rev-walk order is not guaranteed to be time-sorted; sort so `commit_at` is correct.
    commits.sort_by(|a, b| b.time_ms.cmp(&a.time_ms));
    Ok(History { head, workdir, commits })
}

/// Repo-relative paths a commit changed vs its first parent (a root commit diffs vs the
/// empty tree). Forward-slashed. Best-effort: a diff error yields an empty list.
fn changed_files(repo: &gix::Repository, commit: &gix::Commit<'_>) -> anyhow::Result<Vec<String>> {
    use gix::object::tree::diff::Change;

    let this_tree = commit.tree()?;
    let parent_tree = match commit.parent_ids().next() {
        Some(pid) => repo.find_commit(pid.detach())?.tree()?,
        None => repo.empty_tree(),
    };

    let mut out: Vec<String> = Vec::new();
    let mut platform = parent_tree.changes()?;
    platform.options(|opts| {
        opts.track_path();
    });
    platform.for_each_to_obtain_tree(&this_tree, |change| {
        let loc = match change {
            Change::Addition { location, .. }
            | Change::Modification { location, .. }
            | Change::Deletion { location, .. }
            | Change::Rewrite { location, .. } => location,
        };
        out.push(loc.to_str_lossy().replace('\\', "/"));
        Ok::<_, std::convert::Infallible>(ControlFlow::Continue(()))
    })?;

    out.sort();
    out.dedup();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cm(oid: &str, time_ms: i64) -> CommitMeta {
        CommitMeta {
            oid: oid.into(),
            short: oid.chars().take(8).collect(),
            summary: String::new(),
            author: String::new(),
            time_ms,
            changed: Vec::new(),
        }
    }

    #[test]
    fn commit_at_picks_newest_at_or_before() {
        // Newest-first, as `run` sorts them.
        let h = History {
            head: Some("c".into()),
            workdir: None,
            commits: vec![cm("c", 3000), cm("b", 2000), cm("a", 1000)],
        };
        assert_eq!(h.commit_at(2500).unwrap().oid, "b", "contemporaneous = newest ≤ ts");
        assert_eq!(h.commit_at(3000).unwrap().oid, "c", "exact time matches");
        assert_eq!(h.commit_at(5000).unwrap().oid, "c", "after HEAD → HEAD");
        assert_eq!(h.commit_at(500).unwrap().oid, "a", "before all → oldest known");
    }
}
