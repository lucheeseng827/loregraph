//! Ingest seam — one [`SessionSource`] per agent on-disk format.
//!
//! Every connector lowers a vendor's volatile on-disk shape into the [`CanonicalSession`]
//! IR (PLAN.md §5). The PoC ships the Claude Code connector; Codex / aider / Cursor /
//! OpenRouter land in Beta, each behind a feature and gated by the producer's version
//! string (the formats are undocumented and drift). The cardinal rule: **lenient +
//! lossless, never hard-fail** — an unrecognized record becomes a `MessageKind::Unknown`
//! turn, never a parse error that drops the session.

pub mod claude_code;
pub mod codex;

use std::path::{Path, PathBuf};

use crate::model::CanonicalSession;

/// A connector over one agent's transcript format.
pub trait SessionSource {
    /// Stable connector id (`claude_code`, `codex`, …).
    fn id(&self) -> &'static str;

    /// Find candidate transcript files for this source on the local machine.
    fn discover(&self) -> anyhow::Result<Vec<PathBuf>>;

    /// Parse one transcript file into a normalized session (secrets already redacted).
    fn read_session(&self, path: &Path) -> anyhow::Result<CanonicalSession>;
}

/// The connectors available in this build. Codex/aider/Cursor/OpenRouter join here behind
/// their cargo features as they land.
pub fn registry() -> Vec<Box<dyn SessionSource>> {
    vec![
        Box::new(claude_code::ClaudeCode::default()),
        Box::new(codex::Codex::default()),
    ]
}

/// Look up a connector by id.
pub fn by_id(id: &str) -> Option<Box<dyn SessionSource>> {
    registry().into_iter().find(|s| s.id() == id)
}
