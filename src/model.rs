//! The memory-graph data model + the normalized chat IR.
//!
//! Two layers:
//! 1. The **graph** ([`Node`]/[`Edge`]) — the durable artifact. Node ids are
//!    **content-addressed** (`blake3(kind ‖ identity) → u128`) so re-ingesting the same
//!    session/file/decision is an idempotent no-op upsert (PLAN.md §2).
//! 2. The **normalized chat IR** ([`CanonicalSession`]/[`Turn`]) — the volatile-format
//!    insulation layer. Every connector lowers its own on-disk shape into this IR so the
//!    graph never sees a vendor's JSONL directly (PLAN.md §3, §5).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Content-addressed node id (first 16 bytes of a blake3 hash).
pub type NodeId = u128;

/// What a node *is*. Field-less so it casts to a stable discriminant for hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    // chat atoms
    Session,
    Turn,
    // the value nodes (provenance-anchored)
    Decision,
    Implementation,
    // advisory (heuristic; carry confidence + evidence)
    Pattern,
    DebtSignal,
    // code graph
    Repo,
    Module,
    File,
    Symbol,
    // dependency graph (precise)
    Manifest,
    Package,
    // git history
    Commit,
    // context
    Framework,
    Tooling,
    DataSource,
    DesignDoc,
    Concept,
    Person,
}

impl NodeKind {
    /// Stable lowercase tag used in URLs / filters / the wire JSON.
    pub fn tag(self) -> &'static str {
        use NodeKind::*;
        match self {
            Session => "session",
            Turn => "turn",
            Decision => "decision",
            Implementation => "implementation",
            Pattern => "pattern",
            DebtSignal => "debt_signal",
            Repo => "repo",
            Module => "module",
            File => "file",
            Symbol => "symbol",
            Manifest => "manifest",
            Package => "package",
            Commit => "commit",
            Framework => "framework",
            Tooling => "tooling",
            DataSource => "data_source",
            DesignDoc => "design_doc",
            Concept => "concept",
            Person => "person",
        }
    }
}

/// How two nodes relate. `Touches`/`Produced` are the chat↔code fusion edges — the core
/// product promise (PLAN.md §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    // chat ↔ decision
    DecidedIn,
    Implements,
    Supersedes,
    Mentions,
    DerivedFrom,
    // fusion (repo ↔ chat)
    Touches,
    Produced,
    // code graph
    Contains,
    Defines,
    Imports,
    DependsOn,
    References,
    ChangedBy,
    CoChangesWith,
    Exhibits,
    Uses,
    RelatesTo,
}

impl EdgeKind {
    pub fn tag(self) -> &'static str {
        use EdgeKind::*;
        match self {
            DecidedIn => "decided_in",
            Implements => "implements",
            Supersedes => "supersedes",
            Mentions => "mentions",
            DerivedFrom => "derived_from",
            Touches => "touches",
            Produced => "produced",
            Contains => "contains",
            Defines => "defines",
            Imports => "imports",
            DependsOn => "depends_on",
            References => "references",
            ChangedBy => "changed_by",
            CoChangesWith => "co_changes_with",
            Exhibits => "exhibits",
            Uses => "uses",
            RelatesTo => "relates_to",
        }
    }
}

/// Where a node came from — what lets an agent *cite* a recalled memory (PLAN.md §2).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Provenance {
    pub source_tool: String,
    pub native_session_id: Option<String>,
    pub native_path: Option<String>,
    pub connector_version: Option<String>,
    pub commit: Option<String>,
    pub span: Option<(usize, usize)>,
    /// native > proxy > export — used to keep the richest copy on cross-source dedup.
    pub source_priority: u8,
    pub confidence: f32,
    /// `deterministic-v1` | `byo-llm-v1` | `manual` | `repo-heuristic-v1`.
    pub extractor: String,
}

/// A graph node. `first_seen_ms` is immutable (stamped once at ingest, honored across
/// re-ingest/dedup) and drives the canvas time-scrub; `last_updated_ms` moves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    pub label: String,
    /// Full searchable text (already secret-redacted before it reaches here). The drill-down
    /// detail — fetched on demand via `/v1/node/{id}`, not returned in bulk by recall.
    pub body: String,
    /// A short, single-line distillation of `body` (T2, PLAN.md §5.3). What recall and the
    /// canvas return per hit, so a result is a few tokens, not a paragraph; empty for code
    /// atoms whose `label` already is the summary. `#[serde(default)]` keeps stores written
    /// before this field readable.
    #[serde(default)]
    pub summary: String,
    pub provenance: Option<Provenance>,
    pub attrs: BTreeMap<String, String>,
    pub first_seen_ms: i64,
    pub last_updated_ms: i64,
}

impl Node {
    pub fn new(kind: NodeKind, identity: &str, label: impl Into<String>, now_ms: i64) -> Self {
        Node {
            id: node_id(kind, identity),
            kind,
            label: label.into(),
            body: String::new(),
            summary: String::new(),
            provenance: None,
            attrs: BTreeMap::new(),
            first_seen_ms: now_ms,
            last_updated_ms: now_ms,
        }
    }
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }
    /// Set the short single-line summary (T2). Pass through [`summarize`] for the standard
    /// distillation, or supply a pre-distilled string (e.g. a decision's lifted sentence).
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = summary.into();
        self
    }
    pub fn with_provenance(mut self, p: Provenance) -> Self {
        self.provenance = Some(p);
        self
    }
    pub fn attr(mut self, k: &str, v: impl Into<String>) -> Self {
        self.attrs.insert(k.to_string(), v.into());
        self
    }
}

/// A typed, weighted edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub kind: EdgeKind,
    pub weight: f32,
    pub attrs: BTreeMap<String, String>,
}

impl Edge {
    pub fn new(from: NodeId, to: NodeId, kind: EdgeKind) -> Self {
        Edge { from, to, kind, weight: 1.0, attrs: BTreeMap::new() }
    }
}

/// Distill `body` into a short single-line summary (T2, PLAN.md §5.3): collapse all
/// whitespace to single spaces, take up to the first sentence boundary, and cap the length so
/// a recalled memory is a gist (a few tokens) rather than a paragraph. The full text stays in
/// `Node::body` for drill-down.
pub fn summarize(body: &str) -> String {
    const MAX: usize = 180;
    let flat: String = body.split_whitespace().collect::<Vec<_>>().join(" ");
    // First sentence end that falls within the cap.
    let sentence_end = flat
        .char_indices()
        .take(MAX)
        .find(|(_, c)| matches!(c, '.' | '!' | '?'))
        .map(|(i, c)| i + c.len_utf8());
    if let Some(e) = sentence_end {
        return flat[..e].to_string();
    }
    if flat.chars().count() > MAX {
        let truncated: String = flat.chars().take(MAX).collect();
        format!("{truncated}…")
    } else {
        flat
    }
}

/// Content-addressed id: `blake3(kind_discriminant ‖ identity)[..16]` → `u128`.
/// Same `(kind, identity)` always yields the same id, so re-ingest dedups for free.
pub fn node_id(kind: NodeKind, identity: &str) -> NodeId {
    let mut h = blake3::Hasher::new();
    // Hash the STABLE string tag, not `kind as u8` — the enum discriminant shifts when
    // variants are reordered/inserted, which would split a node's id across upgrades.
    h.update(kind.tag().as_bytes());
    h.update(&[0u8]); // separator so tag/identity can't run together
    h.update(identity.as_bytes());
    let bytes = h.finalize();
    let mut arr = [0u8; 16];
    arr.copy_from_slice(&bytes.as_bytes()[..16]);
    u128::from_le_bytes(arr)
}

// ---------------------------------------------------------------------------
// Normalized chat IR (the volatile-format insulation layer)
// ---------------------------------------------------------------------------

/// Which agent produced a session. `Unknown` is load-bearing: connectors must never
/// hard-fail on an unrecognized tool/record (PLAN.md §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentKind {
    ClaudeCode,
    Codex,
    Aider,
    Cursor,
    OpenRouter,
    Otlp,
    Manual,
    Unknown,
}

impl AgentKind {
    pub fn tag(self) -> &'static str {
        match self {
            AgentKind::ClaudeCode => "claude_code",
            AgentKind::Codex => "codex",
            AgentKind::Aider => "aider",
            AgentKind::Cursor => "cursor",
            AgentKind::OpenRouter => "openrouter",
            AgentKind::Otlp => "otlp",
            AgentKind::Manual => "manual",
            AgentKind::Unknown => "unknown",
        }
    }
}

/// Role of a single turn. `Unknown` is the fallback for any unrecognized record type
/// (Claude Code also emits `queue-operation`/`attachment`/`last-prompt`/… — PLAN.md §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageKind {
    User,
    Assistant,
    System,
    Summary,
    Thinking,
    Unknown,
}

/// One normalized turn. `raw` is the lossless original record — retained in memory for
/// format-drift insurance, but never persisted (it may contain unredacted secrets).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub index: usize,
    pub kind: MessageKind,
    /// Secret-redacted text.
    pub text: String,
    pub ts_ms: Option<i64>,
    /// Repo-relative paths this turn edited (parsed from tool_use effects).
    pub touched_files: Vec<String>,
    #[serde(skip)]
    pub raw: serde_json::Value,
}

/// A normalized session — the unit a [`crate::ingest::SessionSource`] produces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalSession {
    pub native_session_id: String,
    pub agent: AgentKind,
    pub repo_root: Option<String>,
    pub commit: Option<String>,
    pub started_at_ms: Option<i64>,
    pub source_path: Option<String>,
    pub connector_version: Option<String>,
    pub turns: Vec<Turn>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_collapses_caps_and_takes_first_sentence() {
        // collapses whitespace + stops at the first sentence boundary
        assert_eq!(summarize("  Use   redb.\nIt is also crash-safe."), "Use redb.");
        // no sentence end within the cap → hard cap with an ellipsis
        let long = "word ".repeat(100);
        let s = summarize(&long);
        assert!(s.chars().count() <= 181, "capped near 180 chars");
        assert!(s.ends_with('…'), "marked as truncated");
        // empty stays empty
        assert_eq!(summarize(""), "");
    }

    #[test]
    fn ids_are_stable_and_kind_separated() {
        let a = node_id(NodeKind::Decision, "storage engine");
        let b = node_id(NodeKind::Decision, "storage engine");
        let c = node_id(NodeKind::Concept, "storage engine");
        assert_eq!(a, b, "same (kind, identity) → same id (idempotent upsert)");
        assert_ne!(a, c, "kind participates in the hash");
    }
}
