//! loregraph (`lore`) — a memory knowledge-graph for AI coding-agent sessions + repos.
//!
//! Reads the chat transcripts your agents already write to disk (Claude Code today; Codex /
//! aider / Cursor / OpenRouter in later phases), fuses them with your repo, and builds a
//! persistent graph of **decisions and implementations** linked by hard provenance edges
//! (`session_id` + `commit`) back to the exact turn and code that produced them — browsable
//! on a canvas and queryable read-only by the agents themselves over MCP, so nobody
//! re-asks the model what was already decided.
//!
//! Status: PoC. See `PLAN.md` for the phased plan and `ARCHITECTURE.md` for the engine.
//!
//! The engine is built from swappable seams, each with a pure-Rust default and a
//! feature-gated heavy impl (the recall/recall pattern):
//! [`ingest::SessionSource`] · [`embed::Embedder`] · [`index::BruteForceIndex`] ·
//! [`graph::MemGraph`] · [`store::Store`].

pub mod api;
pub mod ask;
pub mod coderepo;
pub mod decision;
pub mod embed;
pub mod git;
pub mod graph;
pub mod index;
pub mod ingest;
/// BYO-key LLM decision extractor (T3). The pure request-build / response-parse logic always
/// compiles + is tested; the network `extract()` is gated behind the `byo-llm` feature.
pub mod llm;
pub mod mcp;
pub mod model;
pub mod redact;
pub mod store;
pub mod ui;

/// Crate version, surfaced by `lore version`.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Initialize tracing once (honors `RUST_LOG`; defaults to `info`).
pub fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = fmt().with_env_filter(filter).with_target(false).try_init();
}
