//! MCP memory server (the toil-reduction loop) — PLAN.md §5.
//!
//! The whole point: let the coding agent query the memory graph itself, over MCP, instead
//! of the user re-prompting the model. Built on the official `rmcp` SDK behind the `mcp`
//! cargo feature; **stdio** transport (local = trust, no token, air-gap-safe). The hosted
//! Streamable-HTTP transport is the commercial plane.
//!
//! Without the feature, the default build stays pure-Rust and `lore mcp` just prints the
//! tool contract (so the binary still documents the surface).

/// The read/write tools the MCP server exposes (the contract of record).
pub const TOOLS: &[(&str, &str)] = &[
    ("memory.search", "free-text recall over the memory graph → ranked evidence + provenance"),
    ("memory.recall_decision", "decisions about a topic, with rationale + the session/commit they were made in"),
    ("memory.related", "memories related to a file/symbol/topic (graph neighborhood)"),
    ("memory.timeline", "recent memory, newest first by when it was first seen"),
    ("memory.note", "append-only: record a new decision/note (never deletes; human-curated in the canvas)"),
];

/// Entrypoint for `lore mcp` when built WITHOUT the `mcp` feature: explain the contract and
/// how to query the same memory headlessly today.
pub fn explain() {
    println!("loregraph MCP server — build with `--features mcp` to run it over stdio (rmcp).");
    println!("Tools the agent calls instead of re-prompting the model:");
    for (name, desc) in TOOLS {
        println!("  • {name:<24} {desc}");
    }
    println!();
    println!("Today, without the feature, query the same memory headlessly:");
    println!("  lore ask \"what did we decide about <topic>\"");
    println!("See PLAN.md §5 and ARCHITECTURE.md for the MCP surface + transports.");
}

// ---------------------------------------------------------------------------
// The real rmcp stdio server (feature `mcp`).
// ---------------------------------------------------------------------------

/// Open the store at `data_dir` and serve the MCP memory tools over stdio until the client
/// disconnects. stdout is the JSON-RPC channel, so we deliberately do NOT install a
/// stdout tracing subscriber here.
#[cfg(feature = "mcp")]
pub fn serve_stdio(data_dir: &std::path::Path) -> anyhow::Result<()> {
    use rmcp::ServiceExt;

    let store = crate::store::Store::open(data_dir)?;
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move {
        let service = server::MemoryServer::new(store)
            .serve(rmcp::transport::stdio())
            .await?;
        service.waiting().await?;
        Ok::<(), anyhow::Error>(())
    })
}

#[cfg(feature = "mcp")]
mod server {
    use std::sync::{Arc, Mutex};

    use rmcp::handler::server::router::tool::ToolRouter;
    use rmcp::handler::server::wrapper::Parameters;
    use rmcp::model::*;
    use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler};
    use serde::Deserialize;

    use crate::ask;
    use crate::model::NodeId;
    use crate::store::Store;

    /// The MCP server over a shared, durable [`Store`]. Reads take a brief lock; the single
    /// write tool (`memory.note`) takes it mutably. Clone is cheap (Arc).
    #[derive(Clone)]
    pub struct MemoryServer {
        store: Arc<Mutex<Store>>,
        // Read by the `#[tool_handler]`-generated dispatch at runtime; the dead-code lint
        // doesn't trace into the macro-generated ServerHandler impl, so silence it.
        #[allow(dead_code)]
        tool_router: ToolRouter<Self>,
    }

    #[derive(Debug, Deserialize, schemars::JsonSchema)]
    struct SearchArgs {
        /// Free-text query to recall memory for.
        query: String,
        /// Max results (default 8).
        #[serde(default)]
        k: Option<usize>,
    }

    #[derive(Debug, Deserialize, schemars::JsonSchema)]
    struct TopicArgs {
        /// Topic to recall decisions about, e.g. "storage engine".
        topic: String,
    }

    #[derive(Debug, Deserialize, schemars::JsonSchema)]
    struct RelatedArgs {
        /// A file path, symbol, or free-text anchor to find related memory for.
        anchor: String,
    }

    #[derive(Debug, Deserialize, schemars::JsonSchema)]
    struct TimelineArgs {
        /// Max items, newest-first by first-seen (default 20).
        #[serde(default)]
        limit: Option<usize>,
    }

    #[derive(Debug, Deserialize, schemars::JsonSchema)]
    struct NoteArgs {
        /// The note / decision text to remember (append-only).
        text: String,
        /// Optional topic to scope the note under.
        #[serde(default)]
        topic: Option<String>,
    }

    fn ok_json(v: serde_json::Value) -> Result<CallToolResult, McpError> {
        let text = serde_json::to_string_pretty(&v).unwrap_or_else(|_| "{}".to_string());
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool_router]
    impl MemoryServer {
        pub fn new(store: Store) -> Self {
            Self {
                store: Arc::new(Mutex::new(store)),
                tool_router: Self::tool_router(),
            }
        }

        #[tool(
            name = "memory.search",
            description = "Recall memory (decisions, implementations, chat turns, code) for a free-text query; returns ranked results with provenance you can cite."
        )]
        async fn memory_search(
            &self,
            Parameters(args): Parameters<SearchArgs>,
        ) -> Result<CallToolResult, McpError> {
            let store = self.store.lock().expect("store lock");
            let res = ask::recall(&store, &args.query, args.k.unwrap_or(8).min(50));
            let results: Vec<_> = res
                .results
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "id": e.id,
                        "kind": e.kind,
                        "label": e.label,
                        "snippet": e.snippet,
                        "score": e.score,
                        "why": e.why,
                        "provenance": e.provenance,
                    })
                })
                .collect();
            ok_json(serde_json::json!({
                "query": res.query,
                "degraded": res.degraded,
                "results": results,
            }))
        }

        #[tool(
            name = "memory.recall_decision",
            description = "Recall prior DECISIONS about a topic, with the rationale text and the session/commit they were made in — so you don't re-ask the model what was already decided."
        )]
        async fn memory_recall_decision(
            &self,
            Parameters(args): Parameters<TopicArgs>,
        ) -> Result<CallToolResult, McpError> {
            let store = self.store.lock().expect("store lock");
            let res = ask::recall(&store, &args.topic, 20);
            let decisions: Vec<_> = res
                .results
                .iter()
                .filter(|e| e.kind == "Decision")
                .map(|e| {
                    let p = e.provenance.as_ref();
                    serde_json::json!({
                        "id": e.id,
                        "topic": args.topic,
                        "decision": e.snippet,
                        "session": p.and_then(|p| p.native_session_id.clone()),
                        "commit": p.and_then(|p| p.commit.clone()),
                        "confidence": p.map(|p| p.confidence).unwrap_or(0.0),
                    })
                })
                .collect();
            ok_json(serde_json::json!(decisions))
        }

        #[tool(
            name = "memory.related",
            description = "Find memory related to a file, symbol, or topic: the graph neighborhood around the best match."
        )]
        async fn memory_related(
            &self,
            Parameters(args): Parameters<RelatedArgs>,
        ) -> Result<CallToolResult, McpError> {
            let store = self.store.lock().expect("store lock");
            let res = ask::recall(&store, &args.anchor, 1);
            if let Some(seed) = res.results.first() {
                if let Ok(id) = seed.id.parse::<NodeId>() {
                    let (nodes, edges) = store.graph.neighbors(id, 1);
                    let n: Vec<_> = nodes
                        .iter()
                        .map(|x| serde_json::json!({ "id": x.id.to_string(), "kind": x.kind.tag(), "label": x.label }))
                        .collect();
                    let ed: Vec<_> = edges
                        .iter()
                        .map(|x| serde_json::json!({ "from": x.from.to_string(), "to": x.to.to_string(), "kind": x.kind.tag() }))
                        .collect();
                    return ok_json(serde_json::json!({ "anchor": args.anchor, "seed": seed.id, "nodes": n, "edges": ed }));
                }
            }
            ok_json(serde_json::json!({ "anchor": args.anchor, "nodes": [], "edges": [] }))
        }

        #[tool(
            name = "memory.timeline",
            description = "List recent memory nodes (newest first by when they were first seen) for a quick sense of what was decided/built lately."
        )]
        async fn memory_timeline(
            &self,
            Parameters(args): Parameters<TimelineArgs>,
        ) -> Result<CallToolResult, McpError> {
            let store = self.store.lock().expect("store lock");
            let mut nodes: Vec<_> = store.graph.nodes.values().collect();
            nodes.sort_by(|a, b| b.first_seen_ms.cmp(&a.first_seen_ms));
            nodes.truncate(args.limit.unwrap_or(20).min(200));
            let items: Vec<_> = nodes
                .iter()
                .map(|x| {
                    serde_json::json!({
                        "id": x.id.to_string(),
                        "kind": x.kind.tag(),
                        "label": x.label,
                        "first_seen_ms": x.first_seen_ms,
                    })
                })
                .collect();
            ok_json(serde_json::json!(items))
        }

        #[tool(
            name = "memory.note",
            description = "Remember a new decision/note (append-only — never deletes or overwrites). Record something you figured out so it can be recalled later."
        )]
        async fn memory_note(
            &self,
            Parameters(args): Parameters<NoteArgs>,
        ) -> Result<CallToolResult, McpError> {
            let text = args.text.trim();
            if text.is_empty() {
                return Err(McpError::invalid_params("note text cannot be empty", None));
            }
            let mut store = self.store.lock().expect("store lock");
            match store.add_note(args.topic.as_deref(), text) {
                Ok(id) => ok_json(serde_json::json!({ "ok": true, "id": id.to_string() })),
                Err(e) => Err(McpError::internal_error(format!("note failed: {e}"), None)),
            }
        }
    }

    #[tool_handler]
    impl ServerHandler for MemoryServer {
        fn get_info(&self) -> ServerInfo {
            // ServerInfo (InitializeResult) is #[non_exhaustive] — mutate fields off
            // default() rather than a struct literal.
            let mut info = ServerInfo::default();
            info.capabilities = ServerCapabilities::builder().enable_tools().build();
            info.instructions = Some(
                "loregraph memory: recall prior decisions/implementations from your AI chat \
                 history + repo so you don't re-ask. Tools: memory.search, \
                 memory.recall_decision, memory.related, memory.timeline, memory.note \
                 (append-only)."
                    .to_string(),
            );
            info
        }
    }
}
