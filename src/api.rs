//! HTTP API + canvas SPA (axum) — the renderer-blind JSON contract (PLAN.md §6).
//!
//! Endpoints (neutral JSON so Cytoscape/Sigma/the PoC renderer are interchangeable):
//! - `GET /healthz`
//! - `GET /v1/graph?limit=&types=`           — a bounded slice (canvas boot)
//! - `GET /v1/graph/neighbors?id=&depth=`    — BFS neighborhood (anti-hairball on-open call)
//! - `GET /v1/search?q=&k=`                  — ranked recall (degrades to lexical, never errors)
//! - `GET /v1/node/{id}`                     — side-panel detail + provenance
//! - `GET /v1/timeline?limit=`              — nodes by first-seen (time-scrub)
//! - `*` fallback                            — the embedded SPA (see [`crate::ui`])
//!
//! Node ids are `u128`, so the wire serializes them as **strings** (JS-number-safe).

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::ask;
use crate::model::{Edge, Node, NodeId, Provenance};
use crate::store::Store;

type Shared = Arc<Store>;

#[derive(Serialize)]
struct WireNode {
    id: String,
    kind: String,
    label: String,
    /// The distilled summary, not the full body (T2): bulk graph/neighbors responses stay
    /// small. The full `body` is only included on the `/v1/node/{id}` drill-down.
    summary: String,
    /// Full text — populated only by [`WireNode::detail`] (the per-node fetch); omitted from
    /// bulk responses so a 400-node graph payload doesn't carry every paragraph.
    #[serde(skip_serializing_if = "Option::is_none")]
    body: Option<String>,
    provenance: Option<Provenance>,
    attrs: BTreeMap<String, String>,
    first_seen_ms: i64,
    last_updated_ms: i64,
}

impl WireNode {
    /// Lightweight form for bulk listings (graph / neighbors / timeline / search): summary,
    /// no body.
    fn bulk(n: &Node) -> Self {
        WireNode {
            id: n.id.to_string(),
            kind: n.kind.tag().to_string(), // stable wire tag, not Debug output
            label: n.label.clone(),
            // Prefer the summary; for legacy nodes with no summary, derive from body; for code
            // atoms (no summary, no body) the label already is the gist.
            summary: if !n.summary.is_empty() {
                n.summary.clone()
            } else if !n.body.is_empty() {
                crate::model::summarize(&n.body)
            } else {
                n.label.clone()
            },
            body: None,
            provenance: n.provenance.clone(),
            attrs: n.attrs.clone(),
            first_seen_ms: n.first_seen_ms,
            last_updated_ms: n.last_updated_ms,
        }
    }

    /// Full form for the `/v1/node/{id}` drill-down: includes the complete `body`.
    fn detail(n: &Node) -> Self {
        let mut w = Self::bulk(n);
        w.body = Some(n.body.clone());
        w
    }
}

impl From<&Node> for WireNode {
    fn from(n: &Node) -> Self {
        WireNode::bulk(n)
    }
}

#[derive(Serialize)]
struct WireEdge {
    from: String,
    to: String,
    kind: String,
    weight: f32,
}

impl From<&Edge> for WireEdge {
    fn from(e: &Edge) -> Self {
        WireEdge {
            from: e.from.to_string(),
            to: e.to.to_string(),
            kind: e.kind.tag().to_string(),
            weight: e.weight,
        }
    }
}

#[derive(Serialize)]
struct WireGraph {
    nodes: Vec<WireNode>,
    edges: Vec<WireEdge>,
}

/// Build the axum app (router + state + SPA fallback). Exposed so tests can drive it
/// without binding a socket.
pub fn build_router(store: Shared) -> Router {
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/v1/graph", get(graph))
        .route("/v1/graph/neighbors", get(neighbors))
        .route("/v1/search", get(search))
        .route("/v1/node/{id}", get(node))
        .route("/v1/timeline", get(timeline))
        .fallback(crate::ui::serve_asset)
        .with_state(store)
}

/// Run the API + canvas. Binds loopback by default so it never listens on the network unasked.
pub async fn serve(addr: SocketAddr, store: Store) -> anyhow::Result<()> {
    let app = build_router(Arc::new(store));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("loregraph canvas + API on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

#[derive(Deserialize)]
struct GraphParams {
    limit: Option<usize>,
    /// comma-separated NodeKind debug names (e.g. `Decision,Session`)
    types: Option<String>,
}

async fn graph(State(s): State<Shared>, Query(p): Query<GraphParams>) -> Json<WireGraph> {
    let limit = p.limit.unwrap_or(400).min(5000);
    let want: Option<Vec<String>> = p
        .types
        .map(|t| t.split(',').map(|x| x.trim().to_string()).collect());

    let mut nodes: Vec<&Node> = s
        .graph
        .nodes
        .values()
        .filter(|n| match &want {
            Some(ks) => ks.iter().any(|k| k.eq_ignore_ascii_case(n.kind.tag())),
            None => true,
        })
        .collect();
    // newest first, then cap
    nodes.sort_by(|a, b| b.last_updated_ms.cmp(&a.last_updated_ms));
    nodes.truncate(limit);

    let ids: std::collections::HashSet<NodeId> =
        nodes.iter().map(|n| n.id).collect();
    let edges: Vec<WireEdge> = s
        .graph
        .edges
        .iter()
        .filter(|e| ids.contains(&e.from) && ids.contains(&e.to))
        .map(WireEdge::from)
        .collect();
    Json(WireGraph {
        nodes: nodes.iter().map(|n| WireNode::from(*n)).collect(),
        edges,
    })
}

#[derive(Deserialize)]
struct NeighborParams {
    id: String,
    depth: Option<usize>,
}

async fn neighbors(State(s): State<Shared>, Query(p): Query<NeighborParams>) -> Json<WireGraph> {
    let Ok(id) = p.id.parse::<NodeId>() else {
        return Json(WireGraph { nodes: vec![], edges: vec![] });
    };
    let depth = p.depth.unwrap_or(1).min(3);
    let (ns, es) = s.graph.neighbors(id, depth);
    Json(WireGraph {
        nodes: ns.iter().map(WireNode::from).collect(),
        edges: es.iter().map(WireEdge::from).collect(),
    })
}

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    k: Option<usize>,
}

async fn search(State(s): State<Shared>, Query(p): Query<SearchParams>) -> Json<ask::RecallResult> {
    Json(ask::recall(&s, &p.q, p.k.unwrap_or(12).min(100)))
}

#[derive(Serialize)]
struct NodeResponse {
    node: Option<WireNode>,
}

async fn node(State(s): State<Shared>, Path(id): Path<String>) -> Json<NodeResponse> {
    let node = id
        .parse::<NodeId>()
        .ok()
        .and_then(|nid| s.graph.get(nid))
        .map(WireNode::detail);
    Json(NodeResponse { node })
}

#[derive(Deserialize)]
struct TimelineParams {
    limit: Option<usize>,
}

async fn timeline(State(s): State<Shared>, Query(p): Query<TimelineParams>) -> Json<Vec<WireNode>> {
    let limit = p.limit.unwrap_or(200).min(2000);
    let mut nodes: Vec<&Node> = s.graph.nodes.values().collect();
    nodes.sort_by(|a, b| a.first_seen_ms.cmp(&b.first_seen_ms));
    nodes.truncate(limit);
    Json(nodes.iter().map(|n| WireNode::from(*n)).collect())
}
