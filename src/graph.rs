//! In-memory memory-graph over `petgraph`.
//!
//! `MemGraph` is the traversal/layout view; it is rehydrated from the durable store and is
//! never the source of truth (PLAN.md §3). Upserts are idempotent by content-addressed
//! [`NodeId`], and `neighbors()` is the bounded BFS that powers the canvas's neighbor-first
//! (anti-hairball) expansion and the retriever's graph-proximity signal.

use std::collections::{HashMap, HashSet};

use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::Direction;

use crate::model::{Edge, EdgeKind, Node, NodeId};

#[derive(Default)]
pub struct MemGraph {
    g: StableDiGraph<NodeId, ()>,
    ix: HashMap<NodeId, NodeIndex>,
    pub nodes: HashMap<NodeId, Node>,
    pub edges: Vec<Edge>,
    edge_keys: HashSet<(NodeId, NodeId, &'static str)>,
    /// node id → indices into `edges` incident to it (both directions). Lets `neighbors()`
    /// gather a node's induced edges without scanning all of `edges` (O(degree), not O(E)) —
    /// the canvas fires a neighbor query on every click/search hit.
    adj: HashMap<NodeId, Vec<usize>>,
    /// superseded node → its superseder. Maintained from `Supersedes` edges
    /// (`from` supersedes `to`). Drives the recall freshness filter (R4): a recalled node that
    /// was superseded is redirected to the decision that replaced it.
    superseded_by: HashMap<NodeId, NodeId>,
}

impl MemGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Insert or merge a node. Preserves the earliest `first_seen_ms`, advances
    /// `last_updated_ms`, and merges attrs — so re-ingest never resets provenance time.
    pub fn upsert_node(&mut self, mut node: Node) {
        if let Some(existing) = self.nodes.get(&node.id) {
            node.first_seen_ms = existing.first_seen_ms.min(node.first_seen_ms);
            // last_updated_ms only moves forward, never backward on re-ingest.
            node.last_updated_ms = existing.last_updated_ms.max(node.last_updated_ms);
            if node.body.is_empty() {
                node.body = existing.body.clone();
            }
            if node.summary.is_empty() {
                node.summary = existing.summary.clone();
            }
            // Keep the richer provenance: a lower-or-equal source_priority must not
            // clobber an existing higher-priority (native > proxy > export) citation.
            node.provenance = match (&existing.provenance, node.provenance.take()) {
                (Some(old), Some(new)) if old.source_priority >= new.source_priority => Some(old.clone()),
                (Some(old), None) => Some(old.clone()),
                (_, newer) => newer,
            };
            for (k, v) in &existing.attrs {
                node.attrs.entry(k.clone()).or_insert_with(|| v.clone());
            }
        } else {
            let nx = self.g.add_node(node.id);
            self.ix.insert(node.id, nx);
        }
        self.nodes.insert(node.id, node);
    }

    /// Insert an edge if both endpoints exist and the (from,to,kind) triple is new.
    pub fn upsert_edge(&mut self, edge: Edge) {
        let key = (edge.from, edge.to, edge.kind.tag());
        if !self.ix.contains_key(&edge.from) || !self.ix.contains_key(&edge.to) {
            return;
        }
        if !self.edge_keys.insert(key) {
            return;
        }
        let (a, b) = (self.ix[&edge.from], self.ix[&edge.to]);
        self.g.add_edge(a, b, ());
        let idx = self.edges.len();
        self.adj.entry(edge.from).or_default().push(idx);
        if edge.to != edge.from {
            self.adj.entry(edge.to).or_default().push(idx);
        }
        if edge.kind == EdgeKind::Supersedes {
            // `from` supersedes `to`: record that `to` is now superseded by `from`.
            self.superseded_by.insert(edge.to, edge.from);
        }
        self.edges.push(edge);
    }

    /// Edge degree (incident edge count) — a cheap centrality proxy for the recall prior (R4).
    pub fn degree(&self, id: NodeId) -> usize {
        self.adj.get(&id).map_or(0, |v| v.len())
    }

    /// The current node that supersedes `id`, following the `Supersedes` chain to its head
    /// (newest decision). Returns `None` if `id` was never superseded. Cycle-guarded.
    pub fn superseder_of(&self, id: NodeId) -> Option<NodeId> {
        let mut cur = *self.superseded_by.get(&id)?;
        let mut seen = HashSet::new();
        seen.insert(id);
        seen.insert(cur);
        while let Some(&next) = self.superseded_by.get(&cur) {
            if !seen.insert(next) {
                return None;
            }
            cur = next;
        }
        Some(cur)
    }

    /// Depth-bounded BFS neighborhood (both directions), returning cloned nodes + the
    /// induced edges. This is the on-open `/v1/graph/neighbors` call.
    pub fn neighbors(&self, start: NodeId, depth: usize) -> (Vec<Node>, Vec<Edge>) {
        let mut nodes_out = Vec::new();
        let mut edges_out = Vec::new();
        let Some(&start_ix) = self.ix.get(&start) else {
            return (nodes_out, edges_out);
        };

        let mut visited: HashSet<NodeIndex> = HashSet::new();
        visited.insert(start_ix);
        let mut frontier = vec![start_ix];
        for _ in 0..depth {
            let mut next = Vec::new();
            for &nx in &frontier {
                for dir in [Direction::Outgoing, Direction::Incoming] {
                    for nb in self.g.neighbors_directed(nx, dir) {
                        if visited.insert(nb) {
                            next.push(nb);
                        }
                    }
                }
            }
            frontier = next;
        }

        let visited_ids: HashSet<NodeId> = visited
            .iter()
            .filter_map(|nx| self.g.node_weight(*nx).copied())
            .collect();
        for id in &visited_ids {
            if let Some(n) = self.nodes.get(id) {
                nodes_out.push(n.clone());
            }
        }
        // Induced edges via the adjacency index: only edges incident to a visited node, each
        // added once (dedup by edge index — an edge is listed under both endpoints).
        let mut seen_edge: HashSet<usize> = HashSet::new();
        for id in &visited_ids {
            let Some(eidxs) = self.adj.get(id) else { continue };
            for &i in eidxs {
                let e = &self.edges[i];
                if visited_ids.contains(&e.to)
                    && visited_ids.contains(&e.from)
                    && seen_edge.insert(i)
                {
                    edges_out.push(e.clone());
                }
            }
        }
        (nodes_out, edges_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Edge, EdgeKind, Node, NodeKind};

    #[test]
    fn idempotent_upsert_and_bfs() {
        let mut g = MemGraph::new();
        let a = Node::new(NodeKind::Session, "s1", "session 1", 10);
        let b = Node::new(NodeKind::Decision, "use redb", "use redb", 20);
        let c = Node::new(NodeKind::File, "src/store.rs", "store.rs", 30);
        let (ai, bi, ci) = (a.id, b.id, c.id);
        g.upsert_node(a.clone());
        g.upsert_node(a); // re-ingest is a no-op
        g.upsert_node(b);
        g.upsert_node(c);
        g.upsert_edge(Edge::new(bi, ai, EdgeKind::DecidedIn));
        g.upsert_edge(Edge::new(bi, ci, EdgeKind::Produced));
        g.upsert_edge(Edge::new(bi, ci, EdgeKind::Produced)); // dup ignored
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
        let (ns, es) = g.neighbors(bi, 1);
        assert_eq!(ns.len(), 3, "decision + both neighbors");
        assert_eq!(es.len(), 2);
    }

    #[test]
    fn supersedes_chain_and_degree() {
        let mut g = MemGraph::new();
        let a = Node::new(NodeKind::Decision, "use sled", "use sled", 10);
        let b = Node::new(NodeKind::Decision, "use redb", "use redb", 20);
        let c = Node::new(NodeKind::Decision, "use redb v2", "use redb v2", 30);
        let (ai, bi, ci) = (a.id, b.id, c.id);
        g.upsert_node(a);
        g.upsert_node(b);
        g.upsert_node(c);
        g.upsert_edge(Edge::new(bi, ai, EdgeKind::Supersedes)); // b supersedes a
        g.upsert_edge(Edge::new(ci, bi, EdgeKind::Supersedes)); // c supersedes b
        assert_eq!(g.superseder_of(ai), Some(ci), "follows the chain a→b→c to the head");
        assert_eq!(g.superseder_of(bi), Some(ci));
        assert_eq!(g.superseder_of(ci), None, "the head is not superseded");
        assert_eq!(g.degree(ai), 1, "one incident edge");
        assert_eq!(g.degree(bi), 2, "superseded by c, supersedes a");
    }
}
