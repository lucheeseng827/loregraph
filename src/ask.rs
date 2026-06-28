//! Retrieval — "what do we already know about X?" (PLAN.md §5).
//!
//! Fuses a semantic signal (embedding ANN) with a recency signal, and falls back to a
//! lexical scan when no vector index is present (so the default build always answers, just
//! `degraded`). The full hybrid (semantic + graph-proximity + recency + centrality with the
//! supersession filter) is the MVP `Retriever`; this PoC ships the semantic+recency+lexical
//! core. Every result carries the provenance an agent can **cite**.

use serde::Serialize;

use crate::model::{NodeId, NodeKind, Provenance};
use crate::store::Store;

/// Per-kind ranking prior (R1, PLAN.md §5.3). A recall query asks "what do we already *know*"
/// — the answer is a value node (a `Decision`/`Implementation`/`Turn`), not a code-graph atom.
/// Without this, a 1-token `Symbol` name (e.g. `create_invoice`) yields a sparse vector that
/// scores spuriously high against a short query and out-ranks the actual decision. Down-weight
/// (don't exclude): a query that genuinely is *about* a symbol/file still surfaces it, just
/// below a real memory of equal raw similarity.
fn kind_weight(kind: NodeKind) -> f32 {
    use NodeKind::*;
    match kind {
        Decision | Implementation => 1.0,
        DesignDoc => 0.9,
        Turn => 0.85,
        Concept => 0.8,
        Session => 0.65,
        Pattern | DebtSignal => 0.6,
        Repo | Module | Commit => 0.5,
        File | Manifest | Package | Framework | Tooling | DataSource | Person => 0.4,
        Symbol => 0.3,
    }
}

/// Whether a node belongs in the **semantic** candidate set. PLAN.md §5.1 ranks recall over
/// `Turn`/`Decision`/`Implementation` (chat + value) embeddings, not code atoms. A code
/// atom's embedding is a 1–2-token name bag — pure noise under `HashEmbedder` — so it must
/// not seed semantic recall (it floods "what did we decide" with `Symbol`s of coincidental
/// hash overlap). Code atoms still surface, but only the legitimate way: graph-expanded from
/// a real memory seed, or via an exact lexical name match. (When `localmodel` lands, R2 in
/// §5.3, this set can widen.)
fn is_semantic_memory(kind: NodeKind) -> bool {
    use NodeKind::*;
    matches!(
        kind,
        Decision | Implementation | Turn | Session | DesignDoc | Concept | Pattern | DebtSignal
    )
}

/// One ranked piece of recalled memory.
#[derive(Debug, Clone, Serialize)]
pub struct Evidence {
    pub id: String, // stringified u128 (JS-safe)
    pub score: f32,
    pub kind: String,
    pub label: String,
    pub snippet: String,
    /// Why this surfaced: `semantic` | `lexical` | `recency`.
    pub why: &'static str,
    pub provenance: Option<Provenance>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecallResult {
    pub query: String,
    /// True when no vector index was available and we fell back to a lexical scan.
    pub degraded: bool,
    pub results: Vec<Evidence>,
}

/// Truncate `body` to at most `n` chars, appending an ellipsis when clipped.
fn snippet(body: &str, n: usize) -> String {
    let s: String = body.chars().take(n).collect();
    if body.chars().count() > n {
        format!("{s}…")
    } else {
        s
    }
}

/// Build an `Evidence` entry for `id`; returns `None` if the node no longer exists.
fn evidence(store: &Store, id: NodeId, score: f32, why: &'static str) -> Option<Evidence> {
    let n = store.graph.get(id)?;
    // Return the distilled summary (T2) — a few tokens — not the full body; full detail is a
    // drill-down via `/v1/node/{id}`. Fall back to a body slice, then the label.
    let snippet = if !n.summary.is_empty() {
        n.summary.clone()
    } else if !n.body.is_empty() {
        snippet(&n.body, 240)
    } else {
        n.label.clone()
    };
    Some(Evidence {
        id: id.to_string(),
        score,
        kind: format!("{:?}", n.kind),
        label: n.label.clone(),
        snippet,
        why,
        provenance: n.provenance.clone(),
    })
}

/// Recall the top-`k` memories for a free-text query.
///
/// Hybrid of two signals (PLAN.md §5): a **semantic** pass (embedding ANN + recency) and a
/// **graph-proximity** pass that expands the top semantic seeds one hop. The graph pass is
/// what surfaces the *answer* when the query only matches the *question* — e.g. "what
/// storage engine did we choose" lands on the user's question turn, then expands through its
/// session to the assistant's decision. Falls back to a lexical scan when no vector index
/// is present (so the default build always answers, just `degraded`).
pub fn recall(store: &Store, query: &str, k: usize) -> RecallResult {
    use std::collections::HashMap;

    // Pre-compute query tokens once; reused by both the exact-match lexical step
    // inside the indexed path and by the lexical-only fallback below.
    let tokens_lc: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(|t| t.to_lowercase())
        .collect();

    // `degraded` means the vector index is unavailable — NOT merely that semantic recall
    // returned nothing (in which case we still lexically answer, undegraded).
    let have_index = !store.index.is_empty();
    if have_index {
        let qv = store.embed_query(query);
        let now = chrono::Utc::now().timestamp_millis();
        // id -> (score, why). Keep the highest score; semantic beats graph on ties.
        let mut scored: HashMap<NodeId, (f32, &'static str)> = HashMap::new();
        let bump = |m: &mut HashMap<NodeId, (f32, &'static str)>, id: NodeId, s: f32, why| {
            m.entry(id)
                .and_modify(|e| {
                    if s > e.0 {
                        *e = (s, why);
                    }
                })
                .or_insert((s, why));
        };

        // 1) semantic seeds. Pull a wide candidate pool: most embeddings are code atoms that
        // the `is_semantic_memory` filter drops, so value nodes can sit deep in the cosine
        // ranking. BruteForce scans every vector regardless, so a larger pool is ~free.
        // saturating_mul guards against overflow on a large k.
        let semantic_pool = k.saturating_mul(40).max(256);
        for (id, sem) in store.index.search(&qv, semantic_pool) {
            if sem <= 0.01 {
                continue;
            }
            let Some(n) = store.graph.get(id) else { continue };
            // Only value/chat nodes seed semantic recall; code atoms enter via the
            // graph hop below or the exact-match lexical step, never as a fuzzy semantic hit.
            if !is_semantic_memory(n.kind) {
                continue;
            }
            let rec = recency_boost(now, n.last_updated_ms);
            let s = (0.7 * sem + 0.15 * rec) * kind_weight(n.kind) + centrality_prior(store, id);
            bump(&mut scored, id, s, "semantic");
        }

        // 2) graph proximity — expand the strongest seeds one hop
        let mut seeds: Vec<(NodeId, f32)> = scored.iter().map(|(id, (s, _))| (*id, *s)).collect();
        seeds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (sid, sscore) in seeds.into_iter().take(3) {
            for nb in store.graph.neighbors(sid, 1).0 {
                if nb.id == sid {
                    continue;
                }
                let rec = recency_boost(now, nb.last_updated_ms);
                let s = (0.35 * sscore + 0.15 * rec) * kind_weight(nb.kind) + centrality_prior(store, nb.id);
                bump(&mut scored, nb.id, s, "graph");
            }
        }

        // 3) exact-match lexical pass for code atoms (Symbol/File/…). Merged into
        // `scored` here — not relegated to the fallback — so a direct query for
        // `create_invoice` can surface alongside semantic memories rather than being
        // swallowed by the early return below.
        for n in store.graph.nodes.values() {
            if is_semantic_memory(n.kind) {
                continue; // already covered by semantic seed pass
            }
            let label_lc = n.label.to_lowercase();
            if tokens_lc.iter().any(|t| label_lc.contains(t.as_str())) {
                let rec = recency_boost(now, n.last_updated_ms);
                let s = (0.6 + 0.1 * rec) * kind_weight(n.kind) + centrality_prior(store, n.id);
                bump(&mut scored, n.id, s, "lexical");
            }
        }

        if !scored.is_empty() {
            return finalize(store, scored, k, query, false);
        }
    }

    // Lexical fallback (no usable vector index / no hits from the indexed path): BM25 over
    // node text — IDF-weighted + length-normalized, so rare query terms matter and a long
    // node body doesn't win on sheer length (R3, PLAN.md §5.3).
    let scored = bm25_scored(store, &tokens_lc);
    finalize(store, scored, k, query, !have_index)
}

/// Weight of the degree-centrality prior (R4). `ln(1+degree)` keeps it a nudge, never enough
/// to override a strong direct match.
const CENTRALITY_WEIGHT: f32 = 0.03;

/// Centrality prior — applied ONLY to value nodes (`Decision`/`Implementation`). For those,
/// a high degree means the decision shaped many files / was reached across many sessions: a
/// real importance signal. For structural nodes (a `Session` with hundreds of turns, a `Repo`
/// with thousands of files) degree is just fanout, and boosting it floods recall with hubs —
/// so they get no prior.
fn centrality_prior(store: &Store, id: NodeId) -> f32 {
    match store.graph.get(id).map(|n| n.kind) {
        Some(NodeKind::Decision | NodeKind::Implementation) => {
            (store.graph.degree(id) as f32).ln_1p() * CENTRALITY_WEIGHT
        }
        _ => 0.0,
    }
}

/// Resolve scored candidates into ranked `Evidence`, applying the R4 supersession filter:
/// a superseded node is redirected to the decision that replaced it (keeping the better
/// score), so a stale memory never out-ranks — or hides — its superseder. A no-op until
/// `Supersedes` edges exist (auto-detection is deferred; see PLAN.md §5.3 R4).
fn finalize(
    store: &Store,
    scored: std::collections::HashMap<NodeId, (f32, &'static str)>,
    k: usize,
    query: &str,
    degraded: bool,
) -> RecallResult {
    use std::collections::HashMap;
    let mut fresh: HashMap<NodeId, (f32, &'static str)> = HashMap::with_capacity(scored.len());
    for (id, (s, why)) in scored {
        let (target, why2) = match store.graph.superseder_of(id) {
            Some(sup) => (sup, "supersedes"),
            None => (id, why),
        };
        fresh
            .entry(target)
            .and_modify(|e| {
                if s > e.0 {
                    *e = (s, why2);
                }
            })
            .or_insert((s, why2));
    }
    let mut results: Vec<Evidence> = fresh
        .into_iter()
        .filter_map(|(id, (s, why))| evidence(store, id, s, why))
        .collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    RecallResult { query: query.to_string(), degraded, results }
}

/// Lowercased alphanumeric tokens, length > 1 — the same rule the embedder uses, so lexical
/// and semantic paths agree on what a "term" is.
fn lex_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(|t| t.to_lowercase())
        .collect()
}

/// BM25 over every node's `label + body`, weighted by [`kind_weight`]. One pass collects each
/// doc's length and per-query-term frequency plus the document frequencies; a second computes
/// the IDF-weighted, length-normalized score. Standard Okapi params (k1=1.2, b=0.75).
fn bm25_scored(
    store: &Store,
    query_tokens: &[String],
) -> std::collections::HashMap<NodeId, (f32, &'static str)> {
    use std::collections::HashMap;
    const K1: f32 = 1.2;
    const B: f32 = 0.75;

    let mut out: HashMap<NodeId, (f32, &'static str)> = HashMap::new();
    // Unique query terms (dedup so a repeated query word isn't double-counted).
    let mut qterms: Vec<&str> = query_tokens.iter().map(|s| s.as_str()).collect();
    qterms.sort_unstable();
    qterms.dedup();
    if qterms.is_empty() {
        return out;
    }

    // Pass 1: per-doc length + query-term frequencies; document frequency per query term.
    struct Doc {
        id: NodeId,
        dl: usize,
        tf: Vec<usize>, // per qterm
        kind: crate::model::NodeKind,
    }
    let mut docs: Vec<Doc> = Vec::new();
    let mut df = vec![0usize; qterms.len()];
    let mut total_len = 0usize;
    for n in store.graph.nodes.values() {
        let toks = lex_tokens(&format!("{} {}", n.label, n.body));
        let dl = toks.len();
        total_len += dl;
        let mut tf = vec![0usize; qterms.len()];
        for t in &toks {
            if let Ok(qi) = qterms.binary_search(&t.as_str()) {
                tf[qi] += 1;
            }
        }
        if tf.iter().any(|c| *c > 0) {
            for (qi, c) in tf.iter().enumerate() {
                if *c > 0 {
                    df[qi] += 1;
                }
            }
            docs.push(Doc { id: n.id, dl, tf, kind: n.kind });
        }
    }
    let n_docs = store.graph.nodes.len().max(1) as f32;
    let avgdl = (total_len as f32 / n_docs).max(1.0);

    // Pass 2: BM25 score per matching doc.
    for d in &docs {
        let mut score = 0.0f32;
        for (qi, &tf) in d.tf.iter().enumerate() {
            if tf == 0 {
                continue;
            }
            let dfi = df[qi] as f32;
            let idf = (1.0 + (n_docs - dfi + 0.5) / (dfi + 0.5)).ln();
            let tf = tf as f32;
            let denom = tf + K1 * (1.0 - B + B * d.dl as f32 / avgdl);
            score += idf * (tf * (K1 + 1.0)) / denom;
        }
        let score = score * kind_weight(d.kind) + centrality_prior(store, d.id);
        if score > 0.0 {
            out.insert(d.id, (score, "lexical"));
        }
    }
    out
}

/// Newer memory scores slightly higher (30-day half-life), in [0,1].
fn recency_boost(now_ms: i64, ts_ms: i64) -> f32 {
    let age_days = ((now_ms - ts_ms).max(0) as f64) / 86_400_000.0;
    0.5f64.powf(age_days / 30.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::model::{AgentKind, CanonicalSession, EdgeKind, Edge, MessageKind, NodeKind, Turn};
    use crate::store::Store;

    fn sess(id: &str, turns: &[(MessageKind, &str)]) -> CanonicalSession {
        CanonicalSession {
            native_session_id: id.into(),
            agent: AgentKind::ClaudeCode,
            repo_root: None,
            commit: None,
            started_at_ms: Some(1000),
            source_path: None,
            connector_version: None,
            turns: turns
                .iter()
                .enumerate()
                .map(|(i, (k, t))| Turn {
                    index: i,
                    kind: *k,
                    text: t.to_string(),
                    ts_ms: Some(1000 + i as i64),
                    touched_files: vec![],
                    raw: serde_json::Value::Null,
                })
                .collect(),
        }
    }

    #[test]
    fn bm25_scores_the_term_bearing_doc() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path()).unwrap();
        store
            .ingest_session(&sess(
                "s1",
                &[
                    (MessageKind::User, "how do we expose the gateway"),
                    (MessageKind::Assistant, "we configured kubernetes ingress for the gateway"),
                    (MessageKind::Assistant, "and also a lot of common filler words here"),
                ],
            ))
            .unwrap();
        // "kubernetes" appears in exactly one turn → BM25 must score that doc and not the
        // filler turn (which shares none of the query terms).
        let scored = bm25_scored(&store, &["kubernetes".to_string()]);
        assert!(!scored.is_empty(), "the kubernetes turn is scored");
        let top = scored
            .iter()
            .max_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap())
            .map(|(id, _)| store.graph.get(*id).unwrap().body.clone())
            .unwrap();
        assert!(top.contains("kubernetes"), "top BM25 hit carries the rare term, got: {top}");
    }

    #[test]
    fn supersession_redirects_to_the_newer_decision() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path()).unwrap();
        store
            .ingest_session(&sess(
                "s1",
                &[
                    (MessageKind::Assistant, "Let's use sled for the durable store."),
                    (MessageKind::Assistant, "Switch to redb for the durable store instead of sled."),
                ],
            ))
            .unwrap();
        // Find the two decision nodes and wire a Supersedes edge (redb supersedes sled).
        // The redb decision also mentions "sled" ("instead of sled"), so match on redb
        // explicitly and take the sled decision as the one without "redb".
        let decision = |pred: &dyn Fn(&str) -> bool| {
            store
                .graph
                .nodes
                .values()
                .find(|n| n.kind == NodeKind::Decision && pred(&n.body.to_lowercase()))
                .map(|n| n.id)
                .unwrap()
        };
        let redb = decision(&|b| b.contains("redb"));
        let sled = decision(&|b| b.contains("sled") && !b.contains("redb"));
        store.graph.upsert_edge(Edge::new(redb, sled, EdgeKind::Supersedes));

        let res = recall(&store, "durable store", 10);
        let ids: Vec<NodeId> = res.results.iter().filter_map(|e| e.id.parse().ok()).collect();
        assert!(ids.contains(&redb), "the current (superseding) decision surfaces");
        assert!(!ids.contains(&sled), "the superseded decision is filtered out");
    }

    #[test]
    fn recall_result_serializes_for_the_json_flag() {
        // Locks the `lore ask --json` wire shape: top-level {query, degraded, results[]} and
        // per-hit {id, score, kind, label, snippet, why, provenance}.
        let r = RecallResult {
            query: "storage".into(),
            degraded: false,
            results: vec![Evidence {
                id: "42".into(),
                score: 0.61,
                kind: "Decision".into(),
                label: "Use redb".into(),
                snippet: "Use redb for the durable store.".into(),
                why: "semantic",
                provenance: None,
            }],
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["query"], "storage");
        assert_eq!(v["degraded"], false);
        assert_eq!(v["results"].as_array().map(|a| a.len()), Some(1));
        assert_eq!(v["results"][0]["id"], "42");
        assert_eq!(v["results"][0]["score"].as_f64().unwrap(), 0.61f32 as f64);
        assert_eq!(v["results"][0]["kind"], "Decision");
        assert_eq!(v["results"][0]["label"], "Use redb");
        assert_eq!(v["results"][0]["why"], "semantic");
        assert!(v["results"][0].get("snippet").is_some());
        assert!(v["results"][0].get("provenance").is_some()); // null, but present
    }

    #[test]
    fn value_nodes_outweigh_code_atoms() {
        // A Decision/Turn must out-rank a Symbol/File at equal raw similarity, so a query
        // like "stripe billing" lands on the decision, not a `create_invoice` symbol.
        assert!(kind_weight(NodeKind::Decision) > kind_weight(NodeKind::Symbol));
        assert!(kind_weight(NodeKind::Turn) > kind_weight(NodeKind::File));
        assert_eq!(kind_weight(NodeKind::Decision), kind_weight(NodeKind::Implementation));
        // ...but a symbol is down-weighted, not excluded — it can still surface.
        assert!(kind_weight(NodeKind::Symbol) > 0.0);
    }
}
