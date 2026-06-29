//! Embedding index seam — find near vectors.
//!
//! The default build ships [`BruteForceIndex`]: an exact cosine scan. It is the recall
//! *oracle* and is correct at PoC scale. The at-scale swap is the pure-Rust, zero-dep HNSW
//! (the `recall-index` precedent from recall) behind the `index-hnsw` feature — same
//! trait, sublinear search, no new dependency (PLAN.md §7).

use std::collections::HashMap;

use crate::model::NodeId;
use serde::{Deserialize, Serialize};

/// One indexed vector keyed by the node it embeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub id: NodeId,
    pub vec: Vec<f32>,
}

/// Exact cosine index (vectors are stored already L2-normalized, so dot == cosine).
///
/// `pos` maps a node id to its slot in `entries` so [`BruteForceIndex::upsert`] is O(1).
/// Without it, building/rehydrating an N-vector index is O(N²) (a linear scan per insert) —
/// the dominant cost of `lore index` and every store `open()`. `pos` is derived state, never
/// serialized; the durable source is the per-vector redb table, replayed through `upsert`.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BruteForceIndex {
    entries: Vec<Entry>,
    #[serde(skip)]
    pos: HashMap<NodeId, usize>,
}

impl BruteForceIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no vectors have been inserted.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Idempotent O(1): re-inserting a node id replaces its vector in place.
    pub fn upsert(&mut self, id: NodeId, vec: Vec<f32>) {
        if let Some(&i) = self.pos.get(&id) {
            self.entries[i].vec = vec;
        } else {
            self.pos.insert(id, self.entries.len());
            self.entries.push(Entry { id, vec });
        }
    }

    /// All indexed entries (used by the durable store's checkpoint to persist vectors).
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Top-`k` by cosine similarity, descending.
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        let mut scored: Vec<(NodeId, f32)> = self
            .entries
            .iter()
            .map(|e| (e.id, cosine(q, &e.vec)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

/// Dot product. Inputs are expected L2-normalized (see [`crate::embed::l2_normalize`]).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The vector index the store talks to. One variant per build: the exact [`BruteForceIndex`]
/// (default, zero-dep, the recall *oracle*) or the sublinear [`HnswIndex`] behind
/// `--features index-hnsw`. A compile-time enum (not `dyn`) keeps the default build's hot path
/// monomorphic and free of any HNSW cost (PLAN.md §7). The store rebuilds whichever index from
/// redb on open via [`VectorIndex::upsert`], so the choice is a pure build-time swap — no stored
/// format change.
pub enum VectorIndex {
    Brute(BruteForceIndex),
    #[cfg(feature = "index-hnsw")]
    Hnsw(HnswIndex),
}

impl Default for VectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex {
    /// Pick the index for this build: HNSW when `index-hnsw` is on, else exact brute force.
    pub fn new() -> Self {
        #[cfg(feature = "index-hnsw")]
        {
            VectorIndex::Hnsw(HnswIndex::new())
        }
        #[cfg(not(feature = "index-hnsw"))]
        {
            VectorIndex::Brute(BruteForceIndex::new())
        }
    }

    /// Human-readable backend id for `lore doctor` / `version`.
    pub fn backend(&self) -> &'static str {
        match self {
            VectorIndex::Brute(_) => "brute-force (exact cosine)",
            #[cfg(feature = "index-hnsw")]
            VectorIndex::Hnsw(_) => "hnsw (approximate, --features index-hnsw)",
        }
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        match self {
            VectorIndex::Brute(b) => b.len(),
            #[cfg(feature = "index-hnsw")]
            VectorIndex::Hnsw(h) => h.len(),
        }
    }

    /// True when no vectors have been inserted.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Idempotent upsert. Re-inserting a node id refreshes its vector in place (a re-embed of the
    /// same content yields the same vector, so the HNSW graph stays valid without a relink).
    pub fn upsert(&mut self, id: NodeId, vec: Vec<f32>) {
        match self {
            VectorIndex::Brute(b) => b.upsert(id, vec),
            #[cfg(feature = "index-hnsw")]
            VectorIndex::Hnsw(h) => h.upsert(id, vec),
        }
    }

    /// All indexed entries, borrowed — the store's checkpoint persists these raw vectors to redb.
    /// Borrowing (not cloning) keeps `save()` zero-copy; HNSW keeps the same `Vec<Entry>` as its
    /// vector store so this stays a borrow in both builds.
    pub fn entries(&self) -> &[Entry] {
        match self {
            VectorIndex::Brute(b) => b.entries(),
            #[cfg(feature = "index-hnsw")]
            VectorIndex::Hnsw(h) => h.entries(),
        }
    }

    /// Top-`k` by cosine similarity, descending. Exact for brute force; approximate for HNSW.
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        match self {
            VectorIndex::Brute(b) => b.search(q, k),
            #[cfg(feature = "index-hnsw")]
            VectorIndex::Hnsw(h) => h.search(q, k),
        }
    }
}

#[cfg(feature = "index-hnsw")]
pub use hnsw::HnswIndex;

/// Pure-Rust, zero-dependency HNSW (Hierarchical Navigable Small World) approximate index — the
/// at-scale swap for [`BruteForceIndex`] behind `--features index-hnsw` (PLAN.md §7, the
/// `recall-index` precedent). Single-threaded (the store owns it behind `&mut`, like brute force),
/// insert-only (the store never deletes a vector — re-embeds upsert in place), and it keeps its
/// vectors in the same `Vec<Entry>` shape brute force uses, so `entries()` stays a borrow and the
/// redb checkpoint path is unchanged.
#[cfg(feature = "index-hnsw")]
mod hnsw {
    use std::cmp::{Ordering, Reverse};
    use std::collections::{BinaryHeap, HashMap, HashSet};

    use super::{cosine, Entry};
    use crate::model::NodeId;

    /// Cosine distance for L2-normalized vectors: `1 - dot`. Smaller is closer.
    fn dist(a: &[f32], b: &[f32]) -> f32 {
        1.0 - cosine(a, b)
    }

    /// Graph tunables. Defaults follow the HNSW paper (M=16) with `ef_search` set wide enough to
    /// hold `recall@1` against the brute-force oracle at the 256-dim embedding regime (the
    /// curse-of-dimensionality margin the recall ann-bench surfaced — see the gate test).
    #[derive(Clone, Copy, Debug)]
    pub struct HnswConfig {
        pub m: usize,
        pub m0: usize,
        pub ef_construction: usize,
        pub ef_search: usize,
        pub ml: f64,
        pub seed: u64,
    }

    impl Default for HnswConfig {
        fn default() -> Self {
            let m = 16;
            Self {
                m,
                m0: m * 2,
                ef_construction: 200,
                ef_search: 128,
                ml: 1.0 / (m as f64).ln(),
                seed: 0x243F_6A88_85A3_08D3,
            }
        }
    }

    /// A scored candidate during traversal, ordered by distance with a total order (NaN sorts
    /// farthest) so it can live in a `BinaryHeap`.
    #[derive(Clone, Copy)]
    struct Cand {
        dist: f32,
        id: usize,
    }
    impl PartialEq for Cand {
        fn eq(&self, o: &Self) -> bool {
            self.cmp(o) == Ordering::Equal
        }
    }
    impl Eq for Cand {}
    impl PartialOrd for Cand {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Cand {
        fn cmp(&self, o: &Self) -> Ordering {
            match self.dist.partial_cmp(&o.dist) {
                Some(ord) => ord,
                None => match (self.dist.is_nan(), o.dist.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => Ordering::Equal,
                },
            }
        }
    }

    /// splitmix64 — a deterministic, dependency-free PRNG for level assignment (so a graph is
    /// reproducible from its insert order).
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Internal node id is its slot in [`HnswIndex::entries`] (1:1), so a node's vector is
    /// `entries[id].vec`. `links[layer]` are neighbor node ids at that layer.
    struct GraphNode {
        links: Vec<Vec<usize>>,
    }

    pub struct HnswIndex {
        cfg: HnswConfig,
        /// The vector store (also what `entries()` borrows for the redb checkpoint).
        entries: Vec<Entry>,
        /// node id == `entries` slot, so `nodes` is parallel to `entries`.
        nodes: Vec<GraphNode>,
        pos: HashMap<NodeId, usize>,
        entry: Option<usize>,
        top_layer: usize,
        rng: u64,
    }

    impl Default for HnswIndex {
        fn default() -> Self {
            Self::new()
        }
    }

    impl HnswIndex {
        /// Create an empty index with default [`HnswConfig`].
        pub fn new() -> Self {
            Self::with_config(HnswConfig::default())
        }

        /// Create an empty index with the given config.
        pub fn with_config(cfg: HnswConfig) -> Self {
            Self {
                rng: cfg.seed,
                cfg,
                entries: Vec::new(),
                nodes: Vec::new(),
                pos: HashMap::new(),
                entry: None,
                top_layer: 0,
            }
        }

        /// Number of indexed vectors.
        pub fn len(&self) -> usize {
            self.entries.len()
        }

        /// True when no vectors have been inserted.
        pub fn is_empty(&self) -> bool {
            self.entries.is_empty()
        }

        /// All indexed entries — borrowed for the redb checkpoint path (zero-copy).
        pub fn entries(&self) -> &[Entry] {
            &self.entries
        }

        fn next_unit(&mut self) -> f64 {
            let u = (splitmix64(&mut self.rng) >> 11) as f64 / (1u64 << 53) as f64;
            1.0 - u // (0, 1] so ln is finite
        }

        fn random_level(&mut self) -> usize {
            let r = self.next_unit();
            ((-r.ln()) * self.cfg.ml).floor() as usize
        }

        /// HNSW `SEARCH-LAYER`: the `ef` nearest nodes to `query` reachable on `layer` from
        /// `entry_points`, sorted nearest-first.
        fn search_layer(&self, query: &[f32], entry_points: &[usize], ef: usize, layer: usize) -> Vec<Cand> {
            let mut visited: HashSet<usize> = HashSet::new();
            let mut candidates: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
            let mut results: BinaryHeap<Cand> = BinaryHeap::new();

            for &ep in entry_points {
                let c = Cand { dist: dist(query, &self.entries[ep].vec), id: ep };
                visited.insert(ep);
                candidates.push(Reverse(c));
                results.push(c);
            }

            while let Some(Reverse(c)) = candidates.pop() {
                if let Some(farthest) = results.peek() {
                    if c.dist > farthest.dist && results.len() >= ef {
                        break;
                    }
                }
                for &e in &self.nodes[c.id].links[layer] {
                    if visited.insert(e) {
                        let d = dist(query, &self.entries[e].vec);
                        let admit = results.len() < ef || results.peek().is_some_and(|f| d < f.dist);
                        if admit {
                            let cand = Cand { dist: d, id: e };
                            candidates.push(Reverse(cand));
                            results.push(cand);
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }

            let mut out: Vec<Cand> = results.into_vec();
            out.sort_unstable();
            out
        }

        /// Re-select `id`'s neighbor list at `layer` to the `cap` closest (the simple heuristic).
        fn prune(&mut self, id: usize, layer: usize, cap: usize) {
            if self.nodes[id].links[layer].len() <= cap {
                return;
            }
            let base = self.entries[id].vec.clone();
            let mut cands: Vec<Cand> = self.nodes[id].links[layer]
                .iter()
                .map(|&nb| Cand { dist: dist(&base, &self.entries[nb].vec), id: nb })
                .collect();
            cands.sort_unstable();
            cands.truncate(cap);
            self.nodes[id].links[layer] = cands.into_iter().map(|c| c.id).collect();
        }

        /// Rebuild the full HNSW graph from the current entry list. Used when a vector changes to
        /// relink all neighbors; O(N log N) but vectors change only on re-embeds with a new
        /// backend, which is rare.
        fn rebuild_from_entries(&mut self) {
            let cfg = self.cfg;
            let entries = std::mem::take(&mut self.entries);
            *self = Self::with_config(cfg);
            for entry in entries {
                self.upsert(entry.id, entry.vec);
            }
        }

        /// Idempotent upsert. Re-inserting a known id with the same vector is a no-op (WAL
        /// idempotency). If the vector changed (embedder swap, re-embed), the graph is rebuilt so
        /// stale links from the old position don't persist.
        pub fn upsert(&mut self, key: NodeId, vec: Vec<f32>) {
            if let Some(&id) = self.pos.get(&key) {
                if self.entries[id].vec == vec {
                    return; // identical — no-op
                }
                self.entries[id].vec = vec;
                self.rebuild_from_entries();
                return;
            }

            let level = self.random_level();
            let id = self.entries.len();
            self.entries.push(Entry { id: key, vec });
            self.nodes.push(GraphNode { links: vec![Vec::new(); level + 1] });
            self.pos.insert(key, id);

            let Some(entry_id) = self.entry else {
                self.entry = Some(id);
                self.top_layer = level;
                return;
            };

            let query = self.entries[id].vec.clone();
            let mut ep_ids = vec![entry_id];

            // Phase 1: greedy descent from the top layer down to just above the new node's level.
            if self.top_layer > level {
                for lc in ((level + 1)..=self.top_layer).rev() {
                    let w = self.search_layer(&query, &ep_ids, 1, lc);
                    if let Some(best) = w.first() {
                        ep_ids = vec![best.id];
                    }
                }
            }

            // Phase 2: connect bidirectionally from min(top, level) down to 0.
            let start = self.top_layer.min(level);
            for lc in (0..=start).rev() {
                let w = self.search_layer(&query, &ep_ids, self.cfg.ef_construction, lc);
                let cap = if lc == 0 { self.cfg.m0 } else { self.cfg.m };
                let selected: Vec<usize> = w.iter().take(cap).map(|c| c.id).collect();
                self.nodes[id].links[lc] = selected.clone();
                for &nb in &selected {
                    self.nodes[nb].links[lc].push(id);
                    self.prune(nb, lc, cap);
                }
                ep_ids = w.iter().map(|c| c.id).collect();
            }

            if level > self.top_layer {
                self.entry = Some(id);
                self.top_layer = level;
            }
        }

        /// Approximate top-`k` by cosine similarity, descending.
        pub fn search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)> {
            let Some(entry_id) = self.entry else {
                return Vec::new();
            };
            let mut ep_ids = vec![entry_id];
            for lc in (1..=self.top_layer).rev() {
                let w = self.search_layer(query, &ep_ids, 1, lc);
                if let Some(best) = w.first() {
                    ep_ids = vec![best.id];
                }
            }
            let width = self.cfg.ef_search.max(k);
            self.search_layer(query, &ep_ids, width, 0)
                .into_iter()
                .take(k)
                .map(|c| (self.entries[c.id].id, 1.0 - c.dist))
                .collect()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::index::BruteForceIndex;

        fn unit01(s: &mut u64) -> f32 {
            (splitmix64(s) >> 11) as f32 / (1u64 << 53) as f32
        }

        /// Uniform-random L2-normalized vectors. At the 256-dim embedding width these are
        /// near-orthogonal (all pairwise cosine ≈ 0), so a lightly-perturbed copy of one has an
        /// unambiguous nearest neighbor — the right way to measure ANN navigability at high dim
        /// (uniform-random *queries* against uniform-random corpus would have an ill-defined NN
        /// and make recall@1 meaningless — the "256-dim collapse" artifact the recall ann-bench
        /// flagged; the planted-NN protocol below is what makes the metric honest).
        fn random_unit(n: usize, dim: usize, mut seed: u64) -> Vec<Vec<f32>> {
            (0..n)
                .map(|_| {
                    let mut v: Vec<f32> = (0..dim).map(|_| unit01(&mut seed) * 2.0 - 1.0).collect();
                    normalize(&mut v);
                    v
                })
                .collect()
        }

        fn normalize(v: &mut [f32]) {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in v {
                    *x /= norm;
                }
            }
        }

        #[test]
        fn self_query_returns_self() {
            let mut idx = HnswIndex::new();
            let vecs = random_unit(300, crate::embed::DIM, 1);
            for (i, v) in vecs.iter().enumerate() {
                idx.upsert(i as NodeId, v.clone());
            }
            assert_eq!(idx.len(), 300);
            for (i, v) in vecs.iter().enumerate() {
                let hits = idx.search(v, 1);
                assert_eq!(hits[0].0, i as NodeId, "self-query must return self (connectivity)");
                assert!(hits[0].1 > 0.999, "self-similarity ~1.0");
            }
        }

        #[test]
        fn upsert_replaces_in_place() {
            let mut idx = HnswIndex::new();
            idx.upsert(1, vec![1.0, 0.0]);
            idx.upsert(2, vec![0.0, 1.0]);
            idx.upsert(1, vec![0.0, 1.0]); // re-insert id 1 → replace, not append
            assert_eq!(idx.len(), 2, "idempotent: count stays");
            let hit = idx.search(&[0.0, 1.0], 1);
            assert!((hit[0].1 - 1.0).abs() < 1e-6, "replaced vector is in effect");
        }

        // THE gate (PLAN.md §5.3): at the real 256-dim embedding width, HNSW top-1 must agree
        // with the exact brute-force oracle on >= 98% of queries. Queries are corpus vectors with
        // a small perturbation (the standard planted-NN ANN protocol), so each has a well-defined
        // true nearest neighbor and the metric measures navigability, not a degenerate tie.
        #[test]
        fn recall_at_1_matches_brute_force_oracle_256d() {
            let dim = crate::embed::DIM; // 256
            let corpus = random_unit(800, dim, 0xABCD);

            let mut oracle = BruteForceIndex::new();
            let mut hnsw = HnswIndex::new();
            for (i, v) in corpus.iter().enumerate() {
                oracle.upsert(i as NodeId, v.clone());
                hnsw.upsert(i as NodeId, v.clone());
            }

            let mut noise_seed = 0x1234u64;
            let mut agree = 0usize;
            let trials = 300usize;
            for _ in 0..trials {
                let base = (splitmix64(&mut noise_seed) as usize) % corpus.len();
                let mut q: Vec<f32> = corpus[base]
                    .iter()
                    .map(|x| x + (unit01(&mut noise_seed) * 2.0 - 1.0) * 0.05)
                    .collect();
                normalize(&mut q);
                let truth = oracle.search(&q, 1);
                let approx = hnsw.search(&q, 1);
                if let (Some(t), Some(a)) = (truth.first(), approx.first()) {
                    if t.0 == a.0 {
                        agree += 1;
                    }
                }
            }
            let recall = agree as f64 / trials as f64;
            assert!(recall >= 0.98, "recall@1 vs brute-force oracle = {recall:.4} (< 0.98)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsert_replaces_in_place_no_dup() {
        let mut ix = BruteForceIndex::new();
        ix.upsert(1, vec![1.0, 0.0]);
        ix.upsert(2, vec![0.0, 1.0]);
        ix.upsert(1, vec![0.0, 1.0]); // re-insert id 1 → replace, not append
        assert_eq!(ix.len(), 2, "idempotent: re-insert replaces, count stays");
        // id 1 now matches the [0,1] query exactly (its vector was replaced).
        let hit = ix.search(&[0.0, 1.0], 2);
        assert!((hit[0].1 - 1.0).abs() < 1e-6, "replaced vector is in effect");
        // pos stays consistent: every id resolves to the right slot after replace.
        assert_eq!(ix.entries()[ix.pos[&1]].id, 1);
        assert_eq!(ix.entries()[ix.pos[&2]].id, 2);
    }
}
