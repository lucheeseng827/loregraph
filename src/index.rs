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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

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
