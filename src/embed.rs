//! Embedder seam — turn text into a vector (PLAN.md §7, §5.3 R2).
//!
//! Two backends behind one trait, selected at runtime (no rebuild needed):
//!
//! - [`HashEmbedder`] — the deps-free default. A deterministic hashed token-bag: it captures
//!   **lexical overlap only** (no synonyms), but keeps the default build pure-Rust / air-gap
//!   and makes the retrieval path exercisable end-to-end.
//! - [`StaticEmbedder`] — real **semantic** recall via *static word embeddings* (the
//!   model2vec idea: a precomputed token→vector table, mean-pooled — **no neural-net
//!   inference, no ONNX/C, no network at query time**). Point it at a local vectors file
//!   (GloVe text format, or a model2vec/word2vec export converted to it). This is R2: the
//!   recall quality ceiling-raiser, still air-gap-clean because the model is a local file
//!   the operator supplies, loaded once.
//!
//! Heavier neural backends (fastembed/candle = transformer inference) remain future work
//! behind a cargo feature; the static table needs no extra dependency, so it always compiles
//! and is chosen at runtime via `LORE_EMBED_BACKEND` / `lore.toml`.

use std::collections::HashMap;
use std::path::Path;

/// Embedding dimensionality for [`HashEmbedder`]. [`StaticEmbedder`]'s dim comes from its file.
pub const DIM: usize = 256;

/// Turn text into a vector. `id()` stamps which embedder produced a stored vector (so a store
/// can refuse to mix vectors from different embedders); `dim()` is the vector length.
pub trait Embedder {
    /// Stable id, e.g. `hash-v1` or `static:glove-50d:50`. Persisted with the index.
    fn id(&self) -> String;
    /// Vector length this embedder emits.
    fn dim(&self) -> usize;
    /// Embed `text` into an L2-normalized vector of length [`Embedder::dim`].
    fn embed(&self, text: &str) -> Vec<f32>;
}

/// Tokenize the way both backends agree on: lowercased alphanumeric runs, length > 1. Keeping
/// query and document tokenization identical is what lets a static vocab line up at all.
fn tokens(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(|t| t.to_lowercase())
}

/// Deterministic hashed token-bag embedder (default, deps-free, lexical-only).
#[derive(Clone, Copy, Default)]
pub struct HashEmbedder;

impl Embedder for HashEmbedder {
    fn id(&self) -> String {
        "hash-v1".to_string()
    }

    fn dim(&self) -> usize {
        DIM
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; DIM];
        for t in tokens(text) {
            let b = blake3::hash(t.as_bytes());
            let h = b.as_bytes();
            let idx = (u32::from_le_bytes([h[0], h[1], h[2], h[3]]) as usize) % DIM;
            let sign = if h[4] & 1 == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }
        l2_normalize(&mut v);
        v
    }
}

/// Static word-embedding backend (R2). Loads a token→vector table once and embeds a string by
/// mean-pooling its known tokens. Real semantics (synonyms land near each other) when pointed
/// at trained vectors, with zero inference cost and no network at query time.
pub struct StaticEmbedder {
    id: String,
    dim: usize,
    table: HashMap<String, Vec<f32>>,
}

impl StaticEmbedder {
    /// Load a GloVe-style text vectors file: one token per line, `token f1 f2 … fN`
    /// (space-separated). The dimension is taken from the first data line; lines whose width
    /// disagrees are skipped (lenient). An optional `<vocab> <dim>` header line is tolerated.
    /// Streams the file line-by-line to avoid doubling memory for large models.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        use std::io::{BufRead, BufReader};
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("reading embeddings file {}: {e}", path.display()))?;
        let mut table: HashMap<String, Vec<f32>> = HashMap::new();
        let mut dim = 0usize;
        let mut hasher = blake3::Hasher::new();
        for line in BufReader::new(file).lines() {
            let line = line?;
            hasher.update(line.as_bytes());
            hasher.update(b"\n");
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let mut it = trimmed.split_whitespace();
            let Some(tok) = it.next() else { continue };
            let raw_vals: Vec<&str> = it.collect();
            let mut vals = Vec::with_capacity(raw_vals.len());
            let mut malformed = false;
            for s in raw_vals {
                match s.parse::<f32>() {
                    Ok(x) => vals.push(x),
                    Err(_) => {
                        malformed = true;
                        break;
                    }
                }
            }
            if malformed {
                continue; // skip the whole row — a partial vector corrupts recall
            }
            // A 2-field `<vocab> <dim>` header (word2vec) parses to a 1-float "vector"; skip it.
            if vals.len() < 2 {
                continue;
            }
            if dim == 0 {
                dim = vals.len();
            }
            if vals.len() != dim {
                continue; // ragged line — skip rather than corrupt the table
            }
            table.insert(tok.to_lowercase(), vals);
        }
        if dim == 0 || table.is_empty() {
            anyhow::bail!(
                "no usable vectors parsed from {} (expected GloVe text: `token f1 f2 …`)",
                path.display()
            );
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        let model_hash = hasher.finalize().to_hex().to_string();
        Ok(StaticEmbedder {
            id: format!("static:{stem}:{dim}:{model_hash}"),
            dim,
            table,
        })
    }

    /// Build directly from a token→vector table (used by tests).
    pub fn from_table(id: impl Into<String>, table: HashMap<String, Vec<f32>>) -> Self {
        let dim = table.values().next().map(|v| v.len()).unwrap_or(0);
        StaticEmbedder { id: id.into(), dim, table }
    }
}

impl Embedder for StaticEmbedder {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        let mut n = 0u32;
        for t in tokens(text) {
            if let Some(row) = self.table.get(&t) {
                for (acc, x) in v.iter_mut().zip(row) {
                    *acc += x;
                }
                n += 1;
            }
        }
        if n > 0 {
            let inv = 1.0 / n as f32;
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
        l2_normalize(&mut v);
        v
    }
}

/// Runtime-selected embedder. Default is `Hash`; `Static` is chosen via env/config and loads a
/// local vectors file. An enum (not `dyn`) keeps it object-simple and `Sized` for the store.
pub enum DynEmbedder {
    Hash(HashEmbedder),
    Static(StaticEmbedder),
}

impl DynEmbedder {
    /// Select the embedder from the environment:
    /// - `LORE_EMBED_BACKEND` = `hash` (default) | `static` (alias `localmodel`)
    /// - `LORE_EMBED_MODEL`   = path to the vectors file (required for `static`)
    pub fn from_env() -> anyhow::Result<Self> {
        let backend = std::env::var("LORE_EMBED_BACKEND").unwrap_or_else(|_| "hash".to_string());
        match backend.as_str() {
            "hash" => Ok(DynEmbedder::Hash(HashEmbedder)),
            "static" | "localmodel" => {
                let path = std::env::var("LORE_EMBED_MODEL").map_err(|_| {
                    anyhow::anyhow!(
                        "LORE_EMBED_BACKEND={backend} needs LORE_EMBED_MODEL=<vectors.txt> \
                         (GloVe text format)"
                    )
                })?;
                Ok(DynEmbedder::Static(StaticEmbedder::load(Path::new(&path))?))
            }
            other => anyhow::bail!("unknown LORE_EMBED_BACKEND {other:?} (expected: hash | static)"),
        }
    }
}

impl Embedder for DynEmbedder {
    fn id(&self) -> String {
        match self {
            DynEmbedder::Hash(e) => e.id(),
            DynEmbedder::Static(e) => e.id(),
        }
    }
    fn dim(&self) -> usize {
        match self {
            DynEmbedder::Hash(e) => e.dim(),
            DynEmbedder::Static(e) => e.dim(),
        }
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        match self {
            DynEmbedder::Hash(e) => e.embed(text),
            DynEmbedder::Static(e) => e.embed(text),
        }
    }
}

/// Normalize in place so a dot product is a cosine similarity.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similar_text_scores_higher() {
        let e = HashEmbedder;
        let a = e.embed("we decided to use redb for the durable store");
        let b = e.embed("the durable store uses redb");
        let c = e.embed("the canvas renders nodes with a force simulation");
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
        assert!(dot(&a, &b) > dot(&a, &c), "lexically closer text is nearer");
    }

    #[test]
    fn static_embedder_is_semantic_via_the_table() {
        // Hand-built table where "car" and "automobile" share a vector but "banana" is
        // orthogonal — proves mean-pool + cosine surfaces a synonym the hash backend can't.
        let mut t = HashMap::new();
        t.insert("car".to_string(), vec![1.0, 0.0]);
        t.insert("automobile".to_string(), vec![1.0, 0.0]);
        t.insert("banana".to_string(), vec![0.0, 1.0]);
        let e = StaticEmbedder::from_table("static:test:2", t);
        assert_eq!(e.dim(), 2);
        assert_eq!(e.id(), "static:test:2");
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
        let q = e.embed("car");
        assert!(
            dot(&q, &e.embed("automobile")) > dot(&q, &e.embed("banana")),
            "synonym (shared vector) beats an unrelated word"
        );
        // An all-unknown string yields a zero vector (no panic, no NaN).
        assert!(e.embed("zzz qqq").iter().all(|x| *x == 0.0));
    }
}
