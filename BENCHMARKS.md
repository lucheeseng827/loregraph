# loregraph — Performance & headroom

> Measured numbers for capacity planning + docs. These are **performance** figures (time /
> memory / disk), not recall-quality (recall@k lands separately when the semantic embedder is
> benchmarked). Re-run the methodology below to refresh. Last measured 2026-06-29.

## Test setup

| | |
|---|---|
| Build | **release** (`cargo build --release`) — the shipped size-opt profile (`opt-level="z"`, `lto`, `codegen-units=1`, `strip`) |
| Host | Windows 11, local SSD |
| Embedder | default `HashEmbedder` (256-dim, lexical) + `BruteForceIndex` (exact cosine) |
| Corpus | a real `~/.claude/projects` (72 Claude Code sessions, ~47k turns) **+** the loregraph monorepo |
| Graph | **24,686 nodes · 27,404 edges · 23,558 embeddings** (256×f32 = 1 KB/vector) |

## Headline numbers

| Operation | Time | Notes |
|---|---|---|
| **`lore index` (cold, full)** | **1.76 s** | 47k turns parsed/redacted/embedded + repo crunch + gix + redb checkpoint |
|  — sessions only | 0.92 s | 47k turns → chat nodes (incl. its own redb save) |
|  — repo only | 1.45 s | symbol scan + gix 400-commit history & diffs + save |
| **`lore ask` (open + recall)** | **~0.25 s** | dominated by rehydrating 24.7k nodes + 23.5k vectors from redb; recall itself is sub-ms |
| **`lore serve` resident RSS** | **148 MB** | the whole graph + vector index held in RAM (no spill) |

### `lore serve` API latency (warm, in-RAM)

| Endpoint | Latency | Payload |
|---|---|---|
| `/healthz` | 0.8 ms | 2 B |
| `/v1/graph?limit=400` (bulk) | 2.6 ms | 113 KB (T2: summaries, not full bodies) |
| `/v1/search?q=…&k=8` (recall) | 6.2 ms | 4.9 KB — BruteForce O(N) over 23.5k vectors |
| `/v1/graph/neighbors?id=…&depth=1–2` | 0.7–0.8 ms | adjacency-indexed (O(degree), not O(E)) |
| `/v1/node/{id}` (drill-down) | 0.8 ms | full body |

### On disk

| File | Size | |
|---|---|---|
| `graph.redb` | 129 MB | nodes/edges as JSON, vectors as raw f32 bytes; includes redb free pages, so logical data is less |
| `wal.log` | 0 B | truncated after the checkpoint (`save()`) |

## Overhead model (rules of thumb for headroom planning)

The whole graph is **RAM-resident** (rehydrated from redb at open; there is no paging/spill),
so RAM is the capacity ceiling — not throughput. From the figures above:

| Resource | Per-node | Project to scale |
|---|---|---|
| **RAM (resident)** | ~6 KB/node | 100k nodes ≈ **0.6 GB** · 500k ≈ **3 GB** · 1M ≈ **6 GB** |
| **Disk (redb)** | ~5 KB/node | 100k ≈ 0.5 GB · 1M ≈ 5 GB |
| **Index throughput** | ~14k nodes/s (≈27k turns/s ingest) | 1M nodes ≈ ~70 s cold |
| **Recall (BruteForce)** | ~0.25 µs/vector | 23.5k → 6 ms · 100k ≈ 26 ms · 500k ≈ 130 ms · 1M ≈ 260 ms |

> A node's resident cost is its `id`/kind/timestamps + `label`/`body`/`summary` strings +
> provenance, plus (for embedded nodes) a 1 KB vector in the index. Bodies dominate; a chat-heavy
> corpus skews larger, a code-heavy one smaller.

## Recommended headroom

| Scale (nodes) | RAM | CPU | Disk |
|---|---|---|---|
| Single repo, a few months of sessions (~10–30k) | **256 MB** | 1 core | ~150 MB |
| Busy dev, big repo, ~1 yr (~100k) | **1 GB** | 1–2 cores | ~0.5 GB |
| Monorepo + heavy history (~1M) | **8 GB** | 2–4 cores | ~5 GB |

- **CPU**: `index` is the only real load (1–2 cores, seconds for this corpus, largely
  sequential — extra cores help little). `serve`/`ask` are near-idle (sub-10 ms queries, one
  core). Idle `serve` ≈ 0% CPU (event-driven loopback).
- **Recall ceiling**: BruteForce is exact and fine to ~500k vectors (sub-150 ms). Past ~1M,
  query latency climbs linearly until the planned `index-hnsw` feature lands.
- **The hard limit is RAM** (full-resident design). A graph that doesn't fit RAM won't run; the
  DataFusion/Parquet off-load tier is design-only (not built).
- Release vs debug: index is ~**4–5× faster** in release (1.76 s vs ~8 s debug). Resident RAM is
  build-independent (~148 vs ~150 MB) — the graph dominates, not the code.

## Methodology (to reproduce)

```bash
cargo build --release
LORE=target/release/lore.exe
# cold index against your real corpus
time $LORE index --sessions ~/.claude/projects --repo . --data-dir /tmp/bench
du -sh /tmp/bench                                  # on-disk size
# open + recall latency
time $LORE ask "storage engine decision" --data-dir /tmp/bench
# serve, then measure RSS + endpoint latency
$LORE serve --addr 127.0.0.1:7801 --data-dir /tmp/bench &
curl -s -o /dev/null -w "%{time_total}s %{size_download}B\n" "http://127.0.0.1:7801/v1/search?q=storage&k=8"
# RSS: ps / Get-Process lore | WorkingSet64
```

Numbers scale with graph size and corpus shape; re-measure on your hardware for an exact figure.
