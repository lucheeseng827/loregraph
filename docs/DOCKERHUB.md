# loregraph (`mancube/loregraph`)

**A single static Rust binary that turns the chat transcripts your AI coding agents already write to disk into a persistent, provenance-anchored memory knowledge graph of your decisions and the code they shaped.**

loregraph reads on-disk agent transcripts (Claude Code JSONL, Codex CLI rollouts), fuses them with your repo + git history, and builds a graph of `Decision` and `Implementation` nodes — every value node linked by hard provenance edges back to the exact `session_id` **and** repo `commit`. Browse it on a pan/zoom canvas, or query it headless so an agent self-recalls instead of re-asking the model.

- **Image:** `mancube/loregraph` — static musl binary on **distroless/static**, runs as **nonroot** (uid `65532`), no shell, no package manager.
- **Arch:** `linux/amd64`, `linux/arm64` · **Binary inside:** `/usr/local/bin/lore` (entrypoint) · **Exposes:** `7700` (the `serve` canvas/API)
- **Build:** the **default** feature set — pure Rust, zero Python/C/ML/network. The embedded canvas SPA is baked in (no Node). The opt-in `mcp` / `neural` / `byo-llm` / `index-hnsw` features are **source-only** (build from source if you want them).
- **Source / full docs:** [github.com/lucheeseng827/loregraph](https://github.com/lucheeseng827/loregraph) · Apache-2.0

## Tags

| Tag | Notes |
|---|---|
| `latest` | newest stable release |
| `0.1.0` | first release — Claude Code + Codex ingest, recall R1–R4/T1–T2, canvas, `doctor` |
| `*-rc.*` | pre-release smoke builds (not tagged `latest`) — don't use in production |

Pin a version in production: `mancube/loregraph:0.1.0`.

## Quick start

The binary is the entrypoint, so the `docker` args are just `lore` subcommands (`index` / `serve` / `ask` / `doctor` / `version`). Mount your data dir, transcripts, and repo as volumes — loregraph reads them **read-only** and writes only its graph.

```bash
docker run --rm mancube/loregraph:0.1.0 version
```

**1. Build the graph** — point it at the transcripts your agents already wrote + your repo, and a persistent data dir:

```bash
docker run --rm \
  -v "$HOME/.claude/projects:/sessions:ro" \
  -v "$PWD:/repo:ro" \
  -v "$PWD/.lore:/data" \
  mancube/loregraph:0.1.0 \
  index --sessions /sessions --repo /repo --data-dir /data
```

**2. Browse it** — the embedded canvas + JSON API. Bind `0.0.0.0` inside the container (the default `127.0.0.1` isn't reachable from the host):

```bash
docker run --rm -p 7700:7700 \
  -v "$PWD/.lore:/data" \
  mancube/loregraph:0.1.0 \
  serve --addr 0.0.0.0:7700 --data-dir /data
# open http://127.0.0.1:7700/
```

**3. Ask it** — headless recall with citable provenance:

```bash
docker run --rm \
  -v "$PWD/.lore:/data" \
  mancube/loregraph:0.1.0 \
  ask "what did we decide about retries?" --data-dir /data
# add --json for a machine-readable RecallResult
```

> **Permissions:** the image runs as nonroot (uid `65532`), so the mounted `/data` dir must be writable by that uid — e.g. `mkdir -p .lore && chmod 777 .lore` (or `chown 65532:65532 .lore`) before the first `index`. Transcript/repo mounts are read-only and need no change.

## Commands

| Command | What it does |
|---|---|
| `index --sessions <dir> --repo <dir> --data-dir <dir>` | ingest transcripts + repo into the graph (idempotent; redacts secrets on ingest) |
| `serve --addr 0.0.0.0:7700 --data-dir <dir>` | axum API + embedded pan/zoom canvas |
| `ask "<query>" --data-dir <dir> [-k N] [--json]` | headless recall → decision + session + file + commit |
| `doctor [--source claude_code]` | per-connector discovery + drift report |
| `version` | print the version |

`index` defaults: `--source claude_code`, `--data-dir .lore`. Add `--dry-run` to parse + report without writing.

## What's inside the graph

- **Value nodes** — `Decision` / `Implementation`, each anchored to a `session_id` **and** the contemporaneous repo `commit` (so you can answer *who decided what, when, and which code it shaped*).
- **Provenance edges** — `DecidedIn`, `Implements`, `Touches` (chat turn ↔ file/span), `ChangedBy` (commit), `Supersedes` (an explicit "… instead of X …" supersession).
- **Durable store** — redb + a crash-safe write-ahead log; re-running `index` is a content-addressed, idempotent upsert.

## Air-gap friendly

The default build (this image) makes **zero network calls** — it only reads the local files you mount. It runs disconnected, on a laptop, in a locked-down CI runner, or on an air-gapped host. distroless/static + nonroot keeps the attack surface to just the binary.

## License

Apache-2.0. The name `loregraph` (binary `lore`) is a working title.
