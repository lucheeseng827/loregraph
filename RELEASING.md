# Releasing loregraph

Releases are cut by tag. `.github/workflows/release.yml` does the rest.

## Cut a release

1. Bump `version` in `Cargo.toml`, update `CHANGELOG`/notes, land on `main`.
2. Tag and push:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
3. The `release` workflow then, on that tag:
   - **builds** static-musl (`x86_64`/`aarch64`), macOS (`x86_64`/`arm64`), and Windows
     (`x86_64`) binaries — each archived as `lore-<target>.{tar.gz,zip}` with a `.sha256`;
   - **publishes** the GitHub release with every artifact + an aggregated `SHA256SUMS`;
   - **bumps** `Formula/lore.rb` (version + the four desktop sha256) and commits it to `main`,
     so `brew upgrade lore` serves the new build;
   - **pushes** a multi-arch (`linux/amd64` + `linux/arm64`) distroless image to
     `docker.io/mancube/loregraph:{vX.Y.Z,latest}`.

`workflow_dispatch` re-runs the pipeline against an existing tag (input `tag`).

## Distribution channels

| Channel | Source of truth |
|---|---|
| `cargo binstall loregraph` | release tarballs; asset names from `[package.metadata.binstall]` in `Cargo.toml` |
| `cargo install loregraph` | crates.io (published by the release workflow on each real tag) |
| `brew install lore` | `Formula/lore.rb` (this repo is its own tap) |
| `docker run … mancube/loregraph` | `Dockerfile` (distroless, self-building musl) |
| `cargo install --git …` | the crate source |

## Verify a build locally

```bash
# the exact static target the release ships
cargo build --release --target x86_64-unknown-linux-musl

# the container, both arches
docker buildx build --platform linux/amd64,linux/arm64 -t loregraph:dev .
```
