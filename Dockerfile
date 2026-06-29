# loregraph — distroless image for `lore`.
#
# The default build is pure-Rust with zero C/ML/network deps, so the binary links fully static
# against musl (Alpine's native target) and drops into `distroless/static` — no libc, no shell,
# no package manager, runs as nonroot. Multi-arch (linux/amd64 + linux/arm64) is produced by
# `docker buildx`: the build stage runs per target platform, so `cargo build` emits the native
# static binary for each without a cross-linker.
#
#   docker buildx build --platform linux/amd64,linux/arm64 -t mancube/loregraph .
#
# Default (air-gap) feature set only — the `mcp` / `neural` / `byo-llm` / `index-hnsw` features
# are opt-in and built from source by users who want them.

FROM rust:1-alpine AS build
RUN apk add --no-cache musl-dev
WORKDIR /src
COPY . .
# Alpine's host target is *-unknown-linux-musl → a fully static binary by default.
RUN cargo build --release --bin lore && \
    strip target/release/lore

FROM gcr.io/distroless/static-debian12:nonroot
LABEL org.opencontainers.image.source="https://github.com/lucheeseng827/loregraph" \
      org.opencontainers.image.description="Memory knowledge-graph for AI coding-agent sessions + repos (single binary)" \
      org.opencontainers.image.licenses="Apache-2.0"
COPY --from=build /src/target/release/lore /usr/local/bin/lore
# `lore serve` default bind.
EXPOSE 7700
ENTRYPOINT ["/usr/local/bin/lore"]
CMD ["version"]
