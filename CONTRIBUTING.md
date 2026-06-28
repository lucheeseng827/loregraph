# Contributing to loregraph

Thanks for your interest. loregraph (the engine, the canvas, the connectors) is
**Apache-2.0** and is the same code that runs in production.

## Developer Certificate of Origin (DCO)

Contributions are accepted under the [Developer Certificate of Origin](https://developercertificate.org/).
Sign off every commit:

```bash
git commit -s -m "your message"
```

The `Signed-off-by` line certifies you wrote the patch or have the right to submit it under
the project's license.

## Ground rules that keep the project honest

- **Default build stays pure-Rust.** No ML / network / C dependencies in the default build —
  air-gap is a feature, not an accident. Anything heavy (tree-sitter, local embedding models,
  alternate ANN backends) goes behind a cargo feature, off by default.
- **Lenient + lossless ingest.** Connectors must never hard-fail on an unrecognized record;
  unknown shapes become `Unknown` turns and retain their raw form. Add a golden fixture for
  every producer/version you parse.
- **Redact before persist.** Secrets are redacted in the connector's `lower()` step, before
  anything is written, embedded, or hashed.
- **Be honest in docs.** The README's "what is and isn't a moat" section is load-bearing.
  Don't add overclaims; name real competitors.

## Building & testing

```bash
cargo test                 # the engine + connectors + canvas API
cargo run  -- doctor       # what connectors discover on your machine
cargo run  -- index --sessions ~/.claude/projects --repo .
cargo run  -- serve        # canvas + API on http://127.0.0.1:7700
```

See `PLAN.md` for the phased plan and `ARCHITECTURE.md` for the engine internals.
