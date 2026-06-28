#!/usr/bin/env bash
# Vendor the canvas's prebuilt graph-rendering bundles into frontend/vendor/, pinned by
# version + SHA-256, each with its upstream LICENSE. This runs ONLY on a version bump by a
# maintainer with network — NEVER at a user's `cargo build` (the bundles are committed, so
# the binary builds air-gapped). The PoC ships a hand-rolled Canvas2D renderer and vendors
# nothing; this script lands the MVP/scale renderers (PLAN.md §6).
#
#   MVP default tier : Cytoscape.js (Canvas, MIT)        — ~3-5k nodes
#   Scale tier       : Sigma.js + graphology (WebGL,MIT) — 5k-100k nodes
#
# Usage:  ops/vendor-frontend.sh
set -euo pipefail

VENDOR_DIR="$(cd "$(dirname "$0")/.." && pwd)/frontend/vendor"
mkdir -p "$VENDOR_DIR"

# Portable SHA-256 (GNU sha256sum on Linux, shasum on macOS).
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else echo "need sha256sum or shasum on PATH" >&2; exit 1; fi
}

# name | version | url | sha256 (fill SHAs in when first vendoring; the script verifies them)
PKGS=(
  "cytoscape|3.30.2|https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.umd.js|REPLACE_WITH_SHA256"
  "cytoscape-fcose|2.2.0|https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js|REPLACE_WITH_SHA256"
  # scale tier (uncomment when the Canvas ceiling is hit by a real graph):
  # "sigma|3.0.1|https://unpkg.com/sigma@3.0.1/dist/sigma.min.js|REPLACE_WITH_SHA256"
  # "graphology|0.25.4|https://unpkg.com/graphology@0.25.4/dist/graphology.umd.min.js|REPLACE_WITH_SHA256"
)

for entry in "${PKGS[@]}"; do
  IFS='|' read -r name version url sha <<<"$entry"
  dest_dir="$VENDOR_DIR/$name"
  mkdir -p "$dest_dir"
  out="$dest_dir/$(basename "$url")"
  echo "→ $name@$version"
  curl -fsSL "$url" -o "$out"
  got="$(sha256_of "$out")"
  if [[ "$sha" == "REPLACE_WITH_SHA256" ]]; then
    # No pinned digest yet: refuse to accept an unverified download unless explicitly
    # bootstrapping, so the verified path can never silently pass with a placeholder.
    if [[ "${LORE_VENDOR_BOOTSTRAP:-0}" != "1" ]]; then
      echo "  ! $name has no pinned SHA-256. Re-run with LORE_VENDOR_BOOTSTRAP=1 to accept the" >&2
      echo "    first download, then paste this digest into PKGS to lock it: $got" >&2
      exit 1
    fi
    echo "  (bootstrap) sha256=$got — record it in PKGS to lock $name"
  elif [[ "$sha" != "$got" ]]; then
    echo "  ! SHA-256 mismatch for $name: expected $sha, got $got" >&2
    exit 1
  else
    echo "  sha256=$got verified ($out)"
  fi
  echo "  → drop the upstream LICENSE next to $out."
done

echo "Done. Commit frontend/vendor/ so the binary keeps building air-gapped."
