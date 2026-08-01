#!/usr/bin/env bash
# Reconstruct the patched KernelBench checkout that the cuTile backend runs in.
#
# This repo deliberately does not vendor KernelBench. Instead it carries:
#   overlay/  - the files the cuTile backend adds (93% of the change)
#   patches/  - the ~280 lines that modify KernelBench's own files
#
# Running this clones upstream at the pinned commit, copies the overlay in, and
# applies the patch, producing ./kernelbench/ ready to use.
#
# Usage:  scripts/setup_kernelbench.sh [--force]
set -euo pipefail

FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKOUT="${KERNELBENCH_DIR:-$FORGE/kernelbench}"
UPSTREAM="${KERNELBENCH_REMOTE:-https://github.com/ScalingIntelligence/KernelBench.git}"
PINNED="$(tr -d '[:space:]' < "$FORGE/upstream.lock")"

if [[ -e "$CHECKOUT" ]]; then
    if [[ "${1:-}" == "--force" ]]; then
        echo "removing existing $CHECKOUT"
        rm -rf "$CHECKOUT"
    else
        echo "error: $CHECKOUT already exists (pass --force to recreate)" >&2
        exit 1
    fi
fi

echo "==> cloning KernelBench at $PINNED"
git clone --quiet "$UPSTREAM" "$CHECKOUT"
git -C "$CHECKOUT" checkout --quiet "$PINNED"

echo "==> copying overlay"
cp -r "$FORGE/overlay/." "$CHECKOUT/"
chmod +x "$CHECKOUT"/scripts/*.sh

echo "==> applying patch"
git -C "$CHECKOUT" apply --verbose "$FORGE/patches/0001-cutile-backend.patch"

# The golden solutions are referenced by path from inside the checkout, so make
# them reachable without duplicating them into the overlay. The link must be
# relative: the checkout is bind-mounted into the eval container at a different
# absolute path, and an absolute link would dangle there.
ln -sfn ../golden "$CHECKOUT/golden"

echo
echo "kernelbench ready at: $CHECKOUT"
echo
echo "next steps:"
echo "  1. build the eval image:"
echo "       docker build -f $FORGE/docker/Dockerfile.cutile-eval -t cutile-eval:latest $FORGE"
echo "  2. point CUTILE_WS at a directory holding models/ and runs/ (defaults to $FORGE)"
echo "  3. sanity check:"
echo "       cd $CHECKOUT && GPUS=none scripts/in_container.sh python3 scripts/test_cutile_checker.py"
