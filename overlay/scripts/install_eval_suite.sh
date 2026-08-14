#!/usr/bin/env bash
# Copy the standalone eval suite into the KernelBench checkout and sync the
# frozen prompt (concepts + TILE=1024). The checkout is gitignored, so this
# has to be re-run after setup_kernelbench.sh.
#
# Usage: overlay/scripts/install_eval_suite.sh
set -euo pipefail

FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KB="${KERNELBENCH_DIR:-$FORGE/kernelbench}"
SRC="$FORGE/tasks/eval"

if [[ ! -d "$KB/KernelBench" ]]; then
    echo "error: $KB/KernelBench missing; run scripts/setup_kernelbench.sh" >&2
    exit 1
fi
if [[ ! -d "$SRC/level60" || ! -d "$SRC/level61" ]]; then
    echo "error: $SRC is incomplete; run python3 taskgen/build_eval_suite.py" >&2
    exit 1
fi

for lvl in 60 61; do
    dest="$KB/KernelBench/level$lvl"
    rm -rf "$dest"
    mkdir -p "$dest"
    cp "$SRC/level$lvl"/*.py "$dest/"
    n="$(python3 -c "import os; print(sum(1 for f in os.listdir('$dest') if f.endswith('.py')))")"
    echo "installed $n problems -> $dest"
done

PROMPT_SRC="$FORGE/overlay/src/kernelbench/prompts"
PROMPT_DST="$KB/src/kernelbench/prompts"
for f in model_new_ex_add_cutile.py cutile_concepts.md cutile_api_reference.md; do
    cp "$PROMPT_SRC/$f" "$PROMPT_DST/$f"
done
echo "synced TILE=1024 concepts prompt into the checkout"
