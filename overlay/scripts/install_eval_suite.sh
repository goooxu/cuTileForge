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

python3 - "$SRC" "$KB/KernelBench" <<'PY'
import os, shutil, sys
src_root, kb = sys.argv[1], sys.argv[2]
expect = {60: 770, 61: 250}
for lvl, want in expect.items():
    src = os.path.join(src_root, "level%d" % lvl)
    dest = os.path.join(kb, "level%d" % lvl)
    os.makedirs(dest, exist_ok=True)
    for name in os.listdir(dest):
        path = os.path.join(dest, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    for name in os.listdir(src):
        if name.endswith(".py"):
            shutil.copy2(os.path.join(src, name), os.path.join(dest, name))
    got = sum(1 for n in os.listdir(dest) if n.endswith(".py"))
    print("installed %d problems -> %s" % (got, dest))
    if got != want:
        sys.exit("level %d: expected %d files, got %d" % (lvl, want, got))
PY

PROMPT_SRC="$FORGE/overlay/src/kernelbench/prompts"
PROMPT_DST="$KB/src/kernelbench/prompts"
for f in model_new_ex_add_cutile.py cutile_concepts.md cutile_api_reference.md; do
    cp "$PROMPT_SRC/$f" "$PROMPT_DST/$f"
done
echo "synced TILE=1024 concepts prompt into the checkout"
