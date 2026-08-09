#!/usr/bin/env bash
# Keep refresh_loop.sh going until it reaches the target iteration count.
#
# Long runs on this machine do not survive on their own: containers get reaped
# out from under them and ssh sessions carrying background jobs get dropped. The
# first attempt at a 100-iteration run stopped at 13 with no error anywhere.
# refresh_loop.sh reads its own progress from history.jsonl, so the recovery is
# simply to invoke it again.
#
# Usage: CUTILE_WS=... rl/supervise_grpo.sh <base> <frontier> <out> <total> <window>
set -uo pipefail

BASE="${1:?}"; FRONTIER="${2:?}"; OUT="${3:?}"
TOTAL="${4:-100}"; WINDOW="${5:-10}"
WS="${CUTILE_WS:?}"
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIST="$WS/runs/$(basename "$OUT")/history.jsonl"
MAX_TRIES="${MAX_TRIES:-30}"

count() {
    [ -f "$HIST" ] || { echo 0; return; }
    python3 - "$HIST" <<'PY'
import json, sys
try:
    print(max(json.loads(l)["iteration"] for l in open(sys.argv[1]) if l.strip()) + 1)
except Exception:
    print(0)
PY
}

for try in $(seq 1 "$MAX_TRIES"); do
    n=$(count)
    if [ "$n" -ge "$TOTAL" ]; then
        echo "supervisor: $n/$TOTAL iterations done"
        exit 0
    fi
    echo "supervisor: attempt $try, at $n/$TOTAL"
    bash "$FORGE/rl/refresh_loop.sh" "$BASE" "$FRONTIER" "$OUT" "$TOTAL" "$WINDOW"

    # A retry that gains nothing means the failure is deterministic, not a reaped
    # container; stop rather than spin.
    if [ "$(count)" = "$n" ]; then
        echo "supervisor: attempt $try made no progress; giving up at $n/$TOTAL" >&2
        exit 1
    fi
    sleep 20
done
echo "supervisor: out of tries at $(count)/$TOTAL" >&2
exit 1
