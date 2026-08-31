#!/usr/bin/env bash
# Ten-minute Hybrid search wrapper; formal validation runs after search.
set -euo pipefail

FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HYBRID_PROFILE=600s
export HYBRID_DEADLINE=600
exec bash "$FORGE/rl/run_hybrid_60s.sh" "$@"
