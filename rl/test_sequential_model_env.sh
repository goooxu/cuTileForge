#!/usr/bin/env bash
set -euo pipefail

FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=sequential_model_env.sh
source "$FORGE/rl/sequential_model_env.sh"

configure_sequential_model GLE /tmp/gle
[[ "$MODEL" == /tmp/gle ]]
[[ "$VLLM_IMAGE" == *muse-glimmer ]]
[[ "$MAX_TOKENS" == 32768 ]]
[[ "$REASONING_STRENGTH" == xhigh ]]
[[ "$EXTRA_ARGS" == *muse_glimmer* ]]

configure_sequential_model Q /tmp/q
[[ "$MODEL" == /tmp/q ]]
[[ "$EXTRA_ARGS" == "--moe-backend triton" ]]
[[ "$MAX_LEN" == 65536 ]]
[[ -z "${ENABLE_THINKING:-}" ]]

configure_sequential_model base /tmp/base
[[ "$EXTRA_ARGS" == "--moe-backend triton" ]]

configure_sequential_model G4t /tmp/g4
[[ "$ENABLE_THINKING" == 1 ]]
[[ "$EXTRA_ARGS" == *gemma4* ]]
[[ "$MAX_TOKENS" == 32768 ]]

configure_sequential_model Q38 /tmp/q38
[[ "$ENABLE_THINKING" == 1 ]]
[[ "$REASONING_EFFORT" == xhigh ]]
[[ "$MAX_LEN" == 98304 ]]

export SEQUENTIAL_USE_NATIVE_CONTEXT=1
export SEQUENTIAL_OUTPUT_CAP=131072
configure_sequential_model GLE /tmp/gle-native
[[ "$NATIVE_CONTEXT" == 131072 ]]
[[ "$MAX_LEN" == 131072 ]]
[[ "$MAX_TOKENS" == 131072 ]]
configure_sequential_model Q /tmp/q-native
[[ "$NATIVE_CONTEXT" == 262144 ]]
[[ "$MAX_LEN" == 262144 ]]
[[ "$MAX_TOKENS" == 131072 ]]

echo "sequential model protocol mapping OK"
