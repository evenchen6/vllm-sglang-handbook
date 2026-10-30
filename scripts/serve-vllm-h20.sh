#!/usr/bin/env bash
# H20 track (SM90, Qwen3.8-27B FP8 by default, vLLM's default caching and CUDA graphs). All knobs live in serve-vllm.sh.
here="${BASH_SOURCE[0]%/*}"
[[ "$here" = "${BASH_SOURCE[0]}" ]] && here=.
MACHINE=h20 exec "$BASH" "$here/serve-vllm.sh" "$@"
