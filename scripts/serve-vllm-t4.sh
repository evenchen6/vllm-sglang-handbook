#!/usr/bin/env bash
# T4 track (SM75, Qwen3.5-9B W4A16, conservative first start). All knobs live in serve-vllm.sh.
here="${BASH_SOURCE[0]%/*}"
[[ "$here" = "${BASH_SOURCE[0]}" ]] && here=.
MACHINE=t4 exec "$BASH" "$here/serve-vllm.sh" "$@"
