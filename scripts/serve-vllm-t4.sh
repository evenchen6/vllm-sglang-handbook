#!/usr/bin/env bash
# Qwen3.5 on one T4: a source-checked starting configuration, not a GPU benchmark.
set -euo pipefail

profile="${PROFILE:-9b-w4a16}"
case "$profile" in
  9b-w4a16)
    model='RedHatAI/Qwen3.5-9B-quantized.w4a16'
    revision='a398088c4228b0ae0c8c78df88fd1e4bf445f068'
    served_name='qwen3.5-9b'
    ;;
  4b-fp16)
    model='Qwen/Qwen3.5-4B'
    revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    served_name='qwen3.5-4b'
    ;;
  *) printf 'Unknown PROFILE: %s (use 9b-w4a16 or 4b-fp16)\n' "$profile" >&2; exit 2 ;;
esac

image="${VLLM_IMAGE:-vllm/vllm-openai:v0.18.1}"
revision="${MODEL_REVISION:-$revision}"
cache_dir="${HF_CACHE_DIR:-$PWD/.cache/huggingface}"
[[ "$cache_dir" = /* ]] || cache_dir="$PWD/$cache_dir"

engine_args=(
  "$model" --revision "$revision" --served-model-name "$served_name"
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1
  --dtype half --kv-cache-dtype auto --language-model-only
  --reasoning-parser qwen3
  --attention-backend "${ATTENTION_BACKEND:-TRITON_ATTN}"
  --max-model-len "${MAX_LEN:-4096}"
  --max-num-seqs "${MAX_SEQS:-1}"
  --max-num-batched-tokens "${BATCHED_TOKENS:-1024}"
  --enable-chunked-prefill
  --gpu-memory-utilization "${MEM_UTIL:-0.85}"
)
case "${PREFIX_CACHE:-0}" in
  0) engine_args+=(--no-enable-prefix-caching) ;;
  1) engine_args+=(--enable-prefix-caching) ;;
  *) printf 'PREFIX_CACHE must be 0 or 1\n' >&2; exit 2 ;;
esac
case "${EAGER:-1}" in
  1) engine_args+=(--enforce-eager) ;;
  0) ;;
  *) printf 'EAGER must be 0 or 1\n' >&2; exit 2 ;;
esac
case "${ASYNC_SCHED:-auto}" in
  auto) ;;
  0) engine_args+=(--no-async-scheduling) ;;
  1) engine_args+=(--async-scheduling) ;;
  *) printf 'ASYNC_SCHED must be auto, 0 or 1\n' >&2; exit 2 ;;
esac

command=(docker run --rm --gpus "device=${GPU_DEVICE:-0}" --ipc=host
  --name "${CONTAINER_NAME:-qwen35-t4}"
  -p "127.0.0.1:${PORT:-8000}:8000"
  -v "$cache_dir:/root/.cache/huggingface"
  "$image" "${engine_args[@]}")
printf 'Profile: %s\nModel: %s\nRevision: %s\nImage: %s\n' "$profile" "$model" "$revision" "$image"
printf 'Command: '; printf '%q ' "${command[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" = 1 ]]; then exit 0; fi
command -v docker >/dev/null || { printf 'Docker is required; see docs/t4-qwen35-lab.md\n' >&2; exit 1; }
mkdir -p "$cache_dir"
exec "${command[@]}"
