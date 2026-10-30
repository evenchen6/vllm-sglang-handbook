#!/usr/bin/env bash
# Qwen3.5 on one T4: a source-checked starting configuration, not a GPU benchmark.
# RUNTIME=docker (default) runs the pinned image; RUNTIME=native hands the same engine
# arguments to a local vllm==0.18.1, for GPU pods and hosts without Docker.
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

runtime="${RUNTIME:-docker}"
case "$runtime" in
  # Inside the container vLLM listens on every interface; Docker publishes it on host loopback only.
  docker) listen_host=0.0.0.0 listen_port=8000 ;;
  native) listen_host=127.0.0.1 listen_port="${PORT:-8000}" ;;
  *) printf 'RUNTIME must be docker or native\n' >&2; exit 2 ;;
esac

vllm_version='0.18.1'
image="${VLLM_IMAGE:-vllm/vllm-openai:v$vllm_version}"
revision="${MODEL_REVISION:-$revision}"
cache_dir="${HF_CACHE_DIR:-$PWD/.cache/huggingface}"
[[ "$cache_dir" = /* ]] || cache_dir="$PWD/$cache_dir"

engine_args=(
  "$model" --revision "$revision" --served-model-name "$served_name"
  --host "$listen_host" --port "$listen_port" --tensor-parallel-size 1
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

if [[ "$runtime" = docker ]]; then
  command=(docker run --rm --gpus "device=${GPU_DEVICE:-0}" --ipc=host
    --name "${CONTAINER_NAME:-qwen35-t4}"
    -p "127.0.0.1:${PORT:-8000}:8000"
    -v "$cache_dir:/root/.cache/huggingface"
    "$image" "${engine_args[@]}")
  engine_line="Image: $image"
else
  python="${PYTHON:-python3}"
  python="$(command -v "$python" || printf '%s' "$python")"
  # The image ENTRYPOINT is `vllm serve`; run that CLI module so the interpreter checked below
  # is the one serving. vllm.entrypoints.openai.api_server would ignore the positional model.
  command=(env "CUDA_VISIBLE_DEVICES=${GPU_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}" "HF_HOME=$cache_dir"
    "$python" -m vllm.entrypoints.cli.main serve "${engine_args[@]}")
  engine_line="Python: $python (needs vllm==$vllm_version)"
fi
printf 'Profile: %s\nModel: %s\nRevision: %s\nRuntime: %s\n%s\n' \
  "$profile" "$model" "$revision" "$runtime" "$engine_line"
printf 'Command: '; printf '%q ' "${command[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" = 1 ]]; then exit 0; fi
if [[ "$runtime" = docker ]]; then
  command -v docker >/dev/null || {
    printf 'Docker is required for RUNTIME=docker; in a GPU pod without Docker use RUNTIME=native. See docs/t4-qwen35-lab.md\n' >&2
    exit 1
  }
else
  found="$("$python" -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null)" || found=''
  if [[ "${found%%+*}" != "$vllm_version" ]]; then
    printf 'RUNTIME=native needs vllm==%s in %s (found: %s).\n' "$vllm_version" "$python" "${found:-none}" >&2
    printf "Install it into a Python 3.10-3.13 venv: pip install 'vllm==%s' --only-binary=vllm --extra-index-url https://download.pytorch.org/whl/cu129\n" "$vllm_version" >&2
    printf 'The PyPI wheel needs glibc >= 2.31. Set PYTHON=/path/to/venv/bin/python if the venv is not active; see docs/t4-qwen35-lab.md\n' >&2
    exit 1
  fi
fi
mkdir -p "$cache_dir"
exec "${command[@]}"
