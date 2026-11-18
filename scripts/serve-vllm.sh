#!/usr/bin/env bash
# Qwen hybrid (GDN) models on vLLM 0.34.0, one launcher for both study tracks.
# MACHINE=t4: constrained path (SM75, Qwen3.5-9B W4A16, short context, conservative start).
# MACHINE=h20: current mainline (SM90, Qwen3.8-27B FP8/BF16, engine defaults for caching, graphs, KV sizing).
# serve-vllm-t4.sh / serve-vllm-h20.sh pin MACHINE. RUNTIME=docker (default) runs the pinned image;
# RUNTIME=native hands the same engine arguments to a local vllm==0.34.0 (GPU pods without Docker).
# VLLM_VERSION=0.32.0 is only the fallback for drivers where CUDA 13 forward compatibility fails:
# 0.33+ ships no CUDA 12.9 build, 0.32.0 still has -cu129. Its block sizes and defaults differ.
# Script arguments are appended to the engine arguments. Source-checked starting points, not benchmarks.
set -euo pipefail

machine="${MACHINE:-t4}"
case "$machine" in
  t4)
    profiles='9b-w4a16 4b-fp16' dtype=half container=qwen35-t4
    default_backend=TRITON_ATTN default_len=4096 default_seqs=1 default_batched=1024
    default_mem=0.85 default_prefix=0 default_eager=1
    ;;
  h20)
    profiles='27b-fp8 27b-bf16 9b-w4a16' dtype=bfloat16 container=qwen38-h20
    default_backend=auto default_len=32768 default_seqs=64     default_batched=8192
    default_mem=auto default_prefix=1 default_eager=0
    ;;
  *) printf 'MACHINE must be t4 or h20\n' >&2; exit 2 ;;
esac

profile="${PROFILE:-${profiles%% *}}"
case " $profiles " in
  *" $profile "*) ;;
  *) printf 'Unknown PROFILE for MACHINE=%s: %s (use one of: %s)\n' "$machine" "$profile" "$profiles" >&2; exit 2 ;;
esac
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
  27b-fp8)
    model='Qwen/Qwen3.8-27B-FP8'
    revision='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
    served_name='qwen3.8-27b-fp8'
    ;;
  27b-bf16)
    model='Qwen/Qwen3.8-27B'
    revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    served_name='qwen3.8-27b'
    ;;
esac

runtime="${RUNTIME:-docker}"
case "$runtime" in
  # Inside the container vLLM listens on every interface; Docker publishes it on host loopback only.
  docker) listen_host=0.0.0.0 listen_port=8000 default_cache="$PWD/.cache/huggingface" ;;
  native)
    listen_host=127.0.0.1 listen_port="${PORT:-8000}"
    default_cache="${HF_HOME:-$PWD/.cache/huggingface}"
    ;;
  *) printf 'RUNTIME must be docker or native\n' >&2; exit 2 ;;
esac

vllm_version="${VLLM_VERSION:-0.34.0}"
case "$vllm_version" in
  0.34.0) default_image="vllm/vllm-openai:v$vllm_version" ;;
  0.32.0) default_image="vllm/vllm-openai:v$vllm_version-cu129" ;;
  *) printf 'VLLM_VERSION must be 0.34.0 (default) or 0.32.0 (CUDA 12.9 fallback)\n' >&2; exit 2 ;;
esac
image="${VLLM_IMAGE:-$default_image}"
revision="${MODEL_REVISION:-$revision}"
cache_dir="${HF_CACHE_DIR:-$default_cache}"
[[ "$cache_dir" = /* ]] || cache_dir="$PWD/$cache_dir"

# CUDA 13 builds need an R580+ driver. Forward compatibility (datacenter GPUs on R535/R570) needs
# the compat libcuda on the loader path when each process starts: the engine core and workers are
# forked, so vLLM's in-process VLLM_ENABLE_CUDA_COMPATIBILITY alone does not reach them.
compat_dir="${CUDA_COMPAT_DIR:-/usr/local/cuda/compat}"
compat_env=()
case "${CUDA_COMPAT:-0}" in
  0) ;;
  1)
    if [[ "$runtime" = docker ]]; then
      # `docker run -e` replaces the image's LD_LIBRARY_PATH; keep the official image's entries after compat.
      library_path="$compat_dir:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
    else
      library_path="$compat_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    compat_env=(VLLM_ENABLE_CUDA_COMPATIBILITY=1 "VLLM_CUDA_COMPATIBILITY_PATH=$compat_dir"
      "LD_LIBRARY_PATH=$library_path")
    ;;
  *) printf 'CUDA_COMPAT must be 0 or 1\n' >&2; exit 2 ;;
esac

engine_args=(
  "$model" --revision "$revision" --served-model-name "$served_name"
  --host "$listen_host" --port "$listen_port" --tensor-parallel-size 1
  --dtype "$dtype" --kv-cache-dtype "${KV_DTYPE:-auto}" --language-model-only
  --reasoning-parser qwen3
)
attention_backend="${ATTENTION_BACKEND:-$default_backend}"
[[ "$attention_backend" = auto ]] || engine_args+=(--attention-backend "$attention_backend")
engine_args+=(
  --max-model-len "${MAX_LEN:-$default_len}"
  --max-num-seqs "${MAX_SEQS:-$default_seqs}"
  --max-num-batched-tokens "${BATCHED_TOKENS:-$default_batched}"
  --enable-chunked-prefill
)
# 0.34 sizes the KV cache from memory measured after warmup when --gpu-memory-utilization is unset;
# an explicit value still caps the instance.
mem_util="${MEM_UTIL:-$default_mem}"
[[ "$mem_util" = auto ]] || engine_args+=(--gpu-memory-utilization "$mem_util")
engine_args+=(--enable-prompt-tokens-details)
case "${PREFIX_CACHE:-$default_prefix}" in
  0) engine_args+=(--no-enable-prefix-caching) ;;
  1) engine_args+=(--enable-prefix-caching) ;;
  *) printf 'PREFIX_CACHE must be 0 or 1\n' >&2; exit 2 ;;
esac
case "${EAGER:-$default_eager}" in
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
engine_args+=("$@")

if [[ "$runtime" = docker ]]; then
  docker_env=()
  for assignment in ${compat_env[@]+"${compat_env[@]}"}; do docker_env+=(-e "$assignment"); done
  command=(docker run --rm --gpus "device=${GPU_DEVICE:-0}" --ipc=host
    --name "${CONTAINER_NAME:-$container}"
    -p "127.0.0.1:${PORT:-8000}:8000"
    -v "$cache_dir:/root/.cache/huggingface"
    ${docker_env[@]+"${docker_env[@]}"}
    "$image" "${engine_args[@]}")
  engine_line="Image: $image"
else
  python="${PYTHON:-python3}"
  python="$(command -v "$python" || printf '%s' "$python")"
  # The image ENTRYPOINT is `vllm serve`; run that CLI module so the interpreter checked below
  # is the one serving. vllm.entrypoints.openai.api_server would ignore the positional model.
  command=(env "CUDA_VISIBLE_DEVICES=${GPU_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}" "HF_HOME=$cache_dir"
    ${compat_env[@]+"${compat_env[@]}"}
    "$python" -m vllm.entrypoints.cli.main serve "${engine_args[@]}")
  engine_line="Python: $python (needs vllm==$vllm_version)"
fi
printf 'Machine: %s\nProfile: %s\nModel: %s\nRevision: %s\nRuntime: %s\n%s\n' \
  "$machine" "$profile" "$model" "$revision" "$runtime" "$engine_line"
printf 'Command: '; printf '%q ' "${command[@]}"; printf '\n'

if [[ "${CUDA_COMPAT:-0}" = 0 ]] && command -v nvidia-smi >/dev/null; then
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)" || driver=''
  driver="${driver%%$'\n'*}"
  if [[ "${driver%%.*}" =~ ^[0-9]+$ ]] && (( ${driver%%.*} < 580 )) \
      && ! [[ "$runtime" = docker && "$image" = *-cu12* ]] \
      && ! [[ "$runtime" = native && "$vllm_version" = 0.32.0 ]]; then
    printf 'Note: driver %s is older than R580, which CUDA 13 builds (the default image and PyPI wheel) need.\n' "$driver" >&2
    printf 'On a datacenter GPU with R535/R570 add CUDA_COMPAT=1. vLLM 0.33+ has no CUDA 12.9 build: if forward compatibility fails, upgrade the driver or fall back to VLLM_VERSION=0.32.0 (vllm/vllm-openai:v0.32.0-cu129). See docs/t4-qwen35-lab.md section 2.\n' >&2
  fi
fi
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
    if [[ "$vllm_version" = 0.32.0 ]]; then
      printf 'In a GPU pod, use vllm/vllm-openai:v0.32.0-cu129 as the pod image: its python3 already has it.\n' >&2
      printf "Or a Python 3.11-3.14 venv: uv pip install 'https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl' --torch-backend=cu129\n" >&2
    else
      printf 'In a GPU pod, use vllm/vllm-openai:v%s as the pod image: its python3 already has it.\n' "$vllm_version" >&2
      printf "Or a Python 3.11-3.14 venv: uv pip install 'vllm==%s' --torch-backend=cu130 (CUDA 13 build; 'auto' can pick a CUDA 12 torch on R535).\n" "$vllm_version" >&2
    fi
    printf 'Below glibc 2.31 (EL8 is 2.28) the llguidance dependency has no wheel and builds from source (needs Rust).\n' >&2
    printf 'Set PYTHON=/path/to/venv/bin/python if the venv is not active; see docs/t4-qwen35-lab.md\n' >&2
    exit 1
  fi
  if [[ "${CUDA_COMPAT:-0}" = 1 && ! -d "$compat_dir" ]]; then
    printf 'CUDA_COMPAT=1 needs the CUDA forward-compatibility libraries in %s (set CUDA_COMPAT_DIR).\n' "$compat_dir" >&2
    exit 1
  fi
fi
mkdir -p "$cache_dir"
exec "${command[@]}"
