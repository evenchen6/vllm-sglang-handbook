"""Command assembly checks for scripts/serve-vllm-t4.sh with fake docker/python; no GPU."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/serve-vllm-t4.sh"
BASH = shutil.which("bash")
CACHE = f"{ROOT}/.cache/huggingface"
IMAGE = "vllm/vllm-openai:v0.18.1"
ENGINE_9B = [
    "RedHatAI/Qwen3.5-9B-quantized.w4a16",
    "--revision", "a398088c4228b0ae0c8c78df88fd1e4bf445f068",
    "--served-model-name", "qwen3.5-9b",
    "--host", "0.0.0.0", "--port", "8000", "--tensor-parallel-size", "1",
    "--dtype", "half", "--kv-cache-dtype", "auto", "--language-model-only",
    "--reasoning-parser", "qwen3", "--attention-backend", "TRITON_ATTN",
    "--max-model-len", "4096", "--max-num-seqs", "1", "--max-num-batched-tokens", "1024",
    "--enable-chunked-prefill", "--gpu-memory-utilization", "0.85",
    "--no-enable-prefix-caching", "--enforce-eager",
]
FAKE_PYTHON = """#!/bin/sh
if [ "$1" = -c ]; then echo "$FAKE_VLLM"; exit 0; fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HF_HOME=$HF_HOME"
for arg in "$@"; do echo "arg=$arg"; done
"""
FAKE_DOCKER = """#!/bin/sh
for arg in "$@"; do echo "arg=$arg"; done
"""


def run(path=None, **env):
    base = {"PATH": path or os.environ["PATH"]}
    return subprocess.run([BASH, str(SCRIPT)], env={**base, **env}, cwd=ROOT,
                          capture_output=True, text=True, timeout=10)


def dry_run(**env):
    result = run(DRY_RUN="1", **env)
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("Command: "))
    return shlex.split(line[len("Command: "):])


def engine_args(argv):
    """Engine arguments after the image (docker) or after `serve` (native)."""
    start = argv.index(IMAGE) if argv[0] == "docker" else argv.index("serve")
    return argv[start + 1:]


def without_listen(args):
    args = list(args)
    for flag in ("--host", "--port"):
        index = args.index(flag)
        del args[index:index + 2]
    return args


def fake_executable(directory, name, body):
    path = Path(directory) / name
    path.write_text(body)
    path.chmod(0o755)
    return path


class ServeScriptTests(unittest.TestCase):
    def test_default_runtime_is_the_unchanged_docker_command(self):
        self.assertEqual(dry_run(), [
            "docker", "run", "--rm", "--gpus", "device=0", "--ipc=host", "--name", "qwen35-t4",
            "-p", "127.0.0.1:8000:8000", "-v", f"{CACHE}:/root/.cache/huggingface", IMAGE,
            *ENGINE_9B,
        ])
        self.assertEqual(dry_run(), dry_run(RUNTIME="docker"))

    def test_native_runs_vllm_serve_with_same_engine_args(self):
        cases = [
            {"name": "default", "env": {}},
            {"name": "prefix cache on", "env": {"PREFIX_CACHE": "1"}},
            {"name": "cuda graph and sync scheduling", "env": {"EAGER": "0", "ASYNC_SCHED": "0"}},
            {"name": "4b profile", "env": {"PROFILE": "4b-fp16", "MODEL_REVISION": "abc"}},
            {"name": "scheduler knobs", "env": {"MAX_LEN": "8192", "MAX_SEQS": "4", "BATCHED_TOKENS": "2048",
                                                "MEM_UTIL": "0.9", "ATTENTION_BACKEND": "FLEX_ATTENTION"}},
        ]
        for case in cases:
            with self.subTest(case["name"]):
                docker = dry_run(**case["env"])
                native = dry_run(RUNTIME="native", PYTHON="/opt/venv/bin/python", **case["env"])
                self.assertEqual(native[:6], ["env", "CUDA_VISIBLE_DEVICES=0", f"HF_HOME={CACHE}",
                                              "/opt/venv/bin/python", "-m", "vllm.entrypoints.cli.main"])
                self.assertEqual(native[6], "serve")
                self.assertEqual(without_listen(engine_args(native)), without_listen(engine_args(docker)))

    def test_listen_address_port_gpu_and_cache_mapping(self):
        cases = [
            {"name": "defaults", "env": {}, "port": "8000", "gpu": "0", "cache": CACHE, "docker_cache": CACHE},
            {"name": "port and gpu", "env": {"PORT": "8100", "GPU_DEVICE": "1"}, "port": "8100", "gpu": "1",
             "cache": CACHE, "docker_cache": CACHE},
            {"name": "existing CUDA_VISIBLE_DEVICES", "env": {"CUDA_VISIBLE_DEVICES": "2"}, "port": "8000",
             "gpu": "2", "cache": CACHE, "docker_cache": CACHE},
            {"name": "GPU_DEVICE wins", "env": {"CUDA_VISIBLE_DEVICES": "2", "GPU_DEVICE": "3"}, "port": "8000",
             "gpu": "3", "cache": CACHE, "docker_cache": CACHE},
            {"name": "relative cache", "env": {"HF_CACHE_DIR": "rel/hf"}, "port": "8000", "gpu": "0",
             "cache": f"{ROOT}/rel/hf", "docker_cache": f"{ROOT}/rel/hf"},
            {"name": "existing HF_HOME, native only", "env": {"HF_HOME": "/data/hf"}, "port": "8000", "gpu": "0",
             "cache": "/data/hf", "docker_cache": CACHE},
            {"name": "HF_CACHE_DIR wins", "env": {"HF_HOME": "/data/hf", "HF_CACHE_DIR": "/data/lab"},
             "port": "8000", "gpu": "0", "cache": "/data/lab", "docker_cache": "/data/lab"},
        ]
        for case in cases:
            with self.subTest(case["name"]):
                native = dry_run(RUNTIME="native", **case["env"])
                self.assertEqual(native[1:3], [f"CUDA_VISIBLE_DEVICES={case['gpu']}", f"HF_HOME={case['cache']}"])
                args = engine_args(native)
                self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
                self.assertEqual(args[args.index("--port") + 1], case["port"])
                docker = dry_run(**case["env"])
                self.assertIn(f"127.0.0.1:{case['port']}:8000", docker)
                self.assertIn(f"{case['docker_cache']}:/root/.cache/huggingface", docker)
                args = engine_args(docker)
                self.assertEqual(args[args.index("--port") + 1], "8000")

    def test_prefix_cache_never_forces_mamba_cache_mode(self):
        for runtime in ("docker", "native"):
            with self.subTest(runtime):
                args = dry_run(RUNTIME=runtime, PREFIX_CACHE="1")
                self.assertIn("--enable-prefix-caching", args)
                self.assertNotIn("--no-enable-prefix-caching", args)
                self.assertFalse([arg for arg in args if arg.startswith("--mamba-cache-mode")])

    def test_invalid_knobs_exit_2(self):
        cases = [
            {"RUNTIME": "podman"}, {"PREFIX_CACHE": "2"}, {"EAGER": "yes"}, {"ASYNC_SCHED": "on"},
            {"PROFILE": "9b-bf16"},
        ]
        for env in cases:
            with self.subTest(env):
                self.assertEqual(run(DRY_RUN="1", **env).returncode, 2)

    def test_native_dry_run_does_not_need_vllm(self):
        result = run(DRY_RUN="1", RUNTIME="native", PYTHON="/missing/python")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Runtime: native\nPython: /missing/python (needs vllm==0.18.1)", result.stdout)

    def test_native_execs_checked_interpreter(self):
        cases = [
            {"name": "pinned", "found": "0.18.1", "code": 0},
            {"name": "pinned with local build tag", "found": "0.18.1+cu129", "code": 0},
            {"name": "other version", "found": "0.19.0", "code": 1},
            {"name": "not installed", "found": "", "code": 1},
        ]
        for case in cases:
            with self.subTest(case["name"]), tempfile.TemporaryDirectory() as directory:
                python = fake_executable(directory, "python", FAKE_PYTHON)
                cache = Path(directory) / "hf"
                result = run(RUNTIME="native", PYTHON=str(python), FAKE_VLLM=case["found"],
                             HF_CACHE_DIR=str(cache))
                self.assertEqual(result.returncode, case["code"], result.stderr)
                if case["code"]:
                    self.assertIn("pip install 'vllm==0.18.1'", result.stderr)
                    self.assertIn(f"found: {case['found'] or 'none'}", result.stderr)
                    self.assertFalse(cache.exists())
                    continue
                lines = result.stdout.splitlines()
                self.assertIn("CUDA_VISIBLE_DEVICES=0", lines)
                self.assertIn(f"HF_HOME={cache}", lines)
                served = [line[len("arg="):] for line in lines if line.startswith("arg=")]
                self.assertEqual(served[:3], ["-m", "vllm.entrypoints.cli.main", "serve"])
                self.assertEqual(without_listen(served[3:]), without_listen(ENGINE_9B))
                self.assertTrue(cache.is_dir())

    def test_docker_exec_matches_dry_run_and_missing_docker_points_to_native(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_executable(directory, "docker", FAKE_DOCKER)
            cache = Path(directory) / "hf"
            path = f"{directory}{os.pathsep}{os.environ['PATH']}"
            result = run(path=path, HF_CACHE_DIR=str(cache))
            self.assertEqual(result.returncode, 0, result.stderr)
            ran = [line[len("arg="):] for line in result.stdout.splitlines() if line.startswith("arg=")]
            self.assertEqual(["docker", *ran], dry_run(HF_CACHE_DIR=str(cache)))

        with tempfile.TemporaryDirectory() as empty:
            result = run(path=empty)
            self.assertEqual(result.returncode, 1)
            self.assertIn("RUNTIME=native", result.stderr)


if __name__ == "__main__":
    unittest.main()
