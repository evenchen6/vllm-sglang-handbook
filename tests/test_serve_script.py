"""Command assembly checks for scripts/serve-vllm*.sh with fake docker/python/nvidia-smi; no GPU."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "scripts/serve-vllm.sh"
T4 = ROOT / "scripts/serve-vllm-t4.sh"
H20 = ROOT / "scripts/serve-vllm-h20.sh"
BASH = shutil.which("bash")
CACHE = f"{ROOT}/.cache/huggingface"
IMAGE = "vllm/vllm-openai:v0.32.0"
COMPAT_DIR = "/usr/local/cuda/compat"
ENGINE_T4_9B = [
    "RedHatAI/Qwen3.5-9B-quantized.w4a16",
    "--revision", "a398088c4228b0ae0c8c78df88fd1e4bf445f068",
    "--served-model-name", "qwen3.5-9b",
    "--host", "0.0.0.0", "--port", "8000", "--tensor-parallel-size", "1",
    "--dtype", "half", "--kv-cache-dtype", "auto", "--language-model-only",
    "--reasoning-parser", "qwen3", "--attention-backend", "TRITON_ATTN",
    "--max-model-len", "4096", "--max-num-seqs", "1", "--max-num-batched-tokens", "1024",
    "--enable-chunked-prefill", "--gpu-memory-utilization", "0.85", "--enable-prompt-tokens-details",
    "--no-enable-prefix-caching", "--enforce-eager",
]
ENGINE_H20_27B_FP8 = [
    "Qwen/Qwen3.8-27B-FP8",
    "--revision", "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
    "--served-model-name", "qwen3.8-27b-fp8",
    "--host", "0.0.0.0", "--port", "8000", "--tensor-parallel-size", "1",
    "--dtype", "bfloat16", "--kv-cache-dtype", "auto", "--language-model-only",
    "--reasoning-parser", "qwen3",
    "--max-model-len", "32768", "--max-num-seqs", "64", "--max-num-batched-tokens", "8192",
    "--enable-chunked-prefill", "--gpu-memory-utilization", "0.90", "--enable-prompt-tokens-details",
    "--enable-prefix-caching",
]
FAKE_PYTHON = """#!/bin/sh
if [ "$1" = -c ]; then echo "$FAKE_VLLM"; exit 0; fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HF_HOME=$HF_HOME"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "VLLM_ENABLE_CUDA_COMPATIBILITY=$VLLM_ENABLE_CUDA_COMPATIBILITY"
for arg in "$@"; do echo "arg=$arg"; done
"""
FAKE_DOCKER = """#!/bin/sh
for arg in "$@"; do echo "arg=$arg"; done
"""
FAKE_NVIDIA_SMI = """#!/bin/sh
echo "$FAKE_DRIVER"
echo "$FAKE_DRIVER"
"""


def run(script=T4, *args, path=None, cwd=ROOT, **env):
    base = {"PATH": path or os.environ["PATH"]}
    return subprocess.run([BASH, str(script), *args], env={**base, **env}, cwd=cwd,
                          capture_output=True, text=True, timeout=10)


def dry_run(script=T4, *args, **env):
    result = run(script, *args, DRY_RUN="1", **env)
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("Command: "))
    return shlex.split(line[len("Command: "):])


def engine_args(argv, image=IMAGE):
    """Engine arguments after the image (docker) or after `serve` (native)."""
    start = argv.index(image) if argv[0] == "docker" else argv.index("serve")
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


def docker_command(name, engine):
    return ["docker", "run", "--rm", "--gpus", "device=0", "--ipc=host", "--name", name,
            "-p", "127.0.0.1:8000:8000", "-v", f"{CACHE}:/root/.cache/huggingface", IMAGE, *engine]


class ServeScriptTests(unittest.TestCase):
    def test_default_commands_per_machine(self):
        cases = [
            {"name": "t4 wrapper", "script": T4, "env": {}, "want": docker_command("qwen35-t4", ENGINE_T4_9B)},
            {"name": "shared script defaults to t4", "script": SHARED, "env": {},
             "want": docker_command("qwen35-t4", ENGINE_T4_9B)},
            {"name": "h20 wrapper", "script": H20, "env": {}, "want": docker_command("qwen38-h20", ENGINE_H20_27B_FP8)},
            {"name": "shared script MACHINE=h20", "script": SHARED, "env": {"MACHINE": "h20"},
             "want": docker_command("qwen38-h20", ENGINE_H20_27B_FP8)},
            {"name": "explicit docker runtime", "script": H20, "env": {"RUNTIME": "docker"},
             "want": docker_command("qwen38-h20", ENGINE_H20_27B_FP8)},
        ]
        for case in cases:
            with self.subTest(case["name"]):
                self.assertEqual(dry_run(case["script"], **case["env"]), case["want"])

    def test_wrappers_pin_machine_from_any_directory(self):
        cases = [
            {"name": "t4 ignores MACHINE=h20", "script": T4, "env": {"MACHINE": "h20"}, "machine": "t4"},
            {"name": "h20 ignores MACHINE=t4", "script": H20, "env": {"MACHINE": "t4"}, "machine": "h20"},
        ]
        for case in cases:
            for cwd, script in ((ROOT, case["script"]), (ROOT / "scripts", Path(case["script"].name))):
                with self.subTest(case["name"], cwd=str(cwd)):
                    result = run(script, cwd=cwd, DRY_RUN="1", **case["env"])
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"Machine: {case['machine']}\n", result.stdout)

    def test_profiles_select_pinned_checkpoints(self):
        cases = [
            {"script": T4, "profile": "4b-fp16", "model": "Qwen/Qwen3.5-4B",
             "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "served": "qwen3.5-4b", "dtype": "half"},
            {"script": H20, "profile": "27b-bf16", "model": "Qwen/Qwen3.8-27B",
             "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "served": "qwen3.8-27b", "dtype": "bfloat16"},
            {"script": H20, "profile": "9b-w4a16", "model": "RedHatAI/Qwen3.5-9B-quantized.w4a16",
             "revision": "a398088c4228b0ae0c8c78df88fd1e4bf445f068", "served": "qwen3.5-9b", "dtype": "bfloat16"},
        ]
        for case in cases:
            with self.subTest(case["profile"], script=case["script"].name):
                args = engine_args(dry_run(case["script"], PROFILE=case["profile"]))
                self.assertEqual(args[0], case["model"])
                self.assertEqual(args[args.index("--revision") + 1], case["revision"])
                self.assertEqual(args[args.index("--served-model-name") + 1], case["served"])
                self.assertEqual(args[args.index("--dtype") + 1], case["dtype"])
                args = engine_args(dry_run(case["script"], PROFILE=case["profile"], MODEL_REVISION="abc"))
                self.assertEqual(args[args.index("--revision") + 1], "abc")

    def test_knobs_map_to_engine_flags(self):
        cases = [
            {"name": "t4 prefix on, graphs on, sync scheduling", "script": T4,
             "env": {"PREFIX_CACHE": "1", "EAGER": "0", "ASYNC_SCHED": "0"},
             "present": ["--enable-prefix-caching", "--no-async-scheduling"],
             "absent": ["--no-enable-prefix-caching", "--enforce-eager"]},
            {"name": "h20 prefix off, eager, async on", "script": H20,
             "env": {"PREFIX_CACHE": "0", "EAGER": "1", "ASYNC_SCHED": "1"},
             "present": ["--no-enable-prefix-caching", "--enforce-eager", "--async-scheduling"],
             "absent": ["--enable-prefix-caching"]},
            {"name": "h20 attention backend pinned", "script": H20, "env": {"ATTENTION_BACKEND": "FLASHINFER"},
             "pairs": {"--attention-backend": "FLASHINFER"}},
            {"name": "t4 attention backend left to vLLM", "script": T4, "env": {"ATTENTION_BACKEND": "auto"},
             "absent": ["--attention-backend"]},
            {"name": "h20 scheduler and memory knobs", "script": H20,
             "env": {"MAX_LEN": "65536", "MAX_SEQS": "128", "BATCHED_TOKENS": "16384", "MEM_UTIL": "0.92",
                     "KV_DTYPE": "fp8"},
             "pairs": {"--max-model-len": "65536", "--max-num-seqs": "128", "--max-num-batched-tokens": "16384",
                       "--gpu-memory-utilization": "0.92", "--kv-cache-dtype": "fp8"}},
        ]
        for case in cases:
            with self.subTest(case["name"]):
                args = engine_args(dry_run(case["script"], **case["env"]))
                for flag in case.get("present", []):
                    self.assertIn(flag, args)
                for flag in case.get("absent", []):
                    self.assertNotIn(flag, args)
                for flag, value in case.get("pairs", {}).items():
                    self.assertEqual(args[args.index(flag) + 1], value)

    def test_native_runs_vllm_serve_with_same_engine_args(self):
        cases = [
            {"name": "t4 default", "script": T4, "env": {}},
            {"name": "t4 prefix cache on", "script": T4, "env": {"PREFIX_CACHE": "1"}},
            {"name": "t4 4b profile", "script": T4, "env": {"PROFILE": "4b-fp16", "MODEL_REVISION": "abc"}},
            {"name": "h20 default", "script": H20, "env": {}},
            {"name": "h20 bf16 with fp8 kv", "script": H20, "env": {"PROFILE": "27b-bf16", "KV_DTYPE": "fp8"}},
            {"name": "h20 sync scheduling", "script": H20, "env": {"ASYNC_SCHED": "0", "EAGER": "1"}},
        ]
        for case in cases:
            with self.subTest(case["name"]):
                docker = dry_run(case["script"], **case["env"])
                native = dry_run(case["script"], RUNTIME="native", PYTHON="/opt/venv/bin/python", **case["env"])
                self.assertEqual(native[:6], ["env", "CUDA_VISIBLE_DEVICES=0", f"HF_HOME={CACHE}",
                                              "/opt/venv/bin/python", "-m", "vllm.entrypoints.cli.main"])
                self.assertEqual(native[6], "serve")
                self.assertEqual(without_listen(engine_args(native)), without_listen(engine_args(docker)))

    def test_extra_arguments_are_appended_last(self):
        extra = ["--speculative-config", '{"method":"mtp","num_speculative_tokens":3}']
        for runtime in ("docker", "native"):
            with self.subTest(runtime):
                args = engine_args(dry_run(H20, *extra, RUNTIME=runtime))
                self.assertEqual(args[-2:], extra)
                self.assertEqual(without_listen(args[:-2]), without_listen(ENGINE_H20_27B_FP8))

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
        for script in (T4, H20):
            for case in cases:
                with self.subTest(case["name"], script=script.name):
                    native = dry_run(script, RUNTIME="native", **case["env"])
                    self.assertEqual(native[1:3], [f"CUDA_VISIBLE_DEVICES={case['gpu']}", f"HF_HOME={case['cache']}"])
                    args = engine_args(native)
                    self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
                    self.assertEqual(args[args.index("--port") + 1], case["port"])
                    docker = dry_run(script, **case["env"])
                    self.assertIn(f"127.0.0.1:{case['port']}:8000", docker)
                    self.assertIn(f"{case['docker_cache']}:/root/.cache/huggingface", docker)
                    args = engine_args(docker)
                    self.assertEqual(args[args.index("--port") + 1], "8000")

    def test_cuda_compat_sets_loader_path_before_the_process_starts(self):
        docker_env = ["-e", "VLLM_ENABLE_CUDA_COMPATIBILITY=1",
                      "-e", f"VLLM_CUDA_COMPATIBILITY_PATH={COMPAT_DIR}",
                      "-e", f"LD_LIBRARY_PATH={COMPAT_DIR}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"]
        for script in (T4, H20):
            with self.subTest("docker", script=script.name):
                argv = dry_run(script, CUDA_COMPAT="1")
                image = argv.index(IMAGE)
                self.assertEqual(argv[image - len(docker_env):image], docker_env)
                self.assertEqual(engine_args(argv), engine_args(dry_run(script)))
        cases = [
            {"name": "native, empty LD_LIBRARY_PATH", "env": {}, "path": COMPAT_DIR, "dir": COMPAT_DIR},
            {"name": "native, existing LD_LIBRARY_PATH", "env": {"LD_LIBRARY_PATH": "/opt/lib"},
             "path": f"{COMPAT_DIR}:/opt/lib", "dir": COMPAT_DIR},
            {"name": "native, CUDA_COMPAT_DIR", "env": {"CUDA_COMPAT_DIR": "/usr/local/cuda-13.0/compat"},
             "path": "/usr/local/cuda-13.0/compat", "dir": "/usr/local/cuda-13.0/compat"},
        ]
        for case in cases:
            with self.subTest(case["name"]):
                argv = dry_run(H20, RUNTIME="native", CUDA_COMPAT="1", PYTHON="/opt/venv/bin/python", **case["env"])
                self.assertEqual(argv[3:6], ["VLLM_ENABLE_CUDA_COMPATIBILITY=1",
                                             f"VLLM_CUDA_COMPATIBILITY_PATH={case['dir']}",
                                             f"LD_LIBRARY_PATH={case['path']}"])
                self.assertEqual(argv[6], "/opt/venv/bin/python")

    def test_old_driver_hint_goes_to_stderr_only(self):
        cases = [
            {"name": "535 without compat", "driver": "535.247.01", "env": {}, "hint": True},
            {"name": "535 with compat", "driver": "535.247.01", "env": {"CUDA_COMPAT": "1"}, "hint": False},
            {"name": "535 with cu129 image", "driver": "535.247.01",
             "env": {"VLLM_IMAGE": "vllm/vllm-openai:v0.32.0-cu129"}, "hint": False},
            {"name": "535 native", "driver": "535.247.01", "env": {"RUNTIME": "native"}, "hint": True},
            {"name": "580 driver", "driver": "580.95.05", "env": {}, "hint": False},
            {"name": "unparsable output", "driver": "N/A", "env": {}, "hint": False},
        ]
        for case in cases:
            with self.subTest(case["name"]), tempfile.TemporaryDirectory() as directory:
                fake_executable(directory, "nvidia-smi", FAKE_NVIDIA_SMI)
                path = f"{directory}{os.pathsep}{os.environ['PATH']}"
                result = run(H20, path=path, DRY_RUN="1", FAKE_DRIVER=case["driver"], **case["env"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual("CUDA_COMPAT=1" in result.stderr, case["hint"], result.stderr)
                if case["hint"]:
                    self.assertIn(f"driver {case['driver']} is older than R580", result.stderr)
                plain = run(H20, DRY_RUN="1", **case["env"])
                self.assertEqual(result.stdout, plain.stdout)

    def test_prefix_cache_never_forces_mamba_cache_mode(self):
        for script in (T4, H20):
            for runtime in ("docker", "native"):
                with self.subTest(script.name, runtime=runtime):
                    args = dry_run(script, RUNTIME=runtime, PREFIX_CACHE="1")
                    self.assertIn("--enable-prefix-caching", args)
                    self.assertNotIn("--no-enable-prefix-caching", args)
                    self.assertFalse([arg for arg in args if arg.startswith("--mamba-cache-mode")])

    def test_invalid_knobs_exit_2(self):
        cases = [
            (T4, {"RUNTIME": "podman"}), (T4, {"PREFIX_CACHE": "2"}), (T4, {"EAGER": "yes"}),
            (T4, {"ASYNC_SCHED": "on"}), (T4, {"PROFILE": "9b-bf16"}), (T4, {"CUDA_COMPAT": "yes"}),
            (T4, {"PROFILE": "27b-fp8"}), (H20, {"PROFILE": "4b-fp16"}), (H20, {"PREFIX_CACHE": "on"}),
            (SHARED, {"MACHINE": "a100"}),
        ]
        for script, env in cases:
            with self.subTest(script=script.name, **env):
                self.assertEqual(run(script, DRY_RUN="1", **env).returncode, 2)

    def test_native_dry_run_does_not_need_vllm(self):
        for script, machine in ((T4, "t4"), (H20, "h20")):
            with self.subTest(machine):
                result = run(script, DRY_RUN="1", RUNTIME="native", PYTHON="/missing/python")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"Machine: {machine}\n", result.stdout)
                self.assertIn("Runtime: native\nPython: /missing/python (needs vllm==0.32.0)", result.stdout)

    def test_native_execs_checked_interpreter(self):
        cases = [
            {"name": "pinned", "script": T4, "found": "0.32.0", "code": 0, "engine": ENGINE_T4_9B},
            {"name": "pinned cu129 build", "script": H20, "found": "0.32.0+cu129", "code": 0,
             "engine": ENGINE_H20_27B_FP8},
            {"name": "old pin", "script": T4, "found": "0.18.1", "code": 1},
            {"name": "not installed", "script": H20, "found": "", "code": 1},
        ]
        for case in cases:
            with self.subTest(case["name"]), tempfile.TemporaryDirectory() as directory:
                python = fake_executable(directory, "python", FAKE_PYTHON)
                cache = Path(directory) / "hf"
                result = run(case["script"], RUNTIME="native", PYTHON=str(python), FAKE_VLLM=case["found"],
                             HF_CACHE_DIR=str(cache))
                self.assertEqual(result.returncode, case["code"], result.stderr)
                if case["code"]:
                    self.assertIn("uv pip install 'vllm==0.32.0'", result.stderr)
                    self.assertIn("vllm/vllm-openai:v0.32.0 as the pod image", result.stderr)
                    self.assertIn(f"found: {case['found'] or 'none'}", result.stderr)
                    self.assertFalse(cache.exists())
                    continue
                lines = result.stdout.splitlines()
                self.assertIn("CUDA_VISIBLE_DEVICES=0", lines)
                self.assertIn(f"HF_HOME={cache}", lines)
                self.assertIn("VLLM_ENABLE_CUDA_COMPATIBILITY=", lines)
                served = [line[len("arg="):] for line in lines if line.startswith("arg=")]
                self.assertEqual(served[:3], ["-m", "vllm.entrypoints.cli.main", "serve"])
                self.assertEqual(without_listen(served[3:]), without_listen(case["engine"]))
                self.assertTrue(cache.is_dir())

    def test_native_cuda_compat_needs_the_compat_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            python = fake_executable(directory, "python", FAKE_PYTHON)
            missing = Path(directory) / "missing-compat"
            result = run(H20, RUNTIME="native", PYTHON=str(python), FAKE_VLLM="0.32.0", CUDA_COMPAT="1",
                         CUDA_COMPAT_DIR=str(missing), HF_CACHE_DIR=str(Path(directory) / "hf"))
            self.assertEqual(result.returncode, 1)
            self.assertIn(f"forward-compatibility libraries in {missing}", result.stderr)

            compat = Path(directory) / "compat"
            compat.mkdir()
            result = run(H20, RUNTIME="native", PYTHON=str(python), FAKE_VLLM="0.32.0", CUDA_COMPAT="1",
                         CUDA_COMPAT_DIR=str(compat), LD_LIBRARY_PATH="/opt/lib",
                         HF_CACHE_DIR=str(Path(directory) / "hf"))
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines()
            self.assertIn(f"LD_LIBRARY_PATH={compat}:/opt/lib", lines)
            self.assertIn("VLLM_ENABLE_CUDA_COMPATIBILITY=1", lines)

    def test_docker_exec_matches_dry_run_and_missing_docker_points_to_native(self):
        for script in (T4, H20):
            with self.subTest(script.name), tempfile.TemporaryDirectory() as directory:
                fake_executable(directory, "docker", FAKE_DOCKER)
                cache = Path(directory) / "hf"
                path = f"{directory}{os.pathsep}{os.environ['PATH']}"
                result = run(script, path=path, HF_CACHE_DIR=str(cache))
                self.assertEqual(result.returncode, 0, result.stderr)
                ran = [line[len("arg="):] for line in result.stdout.splitlines() if line.startswith("arg=")]
                self.assertEqual(["docker", *ran], dry_run(script, HF_CACHE_DIR=str(cache)))

            with self.subTest(script.name, docker="missing"), tempfile.TemporaryDirectory() as empty:
                result = run(script, path=empty)
                self.assertEqual(result.returncode, 1)
                self.assertIn("RUNTIME=native", result.stderr)


if __name__ == "__main__":
    unittest.main()
