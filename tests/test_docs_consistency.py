"""Version pins in README/docs/index.html must match scripts/serve-vllm.sh; no GPU or network."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/serve-vllm.sh"
DOCS = [ROOT / "README.md", ROOT / "index.html", *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "examples").glob("*.py"))]
FALLBACK_VERSION = "0.32.0"
IMAGE_TAG = re.compile(r"vllm/vllm-openai:v(\d+\.\d+\.\d+)([-\w.]*)")
PIP_PIN = re.compile(r"vllm==(\d+\.\d+\.\d+)")
CUDA12_BUILD = re.compile(r"v?(\d+)\.(\d+)\.\d+(?:-cu12\d|\+cu12\d)")


def default_version():
    match = re.search(r'vllm_version="\$\{VLLM_VERSION:-([0-9.]+)\}"', SCRIPT.read_text())
    assert match, "default VLLM_VERSION not found in serve-vllm.sh"
    return match.group(1)


def occurrences(pattern):
    for path in DOCS:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in pattern.finditer(line):
                yield f"{path.relative_to(ROOT)}:{lineno}", match


class DocsConsistencyTest(unittest.TestCase):
    def test_image_tags_are_the_pin_or_the_cu129_fallback(self):
        allowed = {(default_version(), ""), (FALLBACK_VERSION, "-cu129")}
        seen = set()
        for where, match in occurrences(IMAGE_TAG):
            tag = (match.group(1), match.group(2).rstrip("."))
            seen.add(tag)
            self.assertIn(tag, allowed, f"{where}: {match.group(0)}")
        self.assertIn((default_version(), ""), seen)

    def test_pip_pins_match_the_script_default(self):
        pins = list(occurrences(PIP_PIN))
        self.assertTrue(pins)
        for where, match in pins:
            self.assertEqual(match.group(1), default_version(), f"{where}: {match.group(0)}")

    def test_no_cuda12_build_after_0_32(self):
        for path, text in [(SCRIPT, SCRIPT.read_text()), *((p, p.read_text(encoding="utf-8")) for p in DOCS)]:
            for match in CUDA12_BUILD.finditer(text):
                major, minor = int(match.group(1)), int(match.group(2))
                if major == 0 and minor > 32:
                    self.fail(f"{path.relative_to(ROOT)}: {match.group(0)} was never published (0.33+ is CUDA 13 only)")

    def test_lab_and_readme_name_the_pinned_version_first(self):
        version = default_version()
        for name in ("README.md", "docs/t4-qwen35-lab.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            first = re.search(r"\*\*vLLM (\d+\.\d+\.\d+)\*\*", text)
            self.assertIsNotNone(first, name)
            self.assertEqual(first.group(1), version, name)

    def test_dates_are_shanghai_time_unless_marked_utc(self):
        # vLLM 0.34.0 was published 2026-11-16T23:22Z, i.e. 2026-11-17 07:22 in Shanghai.
        for where, match in occurrences(re.compile(r".*")):
            line = match.group(0)
            if "2026-11-16" in line:
                self.assertIn("UTC 2026-11-16 23:22", line, where)
            if "UTC" in line:
                self.assertIn("上海时间", line, where)

    def test_hybrid_block_sizes_follow_the_pinned_version(self):
        lab = (ROOT / "docs/t4-qwen35-lab.md").read_text(encoding="utf-8")
        for row in ("| T4，9B，FP16 KV | 2.05 MiB | 4 KiB | 576 | 528 |",
                    "| H20，27B，BF16 KV | 3.06 MiB | 4 KiB | 832 | 784 |",
                    "| H20，27B，`KV_DTYPE=fp8` | 3.06 MiB | 2 KiB | 1600 | 1568 |"):
            self.assertIn(row, lab)


if __name__ == "__main__":
    unittest.main()
