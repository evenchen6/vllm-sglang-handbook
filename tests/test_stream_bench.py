"""Protocol checks using a local HTTP server; no GPU/model downloads."""
import importlib.util
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stream_bench", ROOT / "examples/stream-bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if payload["model"] == "http-error":
            self.send_error(400, "unsupported model")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning_content": "先思考"}}]},
            {"choices": [{"delta": {"content": "两个以上的token"}}]},
            {"choices": [{"delta": {"content": "。"}, "finish_reason": "stop"}]},
        ]
        if payload["model"] != "no-usage":
            usage = {"completion_tokens": 9, "prompt_tokens": 40}
            if payload["model"] == "cached":
                usage["prompt_tokens_details"] = {"cached_tokens": 32}
            events.append({"choices": [], "usage": usage})
        for event in events:
            data = ("data: " + json.dumps(event, ensure_ascii=False) + "\r\n\r\n").encode()
            # Transport write boundaries can split UTF-8 and JSON; they are not tokens.
            for start in range(0, len(data), 7):
                self.wfile.write(data[start:start + 7])
                self.wfile.flush()
        if payload["model"] != "truncated":
            self.wfile.write(b"data: [DONE]\n\n")


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, model="normal"):
        return bench.run_request(self.url, model, [{"role": "user", "content": "test"}])

    def test_reasoning_role_usage_and_chunk_token_distinction(self):
        result = self.request()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["text_chunks"], 3)
        self.assertEqual(result["completion_tokens"], 9)
        self.assertEqual(result["reasoning"], "先思考")
        self.assertEqual(result["content"], "两个以上的token。")
        self.assertGreaterEqual(result["first_answer_ms"], result["ttft_text_ms"])
        self.assertIsNotNone(result["estimated_tpot_ms"])

    def test_missing_usage_does_not_invent_tokens_or_tpot(self):
        result = self.request("no-usage")
        self.assertTrue(result["ok"])
        self.assertIsNone(result["completion_tokens"])
        self.assertIsNone(result["estimated_tpot_ms"])
        self.assertIsNone(result["cached_tokens"])

    def test_cached_tokens_only_when_the_server_reports_them(self):
        cases = [
            {"model": "cached", "want": 32},
            {"model": "normal", "want": None},
        ]
        for case in cases:
            with self.subTest(case["model"]):
                result = self.request(case["model"])
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["cached_tokens"], case["want"])

    def test_incomplete_stream_is_failure(self):
        result = self.request("truncated")
        self.assertFalse(result["ok"])
        self.assertIn("[DONE]", result["error"])

    def test_http_error_is_a_reported_failure(self):
        result = self.request("http-error")
        self.assertFalse(result["ok"])
        self.assertIn("400", result["error"])

    def test_multiline_and_comments(self):
        data = b': heartbeat\r\nevent: message\r\ndata: first\r\ndata: second\r\n\r\ndata: final'
        self.assertEqual(list(bench.sse_data(io.BytesIO(data))), ["first\nsecond", "final"])

    def test_cli_parallel_results_and_missing_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "results.json"
            result = subprocess.run([
                sys.executable, str(ROOT / "examples/stream-bench.py"),
                "--base-url", self.url, "--model", "no-usage", "--requests", "4",
                "--concurrency", "2", "--warmup", "0", "--output", str(target),
            ], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(target.read_text())
            self.assertEqual(record["summary"]["successful"], 4)
            self.assertEqual(record["summary"]["failed"], 0)
            self.assertIsNone(record["summary"]["output_tok_per_s"])
            self.assertIsNone(record["summary"]["estimated_tpot_ms"])
            self.assertIsNone(record["summary"]["cached_tokens"])
            self.assertIsNone(record["summary"]["cache_hit_requests"])

    def test_cli_summarizes_reported_cache_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "results.json"
            result = subprocess.run([
                sys.executable, str(ROOT / "examples/stream-bench.py"),
                "--base-url", self.url, "--model", "cached", "--requests", "3",
                "--warmup", "0", "--shared-prefix", "--output", str(target),
            ], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(target.read_text())["summary"]
            self.assertEqual(summary["cached_tokens"]["n"], 3)
            self.assertEqual(summary["cached_tokens"]["p50"], 32)
            self.assertEqual(summary["cache_hit_requests"], 3)


if __name__ == "__main__":
    unittest.main()
