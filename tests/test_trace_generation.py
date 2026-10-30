"""Cache-shape extraction in examples/trace-generation.py with stand-in objects; no torch or downloads."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trace_generation", ROOT / "examples/trace-generation.py")
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


class Tensor:
    def __init__(self, *shape):
        self.shape = shape


class DynamicLayer:
    """Shaped like transformers 5.x full-attention cache layers."""

    def __init__(self):
        self.keys = Tensor(1, 4, 7, 256)
        self.values = Tensor(1, 4, 7, 256)


class LinearAttentionLayer:
    """Shaped like transformers 5.x linear-attention (GDN) cache layers: one tensor per state index."""

    def __init__(self, conv_states):
        self.conv_states = conv_states
        self.recurrent_states = {0: Tensor(1, 32, 128, 128)}


class Cache:
    def __init__(self, layers):
        self.layers = layers


class CacheSnapshotTests(unittest.TestCase):
    def test_tensor_shapes(self):
        cases = [
            {"name": "tensor", "value": Tensor(2, 3), "want": [2, 3]},
            {"name": "one state per layer", "value": {0: Tensor(1, 8192, 3)}, "want": [1, 8192, 3]},
            {"name": "several states", "value": {0: Tensor(1, 2), 1: Tensor(1, 3)}, "want": [[1, 2], [1, 3]]},
            {"name": "state not allocated yet", "value": {0: None}, "want": None},
            {"name": "list of tensors", "value": [Tensor(4)], "want": [4]},
            {"name": "attribute missing", "value": None, "want": None},
        ]
        for case in cases:
            with self.subTest(case["name"]):
                self.assertEqual(trace.tensor_shapes(case["value"]), case["want"])

    def test_snapshot_reports_both_layer_kinds(self):
        cases = [
            {"name": "gdn then full attention",
             "layers": [LinearAttentionLayer({0: Tensor(1, 8192, 3)}), DynamicLayer()],
             "want": [{"layer": 0, "type": "LinearAttentionLayer",
                       "shapes": {"conv_states": [1, 8192, 3], "recurrent_states": [1, 32, 128, 128]}},
                      {"layer": 1, "type": "DynamicLayer",
                       "shapes": {"keys": [1, 4, 7, 256], "values": [1, 4, 7, 256]}}]},
            {"name": "gdn state before the first forward",
             "layers": [LinearAttentionLayer({0: None})],
             "want": [{"layer": 0, "type": "LinearAttentionLayer",
                       "shapes": {"recurrent_states": [1, 32, 128, 128]}}]},
            {"name": "object without layers", "layers": None, "want": []},
        ]
        for case in cases:
            with self.subTest(case["name"]):
                cache = Cache(case["layers"]) if case["layers"] is not None else object()
                self.assertEqual(trace.cache_snapshot(cache), case["want"])


if __name__ == "__main__":
    unittest.main()
