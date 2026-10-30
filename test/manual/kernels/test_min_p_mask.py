"""Standalone qualification: python test/manual/kernels/test_min_p_mask.py.

Loads the prototype directly so reference tests need only PyTorch, not SRT.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
KERNELS = ROOT / "python/sglang/kernels/ops/sampling"
spec = importlib.util.spec_from_file_location("min_p_mask", KERNELS / "min_p_mask.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Let the lazy GPU import resolve without importing the full SGLang runtime.
if torch.cuda.is_available():
    triton_spec = importlib.util.spec_from_file_location(
        "sglang.kernels.ops.sampling.min_p_mask_triton",
        KERNELS / "min_p_mask_triton.py",
    )
    triton_module = importlib.util.module_from_spec(triton_spec)
    triton_spec.loader.exec_module(triton_module)
    sys.modules[triton_spec.name] = triton_module


class TestMinPMask(unittest.TestCase):
    def test_threshold_semantics(self):
        p = torch.tensor([[0.5, 0.25, 0.125, 0.0]]).expand(3, -1).clone()
        m = torch.tensor([0.0, 0.5, 1.0])
        ids = torch.tensor([3, 1, 2])
        filtered, selected = module.min_p_mask(p, m, ids)
        torch.testing.assert_close(
            filtered,
            torch.tensor(
                [[0.5, 0.25, 0.125, 0.0], [0.5, 0.25, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]]
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            selected, torch.tensor([0.0, 0.25, 0.0]), rtol=0, atol=0
        )
        self.assertEqual(p[1, 2].item(), 0.125)

    def test_empty_batch(self):
        p, w = module.min_p_mask(
            torch.empty(0, 7), torch.empty(0), torch.empty(0, dtype=torch.long)
        )
        self.assertEqual(p.shape, (0, 7))
        self.assertEqual(w.shape, (0,))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_gpu_reference_and_graph(self):
        torch.manual_seed(17)
        for rows, vocab in [(1, 1), (3, 1023), (7, 4097), (1, 128256), (32, 151936)]:
            for strided in (False, True):
                with self.subTest(rows=rows, vocab=vocab, strided=strided):
                    p = torch.rand(rows, vocab * (2 if strided else 1), device="cuda")
                    if strided:
                        p = p[:, ::2]
                    p[0].zero_()
                    m = torch.linspace(0, 1, rows, device="cuda")
                    ids = torch.arange(rows, device="cuda", dtype=torch.int32) % vocab
                    expected = module.min_p_mask_reference(p, m, ids)
                    actual = module.min_p_mask(p, m, ids)
                    for a, e in zip(actual, expected):
                        torch.testing.assert_close(a, e, rtol=0, atol=0)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = module.min_p_mask(p, m, ids)
                    graph.replay()
                    for a, e in zip(captured, expected):
                        torch.testing.assert_close(a, e, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
