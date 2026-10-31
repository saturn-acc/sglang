"""Run the actual kernel on CPU: TRITON_INTERPRET=1 python <this file>.

This checks Triton operation semantics, not GPU compilation or performance.
"""

import importlib.util
import os
import unittest
from pathlib import Path

if os.environ.get("TRITON_INTERPRET") != "1":
    raise RuntimeError("Set TRITON_INTERPRET=1 before starting Python")

import torch
import triton

KERNELS = Path(__file__).resolve().parents[3] / "python/sglang/kernels/ops/sampling"


def load_file(name, filename):
    spec = importlib.util.spec_from_file_location(name, KERNELS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load_file("min_p_reference", "min_p_mask.py").min_p_mask_reference
kernel = load_file("min_p_interpreted", "min_p_mask_triton.py")
if type(kernel._capture).__name__ != "InterpretedFunction":
    raise RuntimeError("Kernel is not using Triton's CPU interpreter")


class TestMinPMaskInterpreter(unittest.TestCase):
    cases = 0

    def compare(self, probs, min_ps, token_ids):
        originals = [x.clone() for x in (probs, min_ps, token_ids)]
        expected = reference(probs, min_ps, token_ids)
        filtered = torch.full(probs.shape, float("nan"), dtype=probs.dtype)
        selected = torch.full((probs.shape[0],), float("nan"), dtype=probs.dtype)
        # Direct launch bypasses min_p_mask()'s PyTorch fallback on CPU.
        if probs.shape[0]:
            kernel.launch(probs, min_ps, token_ids, filtered, selected)
        for actual, wanted in zip((filtered, selected), expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        for actual, original in zip((probs, min_ps, token_ids), originals):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        type(self).cases += 1

    def test_threshold_boundaries(self):
        probs = torch.tensor([[0.5, 0.25, 0.125, 0.0]]).repeat(4, 1)
        min_ps = torch.tensor([0.0, 0.25, 0.5, 1.0])
        for token in range(4):
            self.compare(probs, min_ps, torch.full((4,), token, dtype=torch.long))

    def test_chunk_boundaries_and_token_dtypes(self):
        generator = torch.Generator().manual_seed(42)
        for rows in (1, 3):
            for vocab in (1, 7, 1023, 1024, 1025, 4097, 32000, 128256, 151936):
                for dtype in (torch.int32, torch.int64):
                    with self.subTest(rows=rows, vocab=vocab, dtype=dtype):
                        probs = torch.rand(rows, vocab, generator=generator)
                        min_ps = torch.linspace(0.1, 0.9, rows)
                        ids = torch.tensor(
                            [0, vocab // 2, vocab - 1][:rows], dtype=dtype
                        )
                        self.compare(probs, min_ps, ids)

    def test_strided_inputs(self):
        generator = torch.Generator().manual_seed(3)
        for vocab in (7, 1025, 4097):
            with self.subTest(vocab=vocab):
                probs = torch.rand(6, vocab * 2, generator=generator)[::2, ::2]
                min_ps = torch.tensor([0.1, -1, 0.5, -1, 1.0, -1])[::2]
                ids = torch.tensor([0, -1, vocab // 2, -1, vocab - 1, -1])[::2]
                self.compare(probs, min_ps, ids)

    def test_zeros_ties_and_sparse_support(self):
        for mode in ("zeros", "ties", "sparse"):
            probs = torch.zeros(3, 4097)
            if mode == "ties":
                probs.fill_(0.25)
            elif mode == "sparse":
                probs[:, [0, 1024, 4096]] = torch.tensor([0.5, 0.25, 0.125])
            self.compare(
                probs, torch.tensor([0.0, 0.5, 1.0]), torch.tensor([0, 1024, 4096])
            )

    def test_empty_batch(self):
        self.compare(
            torch.empty(0, 7), torch.empty(0), torch.empty(0, dtype=torch.long)
        )

    @classmethod
    def tearDownClass(cls):
        print(
            f"\nActual Triton interpreter: {cls.cases} cases matched PyTorch exactly."
        )


if __name__ == "__main__":
    print(f"Triton {triton.__version__}; torch {torch.__version__}; CPU interpreter")
    unittest.main(verbosity=2)
