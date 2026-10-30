"""Run alongside test_min_p_mask.py on CUDA; reports graph-replay latency."""

import torch
import triton.testing
from test_min_p_mask import module


def main():
    print(f"GPU: {torch.cuda.get_device_name()}; torch={torch.__version__}")
    print("rows,vocab,reference_us,fused_us,speedup")
    for rows in (1, 8, 32, 128):
        for vocab in (32000, 128256, 151936):
            p = torch.softmax(torch.randn(rows, vocab, device="cuda"), -1)
            m = torch.full((rows,), 0.1, device="cuda")
            ids = p.argmax(-1)
            for a, e in zip(
                module.min_p_mask(p, m, ids), module.min_p_mask_reference(p, m, ids)
            ):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
            ref = triton.testing.do_bench_cudagraph(
                lambda p=p, m=m, ids=ids: module.min_p_mask_reference(p, m, ids)
            )
            fused = triton.testing.do_bench_cudagraph(
                lambda p=p, m=m, ids=ids: module.min_p_mask(p, m, ids)
            )
            print(
                f"{rows},{vocab},{ref * 1000:.3f},{fused * 1000:.3f},{ref / fused:.3f}"
            )


if __name__ == "__main__":
    main()
