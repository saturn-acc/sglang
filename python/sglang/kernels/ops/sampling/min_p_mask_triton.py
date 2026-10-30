"""Untuned, direct fusion of row max, threshold, masked fill, and gather.

One kernel launch handles all four operations. Fixed-size chunks let the same
implementation cover vocabularies larger than a single Triton block. There is
no autotuning, shape-specialized dispatch, or backend-specific fast path.
"""

import triton
import triton.language as tl


@triton.jit
def _capture(
    P,
    M,
    T,
    O,
    S,
    p_row,
    p_col,
    m_stride,
    t_stride,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)

    # 1. capture_probs.max(dim=-1).values
    maxima = tl.full((BLOCK,), -float("inf"), tl.float32)
    for start in range(tl.cdiv(V, BLOCK)):
        col = start * BLOCK + lane
        value = tl.load(P + row * p_row + col * p_col, col < V, other=-float("inf"))
        maxima = tl.maximum(maxima, value)
    row_max = tl.max(maxima, 0)

    # 2. min_p_thresholds = row_max * capture_min_ps
    threshold = row_max * tl.load(M + row * m_stride)
    token = tl.load(T + row * t_stride).to(tl.int64)
    selected_weight = tl.full((), 0.0, tl.float32)
    for start in range(tl.cdiv(V, BLOCK)):
        col = start * BLOCK + lane
        value = tl.load(P + row * p_row + col * p_col, col < V, other=0.0)

        # 3. filtered_probs = capture_probs.masked_fill(probs < threshold, 0)
        filtered = tl.where(value < threshold, 0.0, value)
        tl.store(O + row * V + col, filtered, col < V)

        # 4. selected_weight = filtered_probs.gather(1, capture_tokens)
        # Exactly one valid column matches the selected token across all chunks.
        selected_weight += tl.sum(
            tl.where((col < V) & (col == token), filtered, 0.0), 0
        )
    tl.store(S + row, selected_weight)


def launch(probs, min_ps, token_ids, filtered, selected):
    _capture[(probs.shape[0],)](
        probs,
        min_ps,
        token_ids,
        filtered,
        selected,
        probs.stride(0),
        probs.stride(1),
        min_ps.stride(0),
        token_ids.stride(0),
        V=probs.shape[1],
        BLOCK=1024,
        num_warps=4,
    )
