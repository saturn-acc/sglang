"""One program per captured row; two passes avoid storing the full row in registers."""

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
    maxima = tl.full((BLOCK,), -float("inf"), tl.float32)
    for start in range(tl.cdiv(V, BLOCK)):
        col = start * BLOCK + lane
        value = tl.load(P + row * p_row + col * p_col, col < V, other=-float("inf"))
        maxima = tl.maximum(maxima, value)
    threshold = tl.max(maxima, 0) * tl.load(M + row * m_stride)
    for start in range(tl.cdiv(V, BLOCK)):
        col = start * BLOCK + lane
        value = tl.load(P + row * p_row + col * p_col, col < V, other=0.0)
        tl.store(O + row * V + col, tl.where(value < threshold, 0.0, value), col < V)
    token = tl.load(T + row * t_stride).to(tl.int64)
    weight = tl.load(P + row * p_row + token * p_col)
    tl.store(S + row, tl.where(weight < threshold, 0.0, weight))


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
