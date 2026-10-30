"""Experimental min-p capture fusion; not enabled in the sampler yet.

Inputs are post-top-k/top-p probability weights and already sampled token ids.
This operator neither samples nor renormalizes. GPU qualification and serving
benchmarks are required before replacing Sampler._sample_from_probs's reference.
"""

import torch


def min_p_mask_reference(probs, min_ps, token_ids):
    threshold = probs.max(dim=-1).values * min_ps
    filtered = probs.masked_fill(probs < threshold[:, None], 0)
    selected = filtered.gather(1, token_ids.long().reshape(-1, 1)).squeeze(1)
    return filtered, selected


def min_p_mask(probs, min_ps, token_ids):
    """Return filtered weights and selected weights without modifying inputs.

    Expects finite nonnegative float32 weights [rows, vocab], float32 min_ps
    [rows] in [0, 1], and valid int32/int64 token_ids [rows]. Values equal to
    the threshold are retained, matching the sampler's strict comparison.
    """
    if probs.ndim != 2 or probs.shape[1] == 0:
        raise ValueError("probs must have shape [rows, nonempty vocab]")
    rows, vocab = probs.shape
    if min_ps.shape != (rows,) or token_ids.shape != (rows,):
        raise ValueError("min_ps and token_ids must have shape [rows]")
    if probs.dtype != torch.float32 or min_ps.dtype != torch.float32:
        raise ValueError("probabilities and min_ps must be float32")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ids must be int32 or int64")
    if probs.device != min_ps.device or probs.device != token_ids.device:
        raise ValueError("all inputs must be on the same device")
    if not probs.is_cuda:
        return min_p_mask_reference(probs, min_ps, token_ids)

    # Keep Triton optional for CPU reference tests and unsupported platforms.
    from sglang.kernels.ops.sampling.min_p_mask_triton import launch

    filtered = torch.empty((rows, vocab), device=probs.device, dtype=probs.dtype)
    selected = torch.empty((rows,), device=probs.device, dtype=probs.dtype)
    if rows:
        launch(probs, min_ps, token_ids, filtered, selected)
    return filtered, selected
