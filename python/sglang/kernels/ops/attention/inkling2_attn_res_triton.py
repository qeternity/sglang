"""Triton implementation of the Inkling2 attention-residual combine (attn_res)
for SM8x (Ampere, e.g. the 3090), where the CUTE-DSL variant
(`inkling2_attn_res.AttnResFwdKernel`) does not apply: it uses TMA pipelines,
mbarriers, and elect_one, all sm_90+.

Term-for-term port of the CUTE-DSL math. Per token t and query q:

  x_n[t] = blocks[n][t]  (+ partial_addend[t] for the last source n == N-1)
  rms_inv_n = rsqrt(sum(x_n^2) / D + eps)          (sStat[n, 0] in the cutedsl kernel)
  logit_n   = dot(x_n[t], queries[q]) * rms_inv_n  (sStat[n, q+1] * sStat[n, 0])
  w_n       = softmax over n of logit_n            (max-subtracted, sExp * inv_sum)
  out[q][t] = sum_n w_n * (x_n[t] * rms_inv_n if normalize_values else x_n[t])

All math in fp32 (inputs bf16/fp32, outputs fp32), like the cutedsl kernel.
No CUTLASS/CUTE-DSL import here: this module is safe on boxes without
nvidia-cutlass-dsl installed. Dispatch: inkling2_attn_res.block_attn_res
(model side) selects this when the device is SM8x.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_res_fwd(
    Base,  # element pointer of blocks[0]; sources addressed via Base + Offsets[n]
    Offsets,  # (MAX_S,) int64 element offsets of each source relative to Base
    Addend,  # element pointer of the partial addend (unloaded unless HAS_ADDEND)
    Queries,  # (N_Q, D) fp32 query vectors
    Out,  # (N_Q, T, D) fp32 output
    T,  # tokens (runtime)
    N,  # number of sources (runtime)
    D: tl.constexpr,
    N_Q: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    NORMALIZE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    q = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    qvec = tl.load(Queries + q * D + cols, mask=mask, other=0.0)

    # Pass 1: per-source logit with an online softmax max/sum over the sources.
    m = -float("inf")
    s = 0.0
    for n in range(N):
        off = tl.load(Offsets + n).to(tl.int64)
        x = tl.load(Base + off + t * D + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_ADDEND and n == N - 1:
            a = tl.load(Addend + t * D + cols, mask=mask, other=0.0).to(tl.float32)
            x = x + a
        dot = tl.sum(x * qvec)
        sum_sq = tl.sum(x * x)
        rms_inv = tl.rsqrt(sum_sq / D + EPS)
        logit = dot * rms_inv
        m_new = tl.maximum(m, logit)
        s = s * tl.exp(m - m_new) + tl.exp(logit - m_new)
        m = m_new
    inv = 1.0 / s

    # Pass 2: the weighted combine.
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for n in range(N):
        off = tl.load(Offsets + n).to(tl.int64)
        x = tl.load(Base + off + t * D + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_ADDEND and n == N - 1:
            a = tl.load(Addend + t * D + cols, mask=mask, other=0.0).to(tl.float32)
            x = x + a
        dot = tl.sum(x * qvec)
        sum_sq = tl.sum(x * x)
        rms_inv = tl.rsqrt(sum_sq / D + EPS)
        prob = tl.exp(dot * rms_inv - m) * inv
        if NORMALIZE:
            prob = prob * rms_inv
        acc += prob * x
    tl.store(Out + q * T * D + t * D + cols, acc, mask=mask)


_OFFSETS_CACHE: dict[tuple, torch.Tensor] = {}


def _source_offsets(blocks: list[torch.Tensor]) -> torch.Tensor:
    """Element offsets of each source relative to blocks[0], cached by pointer
    tuple. The blocks are the model's static message-bank views, so their
    data_ptrs are stable across steps and across graph replays; a per-call
    torch.tensor(..., device=...) H2D copy here would be both un-capturable
    (unpinned) and a host-race with the pinned allocator on SM8x decode.
    The cache entry is built with torch.empty + per-element scalar fills
    (fill kernels with baked scalar args: no CPU tensor copy at all, so
    even a cache miss that first happens inside a graph capture is safe)."""
    base = blocks[0]
    key = (
        tuple(block.data_ptr() for block in blocks),
        base.element_size(),
        base.device,
    )
    cached = _OFFSETS_CACHE.get(key)
    if cached is None:
        base_ptr = base.data_ptr()
        values = [
            (block.data_ptr() - base_ptr) // base.element_size() for block in blocks
        ]
        cached = torch.empty(len(values), dtype=torch.int64, device=base.device)
        for i, value in enumerate(values):
            # explicit narrow+fill_: a scalar fill kernel (baked scalar
            # argument), never a CPU-tensor copy
            cached.narrow(0, i, 1).fill_(int(value))
        _OFFSETS_CACHE[key] = cached
    return cached

def attn_res(
    blocks: list[torch.Tensor],
    partial_addend: torch.Tensor | None,
    queries: torch.Tensor,
    *,
    normalize_values: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Same public shape as inkling2_attn_res.attn_res (the cutedsl entry):
    `blocks` are (T, D) bf16/fp32, `partial_addend` (T, D) or None (added to
    the last block), `queries` (N_Q, D) fp32; returns a tuple of N_Q (T, D)
    fp32 tensors."""
    base = blocks[0]
    assert base.is_cuda and base.is_contiguous() and base.ndim == 2
    assert queries.dtype == torch.float32 and queries.is_contiguous()
    assert base.dtype in (torch.bfloat16, torch.float32)
    for block in blocks:
        assert block.shape == base.shape and block.dtype == base.dtype
        assert block.device == base.device and block.is_contiguous()
    if partial_addend is not None:
        assert partial_addend.shape == base.shape and partial_addend.dtype == base.dtype
        assert partial_addend.device == base.device and partial_addend.is_contiguous()
    tokens, width = base.shape
    n_queries = queries.shape[0]
    assert queries.shape[1] == width and queries.device == base.device
    output = torch.empty(
        (n_queries, tokens, width), dtype=torch.float32, device=base.device
    )
    if not tokens:
        return tuple(output.unbind(0))

    offsets = _source_offsets(blocks)
    block = 1 << (width - 1).bit_length()
    _attn_res_fwd[(tokens, n_queries)](
        base,
        offsets,
        base if partial_addend is None else partial_addend,
        queries,
        output,
        tokens,
        len(blocks),
        width,
        n_queries,
        partial_addend is not None,
        normalize_values,
        1e-6,
        block,
        num_warps=8 if block >= 4096 else 4,
    )
    return tuple(output.unbind(0))
