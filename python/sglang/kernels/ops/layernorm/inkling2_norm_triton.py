# type: ignore
"""Triton implementations of the Inkling2 per-layer norm/clip ops, for SM8x
(Ampere, e.g. the 3090) where the CUTE-DSL variants do not apply.

Ported term-for-term from the cutedsl kernels in inkling2_qk_l2_clip.py and
inkling2_rms_norm.py (see inkling2_cute_utils.qk_l2_clip_row_stats /
rmsnorm_row_rstd for the source formulas):

  qk_l2_clip:  x  -> f32 (per-head learnable scale applied first, if present)
               sum_sq = sum(x^2) over the row
               norm   = sum_sq * rsqrt(sum_sq)   (0 if sum_sq == 0)
               coeff  = min(1.0, threshold / (norm + eps))
               y      = x * coeff
               (if output_scale: y = (y -> out_dtype -> f32) * out_scale[row // k])
               store as out_dtype
  rms_norm:    x  -> f32 (pre_weight applied first, if present)
               rstd = rsqrt(sum(x^2) / N + eps)
               y    = x * rstd (* weight, if present)
               store as out_dtype

No CUTLASS/CUTE-DSL import here: this module is safe on boxes without
nvidia-cutlass-dsl installed. The model dispatches to it when the device is
SM8x (see inkling2_norm.RMSNorm.forward / inkling2_attn.QKL2Clip.forward).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_l2_clip_rows(
    X,
    Scale,
    Y,
    OutputScale,
    N: tl.constexpr,
    H: tl.constexpr,
    rows_per_out_scale: tl.constexpr,
    threshold,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    if H > 0:
        # scale is (H, N); row r belongs to head r % H (rows are head-major).
        head = row % H
        s = tl.load(Scale + head * N + cols, mask=mask, other=0.0).to(tl.float32)
        x = x * s
    sum_sq = tl.sum(x * x, axis=0)
    norm = tl.where(sum_sq > 0.0, sum_sq * tl.rsqrt(sum_sq), 0.0)
    coeff = tl.minimum(1.0, threshold / (norm + eps))
    y = x * coeff
    if rows_per_out_scale > 0:
        out_scale = tl.load(OutputScale + row // rows_per_out_scale).to(tl.float32)
        # Exact port: round to the output dtype, back to f32, scale, re-round
        # on store.
        y = y.to(Y.dtype.element_ty).to(tl.float32) * out_scale
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _rms_norm_rows(
    X,
    Weight,
    PreWeight,
    Y,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_PRE_WEIGHT: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_PRE_WEIGHT:
        pw = tl.load(PreWeight + cols, mask=mask, other=0.0).to(tl.float32)
        x = x * pw
    sum_sq = tl.sum(x * x, axis=0)
    rstd = tl.rsqrt(sum_sq / N + eps)
    y = x * rstd
    if HAS_WEIGHT:
        w = tl.load(Weight + cols, mask=mask, other=0.0).to(tl.float32)
        y = y * w
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=mask)


def _as_rows(x: torch.Tensor) -> torch.Tensor:
    """Same contract as inkling2_cute_utils.as_norm_rows for the shapes the
    model feeds these ops (2D/3D, last dim contiguous)."""
    if x.ndim < 2 or x.shape[-1] == 0:
        raise ValueError(
            f"Expected at least two dimensions with a nonempty last dim, got {x.shape=}"
        )
    if x.ndim > 3 or x.stride(-1) != 1 or not x.is_contiguous():
        x = x.contiguous()
    return x.view(-1, x.shape[-1])


def qk_l2_clip_fwd(
    x: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
    *,
    output_dtype: torch.dtype | None = None,
    scale: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton qk_l2_clip for SM8x. Same public shape as
    inkling2_qk_l2_clip.qk_l2_clip_fwd_cutedsl."""
    assert x.is_cuda, f"{x.device=} must be CUDA"
    output_dtype = output_dtype or x.dtype
    if scale is not None:
        assert scale.is_cuda and scale.device == x.device
        assert scale.ndim in (1, 2) and scale.is_contiguous()
    if output_scale is not None:
        assert output_scale.is_cuda and output_scale.is_contiguous()
        assert output_scale.dtype == torch.float32 and output_scale.ndim == 1

    rows = _as_rows(x)
    m, n = rows.shape
    output_scale_len = output_scale.numel() if output_scale is not None else 0
    if output_scale_len:
        assert m % output_scale_len == 0, (output_scale.shape, x.shape)
    rows_per_out_scale = m // output_scale_len if output_scale_len else 0
    h = scale.shape[0] if scale is not None and scale.ndim == 2 else 0
    if scale is not None and scale.ndim == 1:
        # A rank-1 scale indexes mScale[col], one value per column, the same on
        # every row; fold it to the (1, N) form so head = row % 1 == 0 loads it.
        scale = scale.unsqueeze(0)
        h = 1

    y = torch.empty((m, n), device=x.device, dtype=output_dtype)
    if m == 0:
        return y.view(x.shape)
    block = 1 << (n - 1).bit_length()
    _qk_l2_clip_rows[(m,)](
        rows,
        scale if scale is not None else y,
        y,
        output_scale if output_scale is not None else y,
        n,
        h,
        rows_per_out_scale,
        float(threshold),
        float(eps),
        block,
        num_warps=4 if block >= 1024 else 1,
    )
    return y.view(x.shape)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    *,
    pre_weight: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton rms_norm for SM8x. Same public shape as
    inkling2_rms_norm.rms_norm."""
    assert x.is_cuda and x.shape[-1] > 0 and x.shape[-1] % 4 == 0
    for parameter in (weight, pre_weight):
        if parameter is not None:
            assert parameter.is_contiguous() and parameter.shape == (x.shape[-1],)
            assert parameter.device == x.device
    output_dtype = output_dtype or x.dtype
    rows = _as_rows(x)
    m, n = rows.shape
    if out is not None:
        assert out.shape == x.shape
        y = out.view(m, n)
        assert y.dtype == output_dtype and y.is_contiguous()
    else:
        y = torch.empty((m, n), device=x.device, dtype=output_dtype)
    if m == 0:
        return y.view(x.shape)
    block = 1 << (n - 1).bit_length()
    _rms_norm_rows[(m,)](
        rows,
        weight if weight is not None else y,
        pre_weight if pre_weight is not None else y,
        y,
        n,
        float(eps),
        block,
        weight is not None,
        pre_weight is not None,
        num_warps=4 if block >= 1024 else 1,
    )
    return y.view(x.shape)
