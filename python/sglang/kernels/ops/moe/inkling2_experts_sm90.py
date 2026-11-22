"""Activation fake quantization for the Hopper expert path: NVFP4 and MXFP8 values
(NVFP4 without the per-row outer scale) are exact in BF16, so a BF16 MMA against the
dequantized weights forms the block-scaled MMA's products exactly."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.inkling2_moe import _mxfp8_block

ROW_LOOP_MIN_ROWS = 1024


@triton.jit
def _e2m1_round(x):
    """cvt.rn.satfinite.e2m1x2.f32 emulation: nearest-even onto
    {0, .5, 1, 1.5, 2, 3, 4, 6}, saturating at 6."""
    a = tl.abs(x)
    r = tl.where(
        a <= 0.25,
        0.0,
        tl.where(
            a < 0.75,
            0.5,
            tl.where(
                a <= 1.25,
                1.0,
                tl.where(
                    a < 1.75,
                    1.5,
                    tl.where(
                        a <= 2.5,
                        2.0,
                        tl.where(a < 3.5, 3.0, tl.where(a <= 5.0, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where(x < 0, -r, r)


@triton.jit
def _e4m3_round(x):
    """cvt.rn.satfinite.e4m3.f32 via exact fp32 arithmetic: round x onto the
    e4m3fn grid (ties to even, saturating at 448) and return it in fp32.
    Triton has no fp8e4nv cast below sm89, but every e4m3fn value is exact in
    fp32 and each op here is exact, so this is bit-identical to the hardware
    cast (validated against torch's e4m3fn cast on 18M adversarial values,
    including all grid points and all ties)."""
    is_nan = x != x  # noqa: PLR0124 - NaN check
    a = tl.where(is_nan, 0.0, tl.abs(x))
    # Saturation: the midpoint between 448 (1.75 * 2^8) and 512 (2^9) is 480.
    a = tl.where(a >= 480.0, 448.0, a)
    # Subnormal grid m * 2^-9 (m in 0..7): q = a * 512, exact, rounds to m.
    q = a * 512.0
    n = q.to(tl.int32)
    f = q - n
    m = n + (f > 0.5).to(tl.int32) + ((f == 0.5).to(tl.int32) & (n & 1))
    sub = (m << 140).to(tl.float32, bitcast=True)
    # Normal grid 1.mmm * 2^e (e in -6..8): qn = a * 2^(3-e), exact, in [8, 16).
    bits = a.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127
    qn = (bits + (3 - e) * (1 << 23)).to(tl.float32, bitcast=True)
    nn = qn.to(tl.int32)
    fn = qn - nn
    r = nn + (fn > 0.5).to(tl.int32) + ((fn == 0.5).to(tl.int32) & (nn & 1))
    # r * 2^(e-3) in bits: r = 8 + m3; r == 16 carries to the next binade.
    carry = r >> 4
    m3 = r & 7
    norm = (((124 + e + carry) << 23) | (m3 << 20)).to(tl.float32, bitcast=True)
    out = tl.where(a < 2.0**-6, sub, norm)
    out = tl.where(is_nan, float("nan"), out)
    return tl.where(x < 0.0, -out, out)


@triton.jit
def _block_error(x, reconstructed):
    groups: tl.constexpr = x.shape[0]
    error = tl.abs(reconstructed - x)
    for width in tl.static_range(3, -1, -1):
        pairs = tl.reshape(error, (groups, 1 << width, 2))
        low, high = tl.split(pairs)
        error = low + high
    return tl.reshape(error, (groups,))


@triton.jit
def _fake_quantize_rows(
    X,
    Y,
    Amax,
    XS,
    K: tl.constexpr,
    AMAX_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if LOOP:
        amax = tl.zeros((), tl.float32)
        for chunk in range(tl.cdiv(K, BLOCK)):
            cols = chunk * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(X + row * XS + cols, cols < K, other=0).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(x), 0))
        tl.store(Amax + row, amax)
        for chunk in range(tl.cdiv(K, BLOCK)):
            _fake_quantize_chunk(X, Y, row, chunk, amax, XS, K, BLOCK)
    else:
        chunk = tl.program_id(1)
        full = tl.arange(0, AMAX_BLOCK)
        amax = tl.max(
            tl.abs(tl.load(X + row * XS + full, full < K, other=0).to(tl.float32)), 0
        )
        if chunk == 0:
            tl.store(Amax + row, amax)
        _fake_quantize_chunk(X, Y, row, chunk, amax, XS, K, BLOCK)


@triton.jit
def _fake_quantize_chunk(
    X, Y, row, chunk, amax, XS, K: tl.constexpr, BLOCK: tl.constexpr
):
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * XS + cols, cols < K, other=0).to(tl.float32)
    y = _nvfp4_values(tl.reshape(x, (BLOCK // 16, 16)), amax)
    tl.store(
        Y + row * K + cols, tl.reshape(y, (BLOCK,)).to(Y.dtype.element_ty), cols < K
    )


@triton.jit
def _nvfp4_values(x, amax):
    """NVFP4 values (e2m1 x e4m3 block scale, no outer scale) of (blocks, 16) x."""
    encoder_scale = tl.minimum(tl.div_rn(1536.0, amax), 3.4028234663852886e38)
    decoder_scale = tl.div_rn(1.0, encoder_scale)
    outer_scale = amax * (1.0 / 1536.0)
    block_max = tl.max(tl.abs(x), 1)
    scale6 = tl.minimum(block_max * (encoder_scale * (1.0 / 6.0)), 448.0)
    scale4 = tl.minimum(block_max * (encoder_scale * 0.25), 448.0)
    scale6 = _e4m3_round(scale6)
    scale4 = _e4m3_round(scale4)
    inverse6 = tl.minimum(tl.div_rn(1.0, scale6 * decoder_scale), 3.4028234663852886e38)
    inverse4 = tl.minimum(tl.div_rn(1.0, scale4 * decoder_scale), 3.4028234663852886e38)
    rounded6 = _e2m1_round(x * inverse6[:, None])
    rounded4 = _e2m1_round(x * inverse4[:, None])
    reconstructed6 = rounded6 * scale6[:, None]
    reconstructed4 = rounded4 * scale4[:, None]
    error6 = _block_error(x, reconstructed6 * outer_scale)
    error4 = _block_error(x, reconstructed4 * outer_scale)
    choose4 = error4 < error6
    return tl.where(choose4[:, None], reconstructed4, reconstructed6)


def fake_quantize(x: torch.Tensor):
    """Rows of x as BF16 NVFP4 values without the outer scale, plus each row's amax."""
    rows, width = x.shape
    assert x.stride(1) == 1
    y = torch.empty((rows, width), dtype=torch.bfloat16, device=x.device)
    amax = torch.empty(rows, dtype=torch.float32, device=x.device)
    if rows:
        loop = rows >= ROW_LOOP_MIN_ROWS
        _fake_quantize_rows[(rows, 1 if loop else triton.cdiv(width, 256))](
            x,
            y,
            amax,
            x.stride(0),
            width,
            triton.next_power_of_2(width),
            256,
            loop,
            enable_fp_fusion=False,
        )
    return y, amax


@triton.jit
def _fake_quantize_mxfp8_rows(X, Y, XS, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * XS + cols, cols < K, other=0).to(tl.float32)
    values, exponent = _mxfp8_block(tl.reshape(x, (BLOCK // 32, 32)))
    y = _e4m3_round(values) * _ue8m0(exponent)[:, None]
    tl.store(Y + row * K + cols, tl.reshape(y, (BLOCK,)).to(tl.bfloat16), cols < K)


def fake_quantize_mxfp8(x: torch.Tensor):
    """Rows of x as BF16 MXFP8 values."""
    rows, width = x.shape
    assert x.stride(1) == 1
    y = torch.empty((rows, width), dtype=torch.bfloat16, device=x.device)
    if rows:
        _fake_quantize_mxfp8_rows[(rows, triton.cdiv(width, 256))](
            x,
            y,
            x.stride(0),
            width,
            256,
            enable_fp_fusion=False,
        )
    return y


@triton.jit
def _ue8m0(exponent):
    return tl.where(
        exponent == 0,
        2.0**-127,
        (exponent.to(tl.int32) << 23).to(tl.float32, bitcast=True),
    )
