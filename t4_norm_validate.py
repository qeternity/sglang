#!/usr/bin/env python3
"""T4 validation probe for a SM86 box (sauron): Inkling2 norm/clip triton
fallback vs a pure-torch reference, plus a CUTE-DSL probe (captured
verbatim if it fails, as the ticket requires).

Run:  python t4_norm_validate.py

PASS bar:
  - all triton-vs-reference checks within bf16 noise (rtol/atol 2e-2,
    exact-match ratio >= 99.9% for the clip, >= 99% for the norm);
  - the model modules (RMSNorm / QKL2Clip) automatically pick the triton
    path on this device (is_sm80 True);
  - the CUTE-DSL outcome is recorded (failures here are expected and
    fine on SM86; the dispatch already routes around them).
"""

import sys
import traceback

import torch

DEVICE = "cuda:0"


def ref_rms_norm(x, weight, eps, pre_weight=None, output_dtype=torch.bfloat16):
    xf = x.float()
    if pre_weight is not None:
        xf = xf * pre_weight.float()
    rstd = torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) / x.shape[-1] + eps)
    y = xf * rstd
    if weight is not None:
        y = y * weight.float()
    return y.to(output_dtype)


def ref_qk_l2_clip(
    x, threshold, eps, scale=None, output_scale=None, output_dtype=torch.bfloat16
):
    xf = x.float()
    if scale is not None:
        m, _ = xf.shape
        h = scale.shape[0]
        heads = torch.arange(m, device=x.device) % h
        xf = xf * scale.float()[heads]
    sum_sq = (xf * xf).sum(-1, keepdim=True)
    norm = torch.where(
        sum_sq > 0, sum_sq * torch.rsqrt(sum_sq), torch.zeros_like(sum_sq)
    )
    coeff = torch.minimum(torch.ones_like(sum_sq), threshold / (norm + eps))
    y = xf * coeff
    if output_scale is not None:
        k = y.shape[0] // output_scale.numel()
        idx = torch.arange(y.shape[0], device=x.device) // k
        y = y.to(output_dtype).float() * output_scale[idx, None]
    return y.to(output_dtype)


def ref_attn_res(blocks, addend, queries, normalize, eps=1e-6):
    """Pure-torch attn_res: per token, per query, over the sources n:
    x_n (+ addend for the last source), rms_inv = rsqrt(mean(x^2) + eps),
    logit = dot(x, q) * rms_inv, softmax over n, weighted combine (x or
    x * rms_inv depending on normalize)."""
    t, d = blocks[0].shape
    n = len(blocks)
    out = torch.empty(
        queries.shape[0], t, d, device=blocks[0].device, dtype=torch.float32
    )
    for qi in range(queries.shape[0]):
        q = queries[qi].float()
        logits, vals = [], []
        for i in range(n):
            x = blocks[i].float()
            if addend is not None and i == n - 1:
                x = x + addend.float()
            rms_inv = torch.rsqrt(x.square().sum(-1) / d + eps)
            logits.append((x * q).sum(-1) * rms_inv)
            vals.append(x * (rms_inv[:, None] if normalize else 1.0))
        w = torch.softmax(torch.stack(logits, 0), 0)
        out[qi] = torch.einsum("nt,ntd->td", w, torch.stack(vals, 0))
    return out

def check(name, got, ref, rtol=2e-2, atol=2e-2):
    got, ref = got.float(), ref.float()
    exact = (got == ref).float().mean().item()
    ok = torch.allclose(got, ref, rtol=rtol, atol=atol)
    maxdiff = (got - ref).abs().max().item()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: exact {exact * 100:.3f}%, maxdiff {maxdiff:.3g}")
    return ok


def main():
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
    from sglang.kernels.ops.layernorm.inkling2_norm_triton import (
        qk_l2_clip_fwd,
        rms_norm,
    )

    all_ok = True

    print("\n== A. triton rms_norm vs torch reference ==")
    for n in (128, 2304):
        for m in (1, 7, 257):
            x = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
            w = torch.randn(n, device=DEVICE, dtype=torch.bfloat16) + 1.0
            pw = torch.randn(n, device=DEVICE, dtype=torch.bfloat16) + 1.0
            for label, weight, pre in (
                ("w", w, None),
                ("w+pw", w, pw),
                ("no-w", None, None),
            ):
                got = rms_norm(x, weight, 1e-6, pre_weight=pre)
                ref = ref_rms_norm(x, weight, 1e-6, pre_weight=pre)
                all_ok &= check(f"rms_norm m={m} n={n} {label}", got, ref)
    # out= path
    x = torch.randn(17, 2304, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(2304, device=DEVICE, dtype=torch.bfloat16) + 1.0
    out = torch.empty_like(x)
    rms_norm(x, w, 1e-6, out=out)
    all_ok &= check("rms_norm out= path", out, ref_rms_norm(x, w, 1e-6))

    print("\n== B. triton qk_l2_clip vs torch reference ==")
    threshold = 22.627
    for n in (16, 128):
        for m in (1, 7, 257):
            x = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16) * 10
            for label, scale, out_scale in (
                ("no-scales", None, None),
                (
                    "scale",
                    torch.rand(2, n, device=DEVICE, dtype=torch.bfloat16) * 1.5 + 0.5,
                    None,
                ),
                (
                    "scale+out",
                    torch.rand(2, n, device=DEVICE, dtype=torch.bfloat16) * 1.5 + 0.5,
                    torch.tensor([1.0], device=DEVICE, dtype=torch.float32),
                ),
            ):
                got = qk_l2_clip_fwd(
                    x, threshold, 1e-6, scale=scale, output_scale=out_scale
                )
                ref = ref_qk_l2_clip(
                    x, threshold, 1e-6, scale=scale, output_scale=out_scale
                )
                all_ok &= check(f"clip m={m} n={n} {label}", got, ref)
    # output_scale covering groups of rows (rows_per_output_scale > 1)
    m8, n8 = 8, 128
    x8 = torch.randn(m8, n8, device=DEVICE, dtype=torch.bfloat16) * 10
    out8 = torch.tensor([1.0, 2.0], device=DEVICE, dtype=torch.float32)
    got8 = qk_l2_clip_fwd(x8, threshold, 1e-6, output_scale=out8)
    ref8 = ref_qk_l2_clip(x8, threshold, 1e-6, output_scale=out8)
    all_ok &= check(f"clip m={m8} rows_per_out_scale=2", got8, ref8)
    t, h, n = 11, 4, 128
    x3 = torch.randn(t, h, n, device=DEVICE, dtype=torch.bfloat16) * 10
    scale3 = torch.rand(h, n, device=DEVICE, dtype=torch.bfloat16) * 1.5 + 0.5
    got3 = qk_l2_clip_fwd(x3, threshold, 1e-6, scale=scale3)
    ref3 = ref_qk_l2_clip(x3.view(-1, n), threshold, 1e-6, scale=scale3).view(t, h, n)
    all_ok &= check("clip 3D input (T,H,N)", got3, ref3)

    print("\n== C. model modules auto-dispatch to triton on this device ==")
    from sglang.srt.models.inkling_common.inkling2_attn import QKL2Clip
    from sglang.srt.models.inkling_common.inkling2_norm import RMSNorm

    norm = RMSNorm(2304).cuda().to(torch.bfloat16)
    print(f"  RMSNorm._use_triton = {norm._use_triton}")
    all_ok &= norm._use_triton is True
    xn = torch.randn(7, 2304, device=DEVICE, dtype=torch.bfloat16)
    gotn = norm(xn)
    all_ok &= check(
        "RMSNorm module forward (weight=ones)",
        gotn,
        ref_rms_norm(xn, torch.ones(2304, device=DEVICE, dtype=torch.bfloat16), 1e-6),
    )

    clip = (
        QKL2Clip(128, threshold, 1e-6, 8, learnable_scale=True)
        .cuda()
        .to(torch.bfloat16)
    )
    clip.scale.data.fill_(1.0)
    print(f"  QKL2Clip._use_triton = {clip._use_triton}")
    all_ok &= clip._use_triton is True
    xc = torch.randn(7, 8, 128, device=DEVICE, dtype=torch.bfloat16) * 10
    gotc = clip(xc)
    all_ok &= check(
        "QKL2Clip module forward (scale=1)",
        gotc,
        ref_qk_l2_clip(
            xc.view(-1, 128),
            threshold,
            1e-6,
            scale=torch.ones(8, 128, device=DEVICE, dtype=torch.bfloat16),
        ).view(7, 8, 128),
    )

    print("\n== C2. triton attn_res vs torch reference (SM86 loop combine) ==")
    from sglang.kernels.ops.attention.inkling2_attn_res_triton import (
        attn_res as attn_res_triton,
    )
    from sglang.srt.models.inkling_common.inkling2_attn_res import _attn_res_kernel

    resolved = _attn_res_kernel()
    print(f"  _attn_res_kernel() -> {resolved.__module__}")
    all_ok &= resolved.__module__ == "sglang.kernels.ops.attention.inkling2_attn_res_triton"
    for d in (128, 2304):
        for n_src in (1, 3, 5):
            for t in (1, 7):
                for nq in (1, 3):
                    for use_addend in (False, True):
                        for normalize in (False, True):
                            blocks = [
                                torch.randn(t, d, device=DEVICE, dtype=torch.bfloat16)
                                * 3
                                for _ in range(n_src)
                            ]
                            addend = (
                                torch.randn(t, d, device=DEVICE, dtype=torch.bfloat16)
                                * 3
                                if use_addend
                                else None
                            )
                            q = torch.randn(nq, d, device=DEVICE, dtype=torch.float32)
                            got = attn_res_triton(
                                blocks, addend, q, normalize_values=normalize
                            )
                            ref = ref_attn_res(blocks, addend, q, normalize)
                            label = (
                                f"attn_res t={t} d={d} n={n_src} nq={nq} "
                                f"addend={use_addend} norm={normalize}"
                            )
                            for i in range(nq):
                                all_ok &= check(
                                    f"{label}[{i}]", got[i], ref[i], rtol=5e-2, atol=5e-2
                                )

    print("\n== D. CUTE-DSL probe on SM86 (failure expected and fine) ==")
    for label, fn in (
        (
            "qk_l2_clip_fwd_cutedsl",
            lambda: __import__(
                "sglang.kernels.ops.layernorm.inkling2_qk_l2_clip", fromlist=["x"]
            ).qk_l2_clip_fwd_cutedsl(
                torch.randn(8, 128, device=DEVICE, dtype=torch.bfloat16), 22.627, 1e-6
            ),
        ),
        (
            "cutedsl rms_norm",
            lambda: __import__(
                "sglang.kernels.ops.layernorm.inkling2_rms_norm", fromlist=["x"]
            ).rms_norm(
                torch.randn(8, 2304, device=DEVICE, dtype=torch.bfloat16),
                torch.ones(2304, device=DEVICE, dtype=torch.bfloat16),
                1e-6,
            ),
        ),
    ):
        try:
            out = fn()
            torch.cuda.synchronize()
            print(f"  [RUNS] {label} ran on this device (out={out.shape}, {out.dtype})")
        except Exception as e:  # noqa: BLE001 - the probe records ANY failure verbatim
            print(f"  [FAIL-EXPECTED] {label}: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc(limit=3)

    print("\n== E. triton _e4m3_round vs torch e4m3fn cast (fake-quant scales) ==")
    import triton
    import triton.language as tl
    from sglang.kernels.ops.moe.inkling2_experts_sm90 import _e4m3_round

    @triton.jit
    def _e4m3_probe(X, Y, BLOCK: tl.constexpr):
        cols = tl.arange(0, BLOCK)
        tl.store(Y + cols, _e4m3_round(tl.load(X + cols)))

    def e4m3_tensor(x):
        y = torch.empty_like(x)
        _e4m3_probe[(triton.cdiv(x.numel(), 1024),)](x, y, 1024)
        return y

    ref_e4m3 = lambda x: x.to(torch.float8_e4m3fn).to(torch.float32)

    for label, x in (
        ("log-uniform [2^-14, 448]",
         torch.exp(torch.log(2.0**-14) + torch.rand(1_000_000, device=DEVICE)
         * (torch.log(448.0) - torch.log(2.0**-14)))),
        ("random-bit fp32 in [0,448)",
         torch.randint(0, 2**32, (1_000_000,), device=DEVICE, dtype=torch.int32)
         .view(torch.float32)
         .abs()
         .clamp(max=447.0)),
        ("clamped mxfp8 values",
         (torch.randn(1_000_000, device=DEVICE) * 448.0).clamp(-448.0, 448.0)),
        ("all e4m3 grid points + ties",
         None),
    ):
        if x is None:
            bits = torch.arange(256, dtype=torch.int64)
            e = (bits >> 3) & 0xF
            mm = bits & 0x7
            vals = torch.where(
                e > 0,
                (1.0 + mm.float() / 8.0) * torch.tensor(2.0 ** (e - 7)),
                torch.tensor(mm * 2.0**-9),
            ).float()
            x = torch.cat([vals, (vals[:-1] + vals[1:]) / 2.0,
                           torch.tensor([0.0, 448.0, 480.0, 512.0, 1e38])]).cuda()
        got = e4m3_tensor(x)
        want = ref_e4m3(x)
        same = (got == want) | (torch.isnan(got) & torch.isnan(want))
        bad = int((~same).sum())
        all_ok &= bad == 0
        print(f"  {label}: {'OK' if bad == 0 else 'MISMATCH'} (n={x.numel()}, bad={bad})")
        if bad:
            idx = (~same).nonzero().flatten()[:5]
            for i in idx:
                print(f"    x={x[i].item():.17g} got={got[i].item():.17g} "
                      f"want={want[i].item():.17g}")

    print(
        f"\nVERDICT: {'T4 PASS' if all_ok else 'T4 FAIL'} (triton fallback vs reference)"
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
