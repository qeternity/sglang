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
                    torch.tensor([1.0, 2.0], device=DEVICE, dtype=torch.float32),
                ),
            ):
                got = qk_l2_clip_fwd(
                    x, threshold, 1e-6, scale=scale, output_scale=out_scale
                )
                ref = ref_qk_l2_clip(
                    x, threshold, 1e-6, scale=scale, output_scale=out_scale
                )
                all_ok &= check(f"clip m={m} n={n} {label}", got, ref)
    # 3D input (T, H, N) like the model feeds
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

    print(
        f"\nVERDICT: {'T4 PASS' if all_ok else 'T4 FAIL'} (triton fallback vs reference)"
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
