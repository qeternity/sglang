#!/usr/bin/env python3
"""Isolate which op in the triton _e4m3_round bit sequence diverges from the
exact reference, level by level.

For one array of values (10 edge cases + a zero pad + 1M values from the
distribution that failed in t4 section E), a triton kernel computes every
intermediate of the fixed _e4m3_round sequence (levels 0..16) or the repo
kernel itself (level 17); the same operations are computed in vectorized
torch. The first level with mismatches is the divergent op.

Run on the box:  python t4_e4m3_debug.py
"""

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.moe.inkling2_experts_sm90 import _e4m3_round

BLOCK = 1024


@triton.jit
def _levels(X, Y, N, LEVEL: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0)
    is_nan = x != x  # noqa: PLR0124 - NaN check
    a = tl.where(is_nan, 0.0, tl.abs(x))
    a = tl.where(a >= 480.0, 448.0, a)
    q = a * 512.0
    n = q.to(tl.int32)
    f = q - n
    m = n + (f > 0.5).to(tl.int32) + ((f == 0.5).to(tl.int32) & (n & 1))
    sub = m.to(tl.float32) * 2.0**-9
    bits = a.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127
    qn = (bits + (3 - e) * (1 << 23)).to(tl.float32, bitcast=True)
    nn = qn.to(tl.int32)
    fn = qn - nn
    r = nn + (fn > 0.5).to(tl.int32) + ((fn == 0.5).to(tl.int32) & (nn & 1))
    carry = r >> 4
    m3 = r & 7
    norm = (((127 + e + carry) << 23) | (m3 << 20)).to(tl.float32, bitcast=True)
    sel = (a < 2.0**-6).to(tl.float32)
    out = tl.where(a < 2.0**-6, sub, norm)
    out = tl.where(is_nan, float("nan"), out)
    out = tl.where(x < 0.0, -out, out)
    if LEVEL == 0:
        v = x
    elif LEVEL == 1:
        v = a
    elif LEVEL == 2:
        v = q
    elif LEVEL == 3:
        v = n
    elif LEVEL == 4:
        v = f
    elif LEVEL == 5:
        v = m
    elif LEVEL == 6:
        v = sub
    elif LEVEL == 7:
        v = bits
    elif LEVEL == 8:
        v = e
    elif LEVEL == 9:
        v = qn
    elif LEVEL == 10:
        v = nn
    elif LEVEL == 11:
        v = fn
    elif LEVEL == 12:
        v = r
    elif LEVEL == 13:
        v = carry
    elif LEVEL == 14:
        v = m3
    elif LEVEL == 15:
        v = norm
    elif LEVEL == 16:
        v = sel
    else:
        v = out
    tl.store(Y + cols, v, mask=mask)

@triton.jit
def _repo_kernel(X, Y, N, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0)
    tl.store(Y + cols, _e4m3_round(x), mask=mask)


# int levels (stored as int32), rest are float32
INT_LEVELS = {3, 5, 7, 8, 10, 12, 13, 14}
NAMES = {
    0: "x", 1: "a", 2: "q", 3: "n", 4: "f", 5: "m", 6: "sub", 7: "bits",
    8: "e", 9: "qn", 10: "nn", 11: "fn", 12: "r", 13: "carry", 14: "m3",
    15: "norm", 16: "sel", 17: "out(repo _e4m3_round)",
}


def ref_chain(x: torch.Tensor):
    """Vectorized torch mirror of the same op sequence (levels 0..17)."""
    is_nan = torch.isnan(x)
    a = torch.where(is_nan, torch.zeros_like(x), x.abs())
    a = torch.where(a >= 480.0, torch.full_like(a, 448.0), a)
    q = a * 512.0
    n = q.to(torch.int32)
    f = q - n.to(torch.float32)
    m = n + (f > 0.5).to(torch.int32) + ((f == 0.5) & (n & 1)).to(torch.int32)
    sub = m.to(torch.float32) * 2.0**-9
    bits = a.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127
    qn = (bits + (3 - e) * (1 << 23)).view(torch.float32)
    nn = qn.to(torch.int32)
    fn = qn - nn.to(torch.float32)
    r = nn + (fn > 0.5).to(torch.int32) + ((fn == 0.5) & (nn & 1)).to(torch.int32)
    carry = r >> 4
    m3 = r & 7
    norm = (((127 + e + carry) << 23) | (m3 << 20)).view(torch.float32)
    sel = (a < 2.0**-6).to(torch.float32)
    out = torch.where(a < 2.0**-6, sub, norm)
    out = torch.where(is_nan, torch.full_like(out, float("nan")), out)
    out = torch.where(x < 0.0, -out, out)
    return {
        0: x, 1: a, 2: q, 3: n, 4: f, 5: m, 6: sub, 7: bits, 8: e, 9: qn,
        10: nn, 11: fn, 12: r, 13: carry, 14: m3, 15: norm, 16: sel, 17: out,
    }


def main():
    torch.manual_seed(0)
    edge = torch.tensor(
        [447.0, 154.45390319824219, 9.6380294957211518e-37, -150.580078125,
         0.015625, 448.0, -448.0, 0.0, 480.0, 75.561668395996094],
        dtype=torch.float32,
    )
    big = (torch.randn(1_000_000) * 448.0).clamp(-448.0, 448.0)
    xs = torch.cat([edge, torch.zeros(1024 - 10), big])
    pad = (BLOCK - xs.numel() % BLOCK) % BLOCK
    if pad:
        xs = torch.cat([xs, torch.zeros(pad)])
    xt = xs.cuda()
    ref = ref_chain(xs)

    # ---- pre-checks (pure torch, no triton): which side is lying? ----
    print(f"devices: {torch.cuda.device_count()}, using cuda:0")
    fresh = xs.cuda()
    print(f"H2D check torch.equal(xt, xs.cuda()) again: "
          f"{torch.equal(xt, fresh)}")
    print(f"zeros: host={int((xs == 0).sum())}  device={int((xt == 0).sum())}")
    print(f"xt[1020:1034] = {xt[1020:1034].cpu().tolist()[:8]}")
    print(f"xs[1020:1034] = {xs[1020:1034].tolist()[:8]}\n")

    for level in range(18):
        dtype = torch.int32 if level in INT_LEVELS else torch.float32
        out = torch.empty_like(xt, dtype=dtype)
        if level == 17:
            _repo_kernel[(xs.numel() // BLOCK,)](xt, out, xs.numel(), BLOCK)
        else:
            _levels[(xs.numel() // BLOCK,)](xt, out, xs.numel(), level, BLOCK)
        got = out.cpu()
        if level == 0:
            on_dev = int((out != xt).sum())
            print(f"  device-side check: kernel-out vs xt on GPU: "
                  f"{on_dev} mismatches")
        want = ref[level]
        same = got == want
        if level not in INT_LEVELS:
            same = same | (torch.isnan(got.to(torch.float32)) & torch.isnan(want))
        bad = int((~same).sum())
        print(f"level {level:2d} {NAMES[level]:<24} mismatches {bad}/{xs.numel()}")
        if bad:
            idx = (~same).nonzero().flatten()[:3]
            for i in idx:
                xv = xs[i].item()
                gv = got[i].item()
                wv = want[i].item()
                tag = f" x={xv!r} got={gv!r} want={wv!r}"
                print(f"  {tag}")


if __name__ == "__main__":
    main()
