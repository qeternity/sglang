# TICKET-04 — CUTE-DSL SM86 probe + Triton fallbacks (qk_l2_clip, rms_norm)

- **Branch:** `zuma-inkling`
- **Hardware target:** 2× RTX 3090 (SM86)
- **Dependencies:** TICKET-00 (needs the model to import; the kernels themselves can be probed standalone)
- **Estimate:** 1–3 days
- **Status:** implemented locally 2026-11-21 (term-for-term triton ports + arch dispatch + CPU tests; GPU parity probe `t4_norm_validate.py` pending on sauron with T5/T6; CUTE-DSL-on-SM86 probe folded into that script, section D)

## Problem

Two small per-layer ops use the NVIDIA **CUTE-DSL** (CUTLASS Python DSL) with no visible
Python-side architecture gate:

| Op | Module | Used by |
|---|---|---|
| QK L2 clip | `python/sglang/srt/models/inkling_common/kernels/qk_l2_clip.py` (`QKL2Clip`; the cutedsl forward `qk_l2_clip_fwd_cutedsl` is referenced from `python/sglang/srt/models/inkling2.py`) | every attention layer (clip q/k rows at `qk_l2_clip_threshold=22.627` before attention) |
| RMSNorm | `python/sglang/kernels/ops/layernorm/inkling2_rms_norm.py` (`Inkling2RMSNorm`) | all layernorms in the model |

CUTE-DSL kernels JIT-compile for the **current** device arch at first use. Whether the
installed `nvidia-cutlass-dsl` build targets SM80/86 is unknown until probed — the Hopper+
assumption lives in the CUDA code, not in a Python check.

## Steps

1. **Probe (on the 3090, standalone — no model needed):**
   - `import` both modules and run each cutedsl kernel on a small bf16 tensor
     (e.g. 4×128 q/k for the clip; a (256, 2304) x for the norm).
   - Record: does the DSL JIT succeed for `sm_86`? If it produces an arch-unsupported error,
     capture it verbatim for this ticket.
2. **If the probe fails — Triton fallbacks** (both ops are small and fully specified by the
   module parameters in the checkpoint):
   - **QK L2 clip:** per head, for each q/k row of dim 128: `v = x; n = ||v||₂; if n >
     threshold: v = v * (threshold / n)`-style application with the layer's learnable scale
     (read the exact formula from the existing cutedsl kernel — port it term-for-term, do not
     approximate).
   - **RMSNorm:** standard row-wise RMS normalization with the model's `rms_norm_eps=1e-6`
     and the `pre_norm_pre_affine` / `final_norm_pre_affine` / `use_value_norm_no_gamma`
     variants already parameterized in the cutedsl version.
   - Add an arch-dispatch wrapper: capability 8 → triton kernel; capability ≥ 9 → cutedsl
     (keep the fast path untouched on Hopper/Blackwell).
3. **Parity test:** random bf16 inputs; triton output vs cutedsl output (on SM90) within
   bf16 noise; vs a pure-torch reference on SM86.

## Acceptance criteria

- Both ops run on SM86 (cutedsl if the probe passes, otherwise triton fallback) with output
  parity vs the SM90 path.
- Dispatch is automatic (no user flag); logged once at startup.
- Standalone unit tests registered alongside `test/registered/unit/models/test_inkling2.py`.

## Risks

- Low. Both kernels are elementwise/rowwise; the main risk is mis-porting the exact clip
  formula (threshold interaction, learnable scale placement) — mitigated by the term-for-term
  port + parity test.
- If the installed `nvidia-cutlass-dsl` *silently* compiles for the wrong arch (shouldn't
  happen — the DSL validates arch at build time), the parity test on SM90 catches regressions
  in the other direction.
## Implementation log (2026-11-21, local; sauron validation pending)

Approach: capability-based dispatch (SM8x → triton, everything else →
CUTE-DSL untouched), with the triton implementations in a **new module that
does not import cutlass** — a 3090 box may not have `nvidia-cutlass-dsl`
installed, and the CUTE-DSL modules have hard `import cutlass` at module top.
So the arch decision lives in the model's two forward methods (both already
lazily import the kernel), and the triton path never touches cutlass.

New: `python/sglang/kernels/ops/layernorm/inkling2_norm_triton.py`
- `qk_l2_clip_fwd(x, threshold, eps, *, output_dtype, scale, output_scale)` —
  term-for-term port of `inkling2_cute_utils.qk_l2_clip_row_stats`:
  x→fp32, per-head scale `scale[row % H, :]*` first (if present; rank-1
  scales fold to (1, N) so `row % 1 == 0` loads the per-column value,
  matching the cutedsl rank-1 branch), `sum_sq = Σx²`,
  `norm = sum_sq * rsqrt(sum_sq)` (0 when sum_sq == 0 — exactly as the
  cutedsl kernel), `coeff = min(1, threshold / (norm + eps))`, `y = x*coeff`,
  and the output_scale double-rounding
  `(y → out_dtype → fp32) * out_scale[row // k]` preserved.
- `rms_norm(x, weight, eps, *, pre_weight, output_dtype, out)` — port of
  `rmsnorm_row_rstd`: x→fp32, `* pre_weight` first (if present),
  `rstd = rsqrt(Σx²/N + eps)`, `y = x*rstd*weight`, store as out_dtype.
  The `out=` contract (preallocated buffer) is kept.
- One program per row, masked BLOCK = next_pow2(N); fp32 accumulation;
  triton's `rsqrt` compiles to the same CUDA intrinsic as the cutedsl one.

Model dispatch (2 lines each, decision cached in `__init__`):
- `inkling2_norm.py::RMSNorm` — `self._use_triton = get_platform().is_sm80`
  (the T2 probe, lru_cached); forward picks `inkling2_norm_triton.rms_norm`
  on SM8x, `inkling2_rms_norm.rms_norm` (CUTE-DSL) otherwise.
- `inkling2_attn.py::QKL2Clip` — same pattern for
  `qk_l2_clip_fwd` vs `qk_l2_clip_fwd_cutedsl`. All three model instances
  (q/k/rel clips) go through this one forward.

Scope note: SM120 (consumer Blackwell) is deliberately **out** — `is_sm80`
is False there, so it keeps today's CUTE-DSL behavior exactly as on all
non-Ampere boxes; whether cutlass-dsl targets sm_120 is a separate ticket.

Tests: `test/registered/unit/models/test_inkling2_norm_triton_dispatch.py`
(10 CPU tests, `base-a-test-cpu`: subprocess import with `cutlass` absent,
dispatch table {sm80/sm90/sm120} × {RMSNorm, QKL2Clip}, no-scale variant).
Regression: 226 passed / 6 skipped across the five inkling2 suites +
model_overrides; ruff clean.

GPU validation (sauron, next deploy): `t4_norm_validate.py` — sections
A/B run the triton kernels vs the pure-torch reference over
{M=1,7,257}×{N=16,128,2304} with all scale/pre_weight/out combinations
(including the 3D (T,H,N) layout the model feeds); C runs the real model
modules and asserts they auto-selected the triton path on the box; D is the
ticket's "probe the CUTE-DSL kernels on the 3090 and record verbatim" step
(success or failure both fine — the dispatch already routes around it).
PASS bar: bf16-noise parity + auto-dispatch.
