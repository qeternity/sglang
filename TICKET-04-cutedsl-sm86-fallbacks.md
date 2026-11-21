# TICKET-04 — CUTE-DSL SM86 probe + Triton fallbacks (qk_l2_clip, rms_norm)

- **Branch:** `zuma-inkling`
- **Hardware target:** 2× RTX 3090 (SM86)
- **Dependencies:** TICKET-00 (needs the model to import; the kernels themselves can be probed standalone)
- **Estimate:** 1–3 days
- **Status:** pending

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
