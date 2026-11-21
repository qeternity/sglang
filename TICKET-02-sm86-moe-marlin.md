# TICKET-02 — SM86 MoE dispatch via Marlin W4A16/W8A16 (NVFP4 + MXFP8)

- **Branch:** `zuma-inkling`
- **Hardware target:** 2× RTX 3090 (SM86, TP2/EP2)
- **Dependencies:** TICKET-00
- **Estimate:** 1–2 wks
- **Status:** pending

## Problem

`Inkling2Experts` (`python/sglang/srt/models/inkling_common/inkling2_moe.py`) dispatches
expert GEMMs by capability:

- line ~239: `self.sm90 = get_platform().is_sm90`
- line ~311: `if self.ep_size == 1 or self.sm90:` → **Marlin** path (import at line 22:
  `from sglang.kernels.ops.gemm import inkling2_marlin_sm90 as marlin_sm90`)
- `prepare_mxfp8_cutlass_scales` (lines ~325–331) runs when `ep_size > 1 and not self.sm90`
  → **CUTLASS** path with `mma_type = "nvfp4xnvfp4" / "fp8xfp8"` (line ~389)
- `swizzle_nvfp4_scales` (line ~483) raises `"NVFP4 scale swizzling requires SM100/SM103/SM107
  for the CUTLASS GEMM"` for `capability < 10`
- Else (SM86, non-sm90, ep1): hard errors in `kernels/ops/gemm/inkling2_nvfp4.py` /
  `inkling2_mxfp8.py` — *"NVFP4/MXFP8 experts require a Blackwell GPU (SM100/SM103/SM107)"*

**SM86 is a gap between the two supported branches.**

## Why Marlin is the correct SM86 path (already the SM90 production path)

- The H200 cookbook recipe (verified for Inkling-2-Mini) runs the **Marlin** path on SM90
  (`self.sm90 == True` → Marlin for all EP sizes). Extending that same branch to SM86 means
  Ampere gets bit-equivalent numerical behavior to the validated Hopper path.
- The Marlin path is weight-only dequant + **BF16 MMA** (see `inkling2_marlin_sm90.py`
  docstring: "weight-only dequantization, BF16 MMA. Activations are fake-quantized to their
  quantized values (NVFP4 without the per-row outer scale)… the BF16 output is the rounded
  raw accumulator"). No FP4/FP8 tensor cores are needed — exactly what a 3090 has (bf16
  tensor cores, no FP8).
- The kernel floor is **SM80**: `python/sglang/kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh:39`
  → `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800` (rejects only < Ampere).
- The NVFP4/MXFP8 dequant and activation fake-quant are **pure Triton emulation**, not
  hardware instructions: `kernels/ops/moe/inkling2_experts_sm90.py` — `_e2m1_round`
  ("cvt.rn.satfinite.e2m1x2.f32 emulation: nearest-even onto {0,.5,1,1.5,2,3,4,6}"),
  `_nvfp4_values`, `_ue8m0`, and `kernels/ops/moe/inkling2_moe.py::_mxfp8_block` — all
  arch-generic Triton.

## Steps

1. **Extend the dispatch** in `Inkling2Experts.__init__` / `forward`:
   introduce e.g. `self._use_marlin = self.sm90 or is_sm8x_device()` (capability major == 8)
   and change the line ~311 condition to `if self.ep_size == 1 or self._use_marlin:` for
   **both** `ep_size == 1` and `ep_size == 2`. The CUTLASS branches
   (`prepare_mxfp8_cutlass_scales`, `swizzle_nvfp4_scales`, cutlass `routed_expert_matmul`)
   must remain unreachable on SM86.
2. **First 3090 JIT build** of `moe_wna16_marlin` (nvfp4 and mxfp8 variants, ep1 and ep2):
   confirm the `cuh` compiles for `sm_86` and the Triton fake-quant kernels build. The repack
   helpers (`srt/layers/quantization/marlin_utils_fp4.py`: `_repack_moe_fp4_weight_for_marlin`,
   `_permute_moe_fp4_scales_for_marlin`, `nvfp4_marlin_process_global_scale`,
   `nvfp4_marlin_process_scales`; `srt/layers/quantization/marlin_utils_fp8.py`:
   `fp8_fused_exponent_bias_into_scales`, `pack_fp8_to_int32`) are torch/CUDA ops — verify
   arch-neutrality by running load-time repack on the 3090.
3. **GEMV (small-batch decode) path:** `inkling2_marlin_sm90.py` has a `GEMV_MAX_EXPERTS = 128`
   branch (decode GEMV). Confirm the decode GEMV compiles and is selected at batch 1 — this is
   the hot path for single-stream serving.
4. **Tests** mirroring `test/registered/unit/models/test_inkling2.py` `Inkling2MoE` cases:
   nvfp4 and mxfp8 experts, `ep_size=1` and `ep_size=2`, top-8 routing with the model's
   `route_scale=8.0`, `use_gate_bias=True`, `mlp_gate_softcap=4.0` / `mlp_up_softcap=25.0`,
   against a bf16-dequant reference (dequantize checkpoint experts offline).
5. **EP2 on 2×3090:** 128 experts / 2 = 64 experts per rank; verify `num_experts % ep_size`
   check (line ~145) and the per-rank chunking (line ~300 `loaded.chunk(self.ep_size, dim=0)`).

## Numerical contract (do not change)

Activations are fake-quantized to their quantized domain values (NVFP4 values without the
per-row outer scale; MXFP8 values with ue8m0 scales), then the BF16 MMA against dequantized
weights reproduces the block-scaled product **exactly** (the values are exact in BF16). The
outer scales are applied by the consumer (`_capped_swiglu` with the softcaps). Changing this
contract changes the model's numerics — it is what makes the SM86 output match the SM90
reference.

## Acceptance criteria

- Expert GEMM output parity vs SM90 Marlin (the validated reference) within bf16 noise for:
  nvfp4 experts (layers 0–19), mxfp8 experts (layers 20–23), ep1 and ep2, prefill (grouped
  GEMM) and decode (GEMV) shapes.
- Load-time repack succeeds on SM86 with no arch-specific failure.
- `Inkling-2-Mini` weight load completes on 2×3090 TP2/EP2 (~10.7 GB weights per GPU).
- Unit tests pass (registered in the base-a CPU/CUDA suites like the existing `test_inkling2.py`
  registrations).

## Risks

- `moe_wna16_marlin.cuh` may contain instructions beyond the stated SM80 floor in practice
  (unlikely — the `#if` guard is explicit; the first build verifies).
- Marlin on SM86 will be **slower than on SM90** in absolute terms only via lower bf16 FLOPS
  and lower bandwidth (936 GB/s vs ~3.3 TB/s); at batch 1 the MoE decode is GEMV/memory-bound
  and 3090 bandwidth (936 GB/s) is the same order as H200's — so MoE decode should be
  competitive per GPU; prefill will be ~3–5× behind H200. Acceptable for the target workload.
- The "sm90" naming in several files (`inkling2_marlin_sm90.py`, `inkling2_experts_sm90.py`)
  is now misleading after this ticket — rename is optional follow-up, not in scope.
