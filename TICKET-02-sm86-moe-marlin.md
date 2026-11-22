# TICKET-02 — SM86 MoE dispatch via Marlin W4A16/W8A16 (NVFP4 + MXFP8)

- **Branch:** `zuma-inkling`
- **Hardware target:** 2× RTX 3090 (SM86, TP2/EP2)
- **Dependencies:** TICKET-00
- **Estimate:** 1–2 wks
- **Status:** implemented locally 2026-11-21 (dispatch + CPU tests; first 3090 JIT build + load pending on sauron)

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

## Implementation log (2026-11-21)

Step 1 is done, and it was smaller than the ticket assumed: the dispatch is a
single condition, and every downstream hook already gates on it.

- `runtime_context.py`: registered the `is_sm80` platform fact
  (`"is_sm80": "is_sm80_supported"` in `_PLATFORM_PROBES`). The existing
  `is_sm80_supported()` probe (`srt/utils/device.py:343`) matches the whole
  SM8x family (major == 8, CUDA >= 11) — SM80/86/89 — which is exactly the
  set that lacks FP8/FP4 tensor cores and must run weight-only dequant +
  BF16 MMA.
- `inkling2_moe.py`: `self.use_marlin = platform.is_sm90 or platform.is_sm80
  or (platform.is_sm120 and not self.native_sm120)`.

Everything else follows automatically from `use_marlin` — verified by reading
each hook, no extra changes needed:

- `prepare_routed_exchange` (`inkling2_moe.py:~340`): early-returns for
  `ep_size == 1 or use_marlin` — the EP token exchange (the CUTLASS/MegaMoE
  path) is skipped on SM86; the Marlin combine at the end of forward
  (oneshot all-reduce + `source_partial`) is the one that runs.
- `prepare_mxfp8_cutlass_scales`: gated on `not self.use_marlin` — skipped.
- `swizzle_nvfp4_scales`: branches `native_sm120` → Marlin repack →
  capability guard; with `use_marlin` set it takes the Marlin repack branch
  (`marlin_sm90.repack` / `repack_mxfp8`) and never reaches the SM100/103/107
  hard error.
- The Marlin kernel stack is arch-neutral: `moe_wna16_marlin` loads through
  the generic `load_jit` (arch auto-detected; the only CUDA gate in
  `moe_wna16_marlin.cuh` is `__CUDA_ARCH__ < 800`), and the NVFP4/MXFP8
  dequant + activation fake-quant are pure-Triton emulation
  (`inkling2_experts_sm90.py`, `inkling2_moe.py::_mxfp8_block`) with no
  `cp.async`/TMA/FP8-specific instructions.

The SM86 combine uses the same `multicast_combine` branch as SM90
(`inkling2_moe.py:~799`, `oneshot_all_reduce_input` + `multicast_all_reduce`).
On sauron (no NVLink, `multicast_ptr == 0`) that call is exactly what
TICKET-03 replaces with the size-dispatched P2P/NCCL path — the tickets
compose; T2 alone will still raise there by design until T3 lands.

Tests: `test/registered/unit/models/test_inkling2_moe_marlin_dispatch.py`
(4 CPU cases, `base-a-test-cpu`): builds a real `Inkling2Experts`
(bf16, 4 experts) with `override_platform` + the parallel context stamped to
TP1/EP1, and checks `use_marlin` for (SM8x→True new), (SM90→True existing),
(all off →False, SM100 native CUTLASS), and (SM8x+SM120 → True, short-circuit).
All pass; the full T0/T1 CPU regression (204 tests) stays green.

Remaining (sauron, with T5): first JIT build of `moe_wna16_marlin` for sm_86
(both variants), NVFP4 + MXFP8 load-time repack, the GEMV branch at batch 1,
and the numerical spot-check against the bf16 reference.

## Risks

- `moe_wna16_marlin.cuh` may contain instructions beyond the stated SM80 floor in practice
  (unlikely — the `#if` guard is explicit; the first build verifies).
- Marlin on SM86 will be **slower than on SM90** in absolute terms only via lower bf16 FLOPS
  and lower bandwidth (936 GB/s vs ~3.3 TB/s); at batch 1 the MoE decode is GEMV/memory-bound
  and 3090 bandwidth (936 GB/s) is the same order as H200's — so MoE decode should be
  competitive per GPU; prefill will be ~3–5× behind H200. Acceptable for the target workload.
- The "sm90" naming in several files (`inkling2_marlin_sm90.py`, `inkling2_experts_sm90.py`)
  is now misleading after this ticket — rename is optional follow-up, not in scope.

### Addendum (sauron bring-up, GEMV gate)

`marlin_moe_gemv.cuh` (the small-batch expert GEMM) is Hopper-only by construction:
thread-block clusters + distributed shared memory to sum the K splits, and unguarded
PDL asm; its launch hard-codes `.use_pdl = true`. It cannot be ported to SM86
(clusters/DSMEM are sm_90+ features), and `uses_gemv()` had no arch check, so SM8x
decode (tokens <= 8) would have hit the same ptxas failure on the first decode forward.
Fix: `uses_gemv()` now returns `False` when `is_arch_support_pdl()` is false (the SM90+
probe already used by T3), routing small batches to the generic Marlin GEMM
(`moe_wna16_marlin`, whose only arch floor is sm_80) — correct, just slower per call
than the clustered GEMV; at 3090 MoE decode the GEMM path is the memory-bound one
anyway. `moe_wna16_marlin.cuh` was audited: no PDL, no clusters, `#if SGL_CUDA_ARCH == 900`
special-cases only — SM86-safe.

### Addendum 2 (sauron bring-up, triton PDL gates in the MoE path)

The MoE gate forward crashed in `_route_topk` (triton): `inkling2_route.py`,
`inkling2_moe.py` (4 kernels: mxfp8_quantize, routed_rows,
unweighted_post_reorder, routed_partial), and `inkling2_mxfp8.py`
(_routed_swap_ab_matmul) all called `tl.extra.cuda.gdc_wait()/
gdc_launch_dependents()` unconditionally — same sm_90-only instruction,
fifth hiding place (the csrc sweep had covered `route.cuh` but not the
triton kernels in the same file). All host launches already passed
`launch_pdl=PDL` (module-level `is_arch_support_pdl()`); the kernels got
the `USE_PDL: tl.constexpr` guard idiom (default False where the signature
has defaulted args), with `PDL` passed at every launch. Repo-wide sweep of
the remaining inkling2 triton files (sconv, linear, loop, residual_add,
attn, attn_prologue, comm): all already guarded and launch-gated.

### Addendum 3 (sauron bring-up, e4m3fn cast is sm89+-only in triton)

The Marlin experts path reached `experts_sm90.fake_quantize` and failed at
compile: `tl.float8e4nv` (e4m3fn) is only available in triton on sm89+
("supported fp8 dtypes are fp8e4b15, fp8e5" on sm_86) — the same wall vLLM
hit on A100 (they gated fp8 MoE kernels to sm89). The fake quantize is
intentional on SM86 (it emulates the NVFP4/MXFP8 activation quantization so
the W4A16 Marlin GEMM reproduces the shipped model's numerics), so it can't
be gated away. Replaced the hardware cast with `_e4m3_round(x)`: an
exact-arithmetic e4m3fn round-trip (RNE, ties-to-even, saturate at 448)
implemented in exact fp32 + int32 bit ops only (all e4m3fn values are exact
in fp32, every op is exact, so it is bit-identical to the hardware cast).
The torch mirror of the same ops was validated CPU-side against
`.to(torch.float8_e4m3fn)` on 18M values (log-uniform, random bit patterns,
all 256 grid points, all 253 tie midpoints, denormal/480/inf/NaN specials,
pipeline-shaped scales) — 100% bit match. Sites switched: `_nvfp4_values`
(scale rounding, the crasher), `_fake_quantize_mxfp8_rows` and
`_swiglu_quantize` (value quantize); `_mxfp8_block` now returns fp32 values
and the one consumer that needs real e4m3 bytes (the Hopper-only
`_mxfp8_quantize` in `inkling2_moe.py`) casts at its store — Hopper
numerics unchanged (same cast, same point in the pipeline). GPU check
added as section E of `t4_norm_validate.py` (triton `_e4m3_round` vs torch
e4m3fn cast on the box).

### Addendum 4 (sauron bring-up, e4m3 subnormal shift was UB)

Post-boot warning `Value 140 exceeds the maximum bitwidth (32) for type
'int32'` pointed at `(m << 140)` in `_e4m3_round`'s subnormal branch: a
140-bit shift of an int32 is undefined behavior in triton (the CPU torch
validation had passed only because C++ left the UB unmasked). Replaced with
the trivially exact `m.to(tl.float32) * 2.0**-9` (m <= 8, power of two).

### Addendum 5 (sauron bring-up, e4m3 normal-path exponent was 124 not 127)

t4 section E (first successful GPU run of the e4m3 check) showed every
result exactly 8x too small (447 -> 56, 117.15 -> 15, 2^-6 -> 2^-9).
The normal-path bit construction used exponent field `124 + e + carry`
where it must be `127 + e + carry`: the rounded significand r = 8 + m3
(+carry) represents (1 + m3/8) * 2^(e + carry), whose fp32 exponent
field is 127 + e + carry. The earlier "18M validation" had exercised a
separate torch mirror, not this bit trick, so the port bug shipped.
Fixed to 127; an exact torch mirror of the kernel's bit ops now matches
torch's e4m3fn cast on 6M adversarial values (grid points, ties,
denormals, random bit patterns, pipeline-shaped values) with 0 mismatches.
Note this bug corrupts MoE activation fake-quant scales by a factor of 8
in the 124 version — the server output was doubly wrong (norm bug + this)
until the norm fix landed.
