# TICKET-01 — SM86 attention path via triton backend (rel-bias, SWA, LSE)

- **Branch:** `zuma-inkling`
- **Hardware target:** 2× RTX 3090 (SM86)
- **Dependencies:** TICKET-00 (model must exist to verify)
- **Estimate:** 2–3 wks
- **Status:** implemented locally 2026-11-21 (CPU-verified; GPU validation on sauron pending with T2/T3/T5)

## Problem

`Inkling2Attention.forward` (`python/sglang/srt/models/inkling_common/inkling2_attn.py`,
~line 290) hard-requires the FA4 backend:

```python
if serving_attention_backend(forward_batch) == "fa4":
    ...
else:
    raise ... # Inkling2 requires FA4 attention
```

FA4 has no SM86 build: `attention_registry.py:334` maps `"fa4"` →
`FlashAttentionBackend(fa_impl_ver=4)`, and `flashattention_backend.py:_load_fa4_kernels`
only loads kernels for `device_capability[0] in (12,)` (sm120 variant) or the default cute
FA4 path (SM90/SM100). No SM80/86 code path exists.

## Why triton is the right SM86 backend (already verified in-tree)

The generic triton backend (`python/sglang/srt/layers/attention/triton_backend.py`) already
has every capability Inkling2 needs:

| Capability | Evidence |
|---|---|
| `score_mod` / `aux_tensors` on extend | `forward_extend` (line ~1811) accepts both (1820–1821); passes them to the unified and 2-stage kernels (1952–1953). DCP paths raise `NotImplementedError` (1936, 2597) — note: DP is out of scope on 2×3090 anyway |
| `score_mod` / `aux_tensors` on decode | `forward_decode` (line ~2467) accepts both (2476–2477) → `decode_attention_fwd(..., score_mod=score_mod, aux_tensors=aux_tensors)` (line ~2664); signature confirmed at `kernels/ops/attention/decode_attention.py:1188` |
| LSE | `forward_metadata.attn_lse` (line ~145) is filled by the decode kernel — needed for the `gate_sink`/`combine_gate_sink` epilogues |
| Sliding-window KV pools | `sliding_window_size`, window sub-pools, `_window_rail`, `use_sliding_window_kv_pool`, `update_sliding_window_buffer` — already implemented (lines 254, 427, 685) |
| Paged KV with `page_size=128` | `decode_attention_fwd(..., page_size=...)` (kernel arg, default 1) — the recipe's `--page-size 128` is supported |
| Rel-bias via score_mod | `kernels/ops/attention/score_mod.py:45` — `relative_bias_score_mod` (the exact Inkling-1 triton path) + `unpack_aux_tensors` (imported by `extend_attention.py:30`) |

**Prior art:** Inkling-1's `InklingAttention` (`python/sglang/srt/models/inkling_common/attn.py`)
already asserts `backend in ("fa4", "triton")` and uses the triton rel-bias path
(`score_mod=triton_relative_bias_score_mod, aux_tensors=[rel_logits]`). It is the template to
follow — the Inkling2 model just needs the same branching instead of the FA4-only assert.

## Steps

1. **Relax the backend gate** in `Inkling2Attention` (`models/inkling_common/inkling2_attn.py`):
   accept `triton` (and keep `fa4`) — mirror `InklingAttention`'s `assert backend in (...)`.
   Selection is per-layer via `serving_attention_backend(forward_batch)`.
2. **Prefill (extend) path:** replace the FA4 sheared-bias call with
   `score_mod=triton_relative_bias_score_mod` + `aux_tensors=[rel_logits]` where `rel_logits`
   is the per-layer `RelLogitsProj` output (the model already computes it in the prologue).
   Keep `sliding_window_size=1024` on the 20 local layers (full layers get `sliding_window_size
   = -1` / None as today); KV pool dtype bf16.
3. **Decode path:** the FA4 compact path in `Inkling2Attention.forward` passes
   `rel_bias_compact_decode=True, rel_bias_decode_epilogue=...` into FA4's `forward`
   (see `flashattention_backend.py:1655–1665`: `score_mod, aux_tensors, rel_bias,
   rel_bias_event, return_lse, rel_bias_compact_decode, rel_bias_decode_epilogue,
   rel_bias_presheared`). On triton these kwargs are not part of the backend API, so add a
   **kernel-level branch** in the model's forward for the triton case:
   - call `kernels/ops/attention/inkling2_attn.py::rel_bias_partials(...)` directly with the
     triton `inkling2_decode_attention_sm90` kernel (it is plain Triton — the "sm90" in the
     name reflects where it runs, not an arch requirement), then
   - `gate_sink(output, gate, lse, sinks, ...)` / `combine_gate_sink` as today (generic CUDA).
4. **Guard the hardcoded PDL** in `kernels/ops/attention/inkling2_decode_attention_sm90.py:183`
   (`launch_pdl=True`): PDL is SM90+. Change to
   `**({"launch_pdl": True} if is_arch_support_pdl() else {})` — the same pattern already used
   in `inkling2_attn_prologue.py:959`, `models/inkling_common/kernels/sconv.py:10–11,1116–1221`,
   and `kernels/ops/communication/inkling2_comm.py`.
5. **Plumb LSE** out of `triton_backend.forward_decode` for the
   `RadixAttention.return_lse` / `strided_lse` contract (the model's gate_sink epilogues
   consume LSE; `radix_attention.py:84` shows `strided_lse` is part of the layer API).
6. **Drop FA4-only branches on SM86:**
   - `attn_padded` head-padding (FA4 `pack_gqa` optimization) — triton handles GQA
     (24 Q / 8 KV, group 3) natively; set the padded path to None on triton.
   - `q_descale` / MXFP8-KV path — SM86 uses a bf16 KV pool (no FP8 tensor cores on 3090).
7. **CUDA-graph capture** of the triton path (extend + decode) at the recipe's batch sizes;
   the H200 recipe runs with `--disable-cuda-graph` — keep that for the first SM86 bring-up
   and re-enable as a follow-up perf item.

## Acceptance criteria

- Attention outputs on SM86 (triton) match the SM90 (FA4) path for identical KV state and
  layer params — prefill and decode — within bf16 noise (rel-bias logits included, SWA and
  full layers, group=3 GQA).
- LSE from the triton decode matches a torch `logsumexp` reference for the same scores.
- Sliding-window behavior: a token >1024 positions before the query is not visible to local
  layers (spot-check with a needle-in-1k query pattern), full layers see everything.
- Server runs 1-token lossless generation end-to-end on 2×3090 (combined with T2/T3).
- `--page-size 128` works (paged decode with page_size=128 + score_mod).

## Implementation log (2026-11-21)

Implemented with one simplification vs. the plan above: **step 3's kernel-level branch
(`rel_bias_partials` + `inkling2_decode_attention_sm90`) turned out to be unnecessary.**
Reading the triton kernels showed the existing `score_mod` plumbing already has correct
relative-distance semantics on every path (verified by code, not assumption):

- **Decode** (`decode_attention.py`, both the standard and lean kernels): the score_mod
  call site passes `q_pos = cur_batch_seq_len - 1` (KV-entry count − 1) and
  `kv_pos = offs_n` (index into the request's KV array). For a sliding-window rail the
  array holds the last W entries, so the distance `(kv_len − 1) − offs_n` is exactly the
  true query↔key distance — the window start cancels because the window always ends at
  the query. No kernel change needed.
- **Extend** (`extend_attention.py`, 2-stage prefix stage, 2-stage chunk stage, unified
  stage): all three SCORE_MOD call sites use `q_pos`/`kv_pos` in a shared coordinate
  system (prefix-count + chunk offset for both, or window-shifted for both) — the
  window shift cancels in the distance the same way.

Changes made:

1. `models/inkling_common/inkling2_attn.py` — `_forward` now accepts
   `serving_attention_backend(...) in ("fa4", "triton")`; on triton: `compact` and
   `padded` are forced off (both FA4-only features), mxfp8 KV raises
   `RuntimeError` (bf16 KV on SM86), and the rel bias goes through
   `score_mod=triton_relative_bias_score_mod, aux_tensors=[rel_bias]`
   (`sglang.kernels.ops.attention.score_mod.relative_bias_score_mod`, the same helper
   Inkling-1's triton path uses). The `gate_sink` epilogue is unchanged — it only reads
   `lse.stride()`, so the triton LSE layout is fine.
2. `layers/attention/triton_backend.py` — new `return_lse` kwarg on both
   `forward_extend` and `forward_decode`:
   - extend: allocates `lse = empty((q_tokens, tp_q_head_num), fp32)` and passes it as
     `lse_extend` to `extend_attention_fwd` (a hook the 2-stage and unified kernels
     already had for the DCP merge path — `STORE_LSE`); returns `(o, lse)`. The
     dense-fp8-chunked, gfx115-verify, and splitkv-verify fast paths now bail out when
     `return_lse` is set; DCP and the deterministic/unified path raise
     `NotImplementedError` (the unified kernel has the hook but is not plumbed yet).
   - decode: pre-fills the active region of `forward_metadata.attn_lse` with −inf (the
     same trick the DCP branch already uses so stale split slots drop out of the
     reduction), then returns `(o, logsumexp(attn_lse[:n, :h], dim=-1))` — split LSEs
     combine exactly as the DCP branch's `local_lse`. DCP decode raises.
3. `kernels/ops/attention/inkling2_decode_attention_sm90.py` — the hardcoded
   `launch_pdl=True` is now guarded by `is_arch_support_pdl()` (step 4; defensive,
   since this kernel is FA4-only and FA4 does not run on SM86).

CUDA-graph safety: the decode LSE path does a `fill_(-inf)` on a persistent metadata
view (no allocation) and one `logsumexp` (one allocation per layer, same pattern as the
existing per-layer `gate_sink` allocation) — both graph-capture-safe.

Tests: `test/registered/unit/models/test_inkling2_triton_attn.py` (10 CPU tests,
registered `base-a-test-cpu`): triton dispatch uses score_mod/aux_tensors and not
`rel_bias`; FA4 dispatch unchanged; mxfp8+triton raises; unknown backend rejected;
decode LSE = logsumexp of split partials (exact vs torch reference) with −inf
pre-fill; extend LSE allocation/identity/return; DCP + deterministic raise paths.
All pass, and the T0 suites stay green (`test_inkling2.py` 50 passed,
`test_model_overrides.py` 144 passed).

Still to verify on sauron (with T2/T3/T5): the acceptance criteria above — especially
SWA needle behavior, `--page-size 128` decode with score_mod, and bf16-noise parity
against the FA4 path (now cross-checked via gsm8k, see TICKET-06, since there is no
SM90/SM100 reference box).

## Risks

- The triton decode kernel + `score_mod` + `page_size=128` combination may be untested in the
  triton backend (SWA + score_mod together); expect a debugging window on kernel args.
- The 2-stage vs unified extend dispatch in `triton_backend` (`enable_deterministic` branch,
  lines ~1940–1953) must both carry score_mod correctly — test both.
- Multi-step draft / DSpark verify (T7) re-enters this path with `KVWriteLoc` window rails —
  the window-rail assert at `triton_backend.py:~2540` is strict; T7 depends on this being
  solid.
- FA4-only perf optimizations (sheared bias, head padding) are lost on SM86; attention is
  not the throughput bottleneck at batch 1 (decode is memory-bound on expert weights), so
  this is acceptable.

### Addendum (sauron bring-up, sconv PDL failure)

The first sauron run failed JIT-compiling `inkling2_sconv` for sm_86: ptxas rejected
`griddepcontrol.wait` / `griddepcontrol.launch_dependents` ("requires .target sm_90 or
higher"). The six inkling2 csrc files with raw PDL asm — `sconv.cuh`, `prologue.cuh`,
`routed_quant.cuh`, `route.cuh`, `routed_rows.cuh`, `gate_sink.cuh` — bypassed the
repo's own guarded helpers and emitted the instructions unconditionally. Fixed by
replacing every pair with the idiom used everywhere else in the tree
(`sgl_kernel/utils.cuh`): `PDLWaitPrimary<true>()` / `PDLTriggerSecondary<true>()`,
which compile to the same asm on sm_90+ and to nothing on sm_8x
(`#if SGL_ARCH_HOPPER_OR_GREATER` + `if constexpr`). Hopper behavior is bit-identical;
SM8x just loses the PDL fast path, as intended. Full scan for other sm90-only PTX
(TMA, mbarrier, clusters, DSMEM) in `csrc/inkling2/`: clean — the only remaining raw asm
is `mma.sync.aligned.m16n8k16` (sm_75+, safe).
