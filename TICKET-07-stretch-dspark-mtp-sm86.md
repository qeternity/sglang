# TICKET-07 (Stretch) — Enable DSpark/MTP speculative decoding on SM86

- **Branch:** `zuma-inkling`
- **Hardware target:** 2× RTX 3090 (SM86)
- **Dependencies:** TICKET-06 (correctness + baseline must be green first)
- **Estimate:** ~1 wk (high variance — speculative decoding is the least predictable part)
- **Status:** local pre-work done 2026-11-21 (SM-gate absence re-verified; selection arg identified: `--speculative-algorithm dspark`); box work deferred to after T6 is green

## Context

`Inkling-2-Mini`'s `mtp_config`: 3 layers, hidden 2304, intermediate 9216,
`target_layer_ids [5, 23, 41, 48]`, `method: "dspark"`, `share_weights: true` (the MTP head
reuses target-layer state — checkpoint entries `model.layers.{0,1,5,17,23}.input_layernorm.weight`).
MTP head cost: ~0.38 B params bf16 ≈ 0.76 GB (fits easily).

The DSpark machinery in this tree:
- `python/sglang/srt/speculative/inkling2_mtp_worker.py` — `Inkling2MTPWorker(DSparkWorkerV2)`
  + `Inkling2MTPVerifyExecutor(TargetVerifyExecutor)`
- `python/sglang/srt/speculative/dspark_components/*` — V2 worker, verify executor, config
  (`dspark_config.py:244` already maps `Inkling2MMConfig`/`Inkling2MTPConfig`); selection at
  `srt/speculative/spec_info.py:341` (`is_dspark()` + architecture check)
- DSpark semantics (from the model's public description): DSpark = **Markov-RNN drafting +
  per-epoch rejection sampling**; the MTP head drafts multiple epochs per step.

There are **no direct SM gates** in the DSpark components (verified: no `sm90`/`sm100`/
capability checks in `inkling2_mtp_worker.py` or `dspark_components/*`) — the SM86 gaps
inherit from T1 (attention) and T3 (AR overlap). The `SGLANG_OPT_DSPARK_*` flags (e.g.
`SGLANG_OPT_DSPARK_MTP_AR_OVERLAP`) are tuned for Hopper/Blackwell timing.

## Steps

1. Enable MTP on the SM86 server (the speculative args that select `Inkling2MTPWorker`);
   confirm the MTP head loads (~0.76 GB) and the draft/verify path runs on the triton
   attention backend.
2. Verify the **Markov RNN** and per-epoch rejection-sampling triton kernels on SM86 (they are
   generic triton — probe + parity vs SM90).
3. Check the `KVWriteLoc` window-rail path: DSpark multi-step draft writes use the triton
   backend's window rails (`triton_backend.py` ~line 2540 assert is strict about
   `window_locs` shapes on draft steps) — the Inkling1/Inkling2 draft path already supports
   this on triton (`attn_backend_wrapper_for_draft_extend` in `attention_registry.py`
   exists specifically because Inkling's draft has short convs).
4. Retune: draft-step count, `SGLANG_OPT_DSPARK_MTP_AR_OVERLAP` (on — T3 overlap must cover
   the MTP ARs too), and any step-count knobs sensitive to the PCIe AR latency (~20 µs each).
5. Measure accepted-step rate and end-to-end speedup vs the T6 non-speculative baseline.

## Acceptance criteria

- Speculative decoding enabled: **accepted step rate ≥ 2** at batch 1 (otherwise the draft
  overhead exceeds the gain — fall back to non-speculative ship).
- No correctness drift: 2 k-token greedy output identical to the T6 baseline (T6 gate 2)
  with and without speculative decoding.
- End-to-end decode speedup ≥ 1.5× at batch 1 (the 3 B-active model is memory-bound;
  speculation helps most when draft cost < saved prefill-of-drafts — measure, don't assume).

## Risks

- Highest variance of all tickets: speculative decoding interacts with every other component
  (window rails, AR overlap, radix cache, loop state for the MTP feature taps at visits
  [5, 23, 41, 48]).
- If DSpark's tuning (draft depth, rejection sampling thresholds) was fit on Hopper timing,
  the optimum on PCIe-latency ARs may be different — expect an offline tuning pass.
- This is explicitly stretch: the initial ship is non-speculative. A clean "no-go" with
  measured data is an acceptable outcome.
## Pre-work log (2026-11-21, local)

- Re-verified: no SM/arch gates anywhere in
  `speculative/inkling2_mtp_worker.py`, `speculative/dspark_components/*`, or
  the `spec_info.py` selection path (grep for sm80/90/100/120, capability,
  get_device_capability — zero hits). The SM86 gaps are exactly the T1/T3
  inheritance stated in the ticket, nothing more.
- Selection: `--speculative-algorithm dspark` (the `speculative_algorithm`
  arg, `SpeculativeAlgorithm.DSPARK`) + an Inkling2 architecture is enough
  for `spec_info.py` to return `Inkling2MTPWorker` (it checks the arch list
  `INKLING2_ARCHS`, so both Mini and Small get the Inkling2 worker; plain
  DSpark models get `DSparkWorkerV2`). Standard speculative args (steps /
  EAGLE-style knobs) + the `SGLANG_OPT_DSPARK_*` envs are the tuning surface.
- The DSpark SPS/STS offline fit tools exist in-tree
  (`python -m sglang.benchmark.dspark_sps_profiler` /
  `dspark_sts_fit`) — the ticket's "expect an offline tuning pass" is
  supported by shipped tooling; the fitted tables feed
  `--speculative-dspark-sps-table-path` /
  `--speculative-dspark-confidence-sts-path`.

Nothing else is doable without the T6 baseline; box work starts there.
