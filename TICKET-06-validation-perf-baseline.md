# TICKET-06 — Validation harness + perf baseline on 2×3090

- **Branch:** `zuma-inkling`
- **Hardware target:** 2× RTX 3090 (SM86) — sauron is the **only** GPU hardware the operator has; there is no reference box
- **Dependencies:** TICKET-05
- **Estimate:** ~1 wk
- **Status:** pending
- **Note:** per the operator constraint, no tests were run during the investigation phase;
  this ticket is the **planned** validation harness for the post-implementation phase.

## Purpose

Establish the SM86 performance baseline and prove correctness **without a reference box** —
the operator has no SM90/SM100 hardware, so the external anchor is the accuracy the PR #51926
authors published (gsm8k 94.01% GB300 / 93.93% H200, 1319 samples, greedy), and the internal
anchor is self-consistency on sauron (TP1-vs-TP2, determinism). Everything below runs on
sauron only; it is the acceptance gate for the whole effort.

## 1. Correctness (gate; must pass before any perf claim)

1. **gsm8k accuracy vs the published numbers (external anchor; runs first):**
   sgl-eval 0.1.3, 200 samples, greedy, max_tokens 8192, TP2/EP2 on sauron —
   **target ≈94% (the PR's GB300/H200 runs)**: ≥92% pass, <90% = numerical defect in a
   kernel path (AR, quant, routing) → localize with the KL-consistency approach before any
   perf work. This is the stand-in for the missing reference box: an 8-point accuracy gap
   would be visible in any single kernel bug of note.
2. **TP1 vs TP2 self-consistency (if TP1 fits 24 GB — 21.5 GB weights + KV is tight; the PR
   notes quantized experts can be served on a single GPU):** 2048-token greedy generation,
   same prompt/seed, token-for-token match; top-k logit diff at token 1 < 1e-2 (bf16 noise),
   flag > 1e-1 (usually an AR/quant path difference). If TP1 doesn't fit, this gate becomes
   TP2 vs a re-run (see 3) plus gate 1 carries the weight.
3. **Determinism:** repeat the same 2048-token generation on TP2; token-for-token match
   across runs (routing, AR reduction order, and CUDA graphs must all be deterministic at
   these settings).
4. **Loop-state semantics (the highest-risk area from T0):**
   - **Prefix-cache hit:** send a shared 4 k-token prefix twice (two requests); the second
     request must produce identical tokens to a cold start and hit the radix cache
     (check `cache_hit_rate` metric) — loop-state reuse through the radix cache.
   - `--inkling2-loop-share-cache` on/off equivalence: **N/A for Mini** (`is_varloop: false`
     → the flag is rejected by `configure_loop`); the gate is retained for a future
     varloop checkpoint.
5. **SWA window behavior (qualitative, no reference needed):** needle test — distinctive
   fact at position p, query at p+2048 (beyond the 1024 local window: answer must come from
   the full-attention layers) and at p+512 (in-window); both must be answered correctly.
## 2. Performance baseline

| Scenario | Target | Notes |
|---|---|---|
| Decode, batch 1 | **~150–250 tok/s per stream** | memory-bound: ~4 B active params/token, ~4.2–4.5 GB weight traffic/token split over 2 GPUs at ~850 GB/s effective → ~3 ms/token hardware floor |
| Decode, batch 4/8 | report | expect graceful degradation (batched expert GEMMs become compute-bound on 3090) |
| Prefill (8 k prompt) | **~2–5k tok/s aggregate** | ~70 TFLOPS bf16 dense/GPU (no FP8/FP4 tensor cores); experts run bf16 via Marlin |
| Mixed (32 k ctx, 8 concurrent) | report | the real serving scenario for a 48 GB box |

Overhead breakdown to report per scenario:
- attention (triton) share of decode step,
- expert GEMM (Marlin) share,
- **AR overhead per token** (T3 overlap enabled — expect < 1 ms; if > 2 ms, the overlap is
  not working),
- sconv/rel-bias/MTP overhead.

## 3. Operational checks

- `nvidia-smi` memory watermark at steady state (both GPUs), 128 k ctx, share-cache on —
  expected < 21 GB/GPU (T5 budget).
- Server restart + prefix-cache persistence across requests (no state leak).
- CUDA-graph off (bring-up mode) — note the perf gap vs graph-on as the follow-up item.
- Driver-patch canary: log the P2P probe result every startup (T3).

## 4. Deliverables

- A `docs/`-style report (not committed to the repo; delivered to the operator): correctness
  table (all 5 gates pass/fail + measured diffs), perf table (all scenarios + overhead
  breakdown), memory watermarks, known issues list, and a go/no-go recommendation for
  enabling DSpark (T7).

## Acceptance criteria

- Correctness gates pass on sauron alone: gsm8k ≥ 92% vs the published ~94%, TP1/TP2 (or re-run) token-identical, determinism clean, prefix-cache hit verified.
- Perf within 30% of the targets above (decode ≥ 100 tok/s bs1 and prefill ≥ 1.5k tok/s
  aggregate are the hard floors; missing them means a kernel path is not actually on the
  hot path — profile before shipping).
- Report delivered.
