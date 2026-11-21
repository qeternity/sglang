# TICKET-05 — Ampere recipe/flags + model wiring for 2×3090 text-only serving

- **Branch:** `zuma-inkling`
- **Hardware target:** 2× RTX 3090 (SM86, 48 GB total)
- **Dependencies:** TICKET-01, TICKET-02, TICKET-03, TICKET-04
- **Estimate:** 2–3 days
- **Status:** pending

## Scope

Wire the server so that a single documented command serves `thinkingmachines/Inkling-2-Mini`
on 2×3090 with correct defaults, and document the scope limitations.

## 1. Reference command (mirrors the H200 cookbook recipe, minus Blackwell bits)

```bash
sglang serve \
  --model-path thinkingmachines/Inkling-2-Mini \
  --host 127.0.0.1 --port 8000 \
  --tp-size 2 \
  --page-size 128 \
  --attention-backend triton \
  --kv-cache-dtype bfloat16 \
  --grammar-backend xgrammar \
  --enable-metrics \
  --mamba-full-memory-ratio 0.8 \
  --bf16-gemm-backend torch \
  --disable-cuda-graph
```

Differences from the H200 recipe and why:
- `--attention-backend triton` (recipe says `fa4`) — T1 delivers the triton path; FA4 has no
  SM86 build.
- `--kv-cache-dtype bfloat16` — the H200 recipe uses the model's MXFP8-KV path (Blackwell FP8
  tensor cores); 3090 has no FP8 → bf16 KV pool.
- `--disable-cuda-graph` — kept from the H200 recipe; SM86 bring-up starts without graph
  capture (symm-mem warmup + triton-path capture is a follow-up perf item).
- No `--mm-process-config` / no vision — **text-only** (see §4).
- `--mamba-full-memory-ratio 0.8` — same budgeting knob (the model's state pools are allocated
  through the hybrid pool machinery); verify the 48-visit loop state (per-pass vs shared KV,
  sconv state) fits: expect ~16–19 GB/GPU total at 128 k ctx (weights 10.7 GB + KV ≤ 3 GB +
  runtime ~2–3 GB).

## 2. Wiring work

1. **Backend auto-selection:** a helper (in `Inkling2Attention` or the server-args validation)
   that picks `triton` when `serving` an Inkling2 model on capability major == 8, and rejects
   `--attention-backend fa4` on SM86 with a clear error (better than the FA4 build failure).
2. **TP2/EP2 verification:** `Inkling2QuantConfig` enforces `EP == TP` (NVFP4 activation scale
   spans the full expert input width — NVFP4 experts are not TP-shardable). With 128 experts
   and TP2: 64 experts/rank, `num_experts % ep_size == 0` check passes. Add an explicit
   server-start log line: "Inkling2 TP2/EP2: 64 experts per rank, NVFP4 layers 0–19, MXFP8
   layers 20–23".
3. **Loop flags:** `--inkling2-loop-share-cache` (default on per the recipe; per-pass pools
   triple the core-layer KV state), `--inkling2-loop-num-passes 3` (from `loop_config`).
   Verify the KV-pool sizing formula at 1 M max context vs the memory ratio (the pool is
   allocated for `max_num_seqs × context`, not for 1 M — confirm `--context-length`/max
   request length default to something sane for a 48 GB box, e.g. 32 k–128 k).
4. **Collectives mode log** (from T3): startup line reporting P2P/symm-mem mode and the
   MoE-combine implementation in use.
5. **xgrammar + `--bf16-gemm-backend torch`:** carry through from the H200 recipe (grammar
   backend is arch-generic; bf16 small-batch GEMV backend `torch` avoids an unvalidated
   sgl-kernel path on SM86 for bring-up — re-evaluate later).

## 3. Memory budget (to be re-measured in T6)

| Item | Per GPU (TP2/EP2) |
|---|---|
| Expert weights (NVFP4 5.7 GB + MXFP8 2.1 GB) | ~7.8 GB |
| Attention/embed/MTP/sconv bf16 (TP-sharded) | ~2.9 GB |
| KV pool (128 k: full layers 4 × 128k × 4 KB + SWA rings) | ≤ ~3 GB (shared) / up to ~6 GB (per-pass) |
| MTP/DSpark state, CUDA graphs (off), NCCL/symm buffers, activations | ~2–3 GB |
| **Total** | **~16–19 GB < 24 GB** |

## 4. Scope documentation

- **Text-only serving:** the vision encoder is hard-gated to Hopper/Blackwell by design
  (`models/inkling2_vision.py`: "Inkling2 vision attention requires Hopper or Blackwell" —
  the cutedsl `flash_attn_varlen_func` path). Document in the recipe that images/vision are
  unavailable on Ampere; `Inkling2LoopForConditionalGeneration` multimodal input should
  degrade to text-only with a clear warning, not a crash.
- **DSpark/MTP:** disabled for the initial SM86 ship (T7 is the stretch ticket); the server
  must start and serve correctly with speculative decoding off.
- **No new cookbook hardware id yet:** `docs/src/snippets/configs/thinkingmachines/inkling-2-mini.jsx`
  lists `h200, gb300, rtx6000, rtx5090, dgx-spark, mi350x, mi355x, tpu7x`. Adding an
  `rtx3090` id is a follow-up (separate docs PR, after T6 proves the box).

## Acceptance criteria

- One command (above) serves Inkling-2-Mini on 2×3090; `/v1/chat/completions` returns a
  coherent completion.
- Startup log contains: backend selection, TP/EP + per-rank expert count, quant layer
  assignment, collectives mode, KV pool size.
- `--attention-backend fa4` on SM86 fails fast with an actionable error message.
- Memory watermark < 21 GB/GPU at 128 k ctx with share-cache on (leaves headroom for
  per-pass mode).

## Risks

- Low. This is glue + validation; the real work is in T1–T4. Main risk is a server-args
  validation that currently *requires* fa4 at a level above the model (checked in T1; if the
  validation is in `server_args.py`/registry, fix there instead).
