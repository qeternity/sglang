# TICKET-05 — Ampere recipe/flags + model wiring for 2×3090 text-only serving

- **Branch:** `zuma-inkling`
- **Hardware target:** 2× RTX 3090 (SM86, 48 GB total)
- **Dependencies:** TICKET-01, TICKET-02, TICKET-03, TICKET-04
- **Estimate:** 2–3 days
- **Status:** implemented locally 2026-11-21 (SM8x fa4 fail-fast, startup + collectives-mode logs; full-server validation pending on sauron with T6)

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
## Implementation log (2026-11-21, local; full-server run pending on sauron)

Three small code changes; everything else in this ticket is recipe/T6 work.

1. **FA4 on SM8x fails fast** — `Inkling2Attention.__init__` (covers both
   the loop and non-loop models) now checks, only on `is_sm80` devices
   (the T2 platform probe — the check is skipped entirely on every other
   arch, so zero cost/no new runtime-context reads on Hopper/Blackwell):
   the configured `(prefill, decode)` backend pair from
   `runtime_context.attention_backends()`; if either is `fa4`, raise
   `RuntimeError` naming the fix (`--attention-backend triton` +
   `--kv-cache-dtype bfloat16`). There is **no server-args-level fa4
   requirement** in the repo (verified: the only fa4 checks are the
   model-level asserts, which T1 already extended to accept `triton`),
   so this model-level guard is the only place needed. The ticket's
   "auto-select triton" step was deliberately *not* implemented: silently
   swapping a user-specified backend hides configuration; the acceptance
   criteria ask for a fail-fast, which is what ships.
2. **Startup summary line** — one `logger.info` in
   `Inkling2CausalLLM.__init__` (after the layers are built):
   `Inkling2 startup: attn_tp=2 moe_tp=2 moe_ep=2, 24 layers (48 visits)
   128 experts (64 per rank) NVFP4 layers 0-19 MXFP8 layers 20-23`.
   The per-layer precision split is derived from the
   `Inkling2QuantConfig` tensor-name sets (`nvfp4_weights`/
   `mxfp8_weights` → `layers.<i>` ids); the non-loop config (no
   `loop_config`) prints the 48-layer line without visits. Verified with
   the exact Mini shape: line renders as above.
3. **Collectives mode line** (T3 integration) — `inkling2_comm.has_multicast`
   now logs once per group when the multicast probe first resolves:
   `Inkling2 collectives [<group>]: multimem (multicast) one-shot
   all-reduce` on NVLink boxes, or `... P2P-read one-shot all-reduce for
   payloads <= 256 KB, NCCL above that (no multicast on this link)` on
   the 3090. This is the startup line the T3 ticket asked for.

Verified locally: 226 tests pass (T0+T1+T2+T3+T4 suites + overrides),
ruff clean on all touched files (the two pre-existing import-order
findings in the upstream model files are left alone — they predate this
branch's local ruff version and reordering 40 upstream import lines is
diff noise).

Remaining (sauron, with T6): run the reference command from §1, confirm
the three log lines, and the memory watermark. Loop flags
(`--inkling2-loop-num-passes`, `--inkling2-loop-share-cache`) and the
`xgrammar`/`bf16-gemm-backend torch` flags already exist in the arg
surface (verified in `arg_groups/fields/model.py`); no new flags were
added.

Scope decision (ticket §4): image inputs on SM86 keep the existing hard
gate in `inkling2_vision.py` ("Inkling2 vision attention requires Hopper
or Blackwell"). Degrading multimodal prompts to text by dropping the
vision embeddings would silently corrupt the output (the prompt's
placeholder tokens would be answered without their content), which is
worse than an explicit error at encode time. The recipe therefore
documents text-only scope; no image-dropping filter was added.

### Addendum (sauron bring-up, pool rebalance for the 2x3090 recipe)

First sustained serving on sauron shows the recipe's
`--mamba-full-memory-ratio 0.8` over-provisions the mamba/loop-state pool
(~95% idle at 24 running requests; ~1,900 slot capacity vs ~4 slots/request
needed) while the sliding-window KV pool binds hard: the window pool holds
only ~17.9k window-token slots (~17-18 concurrent full-1024 windows; it is
byte-balanced with the full pool but ~10x costlier per token), so any queue
deeper than ~17 requests triggers the scheduler's "KV cache pool is full.
Retract requests." line (a generic message — the binding pool is the window
one, at 96-99% usage, not the full pool at 12%). Retracting under a deep
queue is correct scheduler behavior (T6 gsm8k-style backlogs hit it), but
for interactive serving the fix is to pin the state pool and let the rest
go to KV: add `--max-mamba-cache-size 128` (4 slots x 32 target concurrency)
to the recipe, which grows the window pool ~1.7x (~31k tokens, ~30 full
windows). The T6 perf scenarios (concurrency 1 and 8) never bind the pool.

## Addendum: prefill CUDA graphs are FA4-only — auto-disabled on triton/SM8x

Sauron (triton backend, no --disable-cuda-graph) crashed at startup in
`capture_prefill_graph`: `triton_backend._apply_cuda_graph_metadata` accepts
decode/verify forward modes only and raises "Invalid forward mode:
ForwardMode.EXTEND ... for CUDA Graph replay" for prefill shapes. H200 never
hits this because FA4 implements extend-mode graph replay.

Fix in `cuda_graph_hook.apply_inkling_prefill_cuda_graph_default`: Inkling
used to unconditionally opt into full-graph prefill capture; it now only
does so when the prefill attention backend is not triton and the platform
is not SM8x (every SM8x box is forced to triton by the T5 fail-fast guard).
On triton/SM8x the generic cascade keeps prefill graphs disabled (prefill
runs through the EagerRunner), while DECODE graphs still capture — the
triton backend supports decode graph replay, and the in-graph tensor-parallel
ARs take the capture-time NCCL fallback added in the T3 addendum-6 commit
(the banked P2P path is eager-only). An explicit --cuda-graph-backend-prefill
/ --disable-prefill-cuda-graph still wins (unchanged).

## Addendum 2: diagnosing the async IMA during decode-capture warmups

With prefill graphs disabled, decode capture crashed later at an unrelated
CUDA call (the extend kernel's first module load surfacing a sticky
CUDA_ERROR_ILLEGAL_ADDRESS). That load proves the extend kernel had never
launched in the process, so the faulting kernel executed during the decode
capture's fake-batch warmup forwards (the only executing phase before the
first prefill). The CUDASymmetricMemory ~AllocationRef trace is teardown
noise: the P2P state buffers' destructor rethrowing the sticky error at
interpreter shutdown.

Diagnostic levers added (one pull covers both routes):
- SGLANG_DEBUG_SYNC_INKLING2=1 — torch.cuda.synchronize() after every P2P
  rank-sum and attn_res launch (skipped during capture, where syncs are
  illegal), so an async fault surfaces at the Python launch site that
  caused it.
- Boot with CUDA_LAUNCH_BLOCKING=1 to make every launch synchronous; the
  traceback then names the faulting kernel directly (if capture rejects
  blocking mode, fall back to the sync-probe boot).
- Boot with SGLANG_OPT_INKLING2_P2P_MAX_BYTES=0 to remove the P2P kernels
  from the warmup path entirely; clean run there points at the captured
  decode kernels, not the P2P path.
