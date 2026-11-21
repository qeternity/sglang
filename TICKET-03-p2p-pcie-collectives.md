# TICKET-03 — P2P/PCIe collectives: probe symm_mem on the patched driver, add multicast fallback

- **Branch:** `zuma-inkling`
- **Hardware target:** 2× RTX 3090, PCIe x16 Gen4, **no NVLink**; operator runs **patched drivers with P2P over PCIe enabled**
- **Dependencies:** TICKET-00 (needs a loadable model to exercise the full server path; the probe itself can start immediately)
- **Estimate:** 2 days (probe passed → the "P2P works" branch: 1-shot P2P-read AR for decode payloads + NCCL for large messages)
- **Status:** probe done 2026-11-21 (Mode A + NCCL-for-large); implementation pending

## Problem

Inkling2's TP/EP collectives (`python/sglang/kernels/ops/communication/inkling2_comm.py`)
are built on `torch.distributed._symmetric_memory`:

- `_symm` → `symm_mem.empty` + `symm_mem.rendezvous` (requires GPU-to-GPU memory mapping)
- `_ptrs` → `handle.buffer_ptrs` (direct peer load/store in the Triton kernels, e.g.
  `_rank_sum`)
- `_multicast_buffer` → **hard-raises** `"Inkling2 tensor parallelism requires NVLink multicast"`
  when `handle.multicast_ptr` is None (CUDA multicast / `cuMulticast` — **NVLink-only
  hardware**; it cannot exist over PCIe)
- `all_reduce` constraints: world in {2, 4, 8, 16}, bf16/fp32, 4-byte aligned, 16-byte
  alignment for the SM100 one-shot path; `_RANK_ORDER_MAX_BYTES = {2: 1200 KB, 4: 512 KB,
  8: 288 KB, 16: 64 KB}`
- Model usage: `Inkling2Attention.forward` uses `oneshot_all_reduce_input` + `oneshot_all_reduce`
  around `wo_ud`; the MoE combine uses `multicast_all_reduce`; `reduce_scatter_hidden` for
  embeddings/logits (imports at `models/inkling_common/inkling2_attn.py` top).

On stock 3090s **neither P2P nor multicast exists**. On this deployment the operator has
**patched drivers enabling P2P over PCIe**, which changes the picture:

- **P2P (peer memory mapping)** → symmetric memory *may work* over PCIe (slow, but functional).
  This covers `oneshot_all_reduce` (direct peer load/store — no multicast involved).
- **Multicast (multimem)** → still impossible (NVLink hardware feature). `multicast_all_reduce`
  needs a replacement.

## Step 0 — Probe (method; results below)

```bash
# 1. P2P visibility
python - <<'EOF'
import torch
for a, b in [(0,1), (1,0)]:
    print(a, b, torch.cuda.can_device_access_peer(a, b))
EOF

# 2. Symmetric memory round-trip
python - <<'EOF'
import torch, torch.distributed as d
# in a 2-process launcher (torchrun --nproc_per_node=2)
import torch.distributed._symmetric_memory as sm
d.init_process_group("nccl")
buf = sm.empty(1024, device=f"cuda:{d.get_rank()}", dtype=torch.bfloat16)
h = sm.rendezvous(buf, d.group.WORLD.group_name)
print("multicast_ptr:", h.multicast_ptr)   # expect None over PCIe
h.barrier()
# one-shot all_reduce via sglang's own API (import
# sglang.kernels.ops.communication.inkling2_comm as c)
EOF
```

Expected: `can_device_access_peer == True` (patched driver), `multicast_ptr == None`.
## Step 0 — Probe (DONE 2026-11-21, on `sauron`, 2× RTX 3090 Ti, driver 610.57.04, cu13)

Probe script: `t3_p2p_probe.py` (repo root; self-contained, spawns 2 ranks).

| Check | Result |
|---|---|
| `torch.cuda.can_device_access_peer(0,1)/(1,0)` | **True / True** (patched driver works) |
| `cuDeviceCanAccessPeer` via libcuda | err=1/101, can=False — **stale API on CUDA 13, ignore** (torch is authoritative) |
| `symm_mem.empty` + `rendezvous` (2304×bf16) | **OK** on both ranks; `buffer_ptrs` mapped |
| `handle.multicast_ptr` | **0** (absent — NVLink-only, as expected) |
| `torch.ops.symm_mem.multimem_all_reduce_` | **RuntimeError: "multicast support is required"** (as expected — this is what must be replaced) |
| P2P-read one-shot AR, 4.6 KB (the per-token payload) | **correct (sum 3.000), 7.1–8.2 µs/AR** → 96/token ≈ 0.7–0.8 ms un-overlapped |
| Raw D2D copy, 512 MB | **1.4 GB/s** — slow |
| 32 MB P2P-read AR | ~1.1–2.9 GB/s effective, rank-asymmetric |

**Revised verdict: Mode A for the decode path + NCCL for large messages.**

- The patched driver gives **working, torch-visible P2P**: symm_mem rendezvous succeeds and
  direct peer reads through `buffer_ptrs` work at NVLink-class *latency* (8 µs for 4.6 KB).
- But **large P2P transfers are slow (~1.4 GB/s, CPU-bounce-like, rank-asymmetric)** — the
  patched path does not deliver Gen4 DMA throughput. So:
  - decode-sized one-shot ARs (≤ hidden size, 4.6 KB): **P2P-read one-shot** (measured 8 µs);
  - 256 KB chunked ARs + the lm_head `reduce_scatter`: **NCCL** (box's standard path; T6
    measures it; if slow, `NCCL_P2P_DISABLE=1` + shm tuning is the lever).
- `multicast_all_reduce` (multimem) is **replaced** by the size-dispatched path above; the
  `if not handle.multicast_ptr: raise` guard becomes the dispatch point.

**Status: probe done; implementation pending (est. 2 days — the "2–3 day" branch).**

## Design (branch on probe results — resolved: Mode A + NCCL-for-large)
**Mode A — P2P works (expected with the patched driver):**
1. Keep the existing symm-mem one-shot AR path (`oneshot_all_reduce`, `all_reduce` non-
   multicast branch) — it is already P2P-based and arch-generic.
2. Replace **only** `multicast_all_reduce` (multimem) with a P2P non-multimem equivalent:
   either (a) a direct two-rank load/store sum kernel over `buffer_ptrs` (trivial for
   world==2: each rank loads the peer's result and adds), or (b) batched NCCL all-reduce
   using the existing `batch_all_reduce` 256 KB chunking.
3. Auto-detect at server start: probe `multicast_ptr`; select the MoE-combine implementation;
   log the chosen mode.

**Mode B — P2P fails (fallback safety):**
4. Full NCCL/PCIe fallback behind the same Python API surface
   (`all_reduce`, `oneshot_all_reduce_input`, `oneshot_all_reduce`, `multicast_all_reduce`,
   `symm_mem_all_reduce`, `reduce_scatter_hidden`), batching small messages through
   `batch_all_reduce` (256 KB chunks) to keep message count down.

**Overlap (both modes):**
5. The MoE-combine AR (~4.6 KB per token at hidden 2304, bf16) is issued per layer visit
   (48 visits/token). Overlap it behind expert compute using the
   `SGLANG_OPT_DSPARK_MTP_AR_OVERLAP` pattern (separate stream + event), as DSpark already
   does for MTP ARs. Budget: 96 ARs/token × ~20 µs ≈ ~2 ms/token worst case un-overlapped —
   overlap should cut the visible cost to ~0.

## Acceptance criteria

- Probe report attached to this ticket (P2P yes/no, multicast null, measured AR latency).
- TP2/EP2 server runs on 2×3090 with **no NVLink requirement**; startup logs the collective
  mode.
- Numerics identical to the NVLink/symm path (sum is sum — no precision difference expected).
- Measured AR overhead per token at batch 1 with overlap enabled (report, expected < 1 ms).

## Risks

- Patched-driver P2P over PCIe is **slow** (PCIe Gen4 x16 ≈ 25 GB/s/dir vs NVLink 900 GB/s+):
  one-shot AR (2×4.6 KB load/store) is bandwidth-trivial, but large messages (the 256 KB
  chunked ARs) will be slower than the one-shot path — keep per-token critical-path payloads
  at the hidden size (2304×2B = 4.6 KB) and verify `reduce_scatter_hidden` (hidden-level, not
  logit-level) stays within that budget.
- P2P over PCIe under CUDA-graph capture: symm-mem buffers must be warm before capture
  (`_multicast_buffer` already raises during capture with "Warm up the collective size before
  CUDA graph capture") — keep the warmup step.
- Driver patch stability: if the operator's driver breaks a torch update, the whole plan
  silently degrades — the probe at server start is also the canary.
