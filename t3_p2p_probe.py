#!/usr/bin/env python3
"""T3 probe: P2P/PCIe collectives feasibility on 2x RTX 3090 (patched P2P drivers).

Usage (on the GPU box, single command — spawns 2 ranks internally):

    python t3_p2p_probe.py            # uses GPU 0 and 1
    CUDA_VISIBLE_DEVICES=0,1 python t3_p2p_probe.py

Answers the T3 ticket questions:
  1. Driver-level P2P: can the GPUs see each other, and does peer access enable?
  2. Raw D2D copy bandwidth (with P2P: ~25-28 GB/s on PCIe Gen4 x16).
  3. torch symmetric memory: empty + rendezvous (requires P2P) + multicast_ptr
     (expect 0 / unsupported on PCIe — that is the NVLink-only feature).
  4. P2P-read one-shot all-reduce (the exact pattern inkling2_comm will use as
     the non-multimem fallback): correctness + latency at the per-token payload
     size (2304 bf16 = 4.6 KB, the hidden-size AR Inkling-2 issues 96x/token).
  5. Whether torch's multimem (multicast) all-reduce works (expect failure).
  6. Large-message P2P read bandwidth (the 256 KB+ chunked AR case).

Verdict (printed by rank 0):
  MODE A = P2P works, multicast absent  -> only multicast_all_reduce needs a
           P2P-read fallback (T3 = ~2 days)
  MODE B = P2P absent/broken            -> full NCCL fallback (T3 = ~1 wk)
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

HIDDEN = 2304  # Inkling-2-Mini hidden size; 2304 * 2 B = 4608 B per AR payload
WORLD = 2


# ---------------------------------------------------------------------------
# P2P-read one-shot all-reduce kernel (the fallback inkling2_comm will need;
# same shape as sglang's inkling2_comm._rank_sum, but the peers are read
# through the symmetric-memory pointer map instead of a local all_gather).
# ---------------------------------------------------------------------------
@triton.jit
def _p2p_rank_sum(PTRS, Y, N: tl.constexpr, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0.0, tl.float32)
    for rank in tl.static_range(WORLD):
        ptr = tl.load(PTRS + rank).to(tl.pointer_type(tl.bfloat16))
        value = tl.load(ptr + indices, indices < N, other=0).to(tl.float32)
        total += value
    tl.store(Y + indices, total.to(tl.bfloat16), indices < N)


def p2p_oneshot_ar(ptrs: torch.Tensor, out: torch.Tensor, n: int, grid_block: int = 256):
    _p2p_rank_sum[(triton.cdiv(n, grid_block),)](ptrs, out, n, WORLD, grid_block)


# ---------------------------------------------------------------------------
# Driver-level P2P via ctypes -> libcuda (no extra deps beyond the driver).
# ---------------------------------------------------------------------------
def driver_p2p_check(rank: int):
    libcuda = None
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        try:
            libcuda = ctypes.CDLL("libcuda.so")
        except OSError:
            return {"error": "libcuda not found"}

    results = {}
    for src, dst in [(0, 1), (1, 0)]:
        can = ctypes.c_int(0)
        err = libcuda.cuDeviceCanAccessPeer(ctypes.c_int(src), ctypes.c_int(dst), ctypes.byref(can))
        results[f"canAccessPeer({src}->{dst})"] = {
            "err": err,
            "can": bool(can.value),
        }
    peer = 1 - rank
    # cuDeviceEnablePeerAccess was removed in CUDA 12+ (peer access is now
    # implicit); only probe for the symbol on older toolkits.
    enable_fn = getattr(libcuda, "cuDeviceEnablePeerAccess", None)
    if enable_fn is None:
        results["enablePeerAccess"] = "n/a (implicit since CUDA 12 — real test is sections 2/4/6)"
    else:
        err = enable_fn(ctypes.c_int(peer), ctypes.c_int(0))
        # 0 = success; 704 (AlreadyEnabled) is fine; 219 = NotSupported
        results[f"enablePeerAccess(current={rank} -> peer={peer})"] = err
    return results


# ---------------------------------------------------------------------------
def section(rank, title):
    print(f"[rank {rank}] === {title} " + "=" * 50, flush=True)


def bench_latency(fn, iters: int, warmup: int = 50) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    us = 1e6 * elapsed / iters
    return us, us  # mean == median for this flat kernel; single number


def worker(rank: int, world: int, init_file: str):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=world)

    # ------------------------------------------------------------------ 1
    section(rank, "1. driver-level P2P (ctypes -> libcuda)")
    res = driver_p2p_check(rank)
    print(f"[rank {rank}] {res}", flush=True)
    print(
        f"[rank {rank}] NOTE: on CUDA>=13 the driver API cuDeviceCanAccessPeer is deprecated "
        f"and may report err=1/101 even when P2P works; torch's check is authoritative:", flush=True
    )
    torch_cap = [
        (a, b, bool(torch.cuda.can_device_access_peer(a, b)))
        for a, b in [(0, 1), (1, 0)]
    ]
    print(f"[rank {rank}] torch.cuda.can_device_access_peer: {torch_cap}", flush=True)

    p2p_ok = all(c for _, _, c in torch_cap)


    # ------------------------------------------------------------------ 2
    section(rank, "2. raw D2D copy bandwidth, 512 MB (needs P2P to be fast)")
    n = 512 * 1024 * 1024 // 2  # bf16 elements
    src = torch.randn(n, device=device)
    dst = torch.empty(n, device=f"cuda:{1 - rank}")
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    gbps = (n * 2 * iters) / (time.perf_counter() - t0) / 1e9
    print(f"[rank {rank}] D2D {src.device.index}->{dst.device.index}: {gbps:.1f} GB/s", flush=True)
    del src, dst
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ 3
    section(rank, "3. torch symmetric memory: empty + rendezvous")
    pg = dist.group.WORLD
    group_name = pg.group_name
    buffer = symm_mem.empty(HIDDEN, device=device, dtype=torch.bfloat16)
    try:
        handle = symm_mem.rendezvous(buffer, group_name)
    except Exception as e:  # noqa: BLE001
        print(f"[rank {rank}] rendezvous FAILED: {type(e).__name__}: {e}", flush=True)
        print("[rank 0] VERDICT: MODE B (symm_mem rendezvous failed — no working P2P path)", flush=True)
        dist.destroy_process_group()
        return
    ptrs = torch.tensor(handle.buffer_ptrs, device=device, dtype=torch.int64)
    mc = getattr(handle, "multicast_ptr", 0)
    try:
        handle.barrier()
        barrier_ok = True
    except Exception as e:  # noqa: BLE001
        barrier_ok = False
        print(f"[rank {rank}] symm_mem barrier FAILED: {e}", flush=True)
    print(f"[rank {rank}] rendezvous OK; buffer_ptrs={[hex(p) for p in handle.buffer_ptrs]}", flush=True)
    print(f"[rank {rank}] multicast_ptr={mc!r} (0/None => NVLink multicast UNAVAILABLE, as expected on PCIe)", flush=True)
    print(f"[rank {rank}] symm_mem barrier ok: {barrier_ok}", flush=True)
    multicast_ok = bool(mc)

    # ------------------------------------------------------------------ 4
    section(rank, "4. P2P-read one-shot all-reduce (4.6 KB payload — the per-token AR)")
    buffer.fill_(torch.tensor(float(rank + 1), dtype=torch.bfloat16, device=device))
    try:
        handle.barrier()
    except Exception:
        pass
    out = torch.empty_like(buffer)
    p2p_oneshot_ar(ptrs, out, HIDDEN)
    torch.cuda.synchronize()
    expected = float(world + 1)  # buffers hold rank+1: 1.0 + 2.0 = 3.0
    got = float(out.cpu().mean().item())
    correct = abs(got - expected) < 1e-3
    print(f"[rank {rank}] correctness: sum of ranks' buffers = {got:.3f} (expect {expected:.1f}) -> {'OK' if correct else 'FAIL'}", flush=True)

    iters = 1000
    us, _ = bench_latency(lambda: p2p_oneshot_ar(ptrs, out, HIDDEN), iters)
    per_token_96 = 96 * us / 1e3
    print(f"[rank {rank}] latency (2-rank, 4.6 KB, P2P-read one-shot AR): {us:.1f} us/AR ; 96 AR/token => {per_token_96:.3f} ms/token if un-overlapped", flush=True)
    ar_ok = correct

    # ------------------------------------------------------------------ 5
    section(rank, "5. torch multimem (NVLink multicast) all-reduce — expect failure on PCIe")
    try:
        buffer.fill_(torch.tensor(1.0, dtype=torch.bfloat16, device=device))
        torch.ops.symm_mem.multimem_all_reduce_(buffer.view(-1), "sum", group_name)
        torch.cuda.synchronize()
        multimem_works = True
        print(f"[rank {rank}] multimem_all_reduce_ SUCCEEDED (unexpected on PCIe!)", flush=True)
    except Exception as e:  # noqa: BLE001
        multimem_works = False
        msg = str(e).splitlines()[0][:160]
        print(f"[rank {rank}] multimem_all_reduce_ failed as expected: {type(e).__name__}: {msg}", flush=True)

    # ------------------------------------------------------------------ 6
    section(rank, "6. large-message P2P read bandwidth (32 MB symmetric buffer)")
    big = 32 * 1024 * 1024  # bf16 elements
    big_buf = symm_mem.empty(big, device=device, dtype=torch.bfloat16)
    big_handle = symm_mem.rendezvous(big_buf, group_name)
    big_ptrs = torch.tensor(big_handle.buffer_ptrs, device=device, dtype=torch.int64)
    big_out = torch.empty(big, dtype=torch.bfloat16, device=device)
    big_buf.fill_(1.0)
    big_handle.barrier()
    for _ in range(3):
        p2p_oneshot_ar(big_ptrs, big_out, big)
    torch.cuda.synchronize()
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        p2p_oneshot_ar(big_ptrs, big_out, big)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    # each AR reads (world-1) * 32 MB over the link; effective link bytes = (world-1)*size
    gbps = ((world - 1) * big * 2 * iters) / elapsed / 1e9
    print(f"[rank {rank}] 32 MB P2P-read AR: {1e3 * elapsed / iters:.1f} ms/AR, ~{gbps:.1f} GB/s effective peer-read bandwidth", flush=True)
    large_ok = True
    del big_buf, big_handle, big_ptrs, big_out
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ 7
    if rank == 0:
        print()
        print("=" * 70)
        print("VERDICT")
        print("=" * 70)
        p2p_all = bool(p2p_ok and ar_ok)
        if p2p_all:
            print("MODE A — P2P works (torch-visible + P2P-read one-shot AR correct).")
            if not multicast_ok or not multimem_works:
                print("  NVLink multicast absent (expected): `multicast_all_reduce` needs the")
                print("  P2P-read fallback. T3 estimate ~2 days.")
            else:
                print("  SURPRISE: multimem also works — T3 shrinks to wiring only.")
            if gbps < 5:
                print(f"  WARNING: large P2P transfers are slow (~{gbps:.1f} GB/s): use NCCL for")
                print("  256 KB+ ARs and the lm_head reduce_scatter, P2P-read one-shot only for")
                print("  decode-sized payloads.")
        else:
            print("MODE B — P2P path unavailable. Full NCCL fallback required (T3 ~1 wk).")
        print(f"  p2p_torch_ok={bool(p2p_ok)}  symm_mem_ok={bool(barrier_ok)}")
        print(f"  oneshot_ar_ok={bool(ar_ok)}  multicast_ptr={bool(multicast_ok)}  multimem_works={multimem_works}")
        print(f"  latency per 4.6 KB AR: {us:.1f} us -> 96/token = {per_token_96:.3f} ms un-overlapped")
        print(f"  large (32 MB) P2P read: ~{gbps:.1f} GB/s")
    dist.destroy_process_group()


def main():
    print("=== T3 P2P probe: 2x RTX 3090, patched drivers ===", flush=True)
    try:
        topo = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=30
        )
        print(topo.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(drv.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    print(f"torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__}", flush=True)

    fd, init_file = tempfile.mkstemp(prefix="t3probe_")
    os.close(fd)
    # file:// init uses FileStore, which creates a single lock file at this path.
    torch.multiprocessing.spawn(
        worker,
        args=(WORLD, f"file://{init_file}"),
        nprocs=WORLD,
        start_method="spawn",
    )
    try:
        os.unlink(init_file)
    except OSError:
        pass


if __name__ == "__main__":
    main()
