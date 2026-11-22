
### Addendum 3 (sauron bring-up, bank offset bug in _p2p_rank_sum)

The corrected t3 probe failed exactly the way the server's gibberish
predicted: first P2P call correct, every later call wrong, deterministic.
Root cause in `reduce()`: `self.in_ptrs + self.bank * self.world *
self.capacity` applied the bank offset to the *pointer values* (an int64
CUDA tensor of peer base addresses) in **bytes**, while the buffer layout
is in **elements**. For bf16 4096-capacity with world 2, bank 1 shifted
each peer address by 8192 bytes instead of 16384, so the sum kernel read
stale bank-0 data (rank 0's slot) instead of the current bank's slots.
Bank 0 (offset 0) was always right, which is why the first call per
(group, dtype, capacity) state passed. Since the P2P AR runs on every
layer (attn output + MoE combine), every *second* all-reduce per payload
class summed garbage, deterministically, in the live server.

Fix: pass the bank offset as an element-unit scalar (`BankOff`) to
`_p2p_rank_sum` and apply it to each loaded peer pointer inside the
kernel. The all_reduce P2P branch (copy-into-region then reduce) was
already correct; the probe comparison bugs fixed in the same round
(float-literal allclose, wrong 2i+1 bank-stress expectation) masked
nothing but would have misdiagnosed correct code.

### Addendum 4 (sauron bring-up, P2P branch broke the in-place contract)

With the bank offset fixed, the probe's only remaining FAIL was section D:
the input view `a` still held its input after
`multicast_all_reduce(a, group)`. The P2P branch returned the summed view
from a separate output region without touching x, while the multimem and
NCCL branches (and the docstring: "All-reduce x in place") write the sum
back into x. Both production call sites (inkling2_moe.py:831, the
multicast branch of all_reduce) use the return value, so the fix is
additive: the P2P branch now copies x into the input bank first (making
the call valid for a fresh tensor as well as the input view), reduces,
and copies the sum back into x. Writing the sum into the input bank slot
is race-safe: the double-buffered bank alternation plus the per-call
barrier guarantee no peer still reads slot k when call k+2 overwrites it.
The probe needed no change — its `a`-after-call check was the correct
contract all along; this run also makes the fp32-state sub-check
independently meaningful.

### Addendum 5 (sauron bring-up, the in-place write-back was wrong — reverted)

Addendum 4's in-place write-back (sum copied back into the input bank slot)
is UNSAFE and was reverted. The write-back executes on this rank's stream
immediately after its own rank-sum, but the peer rank's rank-sum of the SAME
call is still reading this rank's bank slot through the peer pointer; nothing
orders the write-back after that in-flight peer read (the next call's
barrier comes after the write-back). The pre-in-place design was safe
precisely because the bank slot kept holding the input until call k+2, at
which point the double buffer + barrier ordering proved no peer could still
be reading it. Observed on sauron: with contended GPUs (a hung probe
instance running in the background stretching calls ~50x) the first P2P sum
came back wrong and section D failed; on idle GPUs the same window wedged
the 30-call bank-rotation loop instead of just corrupting data.

Resolution: the P2P path keeps the return-value contract — x is copied into
the input bank first (which also fixes the latent fresh-tensor bug where a
non-view x would never have been reduced), the sum is returned in the
separate out view, and x is left untouched. The docstring now states the
per-path contract instead of claiming in-place everywhere. The probe was
wrong to test x in place on the P2P path; it now checks the return values
(outa/outb). Both production call sites already used the return value, so
nothing else changed. True in-place on the P2P path would require an extra
synchronization round trip and is not worth ~1 us per call.

### Addendum 6 (sauron bring-up, capture-time NCCL fallback — CUDA graphs at TP2)

The official H200 recipe (TP4/EP4, no --disable-cuda-graph) proved the
earlier claim "Inkling2 is graph-less by design on Hopper" wrong: on NVLink
boxes the decode graph captures the STATELESS path — oneshot_all_reduce_input
returns None during stream capture when its state isn't created yet (and the
oneshot is TP4/8-only by `world_size in (4, 8)`), so the captured decode
graph contains the fixed-buffer multimem/NCCL all-reduce (stateless,
capture-safe), while the banked/rotating fast paths (oneshot, and my P2P)
serve the eager path (prefill, uncaptured batch sizes).

The P2P port now mirrors that exactly: all three P2P entry points
(multicast_all_reduce_input -> None, multicast_all_reduce, all_reduce) skip
the banked P2P path while torch.cuda.is_current_stream_capturing() and take
the NCCL branch instead. The NCCL fallback is the same code the P2P_MAX_BYTES
= 0 diagnostic exercised on sauron (server worked end-to-end with all ARs
on NCCL), so the captured graph runs a proven path. P2P stays the eager fast
path (50 us vs ~2x for NCCL at 4.6 KB). With --disable-cuda-graph dropped
from the recipe, decode should capture and the launch-bound 15 tok/s should
move toward the 60-130 tok/s the box's HBM + in-graph AR budget allows.
