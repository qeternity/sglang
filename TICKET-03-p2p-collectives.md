
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
