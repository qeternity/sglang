
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
