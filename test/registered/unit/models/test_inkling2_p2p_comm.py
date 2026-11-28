"""TICKET-03: multicast-less (PCIe P2P) dispatch in inkling2_comm.

The 3090 box (sauron) maps peers but has no NVLink multicast, so the
collectives must dispatch to the P2P read path at or below P2P_MAX_BYTES and
NCCL above it, instead of the multimem kernels that hard-raise without a
multicast pointer. These tests exercise that dispatch with a fake group and
faked symmetric state (no CUDA). The all_reduce tail is the same 4-line
branch but sits behind its is_cuda gate, so it is covered by the sauron probe
(t3_p2p_validate.py) with real CUDA tensors.

Run: .venv/bin/python -m pytest test/registered/unit/models/test_inkling2_p2p_comm.py -v
CI:   --ci-test-cpu
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.communication import inkling2_comm as comm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _no_capture(monkeypatch):
    # CPU build: torch.cuda.is_current_stream_capturing raises; the production
    # guard is CUDA-only. Fake it as False (not capturing).
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: False, raising=False
    )


def make_group(world=2):
    calls = []

    def all_reduce(x):
        calls.append(("nccl_all_reduce", x.numel(), x.dtype))
        return torch.full_like(x, 1234.0)  # stands in for the summed tensor

    return (
        SimpleNamespace(
            world_size=world,
            device_group=SimpleNamespace(group_name=f"tp{world}"),
            torch_symm_mem_comm=None,
            all_reduce=all_reduce,
        ),
        calls,
    )


class _FakeState:
    def __init__(self, capacity):
        self.capacity = capacity
        self.reductions = []

    def input(self, numel):
        return torch.full((numel,), 0.5, dtype=torch.bfloat16)

    def reduce(self, x):
        self.reductions.append(x.numel())
        return torch.full_like(x, 1024.0)


def _install(monkeypatch, multicast):
    states = {}

    def fake_p2p_state(group, numel, dtype, device):
        capacity = 1 << (numel - 1).bit_length()
        key = (group.device_group.group_name, dtype, capacity)
        if key not in states:
            states[key] = _FakeState(capacity)
        return states[key]

    monkeypatch.setattr(comm, "has_multicast", lambda group: multicast)
    monkeypatch.setattr(comm, "_p2p_state", fake_p2p_state)
    monkeypatch.setattr(comm, "_device", lambda: torch.device("cpu"))
    return states


def test_module_and_symbols():
    assert comm.P2P_MAX_BYTES == 256 * 1024
    assert comm._p2p_rank_sum is not None
    assert comm._P2PAllReduce is not None
    assert comm.has_multicast is not None


def test_multicast_all_reduce_input_dispatch(monkeypatch):
    _install(monkeypatch, multicast=False)
    group, _ = make_group()

    # Decode-sized (1 x 2304 bf16 = 4.6 KB) -> P2P region view.
    view = comm.multicast_all_reduce_input(group, (1, 2304), torch.bfloat16)
    assert view is not None and view.shape == (1, 2304)

    # Above the P2P band -> None, caller falls back to NCCL.
    big_shape = (512, 2304)  # fp32: 4.7 MB > 256 KB
    assert comm.multicast_all_reduce_input(group, big_shape, torch.float32) is None


def test_multicast_all_reduce_input_boundary(monkeypatch):
    _install(monkeypatch, multicast=False)
    group, _ = make_group()
    # bf16: 256 KB / 2 bytes = 131072 elements is exactly at the limit.
    at = comm.multicast_all_reduce_input(group, (64, 2048), torch.bfloat16)
    assert at is not None
    over = comm.multicast_all_reduce_input(
        group, (64, 2049), torch.bfloat16
    )  # 263136 B > limit
    assert over is None


def test_multicast_all_reduce_dispatch(monkeypatch):
    states = _install(monkeypatch, multicast=False)
    group, calls = make_group()
    x = torch.full((1, 2304), 1.5, dtype=torch.bfloat16)

    out = comm.multicast_all_reduce(x, group)
    assert out.shape == x.shape
    key = (f"tp{group.world_size}", x.dtype, 4096)
    assert key in states and states[key].reductions == [2304]
    assert calls == []
    assert out.flatten()[0].item() == 1024.0

    big = torch.full((512, 2304), 1.0, dtype=torch.float32)
    out = comm.multicast_all_reduce(big, group)
    assert out.flatten()[0].item() == 1234.0
    assert len(calls) == 1


def test_multicast_all_reduce_reuses_state_across_calls(monkeypatch):
    states = _install(monkeypatch, multicast=False)
    group, _ = make_group()
    for _ in range(5):
        x = torch.full((1, 2304), 1.5, dtype=torch.bfloat16)
        comm.multicast_all_reduce(x, group)
    key = (f"tp{group.world_size}", torch.bfloat16, 4096)
    assert key in states and len(states[key].reductions) == 5


def test_world_one_passthrough(monkeypatch):
    _install(monkeypatch, multicast=False)
    group, _ = make_group(world=1)
    x = torch.full((1, 2304), 1.5, dtype=torch.bfloat16)
    assert comm.multicast_all_reduce(x, group) is x
    view = comm.multicast_all_reduce_input(group, (1, 2304), torch.bfloat16)
    assert view.shape == (1, 2304)


def test_all_reduce_world_one_passthrough():
    group, calls = make_group(world=1)
    x = torch.full((1, 2304), 1.5, dtype=torch.bfloat16)
    assert comm.all_reduce(x, group) is x
    assert calls == []


def test_all_reduce_rejects_non_cuda_and_bad_dtype():
    # The is_cuda gate (pre-existing) fires first on CPU tensors with the
    # collective contract message; CUDA-tensor cases are covered on sauron.
    group, _ = make_group()
    x = torch.full((1, 2304), 1.5, dtype=torch.float64)
    with pytest.raises(ValueError, match="BF16 or FP32"):
        comm.all_reduce(x, group)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
