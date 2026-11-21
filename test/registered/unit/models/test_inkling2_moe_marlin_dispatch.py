"""Inkling-2 expert dispatch: SM8x (Ampere) must take the Marlin path.

CPU-runnable: builds a real ``Inkling2Experts`` (bf16, unquantized, 4 experts)
with the runtime platform facts overridden and checks ``use_marlin``. The
Marlin path is the SM90 production path (weight-only dequant + BF16 MMA,
kernel floor SM80), so SM86 must route exactly like SM90; SM100/SM103/SM107
stay on the native CUTLASS path.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.inkling_common import inkling2_moe as moe
from sglang.srt.runtime_context import get_parallel, override_platform
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _experts():
    config = SimpleNamespace(
        hidden_size=2048,
        intermediate_size=1152,
        num_experts_per_tok=8,
        route_scale=1.0,
        mlp_gate_softcap=10.0,
        mlp_up_softcap=10.0,
    )
    with set_default_torch_dtype(torch.bfloat16):
        return moe.Inkling2Experts(config, 4, "x.", None)


@pytest.mark.parametrize(
    ("sm80", "sm90", "sm120", "expected"),
    [
        (True, False, False, True),  # SM86 (3090): new Marlin routing
        (False, True, False, True),  # SM90: existing Marlin routing
        (False, False, False, False),  # SM100/103/107: native CUTLASS
        (True, False, True, True),  # is_sm80 short-circuits before SM120
    ],
)
def test_use_marlin_dispatch(sm80, sm90, sm120, expected):
    with (
        override_platform(is_sm80=sm80, is_sm90=sm90, is_sm120=sm120),
        get_parallel().override(
            moe_tp_rank=0, moe_tp_size=1, moe_ep_rank=0, moe_ep_size=1
        ),
    ):
        experts = _experts()
        assert experts.use_marlin is expected
