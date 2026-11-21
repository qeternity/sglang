"""Inkling2 norm/clip SM8x dispatch: on Ampere the model must route the
QK L2 clip and RMSNorm to the triton implementations (CUTE-DSL targets SM90+).

CPU-runnable: the triton modules import without cutlass/cutlass-dsl (checked
in a subprocess), and the dispatch flag is checked with the runtime platform
facts overridden. The triton kernels themselves run on the sauron probe
(``t4_norm_validate.py``) against a pure-torch reference.
"""

import subprocess
import sys

import pytest
from sglang.srt.models.inkling_common.inkling2_attn import QKL2Clip
from sglang.srt.models.inkling_common.inkling2_norm import RMSNorm
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def test_triton_norm_module_imports_without_cutlass():
    code = (
        "import sys; "
        "import sglang.kernels.ops.layernorm.inkling2_norm_triton as m; "
        "assert 'cutlass' not in sys.modules, 'triton fallback must not pull in CUTE-DSL'; "
        "assert hasattr(m, 'qk_l2_clip_fwd') and hasattr(m, 'rms_norm')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize(
    ("sm80", "sm90", "sm120", "expected"),
    [
        (True, False, False, True),  # SM86 (3090): triton port
        (False, True, False, False),  # SM90: CUTE-DSL
        (False, False, False, False),  # SM100/103/107: CUTE-DSL
        (True, False, True, True),  # is_sm80 wins
    ],
)
def test_rms_norm_dispatch(sm80, sm90, sm120, expected):
    with override_platform(is_sm80=sm80, is_sm90=sm90, is_sm120=sm120):
        norm = RMSNorm(64)
        assert norm._use_triton is expected


@pytest.mark.parametrize(
    ("sm80", "sm90", "sm120", "expected"),
    [
        (True, False, False, True),  # SM86 (3090): triton port
        (False, True, False, False),  # SM90: CUTE-DSL
        (False, False, False, False),  # SM100/103/107: CUTE-DSL
        (True, False, True, True),  # is_sm80 wins
    ],
)
def test_qk_l2_clip_dispatch(sm80, sm90, sm120, expected):
    with override_platform(is_sm80=sm80, is_sm90=sm90, is_sm120=sm120):
        clip = QKL2Clip(
            dim=64,
            threshold=22.627,
            eps=1e-6,
            num_heads=4,
            learnable_scale=True,
        )
        assert clip._use_triton is expected


def test_qk_l2_clip_dispatch_no_scale():
    with override_platform(is_sm80=True, is_sm90=False, is_sm120=False):
        clip = QKL2Clip(
            dim=64,
            threshold=22.627,
            eps=1e-6,
            num_heads=4,
            learnable_scale=False,
        )
        assert clip._use_triton is True
        assert clip.scale is None
