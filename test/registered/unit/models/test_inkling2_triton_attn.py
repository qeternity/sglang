"""Inkling-2 on the triton attention backend: dispatch + LSE plumbing.

CPU-runnable: the model dispatch test drives ``Inkling2Attention._forward``
with faked submodules and a fixed serving-backend answer; the backend tests
drive ``TritonAttnBackend.forward_extend`` / ``forward_decode`` with faked
kernel entry points and check the ``return_lse`` contract (allocation, the
split-LSE logsumexp, and the ``(output, lse)`` return) without a GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.score_mod import relative_bias_score_mod
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.inkling_common import inkling2_attn as attn_mod
from sglang.srt.models.inkling_common.inkling2_attn import Inkling2Attention
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

H, KV, DIM, EXTENT = 12, 4, 128, 1024


class _Recorder:
    """Callable that records positional/keyword args and returns canned outputs."""

    def __init__(self, *results):
        self.results = results
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.results[len(self.calls) - 1]


def _make_module(monkeypatch, backend_name, kv_cache_dtype):
    mod = Inkling2Attention.__new__(Inkling2Attention)
    mod.is_local = True
    mod.num_tp_heads = H
    mod.num_tp_kv_heads = KV
    mod.head_dim = DIM
    mod.d_rel = 16
    mod.rel_extent = EXTENT
    mod.fused_decode_prologue = False
    mod.group_all_reduce = False

    rel_bias = torch.randn(2, H, EXTENT)
    mod.rel_logits_proj = _Recorder(rel_bias)
    q = torch.randn(2, H * DIM)
    k = torch.randn(2, KV * DIM)
    v = torch.randn(2, KV * DIM)
    r = torch.randn(2, H, 16)
    gate = torch.randn(2, 2304)
    sinks = torch.randn(2, H)
    mod._project_sources = _Recorder((q, k, v, r, gate, sinks))
    mod._prologue = _Recorder((q, k, v, r))
    out = torch.randn(2, H * DIM)
    lse = torch.randn(2, H)
    mod.wo_ud = _Recorder((out, None))

    monkeypatch.setattr(attn_mod, "serving_attention_backend", lambda fb: backend_name)
    monkeypatch.setattr(
        attn_mod, "get_model", lambda: SimpleNamespace(kv_cache_dtype=kv_cache_dtype)
    )
    monkeypatch.setattr(
        attn_mod,
        "get_platform",
        lambda: SimpleNamespace(is_sm90=False, is_sm120=False),
    )
    monkeypatch.setattr(
        attn_mod,
        "get_parallel",
        lambda: SimpleNamespace(
            moe_ep_size=0, attn_tp_group=SimpleNamespace(world_size=1)
        ),
    )
    gate_sink = _Recorder(out)
    monkeypatch.setattr(attn_mod, "gate_sink", gate_sink)
    return mod, (mod.wo_ud is not None and (rel_bias, gate_sink, lse, out))


def _forward(mod, attn, monkeypatch=None):
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    positions = torch.arange(2)
    hidden = torch.randn(2, 2304)
    return attn_mod.Inkling2Attention._forward(
        mod, hidden, positions, forward_batch, attn=attn, attn_padded=None
    )


def test_triton_dispatch_uses_score_mod(monkeypatch):
    mod, (rel_bias, gate_sink, lse, out) = _make_module(
        monkeypatch, "triton", "bfloat16"
    )
    fake_attn = _Recorder((out, lse))
    result = _forward(mod, fake_attn)

    assert torch.equal(result, out)
    args, kwargs = fake_attn.calls[0]
    assert kwargs["score_mod"] is relative_bias_score_mod
    assert kwargs["aux_tensors"] == [rel_bias]
    assert "rel_bias" not in kwargs
    assert kwargs["return_lse"] is True
    # The unpadded path consumed the attention LSE through gate_sink.
    assert torch.equal(gate_sink.calls[0][0][2], lse)
    assert gate_sink.calls[0][1]["head_groups"] is None


def test_fa4_dispatch_unchanged(monkeypatch):
    mod, (rel_bias, gate_sink, lse, out) = _make_module(monkeypatch, "fa4", "bfloat16")
    fake_attn = _Recorder((out, lse))
    _forward(mod, fake_attn)

    _, kwargs = fake_attn.calls[0]
    assert kwargs["rel_bias"] is rel_bias
    assert "score_mod" not in kwargs
    assert kwargs["return_lse"] is True


def test_triton_mxfp8_raises(monkeypatch):
    mod, _ = _make_module(monkeypatch, "triton", "mxfp8")
    fake_attn = _Recorder((torch.randn(2, H * DIM), torch.randn(2, H)))
    with pytest.raises(RuntimeError, match="FA4"):
        _forward(mod, fake_attn)


def test_triton_rejects_unknown_backend(monkeypatch):
    mod, _ = _make_module(monkeypatch, "flashinfer", "bfloat16")
    fake_attn = _Recorder((torch.randn(2, H * DIM), torch.randn(2, H)))
    with pytest.raises(AssertionError, match="FA4 or triton"):
        _forward(mod, fake_attn)


def _bare_backend(monkeypatch, n, heads, k, v):
    backend = TritonAttnBackend.__new__(TritonAttnBackend)
    backend.dcp_size = 1
    backend.enable_deterministic = False
    backend.enable_lean_attention = False
    backend.use_mla = False
    backend.use_verify_gfx115 = False
    backend.use_verify_shared_kv = False
    backend.use_verify_splitkv = False
    backend.verify_mla_draft_fwd = None
    backend.draft_bf16_dot = False
    backend.dllm_attention = None
    backend.page_size = 1
    backend.use_pdl = False
    backend.max_kv_splits = 4
    backend.allow_bidirectional_attention_in_extend = False
    backend._window_write_locs = lambda: None
    backend._set_kv_buffer = lambda *a, **k: None
    backend._window_rail = lambda layer: (None, None, None)
    backend.forward_metadata = SimpleNamespace(
        attn_lse=torch.zeros(n, heads, 4),
        attn_logits=torch.zeros(n, heads, 4, DIM),
        extra_attn_logits={},
        kv_indptr=torch.tensor([0, n]),
        kv_indices=torch.arange(n * 10),
        custom_mask=None,
        mask_indptr=torch.tensor([0]),
        max_extend_len=n,
        num_kv_splits=torch.ones(n, dtype=torch.int32),
        qo_indptr=torch.arange(0, n + 1),
        seq_lens=None,
        seq_lens_sum=n,
        lean_Mp=1,
        lean_Lp=1,
        lean_Op=1,
        lean_locks=None,
        uneven_batch=torch.zeros(1, dtype=torch.int32),
    )
    kv = SimpleNamespace(
        read=lambda layer_id: SimpleNamespace(key=lambda: k, value=lambda: v)
    )
    monkeypatch.setattr(TritonAttnBackend, "kv_layers", property(lambda self: kv))
    return backend


def _layer(heads):
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=heads,
        tp_k_head_num=KV,
        qk_head_dim=DIM,
        v_head_dim=DIM,
        scaling=1.0 / DIM**0.5,
        logit_capping_method="tanh",
        logit_cap=0.0,
        xai_temperature_len=-1,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        sliding_window_size=-1,
        k_scale=None,
        v_scale=None,
        k_scale_float=1.0,
        v_scale_float=1.0,
    )


def test_decode_return_lse(monkeypatch):
    n, heads = 2, H
    k = torch.randn(8, KV, DIM)
    v = torch.randn(8, KV, DIM)
    backend = _bare_backend(monkeypatch, n, heads, k, v)

    written = {}

    def fake_decode(*args, **kwargs):
        # Emulate the split-KV kernel: fill the active split slots, leave the
        # rest as the -inf the backend pre-filled.
        lse_buf = backend.forward_metadata.attn_lse
        for s in range(3):
            lse_buf[:, :, s] = (
                torch.arange(n * heads, device=lse_buf.device).float().reshape(n, heads)
                + s
            )
        written.update(kwargs)

    backend.decode_attention_fwd = fake_decode
    q = torch.randn(n, heads, DIM)
    score_mod = object()
    aux = [torch.randn(n, heads, EXTENT)]
    o, lse = backend.forward_decode(
        q,
        k[:n],
        v[:n],
        _layer(heads),
        SimpleNamespace(forward_mode=ForwardMode.DECODE),
        save_kv_cache=False,
        score_mod=score_mod,
        aux_tensors=aux,
        return_lse=True,
    )
    assert o.shape == (n, heads * DIM)
    assert lse.shape == (n, heads)
    expected = torch.logaddexp(
        torch.logaddexp(
            torch.arange(n * heads).float().reshape(n, heads),
            torch.arange(n * heads).float().reshape(n, heads) + 1,
        ),
        torch.arange(n * heads).float().reshape(n, heads) + 2,
    )
    assert torch.allclose(lse, expected, atol=1e-5)
    assert written["score_mod"] is score_mod
    assert written["aux_tensors"] is aux


def test_decode_no_return_lse_returns_output_only(monkeypatch):
    n, heads = 2, H
    k = torch.randn(8, KV, DIM)
    v = torch.randn(8, KV, DIM)
    backend = _bare_backend(monkeypatch, n, heads, k, v)
    backend.decode_attention_fwd = lambda *a, **k: None
    q = torch.randn(n, heads, DIM)
    result = backend.forward_decode(
        q,
        k[:n],
        v[:n],
        _layer(heads),
        SimpleNamespace(forward_mode=ForwardMode.DECODE),
        save_kv_cache=False,
    )
    assert isinstance(result, torch.Tensor)


def test_extend_return_lse(monkeypatch):
    n, heads = 2, H
    k = torch.randn(n * KV, DIM)
    v = torch.randn(n * KV, DIM)
    backend = _bare_backend(monkeypatch, n, heads, k, v)

    written = {}

    def fake_extend(*args, **kwargs):
        lse = kwargs["lse_extend"]
        lse[:] = 1.5
        written["lse_extend"] = lse
        written.update(kwargs)

    backend.extend_attention_fwd = fake_extend
    q = torch.randn(n, heads, DIM)
    o, lse = backend.forward_extend(
        q,
        k,
        v,
        _layer(heads),
        SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            mha_one_shot=False,
            extend_seq_lens_cpu=None,
            attn_attend_prefix_cache=False,
        ),
        save_kv_cache=False,
        score_mod=object(),
        aux_tensors=[torch.randn(n, heads, EXTENT)],
        return_lse=True,
    )
    assert o.shape == (n, heads, DIM)
    assert lse.shape == (n, heads)
    assert torch.all(lse == 1.5)
    assert written["lse_extend"] is lse
    assert written["score_mod"] is not None
    assert written["aux_tensors"] is not None


def test_extend_deterministic_return_lse_raises(monkeypatch):
    n, heads = 2, H
    k = torch.randn(n * KV, DIM)
    v = torch.randn(n * KV, DIM)
    backend = _bare_backend(monkeypatch, n, heads, k, v)
    backend.enable_deterministic = True
    q = torch.randn(n, heads, DIM)
    with pytest.raises(NotImplementedError, match="return_lse"):
        backend.forward_extend(
            q,
            k,
            v,
            _layer(heads),
            SimpleNamespace(forward_mode=ForwardMode.EXTEND, mha_one_shot=False),
            save_kv_cache=False,
            return_lse=True,
        )


def test_extend_dcp_return_lse_raises(monkeypatch):
    n, heads = 2, H
    k = torch.randn(n * KV, DIM)
    v = torch.randn(n * KV, DIM)
    backend = _bare_backend(monkeypatch, n, heads, k, v)
    backend.dcp_size = 2
    q = torch.randn(n, heads, DIM)
    with pytest.raises(NotImplementedError, match="return_lse"):
        backend.forward_extend(
            q,
            k,
            v,
            _layer(heads),
            SimpleNamespace(forward_mode=ForwardMode.EXTEND, mha_one_shot=False),
            save_kv_cache=False,
            return_lse=True,
        )


def test_decode_dcp_return_lse_raises(monkeypatch):
    n, heads = 2, H
    k = torch.randn(8, KV, DIM)
    v = torch.randn(8, KV, DIM)
    backend = _bare_backend(monkeypatch, n, heads, k, v)
    backend.dcp_size = 2
    q = torch.randn(n, heads, DIM)
    with pytest.raises(NotImplementedError, match="return_lse"):
        backend.forward_decode(
            q,
            k[:n],
            v[:n],
            _layer(heads),
            SimpleNamespace(forward_mode=ForwardMode.DECODE),
            save_kv_cache=False,
            return_lse=True,
        )
