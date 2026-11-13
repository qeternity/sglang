"""Qwen3.5's deferred MTP allocation with the common draft sharing paths."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models import qwen3_5_mtp as mtp
from sglang.srt.speculative import eagle_worker_v2, pp_draft_embedding
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _embedding(vocab_size, hidden_size, **kwargs):
    return nn.Embedding(vocab_size, hidden_size)


def _head(vocab_size, hidden_size, **kwargs):
    return nn.Linear(hidden_size, vocab_size, bias=False)


def _draft(*, tied=False):
    draft = mtp.Qwen3_5ForCausalLMMTP.__new__(mtp.Qwen3_5ForCausalLMMTP)
    nn.Module.__init__(draft)
    draft.config = SimpleNamespace(
        vocab_size=4, hidden_size=3, tie_word_embeddings=tied
    )
    draft.quant_config = None
    draft.prefix = ""
    draft.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    draft.model = nn.Module()
    draft.model.embed_tokens = mtp.DeferredSharedEmbedding()
    draft.lm_head = draft.model.embed_tokens if tied else mtp.DeferredSharedLMHead()
    return draft


class TestMTPEmbeddingSharing(CustomTestCase):
    def setUp(self):
        self.enterContext(patch.object(mtp, "VocabParallelEmbedding", _embedding))
        self.enterContext(patch.object(mtp, "ParallelLMHead", _head))
        self.enterContext(
            patch.object(mtp, "is_dp_attention_enabled", return_value=False)
        )
        self.enterContext(
            patch.object(
                mtp,
                "current_platform",
                SimpleNamespace(empty_cache=lambda: None, synchronize=lambda: None),
            )
        )

    def test_constructor_defers_vocab_allocations(self):
        def backbone(config, quant_config, **kwargs):
            model = nn.Module()
            model.embed_tokens = kwargs["embed_tokens_module"]
            return model

        config = SimpleNamespace(
            vocab_size=4, hidden_size=3, rms_norm_eps=1e-6, tie_word_embeddings=False
        )
        with (
            patch.object(mtp, "Qwen3_5ForCausalLM", backbone),
            patch.object(
                mtp.Qwen3_5ForCausalLMMTP, "own_quant_config", return_value=None
            ),
            patch.object(mtp, "GemmaRMSNorm", side_effect=lambda *args: nn.Identity()),
            patch.object(mtp, "LogitsProcessor", return_value=nn.Identity()),
            patch.object(
                mtp,
                "get_parallel",
                return_value=SimpleNamespace(
                    pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True)
                ),
            ),
        ):
            draft = mtp.Qwen3_5ForCausalLMMTP(config)

        self.assertIsNone(draft.model.embed_tokens.weight)
        self.assertIsNone(draft.lm_head.weight)
        self.assertEqual(list(dict(draft.named_parameters())), ["fc.weight"])

    def test_single_stage_worker_shares_target_modules(self):
        """Sharing must initialize a usable draft without allocating a second vocab."""
        for tied in (False, True):
            with self.subTest(tied=tied):
                draft = _draft(tied=tied)
                target = nn.Module()
                target.model = nn.Module()
                target.model.embed_tokens = _embedding(4, 3)
                target.lm_head = target.model.embed_tokens if tied else _head(4, 3)
                worker = SimpleNamespace(
                    target_worker=SimpleNamespace(
                        model_runner=SimpleNamespace(model=target)
                    ),
                    draft_runner=SimpleNamespace(model=draft),
                    hot_token_id=None,
                    speculative_algorithm=SimpleNamespace(is_eagle3=lambda: False),
                    _resolve_shared_embed_and_head=lambda: (
                        target.model.embed_tokens.weight,
                        target.lm_head.weight,
                    ),
                )
                with patch.object(
                    eagle_worker_v2,
                    "get_parallel",
                    return_value=SimpleNamespace(pp_size=1),
                ):
                    eagle_worker_v2.EagleDraftWorker.init_lm_head(worker)

                tokens = torch.tensor([0, 2])
                self.assertIs(
                    draft.model.embed_tokens.weight, target.model.embed_tokens.weight
                )
                self.assertIs(
                    draft.lm_head,
                    target.lm_head if not tied else draft.model.embed_tokens,
                )
                torch.testing.assert_close(
                    draft.model.embed_tokens(tokens), target.model.embed_tokens(tokens)
                )

    def test_pipeline_resolver_materializes_before_checkpoint_load(self):
        """A PP draft owns its embedding because its target stage cannot share one."""
        draft = _draft()
        expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        head = _head(4, 3).weight

        def load_embedding(model, *args, **kwargs):
            self.assertIsInstance(model.model.embed_tokens, nn.Embedding)
            with torch.no_grad():
                model.model.embed_tokens.weight.copy_(expected)
            return model.model.embed_tokens.weight

        with (
            patch.object(
                pp_draft_embedding,
                "resolve_target_embed_and_head",
                return_value=(None, head),
            ),
            patch.object(
                pp_draft_embedding,
                "load_draft_embedding_from_checkpoint",
                side_effect=load_embedding,
            ),
        ):
            embed, resolved_head = pp_draft_embedding.resolve_draft_embed_and_head(
                target_model=nn.Module(),
                draft_model=draft,
                model_path="unused",
                revision=None,
                load_config=None,
            )

        draft.set_embed_and_head(embed, resolved_head)
        self.assertIs(draft.lm_head.weight, head)
        torch.testing.assert_close(
            draft.model.embed_tokens(torch.tensor([1, 3])), expected[[1, 3]]
        )

    def test_tensor_fallback_keeps_tied_head_attached(self):
        """Replacing a deferred embedding must also rebind a tied output head."""
        draft = _draft(tied=True)
        embed = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3))
        draft.set_embed_and_head(embed, embed)

        self.assertIs(draft.lm_head, draft.model.embed_tokens)
        self.assertIs(draft.lm_head.weight, embed)
        torch.testing.assert_close(
            draft.model.embed_tokens(torch.tensor([2])), embed[[2]]
        )

    def test_pipeline_worker_keeps_checkpoint_embedding(self):
        draft = _draft()
        draft.materialize_embed_tokens()
        embed = draft.model.embed_tokens.weight
        target = SimpleNamespace(lm_head=_head(4, 3))
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=SimpleNamespace(model=target)),
            draft_runner=SimpleNamespace(model=draft),
            hot_token_id=None,
            speculative_algorithm=SimpleNamespace(is_eagle3=lambda: False),
            _resolve_shared_embed_and_head=lambda: (embed, target.lm_head.weight),
        )
        with patch.object(
            eagle_worker_v2, "get_parallel", return_value=SimpleNamespace(pp_size=2)
        ):
            eagle_worker_v2.EagleDraftWorker.init_lm_head(worker)

        self.assertIs(draft.model.embed_tokens.weight, embed)
        self.assertIs(draft.lm_head, target.lm_head)

    def test_token_map_uses_tensor_fallback_without_mutating_target_head(self):
        draft = _draft()
        target = SimpleNamespace(lm_head=_head(4, 3))
        original = target.lm_head.weight.detach().clone()
        embed = _embedding(4, 3).weight
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=SimpleNamespace(model=target)),
            draft_runner=SimpleNamespace(model=draft),
            hot_token_id=torch.tensor([1, 3]),
            speculative_algorithm=SimpleNamespace(is_eagle3=lambda: False),
            _resolve_shared_embed_and_head=lambda: (embed, target.lm_head.weight),
        )
        eagle_worker_v2.EagleDraftWorker.init_lm_head(worker)

        self.assertIs(draft.model.embed_tokens.weight, embed)
        torch.testing.assert_close(draft.lm_head.weight, original[[1, 3]])
        torch.testing.assert_close(target.lm_head.weight, original)


if __name__ == "__main__":
    unittest.main()
