import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLanguageModelOnlyWeightLoading(unittest.TestCase):
    def test_dense_loader_skips_visual_weights(self):
        model = SimpleNamespace(
            language_model_only=True,
            named_parameters=lambda **_: (),
        )

        loaded = Qwen3_5ForConditionalGeneration.load_weights(
            model,
            [("model.visual.patch_embed.proj.weight", torch.empty(0))],
        )

        self.assertEqual(loaded, set())

    def test_moe_loader_skips_visual_weights(self):
        model = SimpleNamespace(
            config=SimpleNamespace(num_experts=0),
            enable_shared_expert_fusion=False,
            language_model_only=True,
            model=SimpleNamespace(start_layer=0, end_layer=0, layers=[]),
            named_parameters=lambda **_: (),
        )

        loaded = Qwen3_5MoeForConditionalGeneration.load_weights(
            model,
            [("model.visual.patch_embed.proj.weight", torch.empty(0))],
        )

        self.assertEqual(loaded, set())


if __name__ == "__main__":
    unittest.main()
