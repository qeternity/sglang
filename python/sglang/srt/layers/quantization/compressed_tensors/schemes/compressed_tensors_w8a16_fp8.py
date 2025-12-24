# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable, List, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)
from sglang.srt.layers.quantization.utils import convert_to_channelwise

__all__ = ["CompressedTensorsW8A16Fp8"]

logger = logging.getLogger(__name__)

SUPPORTED_STRATEGIES = [QuantizationStrategy.CHANNEL, QuantizationStrategy.TENSOR]


class CompressedTensorsW8A16Fp8(CompressedTensorsScheme):
    def __init__(self, strategy: str, is_static_input_scheme: bool):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    # W8A8-Fp8 kernels support only per-tensor and per-channel cases.
    # So if we have a fused module (QKV, MLP) with per tensor scales,
    # we expand each scale to its shard's channels.
    def process_weights_after_loading(self, layer) -> None:
        if self.strategy == QuantizationStrategy.TENSOR:
            logger.info_once(
                "CompressedTensorsW8A16Fp8: tensor weight scales shape=%s dtype=%s",
                tuple(layer.weight_scale.shape),
                layer.weight_scale.dtype,
            )
            ws_channelwise = convert_to_channelwise(
                layer.weight_scale, layer.logical_widths
            )
            logger.info_once(
                "CompressedTensorsW8A16Fp8: converted channel scales shape=%s dtype=%s",
                tuple(ws_channelwise.shape),
                ws_channelwise.dtype,
            )
            layer.weight_scale = torch.nn.Parameter(ws_channelwise, requires_grad=False)
        else:
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data, requires_grad=False
            )

        logger.info_once(
            "CompressedTensorsW8A16Fp8: weight shape=%s dtype=%s",
            tuple(layer.weight.shape),
            layer.weight.dtype,
        )

        if self.is_static_input_scheme:
            # required by torch.compile to be torch.nn.Parameter
            layer.input_scale = torch.nn.Parameter(
                layer.input_scale.data, requires_grad=False
            )
        size_k_first = False
        transpose_qweight = True
        # Debug: force fallback for all layers to isolate Marlin issues.
        layer.fp8_marlin_fallback = True
        logger.warning(
            "CompressedTensorsW8A16Fp8: enable fp8_marlin_fallback for layer=%s",
            getattr(layer, "prefix", layer.__class__.__name__),
        )
        prepare_fp8_layer_for_marlin(
            layer,
            size_k_first=size_k_first,
            transpose_qweight=transpose_qweight,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size: int,
        output_partition_sizes: List[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        elif self.strategy == QuantizationStrategy.TENSOR:
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
        else:
            raise ValueError(
                f"Unsupported weight strategy={self.strategy}, "
                f"supported strategies are {SUPPORTED_STRATEGIES}"
            )

        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE (to deal with converted checkpoints)
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_scale", input_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if getattr(layer, "fp8_marlin_fallback", False):
            logger.warning(
                "CompressedTensorsW8A16Fp8: using fp8_marlin_fallback for layer=%s",
                getattr(layer, "prefix", layer.__class__.__name__),
            )
            layer.fp8_marlin_fallback_used = True
            # Dequantized fallback for correctness on square RowParallelLinear.
            weight = layer.fp8_weight.to(torch.float16)
            scales = layer.fp8_weight_scale.to(torch.float16)
            weight = weight * scales
            if not hasattr(layer, "fp8_fallback_logged"):
                layer.fp8_fallback_logged = True
                layer_name = getattr(layer, "prefix", layer.__class__.__name__)
                if layer_name == "model.layers.0.mlp.gate_up_proj":
                    w_fp32 = layer.fp8_weight.to(torch.float32)
                    s_fp32 = layer.fp8_weight_scale.to(torch.float32)
                    w_dq = w_fp32 * s_fp32
                    logger.warning(
                        "FP8 fallback stats layer=%s w_min=%s w_max=%s w_mean=%s "
                        "s_min=%s s_max=%s s_mean=%s dq_min=%s dq_max=%s dq_mean=%s dq_norm=%s",
                        layer_name,
                        w_fp32.min().item(),
                        w_fp32.max().item(),
                        w_fp32.mean().item(),
                        s_fp32.min().item(),
                        s_fp32.max().item(),
                        s_fp32.mean().item(),
                        w_dq.min().item(),
                        w_dq.max().item(),
                        w_dq.mean().item(),
                        torch.norm(w_dq).item(),
                    )
            x_fp16 = x.to(torch.float16)
            if getattr(layer, "fp8_fallback_transpose", True):
                out = torch.matmul(x_fp16, weight.t())
            else:
                out = torch.matmul(x_fp16, weight)
            if bias is not None:
                out = out + bias.to(torch.float16)
            return out.to(x.dtype)
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
