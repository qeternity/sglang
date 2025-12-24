# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_make_workspace,
    marlin_permute_bias,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
)
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()
if _is_cuda:
    from sgl_kernel import gptq_marlin_gemm, gptq_marlin_repack

ScalarType, scalar_types = get_scalar_types()

logger = logging.getLogger(__name__)


def fp8_fused_exponent_bias_into_scales(scales):
    # Marlin dequant for FP8 skips exponent-bias correction, so we fold the
    # FP8->FP16 exponent bias (x256) into scales. Use FP16 bias even when
    # scales are BF16 to avoid BF16-scale explosion.
    fp8_exponent = 4
    target_exponent = 5  # FP16 exponent
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp8_exponent - 1)
    if scales.dtype == torch.bfloat16:
        work_scales = scales.to(torch.float16)
    elif scales.dtype == torch.float16:
        work_scales = scales
    else:
        return scales
    s = torch.ones_like(work_scales) * 2
    s = s**exponent_bias
    logger.info("Marlin FP8: folding exponent-bias into scales (x256)")
    return (work_scales * s).to(scales.dtype)


def apply_fp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor],
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    # For GPUs that lack FP8 hardware support, we can leverage the
    # Marlin kernel for fast weight-only FP8 quantization

    out_shape = input.shape[:-1] + (size_n,)
    orig_dtype = input.dtype
    if orig_dtype == torch.bfloat16:
        input = input.to(torch.float16)
        weight_scale = weight_scale.to(torch.float16)
        if bias is not None:
            bias = bias.to(torch.float16)

    reshaped_x = input.reshape(-1, input.shape[-1])

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0), n=size_n, k=size_k, device=input.device, dtype=input.dtype
    )

    output = gptq_marlin_gemm(
        a=reshaped_x,
        c=None,
        b_q_weight=weight,
        b_scales=weight_scale,
        global_scale=None,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float8_e4m3fn,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    if bias is not None:
        output.add_(bias)

    output = output.reshape(out_shape)
    if output.dtype != orig_dtype:
        output = output.to(orig_dtype)
    return output


def prepare_fp8_layer_for_marlin(
    layer: torch.nn.Module, size_k_first: bool = True
) -> None:
    layer_name = getattr(layer, "prefix", None)
    if layer_name is None:
        layer_name = f"{layer.__class__.__name__}@{id(layer)}"
    logger.warning_once(
        "Your GPU does not have native support for FP8 computation but "
        "FP8 quantization is being used. Weight-only FP8 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    weight_block_size = getattr(layer, "weight_block_size", None)

    if size_k_first:
        assert layer.weight.shape == (part_size_k, part_size_n)
    else:
        assert layer.weight.shape == (part_size_n, part_size_k)

    device = layer.weight.device
    logger.info(
        "Marlin FP8 prep: layer=%s weight shape=%s dtype=%s part_size_k=%s part_size_n=%s block_size=%s size_k_first=%s",
        layer_name,
        tuple(layer.weight.shape),
        layer.weight.dtype,
        part_size_k,
        part_size_n,
        weight_block_size,
        size_k_first,
    )

    # WORKSPACE
    layer.workspace = marlin_make_workspace(device)

    # WEIGHT
    # Repack weights to marlin format
    perm = torch.empty(0, dtype=torch.int, device=device)
    orig_weight = layer.weight
    qweight = pack_fp8_to_int32(orig_weight, size_k_first)
    if not size_k_first:
        qweight = qweight.T.contiguous()

    marlin_qweight = gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=8,
    )
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

    # WEIGHT SCALES
    # Permute scales
    if "weight_scale" in dir(layer):
        scales = layer.weight_scale.to(layer.orig_dtype)
    elif "weight_scale_inv" in dir(layer):
        scales = layer.weight_scale_inv.to(layer.orig_dtype)
        del layer.weight_scale_inv

    group_size = -1 if weight_block_size is None else weight_block_size[1]
    if scales.numel() > 0:
        scale_min = scales.min().item()
        scale_max = scales.max().item()
        scale_mean = scales.mean().item()
    else:
        scale_min = scale_max = scale_mean = None
    logger.info(
        "Marlin FP8 prep: layer=%s scales shape=%s dtype=%s group_size=%s min=%s max=%s mean=%s",
        layer_name,
        tuple(scales.shape),
        scales.dtype,
        group_size,
        scale_min,
        scale_max,
        scale_mean,
    )

    # marlin kernel only support channel-wise and group-wise quantization
    # we need to convert the scales
    if weight_block_size is None:
        if scales.nelement() == 1:
            # tensor-wise quantization -> channel-wise quantization
            # (1, 1) =>(repeat)=> (1, size_n)
            scales = scales.view(1, 1).repeat_interleave(part_size_n, 1)
        elif scales.nelement() > 1 and scales.nelement() != part_size_n:
            assert part_size_n % scales.nelement() == 0
            s_size = scales.nelement()
            # tensor-wise quantization (for gate-up proj)
            #     -> channel-wise quantization
            # (1, s_size) =>(repeat)=> (1, size_n)
            scales = scales.view(1, s_size)
            scales = scales.repeat_interleave(part_size_n // s_size, 1)
        else:
            # channel-wise quantization
            # (1, size_n)
            scales = scales.view(1, part_size_n)
    else:
        # block-wise quantization -> group-wise quantization
        # (size_k // block_size[1], ceil(size_n / block_size[0]))
        #  =>(repeat)=> (size_k // block_size[1], size_n)
        if not size_k_first:
            scales = scales.T.contiguous()
        block_n = weight_block_size[0]
        scales = scales.repeat_interleave(block_n, 1)
        # size_n may not divisible by block_size[0]
        scales = scales[:, :part_size_n]

    raw_scales = scales
    marlin_scales = marlin_permute_scales(
        s=scales, size_k=part_size_k, size_n=part_size_n, group_size=group_size
    )
    marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)
    if marlin_scales.numel() > 0:
        scales_min = marlin_scales.min().item()
        scales_max = marlin_scales.max().item()
        scales_mean = marlin_scales.mean().item()
        num_inf = torch.isinf(marlin_scales).sum().item()
        num_nan = torch.isnan(marlin_scales).sum().item()
    else:
        scales_min = scales_max = scales_mean = None
        num_inf = num_nan = 0
    logger.info(
        "Marlin FP8 prep: layer=%s fused scales shape=%s dtype=%s min=%s max=%s mean=%s inf=%s nan=%s",
        layer_name,
        tuple(marlin_scales.shape),
        marlin_scales.dtype,
        scales_min,
        scales_max,
        scales_mean,
        num_inf,
        num_nan,
    )
    layer.weight_scale = torch.nn.Parameter(marlin_scales, requires_grad=False)

    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        bias = marlin_permute_bias(layer.bias)
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)

    with torch.no_grad():
        # Compare Marlin output to reference dequantized matmul to validate scale semantics.
        m = int(getattr(layer, "fp8_marlin_debug_m", 1) or 1)
        x = torch.randn((m, part_size_k), device=device, dtype=torch.float32)
        weight_fp32 = orig_weight.to(torch.float32)
        scales_fp32 = scales.to(torch.float32)
        if weight_fp32.shape[0] != scales_fp32.shape[0]:
            weight_fp32 = weight_fp32.t()
        w_scaled = weight_fp32 * scales_fp32
        w_scaled_inv = weight_fp32 * (1.0 / scales_fp32)
        w_scaled_448 = weight_fp32 * (scales_fp32 * 448.0)
        w_scaled_div_448 = weight_fp32 * (scales_fp32 / 448.0)
        w_scaled_256 = weight_fp32 * (scales_fp32 * 256.0)
        w_scaled_div_256 = weight_fp32 * (scales_fp32 / 256.0)
        w_scaled_t = w_scaled.t()
        w_scaled_inv_t = w_scaled_inv.t()
        w_scaled_448_t = w_scaled_448.t()
        w_scaled_div_448_t = w_scaled_div_448.t()
        w_scaled_256_t = w_scaled_256.t()
        w_scaled_div_256_t = w_scaled_div_256.t()
        ref_t = x @ w_scaled_t if x.shape[1] == w_scaled_t.shape[0] else x @ w_scaled
        ref_t_inv = (
            x @ w_scaled_inv_t
            if x.shape[1] == w_scaled_inv_t.shape[0]
            else x @ w_scaled_inv
        )
        ref_t_448 = (
            x @ w_scaled_448_t
            if x.shape[1] == w_scaled_448_t.shape[0]
            else x @ w_scaled_448
        )
        ref_t_div_448 = (
            x @ w_scaled_div_448_t
            if x.shape[1] == w_scaled_div_448_t.shape[0]
            else x @ w_scaled_div_448
        )
        ref_t_256 = (
            x @ w_scaled_256_t
            if x.shape[1] == w_scaled_256_t.shape[0]
            else x @ w_scaled_256
        )
        ref_t_div_256 = (
            x @ w_scaled_div_256_t
            if x.shape[1] == w_scaled_div_256_t.shape[0]
            else x @ w_scaled_div_256
        )
        ref_no_t = x @ w_scaled if x.shape[1] == w_scaled.shape[0] else x @ w_scaled_t
        marlin_out = gptq_marlin_gemm(
            a=x.to(torch.float16),
            c=None,
            b_q_weight=marlin_qweight,
            b_scales=marlin_scales.to(torch.float16),
            global_scale=None,
            b_zeros=None,
            g_idx=None,
            perm=None,
            workspace=layer.workspace,
            b_q_type=scalar_types.float8_e4m3fn,
            size_m=m,
            size_n=part_size_n,
            size_k=part_size_k,
            use_atomic_add=False,
            use_fp32_reduce=False,
        )
        marlin_out_raw = gptq_marlin_gemm(
            a=x.to(torch.float16),
            c=None,
            b_q_weight=marlin_qweight,
            b_scales=raw_scales.to(torch.float16),
            global_scale=None,
            b_zeros=None,
            g_idx=None,
            perm=None,
            workspace=layer.workspace,
            b_q_type=scalar_types.float8_e4m3fn,
            size_m=m,
            size_n=part_size_n,
            size_k=part_size_k,
            use_atomic_add=False,
            use_fp32_reduce=False,
        )
        marlin_out = marlin_out.to(torch.float32)
        marlin_out_raw = marlin_out_raw.to(torch.float32)
        err = (marlin_out - ref_t).abs().max().item()
        err_inv = (marlin_out - ref_t_inv).abs().max().item()
        err_scale_448 = (marlin_out - ref_t_448).abs().max().item()
        err_scale_div_448 = (marlin_out - ref_t_div_448).abs().max().item()
        err_scale_256 = (marlin_out - ref_t_256).abs().max().item()
        err_scale_div_256 = (marlin_out - ref_t_div_256).abs().max().item()
        err_raw = (marlin_out_raw - ref_t).abs().max().item()
        err_raw_inv = (marlin_out_raw - ref_t_inv).abs().max().item()
        err_no_t = (marlin_out - ref_no_t).abs().max().item()
        logger.info(
            "Marlin FP8 debug: layer=%s max_abs_err perm_scale=%s perm_inv=%s "
            "perm_scale*448=%s perm_scale/448=%s perm_scale*256=%s "
            "perm_scale/256=%s raw_scale=%s raw_inv=%s no_t=%s",
            layer_name,
            err,
            err_inv,
            err_scale_448,
            err_scale_div_448,
            err_scale_256,
            err_scale_div_256,
            err_raw,
            err_raw_inv,
            err_no_t,
        )
        if err > 0.05:
            logger.warning(
                "Marlin FP8 debug: layer=%s large error (perm_scale=%s). "
                "Check scale semantics or packing.",
                layer_name,
                err,
            )


def prepare_moe_fp8_layer_for_marlin(
    layer: torch.nn.Module, size_k_first: bool = True
) -> None:
    logger.warning_once(
        "Your GPU does not have native support for FP8 computation but "
        "FP8 quantization is being used. Weight-only FP8 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )

    e = layer.num_experts
    k = layer.hidden_size
    n = layer.intermediate_size_per_partition
    weight_block_size = getattr(layer, "weight_block_size", None)

    # WORKSPACE
    device = layer.w13_weight.device
    layer.workspace = marlin_make_workspace(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    # WEIGHT
    # Repack weights to marlin format
    for name in ["w13_weight", "w2_weight"]:
        weight = getattr(layer, name)
        tensor_list = []
        if "w13" in name:
            size_n, size_k = n * 2, k
        else:
            size_n, size_k = k, n

        if size_k_first:
            assert weight.shape == (e, size_k, size_n)
        else:
            assert weight.shape == (e, size_n, size_k)

        for i in range(e):
            qweight = pack_fp8_to_int32(weight[i], size_k_first)
            if not size_k_first:
                qweight = qweight.T.contiguous()

            marlin_qweight = gptq_marlin_repack(
                b_q_weight=qweight, perm=perm, size_k=size_k, size_n=size_n, num_bits=8
            )
            tensor_list.append(marlin_qweight)

        weight = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        weight = torch.nn.Parameter(weight, requires_grad=False)

        setattr(layer, name, weight)

    # WEIGHT SCALES
    # Permute scales
    group_size = -1 if weight_block_size is None else weight_block_size[1]

    for name in ["w13", "w2"]:
        if name + "_weight_scale" in dir(layer):
            new_name = name + "_weight_scale"
            scales = getattr(layer, new_name).to(layer.orig_dtype)
            delattr(layer, new_name)
        elif name + "_weight_scale_inv" in dir(layer):
            new_name = name + "_weight_scale_inv"
            scales = getattr(layer, new_name).to(layer.orig_dtype)
            delattr(layer, new_name)

        tensor_list = []
        if "w13" in name:
            size_n, size_k = n * 2, k
        else:
            size_n, size_k = k, n

        # marlin kernel only support channel-wise and group-wise quantization
        # we need to convert the scales
        if weight_block_size is None:
            if scales.nelement() == e:
                # tensor-wise quantization -> channel-wise quantization
                # (e, 1, 1) =>(repeat)=> (e, 1, size_n)
                scales = scales.view(e, 1, 1).repeat_interleave(size_n, 2)
            elif scales.nelement() > e and scales.nelement() != e * size_n:
                assert (e * size_n) % scales.nelement() == 0
                s_size = scales.nelement() // e
                # tensor-wise quantization (for gate-up proj)
                #     -> channel-wise quantization
                # (e, 1, s_size) =>(repeat)=> (e, 1, size_n)
                scales = scales.view(e, 1, s_size)
                scales = scales.repeat_interleave(size_n // s_size, 2)
            else:
                # channel-wise quantization
                # (e, 1, size_n)
                scales = scales.view(e, 1, size_n)
        else:
            # block-wise quantization -> group-wise quantization
            # (e, size_k // block_size[1], ceil(size_n / block_size[0]))
            #  =>(repeat)=> (e, size_k // block_size[1], size_n)
            if not size_k_first:
                scales = scales.permute(0, 2, 1)
            block_n = weight_block_size[0]
            scales = scales.repeat_interleave(block_n, 2)
            # size_n may not divisible by block_size[0]
            scales = scales[..., :size_n].contiguous()

        for i in range(e):
            marlin_scales = marlin_permute_scales(
                s=scales[i], size_k=size_k, size_n=size_n, group_size=group_size
            )
            tensor_list.append(marlin_scales)

        scales = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        scales = fp8_fused_exponent_bias_into_scales(scales)
        scales = torch.nn.Parameter(scales, requires_grad=False)

        setattr(layer, name + "_weight_scale", scales)

    # BIAS
    # Permute bias
    for name in ["w13_bias", "w2_bias"]:
        if not hasattr(layer, name):
            continue
        bias = getattr(layer, name).to(layer.orig_dtype)

        tensor_list = []
        for i in range(e):
            expert_bias = bias[i]

            tensor_list.append(marlin_permute_bias(expert_bias))

        bias = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        bias = torch.nn.Parameter(bias, requires_grad=False)
        setattr(layer, name, bias)


def pack_fp8_to_int32(
    fp8_tensor: torch.Tensor, size_k_first: bool = True
) -> torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements)
    """
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert fp8_tensor.ndim == 2

    fp8_tensor = fp8_tensor.T if size_k_first else fp8_tensor
    fp8_tensor = fp8_tensor.contiguous()
    # fp8_tensor is contiguous and have shape (N, K) now
    # with `.view(torch.int32)`, it become (N, K // 4)
    int32_tensor = fp8_tensor.view(torch.int32)
    return int32_tensor.T.contiguous() if size_k_first else int32_tensor


def marlin_quant_fp8_torch(weight, group_size):
    size_n, size_k = weight.shape
    device = weight.device

    if group_size != -1:
        scales = weight.view(size_n, -1, group_size).abs().max(-1)[0] / 448
        repeated_scales = scales.repeat_interleave(group_size, 1)
        fp8_weight = (weight / repeated_scales).to(torch.float8_e4m3fn)
        weight_ref = fp8_weight.to(weight.dtype) * repeated_scales
    else:
        scales = weight.view(size_n, 1, group_size).abs().max(-1)[0] / 448
        repeated_scales = scales.repeat_interleave(size_k, 1)
        fp8_weight = (weight / repeated_scales).to(torch.float8_e4m3fn)
        weight_ref = fp8_weight.to(weight.dtype) * repeated_scales

    packed_weight = pack_fp8_to_int32(fp8_weight, False).T.contiguous()
    marlin_qweight = gptq_marlin_repack(
        b_q_weight=packed_weight,
        perm=torch.empty(0, dtype=torch.int, device=device),
        size_k=size_k,
        size_n=size_n,
        num_bits=8,
    )

    marlin_scales = marlin_permute_scales(
        s=scales.T, size_k=size_k, size_n=size_n, group_size=group_size
    )

    marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)

    return weight_ref.T, marlin_qweight, marlin_scales
