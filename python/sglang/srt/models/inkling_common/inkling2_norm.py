import torch
from sglang.srt.runtime_context import get_platform
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        pre_affine: bool = False,
        output_dtype: torch.dtype | None = None,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.output_dtype = output_dtype
        # Ampere (SM8x): the CUTE-DSL norm targets SM90+; the triton port is
        # term-for-term (inkling2_norm_triton.rms_norm).
        self._use_triton = get_platform().is_sm80
        self.weight = (
            nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None
        )
        self.pre_weight = nn.Parameter(torch.ones(hidden_size)) if pre_affine else None

    def forward(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if self._use_triton:
            from sglang.kernels.ops.layernorm.inkling2_norm_triton import rms_norm
        else:
            from sglang.kernels.ops.layernorm.inkling2_rms_norm import rms_norm

        return rms_norm(
            x,
            self.weight,
            self.eps,
            pre_weight=self.pre_weight,
            output_dtype=self.output_dtype,
            out=out,
        )
