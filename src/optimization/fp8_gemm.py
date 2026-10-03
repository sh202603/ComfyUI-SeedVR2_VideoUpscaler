"""
FP8 GEMM Module for SeedVR2

Opt-in acceleration (fp8_dit) that runs the linear layers inside the DiT transformer
blocks as FP8 (E4M3) matrix multiplications through torch._scaled_mm.

The *_fp8_e4m3fn checkpoints only store their weights in FP8: every linear call casts
the weight back to the compute dtype and multiplies in BF16. Here the GEMM itself runs
in FP8, which is ~2.5x faster on GPUs where BF16 with FP32 accumulation is half rate.

Key Features:
- Weights: quantized once with a per-tensor scale when the model is materialized
  (FP8 E4M3 checkpoints are used as stored, with scale 1)
- Activations: dynamic per-tensor scale (abs-max) on every call, computed on the GPU
  by two Triton kernels (abs-max reduction, then scale + saturate + cast; no host sync)
- GEMM: torch._scaled_mm with BF16 output; the bias is added outside the GEMM
- MLP fusion: the activation between the MLP projections is folded into the
  quantization of the next GEMM's input (SwiGLU for 3B, tanh-GELU for 7B)

The swap is done by walking the block modules from outside, so the dit_3b and dit_7b
model definitions stay untouched. It frees the original weights: changing the setting
requires reloading the model.

The quantization kernels follow flashvsr-sm89-ops
(https://github.com/aireet/flashvsr-sm89-ops, Apache-2.0).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .compatibility import TRITON_AVAILABLE

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0

# Activation folded into the quantization of a GEMM input
ACT_NONE = 0
ACT_GELU = 1    # gelu_tanh(x + bias)
ACT_SWIGLU = 2  # silu(x) * other

_BLOCK = 8192
_GELU_C0 = 0.7978845608028654  # sqrt(2/pi), as in aten's tanh GELU

try:
    if not TRITON_AVAILABLE:
        raise ImportError("Triton is not installed")
    import triton
    import triton.language as tl

    @triton.jit
    def _round_bf16(x):
        # Round to bf16 like the eager op whose output is folded into the kernel
        return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _load_act(x_ptr, y_ptr, offs, mask, N, ACT: tl.constexpr, HAS_BIAS: tl.constexpr, C0: tl.constexpr):
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if ACT == 2:
            y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            x = _round_bf16(_round_bf16(x * tl.sigmoid(x)) * y)
        else:
            if HAS_BIAS:
                x = _round_bf16(x + tl.load(y_ptr + offs % N, mask=mask, other=0.0).to(tl.float32))
            if ACT == 1:
                # tanh(u) = 1 - 2 / (exp(2u) + 1)
                u = C0 * (x + 0.044715 * x * x * x)
                x = _round_bf16(0.5 * x * (2.0 - 2.0 / (tl.exp(2.0 * u) + 1.0)))
        return tl.where(mask, x, 0.0)

    @triton.jit
    def _absmax_kernel(x_ptr, y_ptr, amax_ptr, n, N, ACT: tl.constexpr, HAS_BIAS: tl.constexpr,
                       C0: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        x = _load_act(x_ptr, y_ptr, offs, offs < n, N, ACT, HAS_BIAS, C0)
        tl.atomic_max(amax_ptr, tl.max(tl.abs(x), axis=0))

    @triton.jit
    def _quant_kernel(x_ptr, y_ptr, amax_ptr, scale_ptr, out_ptr, n, N, ACT: tl.constexpr,
                      HAS_BIAS: tl.constexpr, C0: tl.constexpr, FP8_MAX: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        s = tl.maximum(tl.load(amax_ptr), 1e-12) / FP8_MAX
        if pid == 0:
            tl.store(scale_ptr, s)
        x = _load_act(x_ptr, y_ptr, offs, mask, N, ACT, HAS_BIAS, C0)
        # Saturate: the float -> e4m3fn cast turns overflow into NaN
        v = tl.minimum(tl.maximum(x / s, -FP8_MAX), FP8_MAX)
        tl.store(out_ptr + offs, v.to(tl.float8e4nv), mask=mask)

    FP8_KERNELS_AVAILABLE = True
except Exception:
    FP8_KERNELS_AVAILABLE = False


def get_fp8_gemm_unsupported_reason(device: torch.device, compute_dtype: torch.dtype) -> Optional[str]:
    """
    Check whether FP8 GEMM can run on the given device.

    Args:
        device: Inference device of the DiT
        compute_dtype: Pipeline compute dtype

    Returns:
        None if supported, otherwise a human-readable reason
    """
    device = torch.device(device)
    if device.type != "cuda" or getattr(torch.version, "hip", None) is not None:
        return "requires an NVIDIA CUDA device"
    if torch.cuda.get_device_capability(device) < (8, 9):
        return "requires compute capability 8.9 or higher (RTX 40 series or newer)"
    if compute_dtype != torch.bfloat16:
        return f"requires bfloat16 compute dtype (got {compute_dtype})"
    if not FP8_KERNELS_AVAILABLE:
        return "requires Triton"
    if not hasattr(torch, "_scaled_mm"):
        return "requires torch._scaled_mm"
    return None


# Triton kernel launches are opaque to torch.compile: break the graph here instead of tracing into them
@torch._dynamo.disable
def quantize_fp8(x: torch.Tensor, other: Optional[torch.Tensor] = None,
                 act: int = ACT_NONE) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a GEMM input to FP8 with a dynamic per-tensor scale.

    Args:
        x: Contiguous [M, K] floating point tensor
        other: ACT_GELU: optional [K] bias added before the GELU.
               ACT_SWIGLU: [M, K] tensor multiplied with silu(x)
        act: Activation folded into the quantization (ACT_NONE, ACT_GELU, ACT_SWIGLU),
             so its output is never materialized

    Returns:
        Tuple of (FP8 [M, K] tensor, fp32 scalar scale on the GPU)
    """
    n = x.numel()
    amax = torch.zeros(1, dtype=torch.float32, device=x.device)
    scale = torch.empty((), dtype=torch.float32, device=x.device)
    out = torch.empty(x.shape, dtype=FP8, device=x.device)
    grid = (triton.cdiv(n, _BLOCK),)
    has_bias = act != ACT_SWIGLU and other is not None
    y = x if other is None else other
    N = x.shape[-1]
    _absmax_kernel[grid](x, y, amax, n, N, ACT=act, HAS_BIAS=has_bias, C0=_GELU_C0,
                         BLOCK=_BLOCK, num_warps=8)
    _quant_kernel[grid](x, y, amax, scale, out, n, N, ACT=act, HAS_BIAS=has_bias, C0=_GELU_C0,
                        FP8_MAX=FP8_MAX, BLOCK=_BLOCK, num_warps=8)
    return out, scale


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    """View the input as a contiguous [M, K] matrix."""
    return x.reshape(-1, x.shape[-1]).contiguous()


class FP8Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear: per-tensor FP8 weight, dynamic per-tensor
    FP8 activation, BF16 output.

    The bias is not given to torch._scaled_mm: with a bias epilogue cuBLASLt picks
    a slower kernel than the bias-free one.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight.detach()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        if weight.dtype == FP8:
            # FP8 checkpoints already hold E4M3 codes
            qweight = weight
            w_scale = torch.ones((), dtype=torch.float32, device=weight.device)
        else:
            weight = weight.float()
            w_scale = (weight.abs().amax() / FP8_MAX).clamp(min=1e-12)
            qweight = (weight / w_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        # Stored as integer views: Module.to(dtype=...) casts every floating point
        # buffer (float8 included) and would destroy the FP8 codes / fp32 scale
        self.register_buffer("qweight", qweight.contiguous().view(torch.uint8))
        self.register_buffer("w_scale", w_scale.view(torch.int32))
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().to(torch.bfloat16))

    def gemm(self, x8: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
        """FP8 GEMM without the bias."""
        return torch._scaled_mm(x8, self.qweight.view(FP8).t(), scale_a=x_scale,
                                scale_b=self.w_scale.view(torch.float32), out_dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gemm(*quantize_fp8(_flatten_tokens(x)))
        if self.bias is not None:
            y.add_(self.bias)
        return y.view(*x.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class FP8SwiGLUMLP(nn.Module):
    """
    FP8 replacement for SwiGLUMLP (3B): proj_out(silu(proj_in_gate(x)) * proj_in(x)).

    The two input projections share one quantized input, and silu(gate) * hidden is
    folded into the quantization of proj_out's input.
    """

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.proj_in_gate = FP8Linear(mlp.proj_in_gate)
        self.proj_out = FP8Linear(mlp.proj_out)
        self.proj_in = FP8Linear(mlp.proj_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x8, x_scale = quantize_fp8(_flatten_tokens(x))
        gate = self.proj_in_gate.gemm(x8, x_scale)
        hidden = self.proj_in.gemm(x8, x_scale)
        if self.proj_in_gate.bias is not None:
            gate.add_(self.proj_in_gate.bias)
        if self.proj_in.bias is not None:
            hidden.add_(self.proj_in.bias)
        y = self.proj_out.gemm(*quantize_fp8(gate, hidden, act=ACT_SWIGLU))
        if self.proj_out.bias is not None:
            y.add_(self.proj_out.bias)
        return y.view(*x.shape[:-1], self.proj_out.out_features)


class FP8GELUMLP(nn.Module):
    """
    FP8 replacement for MLP (7B): proj_out(gelu_tanh(proj_in(x))).

    proj_in's bias and the GELU are folded into the quantization of proj_out's input.
    """

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.proj_in = FP8Linear(mlp.proj_in)
        self.proj_out = FP8Linear(mlp.proj_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.proj_in.gemm(*quantize_fp8(_flatten_tokens(x)))
        y = self.proj_out.gemm(*quantize_fp8(hidden, self.proj_in.bias, act=ACT_GELU))
        if self.proj_out.bias is not None:
            y.add_(self.proj_out.bias)
        return y.view(*x.shape[:-1], self.proj_out.out_features)


def _is_convertible_linear(module: Optional[nn.Module]) -> bool:
    """Plain nn.Linear with dimensions torch._scaled_mm accepts (GGUF layers are skipped)."""
    return (
        type(module) is nn.Linear
        and module.in_features % 16 == 0 and module.out_features % 16 == 0
        and module.weight.is_floating_point()
        and not hasattr(module.weight, 'tensor_type')
    )


def _convert_module(module: nn.Module) -> Tuple[Optional[nn.Module], int]:
    """Build the FP8 replacement for a module, or (None, 0) if it is not a conversion target."""
    name = type(module).__name__
    if name == 'SwiGLUMLP' and all(
        _is_convertible_linear(getattr(module, attr, None)) for attr in ('proj_in_gate', 'proj_in', 'proj_out')
    ):
        return FP8SwiGLUMLP(module), 3
    if name == 'MLP' and getattr(getattr(module, 'act', None), 'approximate', None) == 'tanh' and all(
        _is_convertible_linear(getattr(module, attr, None)) for attr in ('proj_in', 'proj_out')
    ):
        return FP8GELUMLP(module), 2
    if _is_convertible_linear(module):
        return FP8Linear(module), 1
    return None, 0


def _convert_children(module: nn.Module) -> int:
    converted = 0
    for name, child in list(module.named_children()):
        replacement, count = _convert_module(child)
        if replacement is not None:
            # The original module (and its weight) is freed as soon as its FP8 copy exists
            setattr(module, name, replacement)
            converted += count
        else:
            converted += _convert_children(child)
    return converted


def convert_dit_to_fp8_gemm(dit_model: nn.Module, debug: Optional['Debug'] = None) -> int:
    """
    Swap the linear layers of the DiT transformer blocks for FP8 GEMM versions.

    Targets the QKV projections, output projections and MLP layers of both the video
    and text branches. Input/output embedding layers are left untouched.

    Args:
        dit_model: Materialized DiT model (unwrapped NaDiT with a `blocks` ModuleList)
        debug: Debug instance for logging

    Returns:
        Number of GEMMs converted
    """
    blocks = getattr(dit_model, 'blocks', None)
    if blocks is None:
        return 0

    if debug:
        debug.start_timer("fp8_gemm_convert")
    with torch.no_grad():
        converted = _convert_children(blocks)
    if debug:
        debug.end_timer("fp8_gemm_convert", f"FP8 GEMM conversion ({converted} linear layers)")

    return converted
