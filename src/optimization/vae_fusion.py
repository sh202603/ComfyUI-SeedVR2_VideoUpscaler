"""
VAE Fused Path Module for SeedVR2

Opt-in acceleration (fused_vae) that swaps the execution path of the VAE without
changing its model definition, using comfy-kitchen kernels:
- GroupNorm + SiLU: one fused per-frame pass (group_norm_silu_pad3d), NDHWC output
- Causal 3D convolutions: fp16-accumulate kernel (fp16_conv3d), NDHWC in and out
- The tensors between them stay in NDHWC (channels_last_3d), including the frames
  a causal convolution carries over from the previous temporal slice and the
  pixel shuffle of the upsamplers
- No intermediate copies in front of a convolution: GroupNorm + SiLU writes straight
  into the zero-padded buffer the convolution reads (which also holds the carried-over
  frames), and the residual of a ResNet block is added in the convolution's epilogue

On GeForce GPUs the tensor core rate with fp32 accumulation is half of the fp16
accumulation rate, so the convolution kernel itself is ~2x faster. It only pays off
when its input is already NDHWC, hence the fused GroupNorm + SiLU in front of it.

Requirements: NVIDIA CUDA device and comfy-kitchen with its CUDA backend. The kernels
need fp16 inputs and weights, so the VAE runs in fp16 instead of the compute dtype.
When comfy-kitchen is missing, lacks the kernels or fails the kernel probe (e.g. after
an incompatible update), the standard path runs unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

_REQUIRED_KERNELS = ("fp16_conv3d", "group_norm_silu_pad3d")

# Lazily imported comfy_kitchen module (None until the fused path is requested)
_kitchen = None

# Kernel probe results per device (None = passed, otherwise the failure reason)
_probe_results = {}


def get_vae_fused_path_unsupported_reason(device: torch.device) -> Optional[str]:
    """
    Check whether the VAE fused path can run on the given device.
    Imports comfy-kitchen on first use.

    Args:
        device: Inference device of the VAE

    Returns:
        None if supported, otherwise a human-readable reason
    """
    global _kitchen

    device = torch.device(device)
    if device.type != "cuda" or getattr(torch.version, "hip", None) is not None:
        return "requires an NVIDIA CUDA device"

    if _kitchen is None:
        try:
            import comfy_kitchen
        except Exception as e:
            return f"requires comfy-kitchen (pip install comfy-kitchen): {e}"

        # Without the CUDA backend comfy-kitchen falls back to plain torch ops,
        # which would be slower than the standard path
        try:
            backend = comfy_kitchen.list_backends().get("cuda", {})
        except Exception as e:
            return f"could not query comfy-kitchen backends: {e}"
        if not backend.get("available", False) or backend.get("disabled", False):
            return f"comfy-kitchen CUDA backend unavailable ({backend.get('unavailable_reason')})"
        missing = [name for name in _REQUIRED_KERNELS if name not in backend.get("capabilities", [])]
        if missing:
            return f"comfy-kitchen CUDA backend lacks {', '.join(missing)} (update comfy-kitchen)"
        _kitchen = comfy_kitchen

    if device not in _probe_results:
        _probe_results[device] = _probe_kernels(device)
    return _probe_results[device]


def _probe_kernels(device: torch.device) -> Optional[str]:
    """
    Run the kernels the way the fused path uses them on a tiny input and compare with torch.

    Guards against comfy-kitchen versions whose kernels changed signature or semantics:
    zero spatial padding written into a frame-offset view, and a residual added in the
    convolution epilogue.

    Returns:
        None if the kernels behave as expected, otherwise a human-readable reason
    """
    try:
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(1, 16, 2, 6, 6, generator=generator).to(device, torch.float16)
        norm = nn.GroupNorm(4, 16).to(device, torch.float16)
        conv_weight = (torch.randn(8, 16, 3, 3, 3, generator=generator) * 0.05).to(device, torch.float16)
        conv_bias = (torch.randn(8, generator=generator) * 0.1).to(device, torch.float16)
        residual = torch.randn(1, 8, 2, 6, 6, generator=generator).to(device, torch.float16)

        # One carried-over frame slot in front, filled with the first frame
        buffer = torch.empty((1, 16, 4, 8, 8), dtype=torch.float16, device=device,
                             memory_format=torch.channels_last_3d)
        fused_group_norm_silu(norm, x, pad=(1, 1), out=buffer[:, :, 2:])
        buffer[:, :, :2] = buffer[:, :, 2:3]
        output = fp16_accum_conv3d(buffer, conv_weight, conv_bias, (1, 1, 1), (0, 0, 0), residual=residual)

        frames = x.float().transpose(1, 2).reshape(2, 16, 6, 6)
        frames = F.silu(F.group_norm(frames, 4, norm.weight.float(), norm.bias.float(), norm.eps))
        reference = F.pad(frames.reshape(1, 2, 16, 6, 6).transpose(1, 2), (1, 1, 1, 1))
        reference = torch.cat([reference[:, :, :1], reference[:, :, :1], reference], dim=2)
        reference = F.conv3d(reference, conv_weight.float(), conv_bias.float()) + residual.float()

        if output.shape != reference.shape:
            return f"comfy-kitchen kernel probe failed: output shape {tuple(output.shape)}"
        error = (output.float() - reference).abs().max().item()
        if not error <= 0.05 * reference.abs().max().item():
            return f"comfy-kitchen kernel probe failed: output differs from torch (max error {error:.3f})"
    except Exception as e:
        return f"comfy-kitchen kernel probe failed: {e}"
    return None


def is_fusable_conv(conv: nn.Module) -> bool:
    """Shapes the fp16-accumulate kernel serves; other convolutions keep the standard path."""
    return (
        conv.groups == 1
        and tuple(conv.dilation) == (1, 1, 1)
        and conv.padding_mode == "zeros"
        and (conv.in_channels % 8 == 0 or conv.in_channels < 8)
        and conv.out_channels % 8 == 0
    )


def enable_vae_fused_path(vae: nn.Module) -> Tuple[int, int]:
    """
    Switch a materialized VAE to the fused execution path.

    Sets the `fused_path` flag on the causal convolutions the kernel serves, on the
    modules that run GroupNorm + SiLU in front of them and on the upsamplers, and stores
    the convolution weights in NDHWC. The model definition and state dict keys are unchanged.
    Modules check the remaining conditions per call and otherwise run their standard path.

    Args:
        vae: Materialized VAE with fp16 weights

    Returns:
        Tuple of (convolutions switched, norm + activation owners switched)
    """
    # Import here to avoid circular dependency
    from ..models.video_vae_v3.modules.attn_video_vae import Decoder3D, Encoder3D, ResnetBlock3D, Upsample3D
    from ..models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d

    convs = 0
    owners = 0
    for module in vae.modules():
        if isinstance(module, InflatedCausalConv3d):
            if is_fusable_conv(module):
                module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last_3d)
                module.fused_path = True
                convs += 1
        elif isinstance(module, (ResnetBlock3D, Encoder3D, Decoder3D)):
            module.fused_path = True
            owners += 1
        elif isinstance(module, Upsample3D):
            # Upscale convolution and pixel shuffle feed the padded buffer of the convolution that follows
            upscale_conv = module.upscale_conv
            upscale_conv.weight.data = upscale_conv.weight.data.contiguous(memory_format=torch.channels_last_3d)
            module.fused_path = True
    vae.fused_path = True
    return convs, owners


def is_fusable_norm_silu(norm_layer: nn.Module, act: nn.Module, x: torch.Tensor) -> bool:
    """Whether norm_layer followed by act on x can run as one fused GroupNorm + SiLU pass."""
    return (
        type(norm_layer) is nn.GroupNorm
        and norm_layer.affine
        and isinstance(act, nn.SiLU)
        and x.ndim == 5
        and x.dtype == torch.float16
    )


def fused_group_norm_silu(norm_layer: nn.GroupNorm, x: torch.Tensor, pad: Tuple[int, int] = (0, 0),
                          out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Per-frame GroupNorm followed by SiLU in one pass, optionally zero-padded spatially.

    Args:
        norm_layer: GroupNorm applied per frame
        x: [B, C, T, H, W] fp16 tensor
        pad: Zero padding (H, W) applied on both sides
        out: Optional NDHWC tensor of the padded shape to write into. For a batch of one
             it may be a temporal slice of a longer buffer

    Returns:
        [B, C, T, H + 2 * pad[0], W + 2 * pad[1]] fp16 tensor in NDHWC (channels_last_3d);
        `out` if given
    """
    return _kitchen.group_norm_silu_pad3d(
        x, norm_layer.weight, norm_layer.bias, norm_layer.num_groups, norm_layer.eps,
        pad=(pad[1], pad[1], pad[0], pad[0], 0), silu=True, zero_pad=True, out=out
    )


def fp16_accum_conv3d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
                      stride: Tuple[int, int, int], padding: Tuple[int, int, int],
                      residual: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Zero-padded 3D convolution with fp16 accumulation.

    Args:
        input: [B, C, T, H, W] fp16 tensor (NDHWC avoids a layout conversion)
        weight: fp16 convolution weight
        bias: fp16 bias or None
        stride: Convolution stride (T, H, W)
        padding: Zero padding (T, H, W) applied on both sides
        residual: Optional fp16 tensor of the output shape, added in the kernel's epilogue

    Returns:
        fp16 convolution output in NDHWC (channels_last_3d)
    """
    if any(padding):
        input = F.pad(input, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]))
    return _kitchen.fp16_conv3d(input, weight, bias, residual=residual, stride=stride)
