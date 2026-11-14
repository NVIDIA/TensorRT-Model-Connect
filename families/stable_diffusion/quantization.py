# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP8 self-quantization of the SDXL-Turbo UNet (Conv2d and Linear layers).

``quantization="fp8"`` quantizes the original fp16 UNet at build time; no FP8 checkpoint is
needed or written. Weights are quantized one layer at a time: ``scale = max(abs(weight)) / 448``
per tensor (448 is the largest finite e4m3 value), and the FP8 bytes are fed to the engine as
a constant. The activation scale of each layer input needs real denoising activations, so it
was calibrated once offline with ``calibrate_fp8_scales.py`` (static per-tensor scale, max over
128 prompts and the whole step schedule, at the native resolution) and is read from
``fp8_activation_scales.json``. Users do not run calibration.

Recipe: the public ModelOpt diffusers / TensorRT demo recipe for UNets. Every Conv2d and Linear
is quantized except the layers in ``EXCLUDED``. Linear layers alone give no speedup (a Quantize
and a cast kernel sit around each linear while the convolutions and norms next to it stay
fp16); the convolutions are what make the plan faster.

Each quantized layer becomes::

    activation -> Quantize -> Dequantize -> MatMul or Convolution -> + bias
    FP8 weight -> Dequantize --------------^

Both Dequantize layers output fp16, as in the public recipe. With fp32 Dequantize outputs
the convolutions and GEMMs write fp32 and a cast kernel follows each; on TensorRT 11.4 that
cancelled most of the speedup (1.04x instead of 1.19x for the 512 UNet, same images).
A 16-bit GEMM or convolution overflows once an output exceeds about
``65504 * input_scale * weight_scale``. The UNet layers stay far below that for the calibrated
inputs: every UNet output of the verification runs (real prompts, all steps) was scanned for NaN
and Inf and none was found.

Stays in fp16:

  ``conv_in``, ``conv_out``    The most error sensitive layers: with them in FP8 the 1-step image
                               is noise (LPIPS 0.6 against the fp16 engine).
  ``conv_shortcut``            1x1 convolution on the skip concatenation.
  resnet ``conv2``             Second 3x3 convolution of every resnet block. Not in the public
                               list: it was added after a sensitivity sweep, because it lowered
                               the 1-step LPIPS against the fp16 engine from 0.10 to 0.07
                               (512) and from 0.11 to 0.085 (1024) for 1 to 3 percent of speed.
  ``time_emb_proj``, ``time_embedding``, ``add_embedding``
                               Tiny GEMMs on the conditioning vector; no gain.
  attention score matmuls      Fused fp16 attention, as in the public recipe.
  GroupNorm, LayerNorm, the text encoders and the VAE.

Only the SDXL-Turbo UNet is covered. SD 1.5 and the other model types of this family are
refused (see ``scale_key``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import ml_dtypes
import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

_SUPPORTED = ("fp8",)
# Build-route knob that turns off TensorRT's fused dual-GEMM kernel.
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"
# Output type of every Dequantize; the matmul or convolution runs in this type, and a result of
# another type than the activation is cast back.
DEQUANTIZE_TYPE = trt.float16
_MIN_ACTIVATION_SCALE = 1e-4 / FP8_E4M3_MAX

# Layers that stay fp16, matched against the module name (the weight key without ".weight").
EXCLUDED = (
    r"^conv_in$",
    r"^conv_out$",
    r"(^|\.)conv_shortcut$",
    r"(^|\.)resnets\.\d+\.conv2$",
    r"(^|\.)time_emb_proj$",
    r"^time_embedding\.",
    r"^add_embedding\.",
)
_EXCLUDED = tuple(re.compile(pattern) for pattern in EXCLUDED)
# A Conv2d weight has 4 dimensions and a Linear weight 2; norms have 1.
_QUANTIZABLE_RANKS = (2, 4)


def is_quantized(name: str, weight_rank: int) -> bool:
    """True when the module ``name`` with a weight of ``weight_rank`` dimensions goes to FP8."""
    return weight_rank in _QUANTIZABLE_RANKS and not any(p.search(name) for p in _EXCLUDED)


def quantized_names(weights) -> list[str]:
    """Module names of the Conv2d and Linear layers of a UNet checkpoint that go to FP8."""
    return sorted(
        key[: -len(".weight")]
        for key, value in weights.items()
        if key.endswith(".weight") and is_quantized(key[: -len(".weight")], np.ndim(value))
    )


def scale_key(cfg: dict) -> str:
    """Key of this model and resolution in the activation-scale file, or raise when unsupported.

    The key names the SDXL-Turbo release: the SDXL UNet config together with the Euler ancestral
    scheduler with trailing timestep spacing that the one-to-four step distillation uses. SDXL
    base (Euler, leading spacing) and Lightning (Euler) have the same UNet config with other
    weights, and SD 1.5 has another config; none of them has calibrated scales.
    """
    unet = cfg["unet"]
    is_sdxl_unet = (
        cfg.get("text_encoder_2") is not None
        and unet.get("addition_embed_type") == "text_time"
        and unet["block_out_channels"] == [320, 640, 1280]
        and unet["layers_per_block"] == 2
        and unet["cross_attention_dim"] == 2048
    )
    if not is_sdxl_unet:
        raise NotImplementedError(
            "stable_diffusion FP8 quantization covers the SDXL-Turbo UNet only; "
            "SD 1.5 and the other model types are not calibrated"
        )
    if (
        cfg["scheduler_class"] != "EulerAncestralDiscreteScheduler"
        or cfg["timestep_spacing"] != "trailing"
    ):
        raise NotImplementedError(
            "stable_diffusion FP8 quantization covers SDXL-Turbo only (Euler ancestral scheduler "
            "with trailing spacing); SDXL base and Lightning are not calibrated"
        )
    return f"sdxl-turbo-s{int(cfg['image_size'])}"


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes and ``scale = max(abs(weight)) / 448``; the same scale is the
    dequantize scale in the engine.
    """
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight_fp32 / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def load_activation_scales(key: str) -> dict[str, float]:
    """Return the committed activation scales for this model and resolution, or raise."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"stable_diffusion FP8 build requires {path}, which was not found"
        ) from error
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: {sorted(table)}. "
            "Run calibrate_fp8_scales.py for this model and image size."
        )
    return table[key]


def _scale_const(network, value, scale_type=trt.float32):
    """A one-element Q/DQ scale constant of the given type."""
    constant = network.add_constant(
        (1,), trt.Weights(np.asarray([value], dtype=np.float32))
    ).get_output(0)
    if scale_type != trt.float32:
        constant = network.add_cast(constant, scale_type).get_output(0)
    return constant


class UNetQuantContext:
    """Routes the quantized UNet layers through FP8 Q/DQ; every other layer stays fp16."""

    def __init__(self, scales: dict[str, float]) -> None:
        self.scales = dict(scales)
        # trt.Weights holds a raw pointer, so the buffers must outlive the build.
        self.keep_alive: list = []

    def should_quantize(self, name: str) -> bool:
        """True when the layer ``name`` (for example ``down_blocks.1.resnets.0.conv1``) is FP8."""
        return name in self.scales

    def _weight_dequantize(self, network, packed: np.ndarray, scale: float):
        """FP8 constant of the given shape and its per-tensor Dequantize."""
        self.keep_alive.append(packed)
        constant = network.add_constant(
            tuple(packed.shape),
            trt.Weights(trt.DataType.FP8, packed.ctypes.data, int(packed.size)),
        ).get_output(0)
        return network.add_dequantize(
            constant, _scale_const(network, scale, DEQUANTIZE_TYPE), DEQUANTIZE_TYPE
        )

    def _activation_dequantize(self, network, x, name: str):
        """Quantize then Dequantize the layer input with its static scale."""
        scale = max(float(self.scales[name]), _MIN_ACTIVATION_SCALE)
        quantized = network.add_quantize(x, _scale_const(network, scale), trt.DataType.FP8)
        return network.add_dequantize(
            quantized.get_output(0), _scale_const(network, scale, DEQUANTIZE_TYPE), DEQUANTIZE_TYPE
        ).get_output(0)

    def linear(self, network, x, weight, bias, name: str, *, dtype=np.float16):
        """Linear over the last axis of ``x``; ``weight`` is [out_features, in_features]."""
        packed, weight_scale = quantize_fp8_weight(np.asarray(weight, dtype=np.float32))
        out_features, in_features = packed.shape
        rank = len(tuple(x.shape))
        rhs = np.ascontiguousarray(packed.T).reshape(
            (1,) * (rank - 2) + (in_features, out_features)
        )
        weight_dq = self._weight_dequantize(network, rhs, weight_scale)
        product = network.add_matrix_multiply(
            self._activation_dequantize(network, x, name),
            trt.MatrixOperation.NONE,
            weight_dq.get_output(0),
            trt.MatrixOperation.NONE,
        ).get_output(0)
        if product.dtype != x.dtype:
            product = network.add_cast(product, x.dtype).get_output(0)
        if bias is None:
            return product
        bias_shape = (1,) * (rank - 1) + (out_features,)
        bias_t = network.add_constant(
            bias_shape,
            trt.Weights(np.ascontiguousarray(np.asarray(bias).reshape(bias_shape), dtype=dtype)),
        ).get_output(0)
        if bias_t.dtype != product.dtype:
            bias_t = network.add_cast(bias_t, product.dtype).get_output(0)
        return network.add_elementwise(product, bias_t, trt.ElementWiseOperation.SUM).get_output(0)

    def conv2d(
        self,
        network,
        x,
        weight,
        bias,
        name: str,
        *,
        stride=(1, 1),
        padding=(0, 0),
        dtype=np.float16,
    ):
        """2-D convolution; ``weight`` is [out, in, kh, kw]. The FP8 weight is the kernel input."""
        packed, weight_scale = quantize_fp8_weight(np.asarray(weight, dtype=np.float32))
        out_channels, _, kernel_h, kernel_w = packed.shape
        weight_dq = self._weight_dequantize(network, packed, weight_scale)
        # The bias stays in the type of the Dequantize outputs, so the convolution is typed
        # consistently; the result is cast back to the activation type below.
        bias_np = np.float32 if DEQUANTIZE_TYPE == trt.float32 else np.float16
        bias_w = (
            trt.Weights(np.ascontiguousarray(np.asarray(bias), dtype=bias_np))
            if bias is not None
            else trt.Weights()
        )
        if bias is not None:
            self.keep_alive.append(bias_w)
        conv = network.add_convolution_nd(
            self._activation_dequantize(network, x, name),
            out_channels,
            (kernel_h, kernel_w),
            trt.Weights(),
            bias_w,
        )
        conv.set_input(1, weight_dq.get_output(0))
        conv.stride_nd = stride
        conv.padding_nd = padding
        out = conv.get_output(0)
        if out.dtype != x.dtype:
            out = network.add_cast(out, x.dtype).get_output(0)
        return out


def configure_build(trt_config) -> None:
    """Apply the FP8 build options and keep layer names for kernel inspection."""
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # TensorRT's fused dual-GEMM kernel fails to compile with scalar-scale FP8.
    # The route is set only when this TensorRT build lists the knob.
    if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def unet_quant_context(weights, cfg: dict, *, quantization: str) -> UNetQuantContext:
    """Pair the committed activation scales with the checkpoint, after validating both."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"stable_diffusion supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    activation_scales = load_activation_scales(scale_key(cfg))
    names = quantized_names(weights)
    missing = [name for name in names if name not in activation_scales]
    if missing:
        raise ValueError(
            f"no activation scale for {len(missing)} UNet layers, e.g. {missing[:3]}; "
            "run calibrate_fp8_scales.py again"
        )
    unknown = sorted(set(activation_scales) - set(names))
    if unknown:
        raise ValueError(
            f"activation scales for layers the checkpoint does not quantize, e.g. {unknown[:3]}"
        )
    for name in names:
        scale = activation_scales[name]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
    return UNetQuantContext({name: float(activation_scales[name]) for name in names})
