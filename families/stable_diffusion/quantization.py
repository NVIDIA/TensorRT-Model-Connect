# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP8 and NVFP4 self-quantization of the SDXL-Turbo UNet (Conv2d and Linear layers).

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

NVFP4 (``quantization="nvfp4"``) is the FP8 recipe above with one change: the GEGLU projection
(``ff.net.0.proj``, the first feed-forward projection of every UNet transformer block, 70 layers)
is NVFP4 W4A4 instead of FP8 (``NVFP4_LAYERS``). Its weight is self-quantized at build time: E2M1
values with one e4m3 scale per 16 inputs and one ``weight_scale_2 = amax / (6 * 448)`` per
tensor. Its input is block-quantized to FP4 at run time (``add_dynamic_quantize``) with a static
global scale, the FP8 activation scale of the same layer divided by 6 (the FP8 scale is
``amax / 448``, the NVFP4 global scale ``amax / (6 * 448)``); no other calibration is needed.
The other 673 layers keep their FP8 recipe. TensorRT has no FP4 convolution, and the other
Linear layers in NVFP4 were slower than FP8 (each FP4 GEMM needs a dynamic quantize step) and
gave two to three times the FP8 image error; only the GEGLU projection made the plan faster than
FP8 (see the family README). The feed-forward output (``ff.net.2``) stays FP8.

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
E2M1_MAX = 6.0
NVFP4_BLOCK_SIZE = 16
NVFP4_MAX = E2M1_MAX * FP8_E4M3_MAX
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

SUPPORTED_QUANTIZATIONS = ("fp8", "nvfp4")
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
# Layers that are NVFP4 under quantization="nvfp4": the GEGLU projection of every transformer
# block. Every other layer of the FP8 recipe stays FP8.
NVFP4_LAYERS = r"(^|\.)transformer_blocks\.\d+\.ff\.net\.0\.proj$"
_NVFP4_LAYERS = re.compile(NVFP4_LAYERS)


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


def is_nvfp4(name: str, weight_rank: int) -> bool:
    """True when the module ``name`` is NVFP4 under ``quantization="nvfp4"``."""
    return weight_rank == 2 and is_quantized(name, weight_rank) and bool(_NVFP4_LAYERS.search(name))


def nvfp4_names(weights) -> list[str]:
    """Module names of the Linear layers of a UNet checkpoint that go to NVFP4."""
    return sorted(
        key[: -len(".weight")]
        for key, value in weights.items()
        if key.endswith(".weight") and is_nvfp4(key[: -len(".weight")], np.ndim(value))
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
            "stable_diffusion FP8 and NVFP4 quantization cover the SDXL-Turbo UNet only; "
            "SD 1.5 and the other model types are not calibrated"
        )
    if (
        cfg["scheduler_class"] != "EulerAncestralDiscreteScheduler"
        or cfg["timestep_spacing"] != "trailing"
    ):
        raise NotImplementedError(
            "stable_diffusion FP8 and NVFP4 quantization cover SDXL-Turbo only (Euler ancestral scheduler "
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


# Midpoints between neighbouring E2M1 magnitudes (0, 0.5, 1, 1.5, 2, 3, 4, 6). A value exactly on
# a midpoint goes to the neighbour with an even code: the first, third, fifth and seventh
# midpoint round down and the others round up.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_MIDPOINT_ROUNDS_UP = (False, True, False, True, False, True, False)


def _e2m1_codes(scaled: np.ndarray) -> np.ndarray:
    """Sign-magnitude E2M1 codes (0..15) of already block-scaled values."""
    magnitude = np.abs(scaled)
    code = np.zeros(scaled.shape, dtype=np.uint8)
    for midpoint, up in zip(_E2M1_MIDPOINTS, _E2M1_MIDPOINT_ROUNDS_UP, strict=True):
        code += (magnitude >= midpoint if up else magnitude > midpoint).astype(np.uint8)
    return code | ((scaled < 0).astype(np.uint8) << 3)


def quantize_nvfp4_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Quantize one ``[out, in]`` weight to NVFP4.

    Returns the packed bytes ``[out, in / 2]`` (first value in the low nibble), the raw e4m3
    block-scale bytes ``[out, in / 16]`` and ``weight_scale_2 = amax / (6 * 448)``. The block
    scale is ``block_amax / 6 / weight_scale_2`` cast to e4m3; an all-zero block (or one whose
    scale underflows e4m3) gets scale 1.
    """
    tensor = np.asarray(weight_fp32, dtype=np.float32)
    if tensor.ndim != 2:
        raise ValueError(f"expected a [out, in] weight, got shape {tensor.shape}")
    out_features, in_features = tensor.shape
    if in_features % NVFP4_BLOCK_SIZE:
        raise ValueError(f"in_features={in_features} is not a multiple of {NVFP4_BLOCK_SIZE}")
    amax = float(np.abs(tensor).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    # amax * fp32(1 / 2688), the ModelOpt operation (a reciprocal multiply).
    weight_scale_2 = float(np.float32(amax) * np.float32(1.0 / NVFP4_MAX))
    blocks = tensor.reshape(out_features, in_features // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    block_scale = np.minimum(
        np.abs(blocks).max(-1) / np.float32(E2M1_MAX) / np.float32(weight_scale_2),
        np.float32(FP8_E4M3_MAX),
    ).astype(ml_dtypes.float8_e4m3fn)
    scale_float = block_scale.astype(np.float32)
    scale_float = np.where(scale_float == 0, np.float32(1.0), scale_float)
    block_scale = scale_float.astype(ml_dtypes.float8_e4m3fn)
    scaled = blocks / (scale_float[..., None] * np.float32(weight_scale_2))
    code = _e2m1_codes(scaled).reshape(out_features, in_features)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return (
        np.ascontiguousarray(packed),
        np.ascontiguousarray(block_scale.view(np.uint8)),
        weight_scale_2,
    )


def load_activation_scales(key: str) -> dict[str, float]:
    """Return the committed activation scales for this model and resolution, or raise."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"stable_diffusion FP8 and NVFP4 builds require {path}, which was not found"
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
    """Routes the quantized UNet layers through FP8 (or NVFP4) Q/DQ; every other layer stays fp16."""

    def __init__(self, scales: dict[str, float], nvfp4: frozenset[str] = frozenset()) -> None:
        self.scales = dict(scales)
        # Linear layers (module names, a subset of ``scales``) that run NVFP4 instead of FP8.
        self.nvfp4 = frozenset(nvfp4)
        unknown = sorted(self.nvfp4 - set(self.scales))
        if unknown:
            raise ValueError(f"NVFP4 layers without an activation scale, e.g. {unknown[:3]}")
        # trt.Weights holds a raw pointer, so the buffers must outlive the build.
        self.keep_alive: list = []

    def should_quantize(self, name: str) -> bool:
        """True when the layer ``name`` (for example ``down_blocks.1.resnets.0.conv1``) is quantized."""
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
        rank = len(tuple(x.shape))
        if name in self.nvfp4:
            out_features = int(np.shape(weight)[0])
            product = self._nvfp4_matmul(network, x, weight, name)
            return self._add_bias(network, product, bias, rank, out_features, dtype)
        packed, weight_scale = quantize_fp8_weight(np.asarray(weight, dtype=np.float32))
        out_features, in_features = packed.shape
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
        return self._add_bias(network, product, bias, rank, out_features, dtype)

    def _add_bias(self, network, product, bias, rank: int, out_features: int, dtype):
        """``product + bias`` over the last axis; ``product`` when there is no bias."""
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

    def _nvfp4_matmul(self, network, x, weight, name: str):
        """``x @ weight.T`` with an NVFP4 weight and a dynamically block-quantized FP4 input.

        The input is flattened to ``[tokens, in_features]`` and quantized at run time with
        ``add_dynamic_quantize`` (per-16 e4m3 block scales, global scale = FP8 activation scale
        / 6). The weight is an FP4 constant that reaches the matmul through a double Dequantize
        (block scale times ``weight_scale_2``, then the weight). TensorRT fuses the pattern into a
        block-scaled FP4 GEMM.
        """
        out_dtype = x.dtype
        packed, block_scale, weight_scale_2 = quantize_nvfp4_weight(
            np.asarray(weight, dtype=np.float32)
        )
        out_features, in_features = int(np.shape(weight)[0]), int(np.shape(weight)[1])
        self.keep_alive += [packed, block_scale]
        w_const = network.add_constant(
            (out_features, in_features),
            trt.Weights(trt.DataType.FP4, packed.ctypes.data, out_features * in_features),
        ).get_output(0)
        bs_const = network.add_constant(
            (out_features, in_features // NVFP4_BLOCK_SIZE),
            trt.Weights(trt.DataType.FP8, block_scale.ctypes.data, int(block_scale.size)),
        ).get_output(0)
        w_scale = network.add_dequantize(
            bs_const, _scale_const(network, weight_scale_2), trt.float32
        )
        w_dq = network.add_dequantize(w_const, w_scale.get_output(0), trt.float32)
        w_dq.axis = 1
        w_compute = network.add_cast(w_dq.get_output(0), out_dtype).get_output(0)

        shape = tuple(int(v) for v in x.shape)
        flat = network.add_shuffle(x)
        flat.reshape_dims = (-1, in_features)
        global_scale = max(float(self.scales[name]), _MIN_ACTIVATION_SCALE) / E2M1_MAX
        act_global = _scale_const(network, global_scale)
        quantized = network.add_dynamic_quantize(
            flat.get_output(0), 1, NVFP4_BLOCK_SIZE, trt.DataType.FP4, trt.DataType.FP8
        )
        quantized.set_input(1, act_global)
        a_scale = network.add_dequantize(
            quantized.get_output(1), _scale_const(network, global_scale, out_dtype), out_dtype
        )
        a_dq = network.add_dequantize(quantized.get_output(0), a_scale.get_output(0), out_dtype)
        a_dq.axis = 1
        product = network.add_matrix_multiply(
            a_dq.get_output(0), trt.MatrixOperation.NONE, w_compute, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        if product.dtype != out_dtype:
            product = network.add_cast(product, out_dtype).get_output(0)
        back = network.add_shuffle(product)
        back.reshape_dims = shape[:-1] + (out_features,)
        return back.get_output(0)

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
    """Pair the committed activation scales with the checkpoint, after validating both.

    ``quantization="nvfp4"`` also selects the NVFP4 layers (``NVFP4_LAYERS``); the other layers
    keep the FP8 recipe.
    """
    if quantization not in SUPPORTED_QUANTIZATIONS:
        raise NotImplementedError(
            f"stable_diffusion supports quantization {list(SUPPORTED_QUANTIZATIONS)} or none; "
            f"got {quantization!r}"
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
    nvfp4 = frozenset()
    if quantization == "nvfp4":
        nvfp4 = frozenset(nvfp4_names(weights))
        if not nvfp4:
            raise ValueError(
                "NVFP4 quantization matched no GEGLU projection in the UNet checkpoint"
            )
        for name in sorted(nvfp4):
            in_features = int(np.shape(weights[f"{name}.weight"])[1])
            if in_features % NVFP4_BLOCK_SIZE:
                raise ValueError(
                    f"{name} has in_features={in_features}, not a multiple of {NVFP4_BLOCK_SIZE}"
                )
    return UNetQuantContext({name: float(activation_scales[name]) for name in names}, nvfp4=nvfp4)
