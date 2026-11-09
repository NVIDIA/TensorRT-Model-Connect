# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_vgg-owned FP8 self-quantization and TensorRT Q/DQ convolution context.

FP8 (``quantization="fp8"``) quantizes the original checkpoint at build time, because no
FP8 checkpoint exists for the timm VGG classifiers. Weights are quantized one
convolution at a time: ``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3
value), and the FP8 bytes are fed to the engine as a constant. FP8 weights are never stored.
The activation scale of each quantized convolution input needs real data, so it was
calibrated once offline with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``, max
calibration, ImageNet validation images) and is read from ``fp8_activation_scales.json``.
Users do not run calibration.

Each quantized convolution becomes::

    activation -> Quantize -> Dequantize(fp16) -> Convolution (+ bias)
    FP8 weight -> Dequantize(fp16) ------------------^

TensorRT fuses the pair into an e4m3 convolution kernel that writes fp16, with the bias in
its epilogue. Both Dequantize layers output fp16; the NaN scan on real ImageNet images and on
noise inputs found no overflow.

Quantized (recipe: no official FP8 config exists for a VGG classifier; the dense convolutions
are quantized and the input convolution is kept, as in the public UNet and ViT FP8 recipes
that keep ``conv_in`` and the patch embedding in 16 bit): every 3x3 convolution of the feature stack after the first
(``features.N``), and the two convolutional head layers ``pre_logits.fc1`` (7x7 over the pooled
feature map) and ``pre_logits.fc2`` (1x1). Kept in fp16: the first convolution (3 input
channels), the max pools, the ReLU and the classifier ``head.fc``.

A convolution whose calibrated input scale is more than 20 times the median of the same kind
has a massive activation: its entry in the scale file is ``null`` and it stays in fp16 on
purpose. A quantized convolution missing from the file is an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
# Smallest normal fp16 value; a smaller Q/DQ scale would lose precision as an fp16 constant.
_FP16_MIN_NORMAL = 2.0**-14

_SUPPORTED = ("fp8",)
# Quantized head convolutions, named as in the checkpoint.
HEAD_PROJECTIONS = ("pre_logits.fc1", "pre_logits.fc2")


@dataclass(frozen=True)
class _FP8Weight:
    """One convolution weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [out, in, kh, kw] raw fp8_e4m3 bytes
    weight_scale: float


@dataclass(frozen=True)
class _LayerScales:
    """Static activation scale and the FP8 weight of one convolution."""

    input_scale: float
    weight: _FP8Weight


def _scale_const(network, value: float):
    """One-element fp16 constant holding a Q/DQ scale (the Dequantize output type)."""
    data = np.asarray([value], dtype=np.float16)
    return network.add_constant((1,), trt.Weights(data)).get_output(0)


@dataclass(frozen=True)
class VggQuantContext:
    """Routes the quantized convolutions through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, module: str) -> bool:
        """True when ``module`` (for example ``features.2`` or ``pre_logits.fc1``) runs as FP8."""
        return module in self.scales

    def convolution(self, network, activation, module: str, bias: np.ndarray, *, padding=(0, 0)):
        """``activation`` convolved with the FP8 weight of ``module``, fp16 output with bias."""
        if activation.dtype != trt.float16:
            raise ValueError(f"timm_vgg FP8 needs an fp16 activation, got {activation.dtype}")
        entry = self.scales[module]
        act_scale = _scale_const(network, entry.input_scale)
        quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
        dequant = network.add_dequantize(quant.get_output(0), act_scale, trt.float16)
        packed = entry.weight.packed
        # trt.Weights holds a raw pointer, so the buffer must outlive the build.
        self.keep_alive.append(packed)
        w_fp8 = trt.Weights(trt.DataType.FP8, packed.ctypes.data, packed.size)
        w_const = network.add_constant(tuple(int(v) for v in packed.shape), w_fp8)
        w_dequant = network.add_dequantize(
            w_const.get_output(0), _scale_const(network, entry.weight.weight_scale), trt.float16
        )
        bias16 = np.ascontiguousarray(bias, dtype=np.float16)
        self.keep_alive.append(bias16)
        layer = network.add_convolution_nd(
            dequant.get_output(0),
            num_output_maps=int(packed.shape[0]),
            kernel_shape=(int(packed.shape[2]), int(packed.shape[3])),
            kernel=trt.Weights(trt.float16),
            bias=trt.Weights(bias16),
        )
        if layer is None:
            raise RuntimeError(f"TensorRT rejected the FP8 convolution {module}")
        layer.set_input(1, w_dequant.get_output(0))
        layer.padding_nd = tuple(padding)
        return layer.get_output(0)


def configure_fp8_build(builder_config, quant_ctx) -> None:
    """Keep per-layer tactic names in a quantized plan so FP8 lowering can be inspected."""
    if quant_ctx is None:
        return
    builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED


def _abs_sum(array) -> float:
    return float(np.abs(np.asarray(array, dtype=np.float32)).sum(dtype=np.float64))


def scale_key(conv_indices: list[int], image_size: tuple[int, int], weights) -> str:
    """Key of this model's entry in the activation-scale file.

    The key holds every config value that changes the activation ranges (the output width of
    every feature convolution, the head width, the input size). Checkpoints of the same
    architecture differ in training, so the key also holds two numeric weight sums (sum of
    absolute values, 5 significant digits): the first convolution and ``pre_logits.fc2``.
    """
    widths = [int(weights[f"features.{index}.weight"].shape[0]) for index in conv_indices]
    first = weights[f"features.{conv_indices[0]}.weight"]
    return (
        f"vgg-c{'x'.join(str(width) for width in widths)}"
        f"-h{int(weights['pre_logits.fc1.weight'].shape[0])}"
        f"-s{image_size[0]}x{image_size[1]}"
        f"-w{_abs_sum(first):.4e}-{_abs_sum(weights['pre_logits.fc2.weight']):.4e}"
    )


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes and ``scale = max(abs(weight)) / 448``; the same scale is
    the dequantize scale in the engine.
    """
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight_fp32 / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def refuse_quantized_checkpoint(model_dir: Path, tensor_names) -> None:
    """Raise when the checkpoint is already quantized; self-quantization needs the original."""
    raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    if (
        (isinstance(raw, dict) and "quantization_config" in raw)
        or (Path(model_dir) / "hf_quant_config.json").is_file()
        or any(str(name).endswith((".weight_scale", ".input_scale")) for name in tensor_names)
    ):
        raise ValueError(
            "the checkpoint is already quantized; FP8 self-quantization needs the original"
        )


def load_activation_scales(
    conv_indices: list[int], image_size: tuple[int, int], weights
) -> dict[str, float | None]:
    """Return the committed activation scales for this model config, or raise when none exist.

    An entry of ``None`` marks a convolution the calibration left unquantized on purpose.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"timm_vgg FP8 build requires {path}, which was not found") from error
    key = scale_key(conv_indices, image_size, weights)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def projection_names(conv_indices: list[int]) -> list[str]:
    """Checkpoint module names of every quantizable convolution of this model config."""
    return [f"features.{index}" for index in conv_indices[1:]] + list(HEAD_PROJECTIONS)


def calibrate_timm_vgg(
    conv_indices: list[int],
    image_size: tuple[int, int],
    weights,
    quantization: str,
    *,
    precision: str,
) -> VggQuantContext:
    """Quantize the loaded weights one convolution at a time.

    Pairs each with its stored activation scale. A convolution with no stored scale, or a
    stored scale for a convolution this model config does not quantize, is an error.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"timm_vgg supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if precision != "fp16":
        raise ValueError(f"timm_vgg FP8 needs precision fp16, got {precision!r}")
    activation_scales = load_activation_scales(conv_indices, image_size, weights)

    names = projection_names(conv_indices)
    unexpected = sorted(set(activation_scales) - set(names))
    if unexpected:
        raise ValueError(
            f"{ACTIVATION_SCALES_FILENAME} has scales for convolutions that are not quantized, "
            f"for example {unexpected[0]}; run calibrate_fp8_scales.py for this model config"
        )
    scales: dict[str, _LayerScales] = {}
    for name in names:
        if name not in activation_scales:
            raise ValueError(
                f"{name} has no activation scale in {ACTIVATION_SCALES_FILENAME}; "
                "run calibrate_fp8_scales.py"
            )
        input_scale = activation_scales[name]
        if input_scale is None:
            continue  # massive-activation convolution, left in fp16 by the calibration
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        key = f"{name}.weight"
        if key not in weights:
            raise ValueError(f"checkpoint has no {key}")
        packed, weight_scale = quantize_fp8_weight(np.asarray(weights[key], dtype=np.float32))
        if min(input_scale, weight_scale) < _FP16_MIN_NORMAL:
            raise ValueError(f"FP8 scale of {name} is below the smallest normal fp16 value")
        scales[name] = _LayerScales(float(input_scale), _FP8Weight(packed, weight_scale))
    if not scales:
        raise RuntimeError(
            "timm_vgg FP8 self-quantization matched no convolution in the checkpoint"
        )
    return VggQuantContext(scales=scales)
