# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_resnet-owned FP8 self-quantization and TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) quantizes the original checkpoint at build time, because no
FP8 checkpoint exists for the timm ResNet classifiers. Each quantized convolution is
quantized on its own: the batch norm after it is folded into the convolution first
(``weight * gamma / sqrt(var + eps)``, shift as the convolution bias, in fp32), then every
output channel gets its own weight scale ``max(abs(channel)) / 448`` (448 is the largest
finite e4m3 value) and the FP8 bytes are fed to the engine as a constant. FP8 weights are
never stored. The activation scale of each convolution input needs real data, so it was
calibrated once offline with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``,
max calibration, ImageNet validation images) and is read from ``fp8_activation_scales.json``.
Users do not run calibration.

Each quantized convolution becomes::

    activation -> Quantize -> Dequantize(fp16) -> Convolution(+ folded bias)
    FP8 weight -> Dequantize(fp16, one fp32 scale per output channel) --^

Both Dequantize layers output fp16 (the weight scales are fp32 constants). TensorRT runs the convolution as an e4m3 implicit GEMM
and writes its output in e4m3 when the next convolution quantizes it. The block input is
quantized once: the first convolution, the downsample convolution (same input, same scale)
and the identity branch of the residual add all read the same Dequantize output, so the add
and ReLU fuse into the epilogue of the last convolution of the block.

Quantized (recipe: the convolutions of the residual stages, as the public CNN recipes):
``conv1``, ``conv2``, ``conv3`` (``conv1``, ``conv2`` for basic blocks) and
``downsample.0`` of every block of the stages ``layer2`` to ``layer4``. Kept in fp16: the
stem convolution (3 input channels), the whole first stage ``layer1`` and the classifier
head. ``layer1`` stays in fp16 because it carries most of the FP8 error of
``resnet50.a1_in1k``: with ``layer1`` quantized too, the KL divergence to the PyTorch model
on held-out ImageNet images is 0.19 and top-1 drops 2.7 points; without it 0.015 and 0.3
to 0.9 point (a PyTorch fake-quant study of each stage and each convolution).

A convolution whose calibrated input scale is more than 20 times the median of the same
convolution kind has a massive activation: its entry in the scale file is ``null`` and it
stays in fp16 on purpose. A convolution missing from the file is an error.
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
# Smallest normal fp16 value; a smaller activation scale would lose precision as an fp16
# constant (the per-channel weight scales are fp32 constants).
_FP16_MIN_NORMAL = 2.0**-14

_SUPPORTED = ("fp8",)
_STAGES = ("layer1", "layer2", "layer3", "layer4")
# Stages whose convolutions run as FP8; the stem, layer1 and the head stay fp16 (see above).
QUANTIZED_STAGES = ("layer2", "layer3", "layer4")
BOTTLENECK_CONVOLUTIONS = ("conv1", "conv2", "conv3")
BASIC_CONVOLUTIONS = ("conv1", "conv2")
DOWNSAMPLE = "downsample.0"


@dataclass(frozen=True)
class _FP8Conv:
    """One BN-folded convolution weight as raw fp8_e4m3 bytes with one scale per channel."""

    packed: np.ndarray  # [out, in / groups, kh, kw] raw fp8_e4m3 bytes
    weight_scale: np.ndarray  # [out] float32
    bias: np.ndarray  # [out] float32, the folded batch-norm shift


@dataclass(frozen=True)
class _ConvScales:
    """Static activation scale and the FP8 weight of one convolution."""

    input_scale: float
    weight: _FP8Conv


def _scale_const(network, values, dtype):
    """Constant holding a Q/DQ scale (one element, or one per output channel)."""
    data = np.ascontiguousarray(np.atleast_1d(np.asarray(values, dtype=dtype)))
    return network.add_constant(data.shape, trt.Weights(data)).get_output(0), data


@dataclass(frozen=True)
class ResnetQuantContext:
    """Routes the quantized convolutions through Q/DQ; the others stay unquantized."""

    scales: dict[str, _ConvScales]
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, module: str) -> bool:
        """True when ``module`` (for example ``layer2.0.conv1``) runs as FP8."""
        return module in self.scales

    def input_scale(self, module: str) -> float:
        """The static activation scale of ``module``."""
        return self.scales[module].input_scale

    def quantize_input(self, network, activation, module: str):
        """``activation -> Quantize -> Dequantize(fp16)`` with the input scale of ``module``."""
        if activation.dtype != trt.float16:
            raise ValueError(f"timm_resnet FP8 needs an fp16 activation, got {activation.dtype}")
        scale, data = _scale_const(network, self.scales[module].input_scale, np.float16)
        self.keep_alive.append(data)
        quant = network.add_quantize(activation, scale, trt.DataType.FP8)
        return network.add_dequantize(quant.get_output(0), scale, trt.float16).get_output(0)

    def convolution(self, network, dequantized_activation, module: str, *, stride, padding, groups):
        """The BN-folded convolution of ``module`` with its FP8 weight and fp16 bias."""
        weight = self.scales[module].weight
        packed = weight.packed
        out_channels = int(packed.shape[0])
        # trt.Weights holds a raw pointer, so the buffers must outlive the build.
        bias = np.ascontiguousarray(weight.bias, dtype=np.float16)
        self.keep_alive.extend([packed, bias])
        w_const = network.add_constant(
            tuple(int(dim) for dim in packed.shape),
            trt.Weights(trt.DataType.FP8, packed.ctypes.data, packed.size),
        ).get_output(0)
        # fp32 scales: a per-channel weight scale can be below the smallest normal fp16 value.
        scale, data = _scale_const(network, weight.weight_scale, np.float32)
        self.keep_alive.append(data)
        w_dequant = network.add_dequantize(w_const, scale, trt.float16)
        w_dequant.axis = 0
        conv = network.add_convolution_nd(
            dequantized_activation,
            num_output_maps=out_channels,
            kernel_shape=(int(packed.shape[2]), int(packed.shape[3])),
            kernel=trt.Weights(),
            bias=trt.Weights(bias),
        )
        conv.set_input(1, w_dequant.get_output(0))
        conv.stride_nd = stride
        conv.padding_nd = padding
        conv.num_groups = groups
        return conv.get_output(0)


def configure_fp8_build(builder_config, quant_ctx) -> None:
    """Keep per-layer tactic names in a quantized plan so FP8 lowering can be inspected."""
    if quant_ctx is None:
        return
    builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED


def _abs_sum(array) -> float:
    return float(np.abs(np.asarray(array, dtype=np.float32)).sum(dtype=np.float64))


def scale_key(cfg: dict, weights) -> str:
    """Key of this model's entry in the activation-scale file.

    The key holds every config value that changes the activation ranges (blocks per stage,
    block type, stem width, width and groups of the first block, input size). Checkpoints of
    the same architecture differ in training, so the key also holds the numeric weight sums
    (sum of absolute values, 5 significant digits) of the stem convolution and of the first
    block's ``conv1``.
    """
    stem = weights["conv1.weight"]
    first = weights["layer1.0.conv1.weight"]
    second = weights["layer1.0.conv2.weight"]
    groups = max(1, int(first.shape[0]) // int(second.shape[1])) if cfg["bottleneck"] else 1
    kind = "bottleneck" if cfg["bottleneck"] else "basic"
    return (
        f"resnet-b{'x'.join(str(count) for count in cfg['blocks'])}-{kind}"
        f"-c{stem.shape[0]}-w{first.shape[0]}-g{groups}"
        f"-s{cfg['image_size_h']}x{cfg['image_size_w']}"
        f"-w{_abs_sum(stem):.4e}-{_abs_sum(first):.4e}"
    )


def conv_names(cfg: dict, weights) -> list[str]:
    """Checkpoint module names of every quantizable convolution of this model config."""
    convolutions = BOTTLENECK_CONVOLUTIONS if cfg["bottleneck"] else BASIC_CONVOLUTIONS
    names = []
    for stage, count in zip(_STAGES, cfg["blocks"]):
        if stage not in QUANTIZED_STAGES:
            continue
        for block in range(count):
            prefix = f"{stage}.{block}"
            names.extend(f"{prefix}.{conv}" for conv in convolutions)
            if f"{prefix}.{DOWNSAMPLE}.weight" in weights:
                names.append(f"{prefix}.{DOWNSAMPLE}")
    return names


def norm_name(module: str) -> str:
    """The batch norm that follows a convolution (``conv2`` -> ``bn2``, downsample.0 -> .1)."""
    if module.endswith(DOWNSAMPLE):
        return module.removesuffix("0") + "1"
    prefix, conv = module.rsplit(".", 1)
    return f"{prefix}.bn{conv.removeprefix('conv')}"


def fold_batch_norm(weight, gamma, beta, mean, variance, eps: float):
    """Fold an inference batch norm into the convolution before it (fp32)."""
    scale = np.asarray(gamma, np.float32) / np.sqrt(np.asarray(variance, np.float32) + eps)
    folded = np.asarray(weight, np.float32) * scale.reshape(-1, 1, 1, 1)
    shift = np.asarray(beta, np.float32) - np.asarray(mean, np.float32) * scale
    return folded.astype(np.float32), shift.astype(np.float32)


def quantize_fp8_conv_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a ``[out, in / groups, kh, kw]`` weight to e4m3 with one scale per channel.

    Returns the raw fp8 bytes and ``scale[c] = max(abs(weight[c])) / 448``; the same scales
    are the dequantize scales in the engine. An all-zero channel gets scale 1.
    """
    weight = np.asarray(weight_fp32, dtype=np.float32)
    if weight.ndim != 4:
        raise ValueError(f"expected a 4-D convolution weight, got shape {weight.shape}")
    amax = np.abs(weight).reshape(weight.shape[0], -1).max(axis=1)
    if not np.all(np.isfinite(amax)) or not np.any(amax > 0):
        raise ValueError("convolution weight has no finite positive amax to scale from")
    scale = np.where(amax > 0, amax / FP8_E4M3_MAX, 1.0).astype(np.float32)
    scaled = np.clip(weight / scale.reshape(-1, 1, 1, 1), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def load_activation_scales(cfg: dict, weights) -> dict[str, float | None]:
    """Return the committed activation scales for this model config, or raise when none exist.

    An entry of ``None`` marks a convolution the calibration left unquantized on purpose.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"timm_resnet FP8 build requires {path}, which was not found") from error
    key = scale_key(cfg, weights)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def calibrate_timm_resnet(
    cfg: dict,
    weights,
    quantization: str,
    *,
    precision: str,
    eps: float,
) -> ResnetQuantContext:
    """Fold and quantize the loaded convolution weights one at a time.

    Pairs each with its stored activation scale. A convolution with no stored scale, or a
    stored scale for a convolution this model config does not quantize, is an error.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"timm_resnet supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if precision != "fp16":
        raise ValueError(f"timm_resnet FP8 needs precision fp16, got {precision!r}")
    if any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights):
        raise ValueError(
            "the checkpoint is already quantized; FP8 self-quantization needs the original"
        )
    activation_scales = load_activation_scales(cfg, weights)

    names = conv_names(cfg, weights)
    unexpected = sorted(set(activation_scales) - set(names))
    if unexpected:
        raise ValueError(
            f"{ACTIVATION_SCALES_FILENAME} has scales for convolutions that are not quantized, "
            f"for example {unexpected[0]}; run calibrate_fp8_scales.py for this model config"
        )
    scales: dict[str, _ConvScales] = {}
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
        norm = norm_name(name)
        folded, shift = fold_batch_norm(
            weights[f"{name}.weight"],
            weights[f"{norm}.weight"],
            weights[f"{norm}.bias"],
            weights[f"{norm}.running_mean"],
            weights[f"{norm}.running_var"],
            eps,
        )
        packed, weight_scale = quantize_fp8_conv_weight(folded)
        if input_scale < _FP16_MIN_NORMAL:
            raise ValueError(f"FP8 input scale of {name} is below the smallest normal fp16 value")
        scales[name] = _ConvScales(float(input_scale), _FP8Conv(packed, weight_scale, shift))
    if not scales:
        raise RuntimeError(
            "timm_resnet FP8 self-quantization matched no convolution in the checkpoint"
        )
    return ResnetQuantContext(scales=scales)
