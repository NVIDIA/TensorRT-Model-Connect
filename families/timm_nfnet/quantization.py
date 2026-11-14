# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_nfnet-owned FP8 self-quantization and TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) quantizes the original checkpoint at build time, because no
FP8 checkpoint exists for the timm NFNet classifiers. Each quantized convolution is
quantized on its own: its weight is standardised first (as the unquantized build does, in
fp32), then every output channel gets its own weight scale ``max(abs(channel)) / 448`` (448
is the largest finite e4m3 value) and the FP8 bytes are fed to the engine as a constant.
FP8 weights are never stored. The activation scale of each convolution input needs real
data, so it was calibrated once offline with ``calibrate_fp8_scales.py`` (ModelOpt
``FP8_DEFAULT_CFG``, max calibration, ImageNet validation images) and is read from
``fp8_activation_scales.json``. Users do not run calibration.

Each quantized convolution becomes::

    activation -> Quantize -> Dequantize(fp16) -> Convolution(+ bias)
    FP8 weight -> Dequantize(fp16, one fp32 scale per output channel) --^

Quantized: every dense (one group) convolution on a spatial map, that is the stem
convolutions after the first, ``conv1``, ``conv3`` and ``downsample.conv`` of every block,
``conv2`` / ``conv2b`` where they have one group (the first stage), and ``final_conv``.
Kept in fp16: the first stem convolution (3 input channels), the grouped 3x3 convolutions
(group width 128), the squeeze gate (1x1 convolutions on a 1x1 map) and the classifier. The
grouped convolutions stay fp16 because TensorRT runs them slower in FP8 than in fp16 at
batch 1 (0.78x to 0.84x for the whole engine on B200 and B300, TensorRT 11.4).

A convolution whose calibrated input scale is more than 20 times the median of the same
convolution kind has a massive activation, and one whose scale is below the smallest normal
fp16 value has an input that is almost zero: its entry in the scale file is ``null`` and it
stays in fp16 on purpose (measured: one of the 112 convolutions of dm_nfnet_f3, none of
dm_nfnet_f0). A convolution missing from the file is an error.
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
FP16_MIN_NORMAL = 2.0**-14

_SUPPORTED = ("fp8",)


@dataclass(frozen=True)
class _FP8Conv:
    """One standardised convolution weight as raw fp8_e4m3 bytes with one scale per channel."""

    packed: np.ndarray  # [out, in, kh, kw] raw fp8_e4m3 bytes
    weight_scale: np.ndarray  # [out] float32


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
class NfnetQuantContext:
    """Routes the quantized convolutions through Q/DQ; the others stay unquantized."""

    scales: dict[str, _ConvScales]
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, module: str) -> bool:
        """True when ``module`` (for example ``stages.1.0.conv1``) runs as FP8."""
        return module in self.scales

    def input_scale(self, module: str) -> float:
        """The static activation scale of ``module``."""
        return self.scales[module].input_scale

    def quantize_input(self, network, activation, module: str):
        """``activation -> Quantize -> Dequantize(fp16)`` with the input scale of ``module``."""
        if activation.dtype != trt.float16:
            raise ValueError(f"timm_nfnet FP8 needs an fp16 activation, got {activation.dtype}")
        scale, data = _scale_const(network, self.scales[module].input_scale, np.float16)
        self.keep_alive.append(data)
        quant = network.add_quantize(activation, scale, trt.DataType.FP8)
        return network.add_dequantize(quant.get_output(0), scale, trt.float16).get_output(0)

    def convolution(self, network, dequantized_activation, module: str, bias, *, stride, padding):
        """The convolution of ``module`` with its FP8 weight and fp16 bias.

        ``padding`` is (top, left, bottom, right), as the unquantized convolution takes it.
        """
        weight = self.scales[module].weight
        packed = weight.packed
        # trt.Weights holds a raw pointer, so the buffers must outlive the build.
        bias = np.ascontiguousarray(bias, dtype=np.float16)
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
            num_output_maps=int(packed.shape[0]),
            kernel_shape=(int(packed.shape[2]), int(packed.shape[3])),
            kernel=trt.Weights(),
            bias=trt.Weights(bias),
        )
        if conv is None:
            raise RuntimeError(f"TensorRT rejected the FP8 NFNet convolution {module}")
        conv.set_input(1, w_dequant.get_output(0))
        conv.stride_nd = (stride, stride)
        conv.pre_padding = (padding[0], padding[1])
        conv.post_padding = (padding[2], padding[3])
        return conv.get_output(0)


def configure_fp8_build(builder_config, quant_ctx) -> None:
    """Keep per-layer tactic names in a quantized plan so FP8 lowering can be inspected."""
    if quant_ctx is None:
        return
    builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED


def _abs_sum(array) -> float:
    return float(np.abs(np.asarray(array, dtype=np.float32)).sum(dtype=np.float64))


def groups_of(weight) -> int:
    """Group count of a square convolution, as the family computes it (out / in per group)."""
    return max(1, int(weight.shape[0]) // int(weight.shape[1]))


def scale_key(config: dict, blocks: list[dict], checkpoint) -> str:
    """Key of this model's entry in the activation-scale file.

    The key holds every config value that changes the activation ranges (blocks per stage,
    stem width, width of the first block, input size). Checkpoints of the same architecture
    differ in training, so the key also holds the numeric weight sums (sum of absolute
    values, 5 significant digits) of the first stem convolution and of the first block's
    ``conv1``, read from the original checkpoint.
    """
    depths: dict[int, int] = {}
    for block in blocks:
        depths[int(block["stage"])] = depths.get(int(block["stage"]), 0) + 1
    stem = checkpoint.tensor("stem.conv1.weight")
    first = checkpoint.tensor(f"{blocks[0]['prefix']}.conv1.weight")
    return (
        f"nfnet-d{'x'.join(str(depths[stage]) for stage in sorted(depths))}"
        f"-c{stem.shape[0]}-w{first.shape[0]}"
        f"-s{config['image_height']}x{config['image_width']}"
        f"-w{_abs_sum(stem):.4e}-{_abs_sum(first):.4e}"
    )


def conv_names(blocks: list[dict], checkpoint) -> list[str]:
    """Checkpoint module names of every quantized convolution of this model config."""
    names = sorted(
        (
            name.removesuffix(".weight")
            for name in checkpoint.names
            if name.startswith("stem.conv") and name.endswith(".weight")
        ),
        key=lambda name: int(name.removeprefix("stem.conv")),
    )[1:]  # the first stem convolution reads the 3-channel image
    for block in blocks:
        prefix = str(block["prefix"])
        names.append(f"{prefix}.conv1")
        for leaf in ("conv2", "conv2b"):
            name = f"{prefix}.{leaf}"
            if f"{name}.weight" in checkpoint.names:
                if groups_of(checkpoint.tensor(f"{name}.weight")) == 1:
                    names.append(name)
        names.append(f"{prefix}.conv3")
        if bool(block["has_downsample"]):
            names.append(f"{prefix}.downsample.conv")
    names.append("final_conv")
    return names


def quantize_fp8_conv_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a ``[out, in, kh, kw]`` weight to e4m3 with one scale per channel.

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


def load_activation_scales(config: dict, blocks: list[dict], checkpoint) -> dict:
    """Return the committed activation scales for this model config, or raise when none exist.

    An entry of ``None`` marks a convolution the calibration left unquantized on purpose.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"timm_nfnet FP8 build requires {path}, which was not found") from error
    key = scale_key(config, blocks, checkpoint)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def calibrate_timm_nfnet(
    config: dict,
    blocks: list[dict],
    checkpoint,
    quantization: str,
    *,
    precision: str,
    standardise,
) -> NfnetQuantContext:
    """Standardise and quantize the convolution weights one at a time.

    Pairs each with its stored activation scale. A convolution with no stored scale, or a
    stored scale for a convolution this model config does not quantize, is an error.
    ``standardise(weight, gain, dtype)`` is the weight standardisation of the family.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"timm_nfnet supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if precision != "fp16":
        raise ValueError(f"timm_nfnet FP8 needs precision fp16, got {precision!r}")
    if any(str(name).endswith((".weight_scale", ".input_scale")) for name in checkpoint.names):
        raise ValueError(
            "the checkpoint is already quantized; FP8 self-quantization needs the original"
        )
    activation_scales = load_activation_scales(config, blocks, checkpoint)

    names = conv_names(blocks, checkpoint)
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
        if input_scale < FP16_MIN_NORMAL:
            raise ValueError(f"FP8 input scale of {name} is below the smallest normal fp16 value")
        weight = standardise(
            checkpoint.tensor(f"{name}.weight"), checkpoint.tensor(f"{name}.gain"), np.float32
        )
        packed, weight_scale = quantize_fp8_conv_weight(weight)
        scales[name] = _ConvScales(float(input_scale), _FP8Conv(packed, weight_scale))
    if not scales:
        raise RuntimeError("timm_nfnet FP8 self-quantization matched no convolution")
    return NfnetQuantContext(scales=scales)
