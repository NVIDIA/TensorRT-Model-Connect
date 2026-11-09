# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_inception-owned FP8 self-quantization of the convolutions.

No FP8 checkpoint exists for the timm Inception-v3 classifier, so ``quantization="fp8"``
quantizes the original checkpoint at build time. Each quantized convolution has its batch
norm folded in first; the folded weight is then quantized with one scale per output
channel, ``scale = max(abs(weight[channel])) / 448`` (448 is the largest finite e4m3
value), and the FP8 bytes are fed to the engine as a constant. FP8 weights are never
stored. The input scale of each convolution needs real data, so it was calibrated once
offline with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``, max calibration,
ImageNet validation images) and is read from ``fp8_activation_scales.json``. Users do not
run calibration.

Each quantized convolution becomes::

    activation -> Quantize -> Dequantize(fp16) -> Convolution(+ folded bias) -> ReLU
    FP8 weight -> Dequantize(fp16, per output channel) ---^

Quantized: every convolution of the 17x17 and 8x8 stages, that is the Mixed blocks of the
InceptionB, InceptionC, InceptionD and InceptionE topologies (``Mixed_6a`` onward).
Convolutions that read the same tensor share one Quantize/Dequantize (their calibrated
scales are equal), and the pooling branch of a quantized block reads that dequantized copy
too, so the block input exists only in FP8 and the producing convolutions write FP8
directly. The stem convolutions, the 35x35 InceptionA blocks (``Mixed_5b`` to
``Mixed_5d``) and the classifier stay in fp16: with them quantized the KL divergence to
the PyTorch model was 0.15 (stem quantized) and 0.022 (stem in fp16), against 0.009 with
both in fp16, for about the same speed.

A convolution whose calibrated input scale is more than 20 times the median of all
quantized convolution inputs would be stored as ``null`` and left in fp16; a convolution
missing from the file is an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import ml_dtypes
import numpy as np
import tensorrt as trt


FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
# Smallest normal fp16 value: the Q/DQ scales are fp16 constants (the Dequantize output
# type), so a scale must not be smaller.
FP16_MIN_NORMAL = 2.0**-14
SUPPORTED_QUANTIZATIONS = ("fp8",)
# Block topologies (``model._BLOCK_MARKERS``) whose convolutions are quantized.
QUANTIZED_BLOCK_KINDS = frozenset({"inception_b", "inception_c", "inception_d", "inception_e"})


@dataclass(frozen=True)
class _FP8Convolution:
    """One folded convolution: fp8_e4m3 weight bytes, per-output-channel scales, fp16 bias."""

    packed: np.ndarray  # [out, in, kernel_h, kernel_w] raw fp8_e4m3 bytes
    weight_scales: np.ndarray  # [out] float16
    bias: np.ndarray  # [out] float16, the folded batch norm shift
    input_scale: float


def _scale_constant(network, values):
    """An fp16 constant holding Q/DQ scales (the Dequantize output type)."""
    data = np.ascontiguousarray(np.asarray(values, dtype=np.float16).reshape(-1))
    return data, network.add_constant(data.shape, trt.Weights(data)).get_output(0)


@dataclass
class InceptionQuantContext:
    """Routes the quantized convolutions through Q/DQ; other convolutions stay fp16."""

    convolutions: dict[str, _FP8Convolution]
    keep_alive: list = field(default_factory=list)
    _dequantized: dict = field(default_factory=dict)

    def should_quantize(self, prefix: str) -> bool:
        """True when convolution ``prefix`` (for example ``Mixed_6b.branch1x1``) is FP8."""
        return prefix in self.convolutions

    def dequantized_input(self, network, tensor, prefix: str):
        """``tensor -> Quantize -> Dequantize`` with the input scale of ``prefix``.

        Shared by every consumer of ``tensor`` that uses the same scale, so one tensor gets
        one Quantize.
        """
        if tensor.dtype != trt.float16:
            raise ValueError(f"timm_inception FP8 needs an fp16 activation, got {tensor.dtype}")
        scale = self.convolutions[prefix].input_scale
        key = (tensor.name, scale)
        if key not in self._dequantized:
            data, scale_tensor = _scale_constant(network, [scale])
            self.keep_alive.append(data)
            quantized = network.add_quantize(tensor, scale_tensor, trt.DataType.FP8)
            dequantized = network.add_dequantize(quantized.get_output(0), scale_tensor, trt.float16)
            self._dequantized[key] = dequantized.get_output(0)
        return self._dequantized[key]

    def convolution(self, network, tensor, prefix: str):
        """The FP8 convolution of ``tensor`` with its folded bias; returns the layer."""
        conv = self.convolutions[prefix]
        out_channels = int(conv.packed.shape[0])
        kernel = (int(conv.packed.shape[2]), int(conv.packed.shape[3]))
        # trt.Weights holds raw pointers, so the buffers must outlive the build.
        self.keep_alive += [conv.packed, conv.bias]
        weight = network.add_constant(
            tuple(int(value) for value in conv.packed.shape),
            trt.Weights(trt.DataType.FP8, conv.packed.ctypes.data, conv.packed.size),
        ).get_output(0)
        data, weight_scale = _scale_constant(network, conv.weight_scales)
        self.keep_alive.append(data)
        weight = network.add_dequantize(weight, weight_scale, trt.float16)
        weight.axis = 0
        layer = network.add_convolution_nd(
            self.dequantized_input(network, tensor, prefix),
            num_output_maps=out_channels,
            kernel_shape=kernel,
            kernel=trt.Weights(),
            bias=trt.Weights(conv.bias),
        )
        if layer is None:
            raise RuntimeError(f"TensorRT rejected the FP8 convolution {prefix}")
        layer.set_input(1, weight.get_output(0))
        return layer


def configure_fp8_build(builder_config, quant_ctx) -> None:
    """Keep per-layer tactic names in a quantized plan so FP8 lowering can be inspected."""
    if quant_ctx is not None:
        builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED


def quantized_convolutions(blocks: list[dict], names) -> list[str]:
    """Prefixes (``Mixed_6a.branch3x3``) of every convolution this model quantizes, sorted."""
    tops = {block["prefix"] for block in blocks if block["kind"] in QUANTIZED_BLOCK_KINDS}
    return sorted(
        name[: -len(".conv.weight")]
        for name in names
        if name.endswith(".conv.weight") and name.split(".", 1)[0] in tops
    )


def _abs_sum(array) -> float:
    return float(np.abs(np.asarray(array, dtype=np.float32)).sum(dtype=np.float64))


def scale_key(config: dict, reader) -> str:
    """Key of this model's entry in the activation-scale file.

    ``config`` is the resolved family config (input size and Mixed block layout). The key
    holds the model config values that change the activation ranges (input size, block
    topologies, number of quantized convolutions). Checkpoints of the same architecture
    differ in training, so the key also holds numeric weight sums (sum of absolute values,
    5 significant digits) of the first stem convolution and of the first quantized
    convolution in name order.
    """
    names = quantized_convolutions(config["blocks"], reader.keys())
    if not names:
        raise ValueError("Inception checkpoint has no convolution to quantize")
    layout = "".join(str(block["kind"])[-1] for block in config["blocks"])
    return (
        f"inception_v3-{layout}-n{len(names)}"
        f"-s{config['image_size_h']}x{config['image_size_w']}"
        f"-w{_abs_sum(reader.get_tensor('Conv2d_1a_3x3.conv.weight')):.4e}"
        f"-{_abs_sum(reader.get_tensor(names[0] + '.conv.weight')):.4e}"
    )


def quantize_fp8_weight(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a convolution weight ``[out, ...]`` to FP8 e4m3, one scale per output channel.

    Returns the raw fp8 bytes and the float16 scales ``max(abs(weight[channel])) / 448``;
    the same scales are the dequantize scales in the engine. An all-zero channel gets scale
    1, and a scale below the smallest normal fp16 value is raised to it (such a channel's
    weights are all below 0.0274, so they still fit in e4m3 with fewer significant bits).
    """
    weight = np.asarray(weight, dtype=np.float32)
    if not np.all(np.isfinite(weight)):
        raise ValueError("weight tensor has non-finite values")
    amax = np.abs(weight.reshape(weight.shape[0], -1)).max(axis=1)
    if not np.any(amax > 0):
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scales = np.where(amax > 0, amax / FP8_E4M3_MAX, 1.0)
    # Quantize with the fp16 value the engine dequantizes with.
    scales = np.maximum(scales, FP16_MIN_NORMAL).astype(np.float16)
    shape = (-1,) + (1,) * (weight.ndim - 1)
    scaled = np.clip(weight / scales.astype(np.float32).reshape(shape), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scales


def fold_batch_norm(reader, prefix: str, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """The float32 weight and bias of convolution ``prefix`` with its batch norm folded in."""

    def tensor(name: str) -> np.ndarray:
        return np.asarray(reader.get_tensor(f"{prefix}.{name}"), dtype=np.float32)

    scale = tensor("bn.weight") / np.sqrt(tensor("bn.running_var") + eps)
    weight = tensor("conv.weight") * scale.reshape(-1, 1, 1, 1)
    return weight, tensor("bn.bias") - tensor("bn.running_mean") * scale


def load_activation_scales(key: str) -> dict[str, float | None]:
    """Return the committed activation scales for this model config, or raise when none exist.

    An entry of ``None`` marks a convolution the calibration left unquantized on purpose.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"timm_inception FP8 build requires {path}, which was not found"
        ) from error
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def check_not_quantized(model_dir: Path, raw: dict, names) -> None:
    """Refuse a checkpoint that is already quantized: self-quantization needs the original."""
    if (
        "quantization_config" in raw
        or (Path(model_dir) / "hf_quant_config.json").is_file()
        or any(name.endswith((".weight_scale", ".input_scale")) for name in names)
    ):
        raise ValueError(
            "the checkpoint is already quantized; FP8 self-quantization needs the original"
        )


def create_quant_context(
    config: dict, reader, quantization: str, *, precision: str, bn_eps: float
) -> InceptionQuantContext:
    """Quantize the folded convolution weights one at a time and pair them with their scales.

    ``config`` is the resolved family config. A convolution with no stored scale, or a stored
    scale for a convolution this model does not quantize, is an error.
    """
    if quantization not in SUPPORTED_QUANTIZATIONS:
        raise NotImplementedError(
            f"timm_inception supports quantization {list(SUPPORTED_QUANTIZATIONS)} or none; "
            f"got {quantization!r}"
        )
    if precision != "fp16":
        raise ValueError(f"timm_inception FP8 needs precision fp16, got {precision!r}")
    activation_scales = load_activation_scales(scale_key(config, reader))
    names = quantized_convolutions(config["blocks"], reader.keys())
    unexpected = sorted(set(activation_scales) - set(names))
    if unexpected:
        raise ValueError(
            f"{ACTIVATION_SCALES_FILENAME} has scales for convolutions that are not quantized, "
            f"for example {unexpected[0]}; run calibrate_fp8_scales.py for this model config"
        )
    convolutions: dict[str, _FP8Convolution] = {}
    for name in names:
        if name not in activation_scales:
            raise ValueError(
                f"{name} has no activation scale in {ACTIVATION_SCALES_FILENAME}; "
                "run calibrate_fp8_scales.py"
            )
        input_scale = activation_scales[name]
        if input_scale is None:
            continue  # outlier input, left in fp16 by the calibration
        if not np.isfinite(input_scale) or input_scale < FP16_MIN_NORMAL:
            raise ValueError(
                f"activation scale of {name} must be finite and at least {FP16_MIN_NORMAL}"
            )
        weight, bias = fold_batch_norm(reader, name, bn_eps)
        packed, weight_scales = quantize_fp8_weight(weight)
        convolutions[name] = _FP8Convolution(
            packed,
            weight_scales,
            np.ascontiguousarray(bias, dtype=np.float16),
            float(np.float16(input_scale)),
        )
    if not convolutions:
        raise RuntimeError(
            "timm_inception FP8 self-quantization matched no convolution in the checkpoint"
        )
    return InceptionQuantContext(convolutions)
