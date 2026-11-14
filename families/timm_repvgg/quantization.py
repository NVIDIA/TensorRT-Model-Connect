# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_repvgg-owned FP8 self-quantization and TensorRT Q/DQ convolution context.

FP8 (``quantization="fp8"``) quantizes the original checkpoint at build time, because no
FP8 checkpoint exists for the timm RepVGG classifiers. The build first fuses each block's
3x3, 1x1 and identity branches (with their batch norms) into one 3x3 convolution, as for the
unquantized engine; that fused weight is then quantized one block at a time and fed to the
engine as FP8 bytes. FP8 weights are never stored.

The weight scale is per output channel: ``scale[c] = max(abs(weight[c])) / 448`` (448 is the
largest finite e4m3 value), at least the smallest normal fp16 value. Folding the batch norms gives the output channels of a fused weight
very different ranges, and one scale per tensor loses the small channels: on RepVGG-A2 a
per-tensor weight scale gave a KL divergence to PyTorch of 0.049 (92% top-1 agreement), the
per-channel scale 0.014 (96%). The activation scale of each block input needs real data, so it
was calibrated once offline with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``, max
calibration, ImageNet validation images) and is read from ``fp8_activation_scales.json``. The
block input is the input of all three branches, so its scale is the ``conv_kxk.conv`` input
scale of the unfused model. Users do not run calibration.

Each quantized block becomes::

    activation -> Quantize -> Dequantize(fp16) -> Convolution 3x3 (+ fused bias)
    FP8 weight -> Dequantize(fp16, per channel) ------^

TensorRT fuses the pair into an e4m3 convolution kernel that writes fp16, with the bias and the
ReLU in its epilogue. Both Dequantize layers output fp16; the NaN scan on real ImageNet images
and on noise inputs found no overflow.

Quantized (recipe: no official FP8 config exists for a RepVGG classifier; the dense
convolutions are quantized and the input convolution is kept, as public UNet and ViT FP8
recipes keep ``conv_in`` and the patch embedding in 16 bit): the fused convolution of every
block (``stages.S.B``). Kept in fp16: the stem (3 input channels), the ReLU, the global pool
and the classifier head.

A block whose calibrated input scale is more than 20 times the median of all blocks has a
massive activation: its entry in the scale file is ``null`` and it stays in fp16 on purpose.
A block missing from the file is an error.
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


@dataclass(frozen=True)
class _FP8Weight:
    """One fused convolution weight as raw fp8_e4m3 bytes with one scale per output channel."""

    packed: np.ndarray  # [out, in, 3, 3] raw fp8_e4m3 bytes
    weight_scale: np.ndarray  # [out] float32


@dataclass(frozen=True)
class _LayerScales:
    """Static activation scale and the FP8 weight of one block."""

    input_scale: float
    weight: _FP8Weight


def _scale_const(network, value: float):
    """One-element fp16 constant holding a Q/DQ scale (the Dequantize output type)."""
    data = np.asarray([value], dtype=np.float16)
    return network.add_constant((1,), trt.Weights(data)).get_output(0)


@dataclass(frozen=True)
class RepVggQuantContext:
    """Routes the quantized blocks through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, module: str) -> bool:
        """True when ``module`` (for example ``stages.1.0``) runs as FP8."""
        return module in self.scales

    def convolution(self, network, activation, module: str, bias: np.ndarray, *, stride: int):
        """``activation`` convolved (3x3, padding 1) with the FP8 weight of ``module``."""
        if activation.dtype != trt.float16:
            raise ValueError(f"timm_repvgg FP8 needs an fp16 activation, got {activation.dtype}")
        entry = self.scales[module]
        act_scale = _scale_const(network, entry.input_scale)
        quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
        dequant = network.add_dequantize(quant.get_output(0), act_scale, trt.float16)
        packed = entry.weight.packed
        channel_scale = np.ascontiguousarray(entry.weight.weight_scale, dtype=np.float16)
        bias16 = np.ascontiguousarray(bias, dtype=np.float16)
        # trt.Weights holds a raw pointer, so the buffers must outlive the build.
        self.keep_alive.extend((packed, channel_scale, bias16))
        w_fp8 = trt.Weights(trt.DataType.FP8, packed.ctypes.data, packed.size)
        w_const = network.add_constant(tuple(int(v) for v in packed.shape), w_fp8)
        scale_const = network.add_constant((int(packed.shape[0]),), trt.Weights(channel_scale))
        w_dequant = network.add_dequantize(
            w_const.get_output(0), scale_const.get_output(0), trt.float16
        )
        w_dequant.axis = 0
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
        layer.stride_nd = (stride, stride)
        layer.padding_nd = (1, 1)
        return layer.get_output(0)


def configure_fp8_build(builder_config, quant_ctx) -> None:
    """Keep per-layer tactic names in a quantized plan so FP8 lowering can be inspected."""
    if quant_ctx is None:
        return
    builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED


def _abs_sum(array) -> float:
    return float(np.abs(np.asarray(array, dtype=np.float32)).sum(dtype=np.float64))


def scale_key(blocks: list[dict[str, object]], image_size: tuple[int, int], weights) -> str:
    """Key of this model's entry in the activation-scale file.

    The key holds every config value that changes the activation ranges (the output width of
    the stem and of every block, the input size). Checkpoints of the same architecture differ in
    training, so the key also holds two numeric weight sums (sum of absolute values, 5
    significant digits) of the fused stem and first block weights.
    """
    names = [str(block["prefix"]) for block in blocks]
    widths = [int(weights["stem.weight"].shape[0])]
    widths += [int(weights[f"{name}.weight"].shape[0]) for name in names]
    return (
        f"repvgg-c{'x'.join(str(width) for width in widths)}"
        f"-s{image_size[0]}x{image_size[1]}"
        f"-w{_abs_sum(weights['stem.weight']):.4e}-{_abs_sum(weights[f'{names[0]}.weight']):.4e}"
    )


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize one convolution weight to FP8 e4m3 with one scale per output channel.

    Returns the raw fp8 bytes and ``scale[c] = max(abs(weight[c])) / 448``; the same scales
    are the dequantize scales in the engine. The scales are fp16 constants in the engine, so a
    scale is raised to the smallest normal fp16 value: an output channel whose weights are all
    below about 0.027 (448 times that value) then uses fewer of the e4m3 steps, and an all-zero
    channel stays zero.
    """
    flat = np.asarray(weight_fp32, dtype=np.float32).reshape(weight_fp32.shape[0], -1)
    amax = np.abs(flat).max(axis=1)
    if not np.isfinite(amax).all() or not (amax > 0).any():
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = np.maximum(amax / FP8_E4M3_MAX, _FP16_MIN_NORMAL).astype(np.float32)
    scaled = np.clip(flat / scale[:, None], -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8).reshape(weight_fp32.shape)
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
    blocks: list[dict[str, object]], image_size: tuple[int, int], weights
) -> dict[str, float | None]:
    """Return the committed activation scales for this model config, or raise when none exist.

    An entry of ``None`` marks a block the calibration left unquantized on purpose.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"timm_repvgg FP8 build requires {path}, which was not found") from error
    key = scale_key(blocks, image_size, weights)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def projection_names(blocks: list[dict[str, object]]) -> list[str]:
    """Names of every quantizable fused block convolution of this model config."""
    return [str(block["prefix"]) for block in blocks]


def calibrate_timm_repvgg(
    blocks: list[dict[str, object]],
    image_size: tuple[int, int],
    weights,
    quantization: str,
    *,
    precision: str,
) -> RepVggQuantContext:
    """Quantize the fused block weights one block at a time.

    Pairs each with its stored activation scale. A block with no stored scale, or a stored
    scale for a block this model config does not have, is an error.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"timm_repvgg supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if precision != "fp16":
        raise ValueError(f"timm_repvgg FP8 needs precision fp16, got {precision!r}")
    activation_scales = load_activation_scales(blocks, image_size, weights)

    names = projection_names(blocks)
    unexpected = sorted(set(activation_scales) - set(names))
    if unexpected:
        raise ValueError(
            f"{ACTIVATION_SCALES_FILENAME} has scales for blocks that are not quantized, "
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
            continue  # massive-activation block, left in fp16 by the calibration
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        packed, weight_scale = quantize_fp8_weight(
            np.asarray(weights[f"{name}.weight"], dtype=np.float32)
        )
        if input_scale < _FP16_MIN_NORMAL:
            raise ValueError(f"FP8 scale of {name} is below the smallest normal fp16 value")
        scales[name] = _LayerScales(float(input_scale), _FP8Weight(packed, weight_scale))
    if not scales:
        raise RuntimeError("timm_repvgg FP8 self-quantization matched no block in the checkpoint")
    return RepVggQuantContext(scales=scales)
