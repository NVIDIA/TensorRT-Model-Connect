# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_swin-owned FP8 self-quantization and TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) quantizes the original checkpoint at build time, because no
FP8 checkpoint exists for the timm Swin classifiers. Weights are quantized one projection at
a time: ``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and the
FP8 bytes are fed to the engine as a constant. FP8 weights are never stored. The activation
scale of each projection input needs real data, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``, max calibration, ImageNet
validation images) and is read from ``fp8_activation_scales.json``. Users do not run
calibration.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize(fp16) -> MatMul
    FP8 weight -> Dequantize(fp16) -----------------^

Both Dequantize layers output fp16, so the FP8 GEMM writes fp16 and the bias add fuses into
its epilogue. The risk of fp16 is an overflow above 65504 in a projection output; the NaN
scan on real ImageNet images found none.

Quantized (recipe: the linears of the encoder, as in the public ViT FP8 recipes and the
timm_vit, siglip and dinov3 families): per block the fused query/key/value projection
(``attn.qkv``, one weight, one scale) and the two MLP projections (``mlp.fc1``,
``mlp.fc2``), and the patch-merging reduction (``layers.N.downsample.reduction``) of every
stage after the first. The attention output projection (``attn.proj``) stays in fp16: the
window attention is one fused attention layer, and TensorRT 11.4 cannot compile an FP8
Quantize right after a fused attention (the bias add then sees an fp8 operand and the build
fails with "Could not find any implementation"); retest it on every newer TensorRT release.
The patch embedding convolution, layer norms, the window attention (matmul, bias, softmax),
the roll and window reshapes, and the classifier head stay in fp16.

A projection whose calibrated input scale is more than 20 times the median of the same
projection kind has a massive activation: its entry in the scale file is ``null`` and it
stays in fp16 on purpose. A projection missing from the file is an error.
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
# Quantized modules of one block, named as in the checkpoint.
PROJECTIONS = ("attn.qkv", "mlp.fc1", "mlp.fc2")
MERGE_PROJECTION = "downsample.reduction"


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [in_features, out_features] raw fp8_e4m3 bytes
    weight_scale: float
    in_features: int
    out_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Static activation scale and the FP8 weight of one module."""

    input_scale: float
    weight: _FP8Weight


def _scale_const(network, value: float):
    """One-element fp16 constant holding a Q/DQ scale (the Dequantize output type)."""
    data = np.asarray([value], dtype=np.float16)
    return network.add_constant((1,), trt.Weights(data)).get_output(0)


def _quantize_dequantize(network, activation, input_scale: float):
    """``activation -> Quantize -> Dequantize`` with the static per-tensor scale."""
    if activation.dtype != trt.float16:
        raise ValueError(f"timm_swin FP8 needs an fp16 activation, got {activation.dtype}")
    act_scale = _scale_const(network, input_scale)
    quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    return network.add_dequantize(quant.get_output(0), act_scale, trt.float16).get_output(0)


def _fp8_weight_matmul(network, dequantized_activation, weight: _FP8Weight, *, keep_alive: list):
    """Add ``activation @ weight`` for an FP8 weight constant with fp16 Dequantize."""
    rank = len(tuple(dequantized_activation.shape))
    shape = (1,) * max(0, rank - 2) + (weight.in_features, weight.out_features)
    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(weight.packed)
    w_fp8 = trt.Weights(
        trt.DataType.FP8, weight.packed.ctypes.data, weight.in_features * weight.out_features
    )
    w_const = network.add_constant(shape, w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, weight.weight_scale), trt.float16
    )
    return network.add_matrix_multiply(
        dequantized_activation,
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.NONE,
    ).get_output(0)


@dataclass(frozen=True)
class SwinQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, module: str) -> bool:
        """True when ``module`` (for example ``layers.1.blocks.0.mlp.fc1``) runs as FP8."""
        return module in self.scales

    def quantize_input(self, network, activation, module: str):
        """The Quantize/Dequantize pair on the input of ``module``."""
        return _quantize_dequantize(network, activation, self.scales[module].input_scale)

    def matmul(self, network, dequantized_activation, module: str):
        """``activation @ weight`` for the FP8 weight of ``module``."""
        return _fp8_weight_matmul(
            network,
            dequantized_activation,
            self.scales[module].weight,
            keep_alive=self.keep_alive,
        )


def configure_fp8_build(builder_config, quant_ctx) -> None:
    """Keep per-layer tactic names in a quantized plan so FP8 lowering can be inspected."""
    if quant_ctx is None:
        return
    builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED


def _abs_sum(array) -> float:
    return float(np.abs(np.asarray(array, dtype=np.float32)).sum(dtype=np.float64))


def scale_key(depths: list[int], image_size: tuple[int, int], weights) -> str:
    """Key of this model's entry in the activation-scale file.

    The key holds every config value that changes the activation ranges (embedding width,
    depths, heads of the first stage, patch size, window size, input size). Checkpoints of
    the same architecture differ in training, so the key also holds the numeric weight sums
    (sum of absolute values, 5 significant digits) of the patch embedding and of the first
    block's qkv.
    """
    patch_weight = weights["patch.weight"]
    area = int(round(np.sqrt(np.asarray(weights["layers.0.blocks.0.rel_index"]).size)))
    window = int(round(np.sqrt(area)))
    heads = int(np.asarray(weights["layers.0.blocks.0.rel_table"]).shape[1])
    return (
        f"swin-c{patch_weight.shape[0]}-d{'x'.join(str(depth) for depth in depths)}"
        f"-n{heads}-p{patch_weight.shape[2]}-win{window}-s{image_size[0]}x{image_size[1]}"
        f"-w{_abs_sum(patch_weight):.4e}"
        f"-{_abs_sum(weights['layers.0.blocks.0.attn.qkv.weight']):.4e}"
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


def load_activation_scales(
    depths: list[int], image_size: tuple[int, int], weights
) -> dict[str, float | None]:
    """Return the committed activation scales for this model config, or raise when none exist.

    An entry of ``None`` marks a projection the calibration left unquantized on purpose.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"timm_swin FP8 build requires {path}, which was not found") from error
    key = scale_key(depths, image_size, weights)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def projection_names(depths: list[int], weights) -> list[str]:
    """Checkpoint module names of every quantizable projection of this model config."""
    names = []
    for layer, depth in enumerate(depths):
        if f"layers.{layer}.downsample.weight" in weights:
            names.append(f"layers.{layer}.{MERGE_PROJECTION}")
        for index in range(depth):
            names.extend(f"layers.{layer}.blocks.{index}.{stem}" for stem in PROJECTIONS)
    return names


def _weight_key(name: str) -> str:
    """Key of the ``[out, in]`` weight of a projection in the loaded weight dict."""
    if name.endswith(MERGE_PROJECTION):
        return name.removesuffix("reduction") + "weight"
    return f"{name}.weight"


def calibrate_timm_swin(
    depths: list[int],
    image_size: tuple[int, int],
    weights,
    quantization: str,
    *,
    precision: str,
) -> SwinQuantContext:
    """Quantize the loaded weights (``[out, in]`` layout) one projection at a time.

    Pairs each with its stored activation scale. A projection with no stored scale, or a
    stored scale for a projection this model config does not quantize, is an error.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"timm_swin supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if precision != "fp16":
        raise ValueError(f"timm_swin FP8 needs precision fp16, got {precision!r}")
    if any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights):
        raise ValueError(
            "the checkpoint is already quantized; FP8 self-quantization needs the original"
        )
    activation_scales = load_activation_scales(depths, image_size, weights)

    names = projection_names(depths, weights)
    unexpected = sorted(set(activation_scales) - set(names))
    if unexpected:
        raise ValueError(
            f"{ACTIVATION_SCALES_FILENAME} has scales for projections that are not quantized, "
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
            continue  # massive-activation projection, left in fp16 by the calibration
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        key = _weight_key(name)
        if key not in weights:
            raise ValueError(f"checkpoint has no {key}")
        matrix = np.asarray(weights[key], dtype=np.float32).T  # [in, out]
        packed, weight_scale = quantize_fp8_weight(matrix)
        if min(input_scale, weight_scale) < _FP16_MIN_NORMAL:
            raise ValueError(f"FP8 scale of {name} is below the smallest normal fp16 value")
        scales[name] = _LayerScales(
            float(input_scale), _FP8Weight(packed, weight_scale, matrix.shape[0], matrix.shape[1])
        )
    if not scales:
        raise RuntimeError(
            "timm_swin FP8 self-quantization matched no projection in the checkpoint"
        )
    return SwinQuantContext(scales=scales)
