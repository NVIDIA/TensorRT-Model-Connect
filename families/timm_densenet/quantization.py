# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_densenet-owned FP8 self-quantization and TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) quantizes the original checkpoint at build time, because no
FP8 checkpoint exists for the timm DenseNet classifiers. Weights are quantized one
convolution at a time with one scale per output channel: ``scale[c] = max(abs(weight[c])) /
448`` (448 is the largest finite e4m3 value). The FP8 bytes are fed to the engine as a
constant; FP8 weights are never stored. The activation scale of each convolution input needs
real data, so it was calibrated once offline with ``calibrate_fp8_scales.py`` (ModelOpt
``FP8_DEFAULT_CFG``, max calibration, ImageNet validation images) and is read from
``fp8_activation_scales.json``. Users do not run calibration.

Each quantized convolution becomes::

    activation -> Quantize -> Dequantize(fp16) -> Convolution
    FP8 weight -> Dequantize(fp16, per output channel) -^

Both Dequantize layers output fp16. TensorRT fuses the preceding batch norm and ReLU into
the Quantize, and the batch norm and ReLU after a 1x1 convolution plus the next Quantize
into the epilogue of the FP8 convolution, so the 1x1 convolution writes e4m3 directly. The
risk of fp16 is an overflow above 65504; the NaN scan on real ImageNet images found none.

Quantized (recipe: every convolution of the feature extractor except the stem, the public
CNN recipe of ModelOpt and the TensorRT samples): per dense layer the 1x1 bottleneck
(``conv1``) and the 3x3 convolution (``conv2``), and the 1x1 convolution of every
transition. The stem convolution (``features.conv0``, 3 input channels), batch norms,
pooling and the classifier stay in fp16. Per output channel weight scales are used because
they halve the error of one per-tensor weight scale at the same speed.

A convolution whose calibrated input scale is more than 20 times the median of the same kind
has a massive activation: its entry in the scale file is ``null`` and it stays in fp16 on
purpose. A convolution missing from the file is an error.
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
# Quantized convolutions of one dense layer, named as in the checkpoint.
LAYER_CONVS = ("conv1", "conv2")
TRANSITION_CONV = "conv"


@dataclass(frozen=True)
class _FP8Conv:
    """One convolution weight stored as raw fp8_e4m3 bytes with one scale per output channel."""

    packed: np.ndarray  # [out, in, kh, kw] raw fp8_e4m3 bytes
    weight_scale: np.ndarray  # [out] float32


@dataclass(frozen=True)
class _LayerScales:
    """Static activation scale and the FP8 weight of one convolution."""

    input_scale: float
    weight: _FP8Conv


def _scale_const(network, values: np.ndarray):
    """fp16 constant holding Q/DQ scales (the Dequantize output type)."""
    data = np.ascontiguousarray(values, dtype=np.float16).reshape(-1)
    return network.add_constant(data.shape, trt.Weights(data)).get_output(0)


@dataclass(frozen=True)
class DensenetQuantContext:
    """Routes the quantized convolutions through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, module: str) -> bool:
        """True when ``module`` (for example ``features.transition1.conv``) runs as FP8."""
        return module in self.scales

    def conv2d(self, network, activation, module: str, kernel_size, padding=(0, 0)):
        """FP8 convolution of ``activation`` with the weight of ``module`` (stride 1)."""
        if activation.dtype != trt.float16:
            raise ValueError(f"timm_densenet FP8 needs an fp16 activation, got {activation.dtype}")
        entry = self.scales[module]
        act_scale = _scale_const(network, np.asarray([entry.input_scale]))
        quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
        dequant = network.add_dequantize(quant.get_output(0), act_scale, trt.float16)

        packed = entry.weight.packed
        # trt.Weights holds a raw pointer, so the buffer must outlive the build.
        self.keep_alive.append(packed)
        w_const = network.add_constant(
            packed.shape, trt.Weights(trt.DataType.FP8, packed.ctypes.data, packed.size)
        ).get_output(0)
        w_dequant = network.add_dequantize(
            w_const, _scale_const(network, entry.weight.weight_scale), trt.float16
        )
        w_dequant.axis = 0
        conv = network.add_convolution_nd(
            dequant.get_output(0),
            int(packed.shape[0]),
            tuple(kernel_size),
            trt.Weights(),
            trt.Weights(),
        )
        conv.set_input(1, w_dequant.get_output(0))
        conv.padding_nd = tuple(padding)
        return conv.get_output(0)


def configure_fp8_build(builder_config, quant_ctx) -> None:
    """Keep per-layer tactic names in a quantized plan so FP8 lowering can be inspected."""
    if quant_ctx is None:
        return
    builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED


def _abs_sum(array) -> float:
    return float(np.abs(np.asarray(array, dtype=np.float32)).sum(dtype=np.float64))


def scale_key(block_layers: list[int], image_size: tuple[int, int], weights) -> str:
    """Key of this model's entry in the activation-scale file.

    The key holds every config value that changes the activation ranges (stem width, growth
    rate, layers per dense block, input size). Checkpoints of the same architecture differ in
    training (``densenet121.ra_in1k`` and ``.tv_in1k``), so the key also holds the numeric
    weight sums (sum of absolute values, 5 significant digits) of the stem convolution and of
    the first dense layer's ``conv2``.
    """
    stem = weights["features.conv0.weight"]
    first = weights["features.denseblock1.denselayer1.conv2.weight"]
    return (
        f"densenet-c{stem.shape[0]}-g{first.shape[0]}"
        f"-b{'x'.join(str(count) for count in block_layers)}"
        f"-s{image_size[0]}x{image_size[1]}"
        f"-w{_abs_sum(stem):.4e}-{_abs_sum(first):.4e}"
    )


def quantize_fp8_conv_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize one convolution weight to FP8 e4m3 with one scale per output channel.

    Returns the raw fp8 bytes and ``scale[c] = max(abs(weight[c])) / 448``; the same scales
    are the dequantize scales in the engine. A scale is never below the smallest normal fp16
    value, so a channel of zeros gets that floor and quantizes to zeros.
    """
    weight = np.asarray(weight_fp32, dtype=np.float32)
    amax = np.abs(weight.reshape(weight.shape[0], -1)).max(axis=1)
    if not np.all(np.isfinite(amax)) or not np.any(amax > 0):
        raise ValueError("weight tensor has no finite positive amax to scale from")
    # The scale is an fp16 constant in the engine: keep it a normal fp16 value. A channel with
    # tiny (or all-zero) weights then uses less of the e4m3 range, which costs no accuracy.
    scale = np.maximum(amax / FP8_E4M3_MAX, _FP16_MIN_NORMAL).astype(np.float32)
    shape = (-1,) + (1,) * (weight.ndim - 1)
    scaled = np.clip(weight / scale.reshape(shape), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def load_activation_scales(
    block_layers: list[int], image_size: tuple[int, int], weights
) -> dict[str, float | None]:
    """Return the committed activation scales for this model config, or raise when none exist.

    An entry of ``None`` marks a convolution the calibration left unquantized on purpose.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"timm_densenet FP8 build requires {path}, which was not found"
        ) from error
    key = scale_key(block_layers, image_size, weights)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def conv_names(block_layers: list[int]) -> list[str]:
    """Checkpoint module names of every quantizable convolution of this model config."""
    names = []
    for block, count in enumerate(block_layers, start=1):
        for layer in range(1, count + 1):
            prefix = f"features.denseblock{block}.denselayer{layer}"
            names.extend(f"{prefix}.{conv}" for conv in LAYER_CONVS)
        if block < len(block_layers):
            names.append(f"features.transition{block}.{TRANSITION_CONV}")
    return names


def is_already_quantized(model_dir: Path, weights) -> bool:
    """True when the checkpoint carries quantization metadata or quantization scales."""
    model_dir = Path(model_dir)
    if (model_dir / "hf_quant_config.json").is_file():
        return True
    config = model_dir / "config.json"
    if config.is_file():
        raw = json.loads(config.read_text(encoding="utf-8"))
        nested = raw.get("pretrained_cfg") if isinstance(raw.get("pretrained_cfg"), dict) else {}
        if "quantization_config" in raw or "quantization_config" in nested:
            return True
    return any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights)


def calibrate_timm_densenet(
    block_layers: list[int],
    image_size: tuple[int, int],
    weights,
    quantization: str,
    *,
    precision: str,
) -> DensenetQuantContext:
    """Quantize the loaded float32 convolution weights one at a time.

    Pairs each with its stored activation scale. A convolution with no stored scale, or a
    stored scale for a convolution this model config does not have, is an error.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"timm_densenet supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if precision != "fp16":
        raise ValueError(f"timm_densenet FP8 needs precision fp16, got {precision!r}")
    if any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights):
        raise ValueError(
            "the checkpoint is already quantized; FP8 self-quantization needs the original"
        )
    activation_scales = load_activation_scales(block_layers, image_size, weights)

    names = conv_names(block_layers)
    unexpected = sorted(set(activation_scales) - set(names))
    if unexpected:
        raise ValueError(
            f"{ACTIVATION_SCALES_FILENAME} has scales for convolutions this model does not have, "
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
        packed, weight_scale = quantize_fp8_conv_weight(weights[key])
        if input_scale < _FP16_MIN_NORMAL:
            raise ValueError(f"FP8 scale of {name} is below the smallest normal fp16 value")
        scales[name] = _LayerScales(float(input_scale), _FP8Conv(packed, weight_scale))
    if not scales:
        raise RuntimeError(
            "timm_densenet FP8 self-quantization matched no convolution in the checkpoint"
        )
    return DensenetQuantContext(scales=scales)
