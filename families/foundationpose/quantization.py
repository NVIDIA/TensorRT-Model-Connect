# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP8 self-quantization of the FoundationPose refiner and scorer convolutions.

``quantization="fp8"`` quantizes the NGC ONNX weights at build time; no FP8 weights are
stored. Each quantized weight gets one scale, ``max(abs(weight)) / 448`` (448 is the largest
finite e4m3 value). The activation scale of each layer input is static per tensor; it was
calibrated once offline with ``calibrate_fp8_scales.py`` (``max(abs(input)) / 448`` over RGB-XYZ
crops of held-out-separated scenes, the statistic of ModelOpt's max calibration) and is read from
``fp8_activation_scales.json`` under the engine's model config key, next to the ``max(abs(weight))``
of each quantized convolution: a build whose ONNX weights do not reproduce those values (other
weights than the NGC 1.0.1 release) is refused. Users do not run calibration.

Recipe: the 3x3 convolutions of the paired-feature trunk (``encodeAB`` in the refiner,
``encoderAB`` in the scorer: four residual blocks and the stride-2 downsample, 9 convolutions
per engine) go to FP8. Each becomes::

    activation -> Quantize -> Dequantize -> Convolution -> + bias
    FP8 weight -> Dequantize -------------^

Both Dequantize layers output fp16, so the convolution writes fp16 and TensorRT fuses bias,
ReLU and the residual add into it.

Stays fp16 (measured on held-out crops, see the PR):

  image encoders ``encodeA`` / ``encoderA``  run once per input image on the raw RGB-XYZ
                                            crop; in FP8 they doubled the pose error for 2 to
                                            5 percent of speed.
  transformer heads and attention           q/k/v, output, feed-forward linears: FP8 linears
                                            made the refiner slower (extra Quantize kernels on
                                            400-token GEMMs).
  score cross attention, output projections, layer norms.
"""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes
import numpy as np
import tensorrt as trt


FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
SUPPORTED = ("fp8",)
# Output type of every Dequantize; the convolution runs in this type.
DEQUANTIZE_TYPE = trt.float16
_MIN_ACTIVATION_SCALE = 1e-4 / FP8_E4M3_MAX
# A calibrated scale this many times above the median of the quantized layers is an outlier;
# the calibration tool stores it as null and the layer stays fp16.
OUTLIER_RATIO = 20.0
_QUANTIZED_TRUNKS = ("/encodeAB/", "/encoderAB/")
_QUANTIZE_OPS = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"}


def quantized_nodes(nodes) -> list[str]:
    """ONNX names of the convolutions that go to FP8, in graph order."""
    return [
        node.name
        for node in nodes
        if node.op_type == "Conv" and node.name.startswith(_QUANTIZED_TRUNKS)
    ]


def scale_key(kind: str) -> str:
    """Key of one engine kind (model config) in the activation-scale file."""
    return f"foundationpose-{kind}-h512"


def weight_amax(nodes, arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """``max(abs(weight))`` of every quantized convolution, keyed by ONNX node name."""
    by_name = {node.name: node for node in nodes}
    return {
        name: float(np.abs(arrays[by_name[name].input[1]]).max()) for name in quantized_nodes(nodes)
    }


def quantize_fp8_weight(weight: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with one scale.

    Returns the raw fp8 bytes and ``scale = max(abs(weight)) / 448``, which is also the
    Dequantize scale in the engine.
    """
    weight = np.asarray(weight, dtype=np.float32)
    amax = float(np.abs(weight).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def load_activation_scales(key: str, path: Path | None = None) -> dict[str, dict]:
    """Return the committed entry (``activation_scales``, ``weight_amax``) of one engine."""
    path = path or Path(__file__).with_name(ACTIVATION_SCALES_FILENAME)
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"foundationpose FP8 build requires {path}") from error
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: {sorted(table)}. "
            "Run calibrate_fp8_scales.py for this model."
        )
    return table[key]


class QuantContext:
    """Routes the calibrated convolutions through FP8 Q/DQ; every other layer is unchanged."""

    def __init__(self, scales: dict[str, float]) -> None:
        self.scales = dict(scales)
        # trt.Weights keeps a raw pointer, so every host buffer must outlive the build.
        self.keep_alive: list[np.ndarray] = []

    def has(self, node_name: str) -> bool:
        return node_name in self.scales

    def _scale(self, network, value: float):
        array = np.asarray([value], dtype=np.float32)
        self.keep_alive.append(array)
        tensor = network.add_constant((1,), trt.Weights(array)).get_output(0)
        return network.add_cast(tensor, DEQUANTIZE_TYPE).get_output(0)

    def _activation(self, network, tensor, node_name: str, label: str):
        scale = max(float(self.scales[node_name]), _MIN_ACTIVATION_SCALE)
        quantize = network.add_quantize(tensor, self._scale(network, scale), trt.DataType.FP8)
        quantize.name = f"{label}.input_quantize"
        dequantize = network.add_dequantize(
            quantize.get_output(0), self._scale(network, scale), DEQUANTIZE_TYPE
        )
        dequantize.name = f"{label}.input_dequantize"
        return dequantize.get_output(0)

    def _weight(self, network, weight: np.ndarray, label: str):
        packed, scale = quantize_fp8_weight(weight)
        self.keep_alive.append(packed)
        constant = network.add_constant(
            tuple(packed.shape),
            trt.Weights(trt.DataType.FP8, packed.ctypes.data, int(packed.size)),
        )
        constant.name = f"{label}.weight_fp8"
        dequantize = network.add_dequantize(
            constant.get_output(0), self._scale(network, scale), DEQUANTIZE_TYPE
        )
        dequantize.name = f"{label}.weight_dequantize"
        return dequantize.get_output(0)

    def convolution(
        self,
        network,
        tensor,
        weight,
        bias,
        node_name: str,
        label: str,
        *,
        stride: int,
        padding: int,
    ):
        """Convolution with an FP8 weight and an FP8 input; returns the input's type."""
        bias_array = np.ascontiguousarray(bias, dtype=np.float16)
        self.keep_alive.append(bias_array)
        convolution = network.add_convolution_nd(
            self._activation(network, tensor, node_name, label),
            int(weight.shape[0]),
            tuple(int(value) for value in weight.shape[2:]),
            trt.Weights(),
            trt.Weights(bias_array),
        )
        convolution.set_input(1, self._weight(network, weight, label))
        convolution.stride_nd = (stride, stride)
        convolution.padding_nd = (padding, padding)
        convolution.name = label
        output = convolution.get_output(0)
        if output.dtype != tensor.dtype:
            output = network.add_cast(output, tensor.dtype).get_output(0)
        return output


def refuse_quantized_artifact(nodes, arrays: dict[str, np.ndarray]) -> None:
    """Raise when the ONNX weight artifact already carries quantization."""
    if any(node.op_type in _QUANTIZE_OPS for node in nodes) or any(
        name.endswith(("_scale", "_zero_point")) for name in arrays
    ):
        raise ValueError(
            "FoundationPose weight artifact is already quantized (Q/DQ nodes or scale "
            "tensors); FP8 self-quantization needs the fp32 NGC ONNX files"
        )


def quant_context(nodes, arrays, kind: str, *, scales_path: Path | None = None) -> QuantContext:
    """Pair the committed activation scales with this ONNX weight artifact."""
    refuse_quantized_artifact(nodes, arrays)
    entry = load_activation_scales(scale_key(kind), scales_path)
    table = entry["activation_scales"]
    names = quantized_nodes(nodes)
    expected = entry["weight_amax"]
    actual = weight_amax(nodes, arrays)
    if set(expected) != set(actual) or any(
        not np.isclose(actual[name], expected[name], rtol=1e-6, atol=0.0) for name in actual
    ):
        raise NotImplementedError(
            f"the FoundationPose {kind} FP8 scales were calibrated for the NGC 1.0.1 weights and "
            "do not match these weights; run calibrate_fp8_scales.py for them"
        )
    missing = [name for name in names if name not in table]
    if missing:
        raise ValueError(
            f"no FP8 activation scale entry for {len(missing)} FoundationPose {kind} layers, "
            f"e.g. {missing[:3]}; run calibrate_fp8_scales.py again"
        )
    unknown = sorted(set(table) - set(names))
    if unknown:
        raise ValueError(f"FP8 scales for layers that are not quantized: {unknown[:3]}")
    selected = {}
    for name in names:
        value = table[name]
        if value is None:
            continue  # outlier stored as null: the layer stays fp16
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"FP8 activation scale of {name} must be finite and positive")
        selected[name] = float(value)
    return QuantContext(selected)


def configure_build(config) -> None:
    """Keep layer names and kernel names in the plan for inspection."""
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
