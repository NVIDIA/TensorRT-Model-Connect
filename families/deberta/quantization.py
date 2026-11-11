# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeBERTa-owned FP8 TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) is self-quantization of the original checkpoint, because no
ModelOpt FP8 checkpoint exists for DeBERTa. Weights are quantized at build time, one
projection at a time: ``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3
value), and the FP8 bytes are fed to the engine as a constant. FP8 weights are never stored.
The activation scale of each projection input needs real data, so it was calibrated once
offline with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``, max calibration) and
is read from ``fp8_activation_scales.json``, one table per model config and checkpoint.
Users do not run calibration. Each quantized projection becomes
``activation -> Quantize -> Dequantize -> MatMul`` with ``FP8 weight -> Dequantize`` on the
other input.

Recipe (the public recipe for small text encoders, adapted to the per-tensor scales of this
path): every linear of every encoder layer that reads the hidden rows is quantized, that is
the fused ``in_proj`` (query, key and value: one input, so one input scale, and one weight
scale for each of the three parts), the attention output projection and the two
feed-forward projections. The embeddings, the relative position table and its projections
(``pos_proj``, ``pos_q_proj``: they do not depend on the input), the layer norms, the GELU
and the attention matmuls stay in the work type. A projection whose calibrated input scale
is more than 20 times the median of its kind is stored as ``null`` in the table and stays
unquantized.

A model config with no table, or a projection with no entry in the table (a ``null`` entry
is an explicit entry), raises and names the calibration tool: there is no silent fallback.

FP8 builds use the fp16 work type (``model.build`` refuses fp32 and bf16). The Dequantize
layers and the matmul run in the type of the activation, so the GEMM writes fp16 and no Cast
layers are added.
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
CALIBRATION_TOOL = "families/deberta/calibrate_fp8_scales.py"

_SUPPORTED = ("fp8",)
# Table name of each quantized projection (the Hugging Face module name without
# ``deberta.encoder.layer.N.``) and the internal weight names it feeds.
LAYER_PROJECTIONS = {
    "attention.self.in_proj": ("w_q", "w_k", "w_v"),
    "attention.output.dense": ("w_o",),
    "intermediate.dense": ("w_fc1",),
    "output.dense": ("w_fc2",),
}
# A projection whose input scale exceeds this multiple of the median of its kind stays
# unquantized.
OUTLIER_RATIO = 20.0


@dataclass(frozen=True)
class _FP8Weight:
    """One weight stored as raw fp8_e4m3 bytes ``[out, in]`` with one per-tensor scale."""

    packed: np.ndarray
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _Projection:
    """Activation scale and the packed weights of one quantized projection."""

    input_scale: float
    weights: tuple[_FP8Weight, ...]


def _scale_const(network, value: float, dtype):
    """One-element constant holding a Q/DQ scale in the Dequantize output type."""
    np_dtype = np.float16 if dtype == trt.float16 else np.float32
    data = np.asarray([value], dtype=np_dtype)
    return network.add_constant((1,), trt.Weights(data)).get_output(0)


@dataclass
class DebertaQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    projections: dict[str, _Projection]
    keep_alive: list = field(default_factory=list)

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection is FP8 (selects the FP8 build options)."""
        return bool(self.projections)

    def should_quantize(self, name: str) -> bool:
        """True when ``name`` (for example ``encoder.layer.3.output.dense``) is quantized."""
        return name in self.projections

    def project(self, network, activation, name: str):
        """``activation @ W`` for every weight of projection ``name``, or None if unquantized.

        The activation is quantized once and shared by every weight of the projection (the
        query, key and value parts of ``in_proj``). Weights keep the checkpoint's
        ``[out_features, in_features]`` layout, so the matmul transposes the weight operand.
        """
        entry = self.projections.get(name)
        if entry is None:
            return None
        dtype = activation.dtype
        if dtype not in (trt.float16, trt.float32):
            raise ValueError(f"DeBERTa FP8 supports fp16 or fp32 activations, got {dtype}")
        act_scale = _scale_const(network, entry.input_scale, dtype)
        a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
        a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, dtype).get_output(0)
        rank = len(tuple(activation.shape))
        outputs = []
        for weight in entry.weights:
            shape = (1,) * max(0, rank - 2) + (weight.out_features, weight.in_features)
            # trt.Weights holds a raw pointer, so the buffer must outlive the build.
            self.keep_alive.append(weight.packed)
            w_fp8 = trt.Weights(
                trt.DataType.FP8,
                weight.packed.ctypes.data,
                weight.out_features * weight.in_features,
            )
            w_const = network.add_constant(shape, w_fp8).get_output(0)
            w_dequant = network.add_dequantize(
                w_const, _scale_const(network, weight.weight_scale, dtype), dtype
            ).get_output(0)
            outputs.append(
                network.add_matrix_multiply(
                    a_dequant,
                    trt.MatrixOperation.NONE,
                    w_dequant,
                    trt.MatrixOperation.TRANSPOSE,
                ).get_output(0)
            )
        return outputs


def _abs_sum(values) -> float:
    return float(np.abs(np.asarray(values, dtype=np.float64)).sum())


def scale_key(config, weights: dict) -> str:
    """Key of this model's table in the activation-scale file.

    The key holds every config value that changes the activation ranges (hidden size,
    layers, feed-forward width, heads, vocabulary, relative positions). Fine-tuned
    checkpoints share all of them with their base model, so the key also holds two numeric
    weight sums (sum of absolute values, 5 significant digits) of the relative position
    table and of the first layer's query projection.
    """
    deberta_cfg = weights.get("_deberta_config", {})
    return (
        f"deberta-h{config.hidden_size}-l{config.num_hidden_layers}"
        f"-i{config.intermediate_size}-n{config.num_attention_heads}-v{config.vocab_size}"
        f"-r{deberta_cfg.get('max_relative_positions', 0)}"
        f"-w{_abs_sum(weights['rel_embeddings']):.4e}-{_abs_sum(weights['layer.0.w_q']):.4e}"
    )


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes and ``scale = max(abs(weight)) / 448``; the same scale is
    the dequantize scale in the engine.
    """
    weight_fp32 = np.asarray(weight_fp32, dtype=np.float32)
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight_fp32 / np.float32(scale), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def load_activation_scales(key: str) -> dict[str, float | None]:
    """Return the committed activation scales for this model, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"DeBERTa FP8 build requires {path}, which was not found") from error
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run {CALIBRATION_TOOL} for this checkpoint."
        )
    return table[key]


def projections(num_layers: int) -> list[str]:
    """Table names of every quantized projection of the encoder."""
    return [
        f"encoder.layer.{layer}.{stem}" for layer in range(num_layers) for stem in LAYER_PROJECTIONS
    ]


def check_not_quantized(model_dir: Path, config) -> None:
    """Refuse a checkpoint that is already quantized (self-quantization needs the original)."""
    if (
        config.raw.get("quantization_config")
        or (Path(model_dir) / "hf_quant_config.json").is_file()
    ):
        raise ValueError(
            f"{model_dir} is an already quantized checkpoint; DeBERTa fp8 self-quantization "
            "needs the original checkpoint"
        )


def create_quant_context(weights: dict, config, quantization: str) -> DebertaQuantContext:
    """Quantize the loaded weights one projection at a time and pair them with the scales."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"DeBERTa supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    key = scale_key(config, weights)
    activation_scales = load_activation_scales(key)
    names = projections(config.num_hidden_layers)
    missing = sorted(set(names) - set(activation_scales))
    unused = sorted(set(activation_scales) - set(names))
    if missing or unused:
        raise ValueError(
            f"FP8 scale table {key!r} does not match the model: "
            f"{len(missing)} projections without an entry (first: {missing[:3]}), "
            f"{len(unused)} entries without a projection (first: {unused[:3]}). "
            f"Re-run {CALIBRATION_TOOL} for this checkpoint."
        )
    result: dict[str, _Projection] = {}
    for name in names:
        input_scale = activation_scales[name]
        if input_scale is None:
            continue  # outlier projection, stays unquantized
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"FP8 activation scale of {name!r} must be finite and positive")
        layer = int(name.split(".")[2])
        stem = name.split(".", 3)[3]
        packed = []
        for internal in LAYER_PROJECTIONS[stem]:
            tensor = f"layer.{layer}.{internal}"
            if tensor not in weights:
                raise ValueError(f"loaded weights have no {tensor}")
            # Loaded weights are [in, out]; FP8 constants keep the checkpoint [out, in].
            weight = np.ascontiguousarray(np.asarray(weights[tensor], dtype=np.float32).T)
            data, weight_scale = quantize_fp8_weight(weight)
            out_features, in_features = weight.shape
            packed.append(_FP8Weight(data, weight_scale, out_features, in_features))
        result[name] = _Projection(float(input_scale), tuple(packed))
    if not result:
        raise RuntimeError("DeBERTa FP8 self-quantization matched no projection")
    return DebertaQuantContext(projections=result)
