# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apertus FP8 self-quantization of the BF16 checkpoint at build time.

No ModelOpt FP8 checkpoint of an Apertus model exists (the public quantized repositories are
compressed-tensors exports), so the original BF16 checkpoint is quantized while the engine is
built. FP8 weights are never stored.

Projections: the q, k, v and output projections of the attention and the up and down
projections of the xIELU MLP (Apertus has no gate projection) of every layer. The embedding,
the norms and ``lm_head`` stay BF16. The attention of this family is explicit (matmul, FP32
softmax, matmul), so the output projection can take an FP8 input as well.

W8A8: ``weight_scale = amax(W) / 448`` per projection, computed from the BF16 weight on every
build. The static per-tensor activation scale of each projection input was calibrated once
offline with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``, max calibration) and is
read from ``fp8_activation_scales.json``, keyed by the model config. A projection stored as
``null`` (its input has a massive activation) stays BF16; a missing entry is an error. Graph::

    activation -> Quantize -> Dequantize(bf16) -> MatMul -> bf16
    FP8 weight -> Dequantize(bf16) ------------^
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import ml_dtypes
import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
FORMATS = ("fp8",)

# Hugging Face module names under ``model.layers.N``; they are also the scale-file names.
PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def scale_key(config) -> str:
    """Key of this model config in the activation-scale file."""
    return (
        f"apertus-h{config.hidden_size}-l{config.num_hidden_layers}"
        f"-i{config.intermediate_size}-kv{config.num_key_value_heads}-v{config.vocab_size}"
    )


def projection_name(layer: int, projection: str) -> str:
    """Scale-file name of one projection, ``layers.N.<module>``."""
    return f"layers.{layer}.{projection}"


@dataclass(frozen=True)
class FP8Weight:
    """One projection weight as raw e4m3 bytes ``[out, in]`` with its scales."""

    packed: np.ndarray
    weight_scale: float
    input_scale: float


def quantize_fp8_weight(weight: np.ndarray) -> tuple[np.ndarray, float]:
    """Raw e4m3 bytes of ``weight / scale`` and ``scale = amax / 448``."""
    weight = np.asarray(weight, dtype=np.float32)
    amax = float(np.abs(weight).max()) if weight.size else 0.0
    if not np.isfinite(amax) or amax <= 0.0:
        raise ValueError("weight has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight / np.float32(scale), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    return np.ascontiguousarray(scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)), scale


def load_activation_scales(config) -> dict[str, float | None]:
    """The committed activation scales of this model config; raise if it was not calibrated."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    key = scale_key(config)
    if key not in table:
        raise NotImplementedError(
            f"apertus has no calibrated FP8 activation scales for {key!r} (available: "
            f"{sorted(table)}); run families/apertus/calibrate_fp8_scales.py for this model"
        )
    return table[key]


_QUANTIZED_NAME = re.compile(r"\.(weight_scale|weight_scale_2|input_scale|weight_packed)$")


def reject_quantized_checkpoint(model_dir: Path, tensor_names) -> None:
    """Self-quantization needs the original BF16 checkpoint, not a quantized export."""
    config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    if config.get("quantization_config") or (Path(model_dir) / "hf_quant_config.json").exists():
        raise ValueError(
            "the apertus checkpoint is already quantized; the FP8 build quantizes the original "
            "BF16 checkpoint"
        )
    scaled = sorted(name for name in tensor_names if _QUANTIZED_NAME.search(name))
    if scaled:
        raise ValueError(
            "the apertus checkpoint is already quantized (it has tensors such as "
            f"{scaled[0]}); the FP8 build quantizes the original BF16 checkpoint"
        )


@dataclass
class QuantContext:
    """FP8 projection weights by scale-file name; other projections stay BF16."""

    weights: dict[str, FP8Weight]
    keep_alive: list = field(default_factory=list)

    def matmul(self, network: trt.INetworkDefinition, activation: trt.ITensor, name: str):
        """``activation @ W.T`` as an FP8 matmul, or ``None`` when the projection stays BF16."""
        weight = self.weights.get(name)
        if weight is None:
            return None
        dtype = activation.dtype
        out_features, in_features = weight.packed.shape
        # trt.Weights keeps a raw pointer: the buffers must live until serialization.
        self.keep_alive.append(weight.packed)
        constant = network.add_constant(
            (out_features, in_features),
            trt.Weights(trt.DataType.FP8, weight.packed.ctypes.data, weight.packed.size),
        ).get_output(0)
        w = network.add_dequantize(constant, self._scalar(network, weight.weight_scale), dtype)
        scale = self._scalar(network, weight.input_scale)
        q = network.add_quantize(activation, scale, trt.DataType.FP8)
        a = network.add_dequantize(q.get_output(0), scale, dtype)
        return network.add_matrix_multiply(
            a.get_output(0),
            trt.MatrixOperation.NONE,
            w.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)

    def _scalar(self, network, value: float) -> trt.ITensor:
        array = np.asarray([value], dtype=np.float32)
        self.keep_alive.append(array)
        return network.add_constant((1,), trt.Weights(array)).get_output(0)


def configure_build(builder_config, quant: QuantContext | None) -> None:
    """Quantized plans keep per-layer kernel names, so the FP8 lowering can be inspected."""
    if quant is not None:
        builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED


def build_quant_context(config, weights, fmt: str) -> QuantContext:
    """Quantize the loaded BF16 projection weights to FP8, one at a time."""
    if fmt not in FORMATS:
        raise ValueError(f"apertus quantization must be one of {FORMATS}, got {fmt!r}")
    scales = load_activation_scales(config)
    quantized: dict[str, FP8Weight] = {}
    for index, layer in enumerate(weights.layers):
        for projection in PROJECTIONS:
            name = projection_name(index, projection)
            if name not in scales:
                raise ValueError(
                    f"{name} has no entry in {ACTIVATION_SCALES_FILENAME}; "
                    "run calibrate_fp8_scales.py"
                )
            input_scale = scales[name]
            if input_scale is None:
                continue  # left BF16 on purpose by the calibration (massive activation)
            if not np.isfinite(input_scale) or input_scale <= 0.0:
                raise ValueError(f"activation scale of {name} must be finite and positive")
            packed, weight_scale = quantize_fp8_weight(getattr(layer, projection.split(".")[1]))
            quantized[name] = FP8Weight(packed, weight_scale, float(input_scale))
    if not quantized:
        raise RuntimeError("apertus FP8 quantization matched no projection")
    return QuantContext(weights=quantized)
