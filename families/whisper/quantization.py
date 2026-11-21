# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whisper-owned FP8 TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) is self-quantization of the original checkpoint, because
no ModelOpt FP8 checkpoint exists for Whisper. Weights are quantized at build time,
one projection at a time: ``scale = max(abs(weight)) / 448`` (448 is the largest
finite e4m3 value), and the FP8 bytes are fed to the engine as a constant. The
activation scale of each projection input needs real audio, so it was calibrated once
offline with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is read
from ``fp8_activation_scales.json``. Users do not run calibration. A projection with no
entry in that file stays unquantized. Each quantized projection becomes
``activation -> Quantize -> Dequantize -> MatMul`` with ``FP8 weight -> Dequantize`` on
the other input.

Quantized projections (internal weight names, ``N`` is the layer index):

  encoder  ``enc_layer.N.w_{q,k,v,fc1,fc2}``
  decoder  ``layer.N.w_{q,k,v,fc1,fc2}`` (self-attention and MLP) and
           ``layer.N.cross_w_{q,k,v,o}`` (cross-attention)

The self-attention output projections ``enc_layer.N.w_o`` and ``layer.N.w_o`` stay
unquantized (see ``_UNQUANTIZED``).

The convolution stem, the embeddings, the layer norms, the biases and the output
projection stay unquantized. Weights keep the checkpoint's ``[out_features,
in_features]`` layout, so the matmul transposes the weight operand.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import tensorrt as trt

from .checkpoint_mapper import _has_tensor, _load_tensor, _open_safetensors

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

_SUPPORTED = ("fp8",)
# (HF projection stem, internal weight-dict stem)
_ATTENTION_PROJECTIONS = (
    ("q_proj", "q"),
    ("k_proj", "k"),
    ("v_proj", "v"),
    ("out_proj", "o"),
)
_MLP_PROJECTIONS = (("fc1", "fc1"), ("fc2", "fc2"))
# The self-attention output projections (encoder and decoder) stay unquantized: TensorRT 11.4
# fails to compile an FP8 Quantize right after the fused attention. Cross-attention and the
# MLP are not affected. They hold about 1/16 (decoder) and 1/12 (encoder) of a layer's weights.
_UNQUANTIZED = re.compile(r"^(enc_)?layer\.\d+\.w_o$")


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Activation scale and the packed weight of one quantized projection."""

    input_scale: float
    weight: _FP8Weight


def _scale_const(network, graph_ops, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    return graph_ops.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
    """Add ``activation @ weight.T`` with FP8 weights and a static FP8 activation scale.

    The Dequantize layers and the matmul run in float32 and the result is cast back.
    With fp16 Dequantize outputs, TensorRT's FP8 GEMM keeps the raw accumulator in
    fp16 and applies the scales afterwards, so it overflows to NaN as soon as an
    output exceeds about ``65504 * input_scale * weight_scale``. Large outputs do
    occur in real models.
    """
    w = scales.weight
    n, k = w.out_features, w.in_features

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, graph_ops, w.weight_scale), trt.float32
    )

    act_scale = _scale_const(network, graph_ops, scales.input_scale)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, trt.float32)

    output = network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    if output.dtype == activation.dtype:
        return output
    return network.add_cast(output, activation.dtype).get_output(0)


@dataclass(frozen=True)
class WhisperQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection is FP8 (selects the FP8 build options)."""
        return bool(self.scales)

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        """True when ``weight_name`` (for example ``layer.3.cross_w_k``) is a quantized projection."""
        return weight_name in self.scales

    def maybe_quantized_matmul(
        self,
        network,
        lhs,
        lhs_width: int,
        rhs_width: int,
        rhs_weights: np.ndarray,
        weight_name: str,
        dtype: np.dtype = np.float32,
    ):
        """Quantized matmul for quantized projections; the regular constant matmul otherwise."""
        entry = self.scales.get(weight_name)
        if entry is None:
            return self.graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        return _wrap_fp8_matmul(
            network, lhs, entry, graph_ops=self.graph_ops, keep_alive=self.keep_alive
        )


def scale_key(config) -> str:
    """Key of this model's entry in the activation-scale file."""
    raw = config.raw
    return (
        f"{str(config.model_type).lower()}"
        f"-h{int(config.hidden_size)}"
        f"-e{int(raw.get('encoder_layers', config.num_hidden_layers))}"
        f"-d{int(raw.get('decoder_layers', config.num_hidden_layers))}"
        f"-m{int(raw.get('num_mel_bins', 80))}"
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


def load_activation_scales(config) -> dict[str, float]:
    """Return the committed activation scales for this model shape, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Whisper FP8 build requires {path}, which was not found") from error
    key = scale_key(config)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model shape."
        )
    return table[key]


def _projections(config) -> list[tuple[str, str]]:
    """(HF module path, internal weight name) of every quantizable projection."""
    raw = config.raw
    enc_layers = int(raw.get("encoder_layers", config.num_hidden_layers))
    dec_layers = int(raw.get("decoder_layers", config.num_hidden_layers))
    found: list[tuple[str, str]] = []
    for layer in range(enc_layers):
        hf = f"model.encoder.layers.{layer}"
        for hf_stem, stem in _ATTENTION_PROJECTIONS:
            found.append((f"{hf}.self_attn.{hf_stem}", f"enc_layer.{layer}.w_{stem}"))
        for hf_stem, stem in _MLP_PROJECTIONS:
            found.append((f"{hf}.{hf_stem}", f"enc_layer.{layer}.w_{stem}"))
    for layer in range(dec_layers):
        hf = f"model.decoder.layers.{layer}"
        for hf_stem, stem in _ATTENTION_PROJECTIONS:
            found.append((f"{hf}.self_attn.{hf_stem}", f"layer.{layer}.w_{stem}"))
            found.append((f"{hf}.encoder_attn.{hf_stem}", f"layer.{layer}.cross_w_{stem}"))
        for hf_stem, stem in _MLP_PROJECTIONS:
            found.append((f"{hf}.{hf_stem}", f"layer.{layer}.w_{stem}"))
    return found


def calibrate_whisper(model_dir: Path, graph_ops, quantization: str, config) -> WhisperQuantContext:
    """Quantize the checkpoint one projection at a time and pair it with the stored scales."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"Whisper supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    activation_scales = load_activation_scales(config)
    readers = _open_safetensors(Path(model_dir))
    scales: dict[str, _LayerScales] = {}
    for hf_prefix, name in _projections(config):
        if _has_tensor(readers, f"{hf_prefix}.weight_scale"):
            raise ValueError(
                f"{hf_prefix} is already quantized; fp8 self-quantization needs the original checkpoint"
            )
        input_scale = activation_scales.get(name)
        if input_scale is None or _UNQUANTIZED.match(name):
            continue
        key = f"{hf_prefix}.weight"
        if not _has_tensor(readers, key):
            raise ValueError(f"checkpoint has no {key}")
        weight = _load_tensor(readers, key)
        packed, weight_scale = quantize_fp8_weight(weight)
        out_features, in_features = weight.shape
        scales[name] = _LayerScales(
            float(input_scale), _FP8Weight(packed, weight_scale, out_features, in_features)
        )
    if not scales:
        raise RuntimeError("Whisper FP8 self-quantization matched no projection in the checkpoint")
    return WhisperQuantContext(scales=scales, graph_ops=graph_ops)
