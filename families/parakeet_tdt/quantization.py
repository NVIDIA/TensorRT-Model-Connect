# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parakeet TDT-owned FP8 TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) is self-quantization of the original checkpoint, because
no ModelOpt FP8 checkpoint exists for Parakeet TDT. Weights are quantized at build time,
one projection at a time: ``scale = max(abs(weight)) / 448`` (448 is the largest finite
e4m3 value), and the FP8 bytes are fed to the engine as a constant. The activation scale
of each projection input needs real audio, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is read from
``fp8_activation_scales.json``. Users do not run calibration. A projection with no entry
in that file is an error. Each quantized projection becomes
``activation -> Quantize -> Dequantize -> MatMul`` with ``FP8 weight -> Dequantize`` on
the other input.

Quantized projections (internal weight names, ``N`` is the encoder layer index):

  ``el.N.ff{1,2}.w{1,2}``             the two macaron feed-forward halves (about 2/3 of the
                                      encoder weights)
  ``el.N.w_{q,k,v,o}``                the self-attention projections (the attention itself is
                                      built from plain matmuls, not a fused kernel, so the
                                      output projection quantizes on TensorRT 11.1 and 11.4)

The convolution subsampling stem and its output linear, the relative-position projection
(folded into a constant at build time), the convolution module (pointwise and depthwise
convolutions), the layer norms, the biases, the predictor LSTM and the joint network
stay unquantized, as the audio recipe of the team quantization flow prescribes. Weights
are stored ``[in, out]`` in the family weight dict; the FP8 constant is built from the
transposed ``[out, in]`` view and the matmul transposes it back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

_SUPPORTED = ("fp8",)
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"
# (checkpoint module path inside ``encoder.layers.N``, internal weight stem inside ``el.N``)
_PROJECTIONS = (
    ("feed_forward1.linear1", "ff1.w1"),
    ("feed_forward1.linear2", "ff1.w2"),
    ("feed_forward2.linear1", "ff2.w1"),
    ("feed_forward2.linear2", "ff2.w2"),
    ("self_attn.q_proj", "w_q"),
    ("self_attn.k_proj", "w_k"),
    ("self_attn.v_proj", "w_v"),
    ("self_attn.o_proj", "w_o"),
)


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

    Both Dequantize layers output the activation type (fp16), so the FP8 GEMM writes fp16
    and TensorRT fuses the bias and the activation function into its epilogue. float32
    Dequantize makes the GEMM write float32 and adds a cast pass; it was 1 to 6 percent
    slower in the encoder. The risk of fp16 is overflow to Inf or NaN when an output is
    above 65504. A stress scan of the fp16 engine (waveform gain up to 16x, hard clipping at
    0.1 and 0.3 of full scale, full-scale noise, near-silence, and scaled or spiked mel
    features, on 200 LibriSpeech test clips) found no overflow within the range the
    frontend can produce: the largest projection output was 11080 (17% of 65504, a
    feed-forward ``w2``; the attention projections stayed below 2600). Inputs beyond that
    range (mel features above the per-feature normalization bound) gave Inf in the fp16
    engine without FP8 as well.
    """
    w = scales.weight
    n, k = w.out_features, w.in_features

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, graph_ops, w.weight_scale), activation.dtype
    )

    act_scale = _scale_const(network, graph_ops, scales.input_scale)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, activation.dtype)

    return network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)


@dataclass(frozen=True)
class ParakeetQuantContext:
    """Routes the quantized encoder projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        """True when ``weight_name`` (for example ``el.3.ff1.w1``) is a quantized projection."""
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


def configure_fp8_build(trt_config, quant_ctx) -> None:
    """Apply the FP8 build options and keep layer names for kernel inspection."""
    if quant_ctx is None:
        return
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        # TensorRT's fused dual-GEMM kernel fails to compile with scalar-scale FP8.
        # The route is set only when this TensorRT build lists the knob.
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def scale_key(config) -> str:
    """Key of this model's entry in the activation-scale file."""
    return (
        f"{str(config.model_type).lower()}"
        f"-h{int(config.encoder_hidden_size)}"
        f"-e{int(config.encoder_layers)}"
        f"-f{int(config.encoder_ffn_size)}"
        f"-m{int(config.num_mel_bins)}"
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
    """Return the committed activation scales for this model config, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Parakeet TDT FP8 build requires {path}, which was not found"
        ) from error
    key = scale_key(config)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def projections(config) -> list[tuple[str, str]]:
    """(checkpoint module path, internal weight name) of every quantizable projection."""
    return [
        (f"encoder.layers.{layer}.{hf_stem}", f"el.{layer}.{stem}")
        for layer in range(int(config.encoder_layers))
        for hf_stem, stem in _PROJECTIONS
    ]


def reject_prequantized(state: dict) -> None:
    """Refuse a checkpoint that already stores quantization scales."""
    scaled = sorted(key for key in state if key.endswith((".weight_scale", ".input_scale")))
    if scaled:
        raise ValueError(
            f"{scaled[0]} is a quantization scale; fp8 self-quantization needs the original "
            "checkpoint (already quantized checkpoints are not supported)"
        )


def calibrate_parakeet(weights, graph_ops, quantization: str, config) -> ParakeetQuantContext:
    """Quantize the encoder projections one at a time and pair them with the stored scales.

    ``weights`` is the family weight dict, whose projection matrices are ``[in, out]``.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"Parakeet TDT supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    activation_scales = load_activation_scales(config)
    scales: dict[str, _LayerScales] = {}
    for _, name in projections(config):
        input_scale = activation_scales.get(name)
        if input_scale is None:
            raise ValueError(
                f"{name} has no activation scale in {ACTIVATION_SCALES_FILENAME}; "
                "run calibrate_fp8_scales.py"
            )
        if name not in weights:
            raise ValueError(f"weight dict has no {name}")
        weight = np.ascontiguousarray(np.asarray(weights[name], dtype=np.float32).T)
        packed, weight_scale = quantize_fp8_weight(weight)
        out_features, in_features = weight.shape
        scales[name] = _LayerScales(
            float(input_scale), _FP8Weight(packed, weight_scale, out_features, in_features)
        )
    if not scales:
        raise RuntimeError("Parakeet TDT FP8 self-quantization matched no projection")
    return ParakeetQuantContext(scales=scales, graph_ops=graph_ops)
