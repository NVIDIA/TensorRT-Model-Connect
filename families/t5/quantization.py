# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T5-owned FP8 self-quantization and TensorRT Q/DQ graph context.

The original T5 checkpoint is used as is; no FP8 checkpoint is needed.

Weights are quantized on the fly at build time, one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and the FP8 bytes
are fed to the engine as a constant. The weights are taken from ``_T5Model.load_weights``
(``[in, out]`` layout), so the quantized weight is the one the unquantized engine uses. The
activation scale of each projection input needs real data, so it was calibrated once offline
with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is read from
``fp8_activation_scales.json``. Users do not run calibration. A projection with no entry in that
file is an error.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize(act type) -> MatMul(act type)
    FP8 weight -> Dequantize(act type) -----------------------^

Both Dequantize layers output the activation type of the build, so the FP8 GEMM writes that type
and the ReLU after ``wi`` stays in the GEMM epilogue. FP8 works with both precisions of the family:
``fp16`` (activation type fp16) and ``fp32`` (activation type fp32, the FP8 GEMM writes fp32). Use
``fp32`` for t5-base and larger: their activations exceed the fp16 range (the feed-forward output
``wo`` and the residual stream reach 1e5 and more), so the fp16 engine of those models gives
non-finite logits with or without FP8.

Quantized projections (internal weight names, ``N`` is the layer index):

  encoder  ``enc_layer.N.w_{q,k,v,fc1,fc2}`` (self-attention q/k/v, ``DenseReluDense.wi/wo``)
  decoder  ``layer.N.w_{q,k,v,fc1,fc2}`` (self-attention q/k/v, ``DenseReluDense.wi/wo``) and
           ``layer.N.cross_w_{q,k,v}`` (cross-attention q/k/v)

A projection whose input has a massive activation (a scale more than 20 times the median of the
same projection on the same side, see ``calibrate_fp8_scales.split_outlier_scales``) is not
quantized; its entry in the scale file is ``null``. Not quantized: the attention output projections
``enc_layer.N.w_o``, ``layer.N.w_o`` and ``layer.N.cross_w_o`` (TensorRT 11.4 cannot compile an FP8
Quantize right after the fused attention; TensorRT 11.1 builds them), the shared embedding, the
relative position bias, the RMSNorms and ``lm_head``.

Scale-file key: the model config plus the fp16 bits of the first three encoder query weights,
so two checkpoints that share a config but not weights do not share activation scales.
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

ENCODER_STEMS = ("w_q", "w_k", "w_v", "w_fc1", "w_fc2")
DECODER_STEMS = (
    "w_q",
    "w_k",
    "w_v",
    "cross_w_q",
    "cross_w_k",
    "cross_w_v",
    "w_fc1",
    "w_fc2",
)
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"


def weights_fingerprint(weights) -> str:
    """Return the fp16 bit patterns of the first encoder query weights; they tell checkpoints apart."""
    first = np.ascontiguousarray(
        np.asarray(weights["enc_layer.0.w_q"]).reshape(-1)[:3], dtype=np.float16
    )
    return "".join(f"{bits:04x}" for bits in first.view(np.uint16))


def _layer_counts(config) -> tuple[int, int]:
    raw = config.raw
    layers = int(raw.get("num_layers", config.num_hidden_layers))
    return (
        int(raw.get("num_encoder_layers", layers)),
        int(raw.get("num_decoder_layers", layers)),
    )


def scale_key(config, weights) -> str:
    """Key of this checkpoint's entry in the activation-scale file."""
    raw = config.raw
    enc_layers, dec_layers = _layer_counts(config)
    return (
        f"{str(config.model_type).lower()}"
        f"-h{int(raw.get('d_model', config.hidden_size))}"
        f"-e{enc_layers}"
        f"-d{dec_layers}"
        f"-f{int(raw.get('d_ff', config.intermediate_size))}"
        f"-a{int(raw.get('num_heads', config.num_attention_heads))}"
        f"x{int(raw.get('d_kv', 64))}"
        f"-{weights_fingerprint(weights)}"
    )


def projection_names(config) -> list[str]:
    """Internal weight names of every quantized projection of this model config."""
    enc_layers, dec_layers = _layer_counts(config)
    names = [f"enc_layer.{layer}.{stem}" for layer in range(enc_layers) for stem in ENCODER_STEMS]
    names += [f"layer.{layer}.{stem}" for layer in range(dec_layers) for stem in DECODER_STEMS]
    return names


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Static activation scale and FP8 weight of one quantized projection."""

    input_scale: float
    weight: _FP8Weight


def _scale_const(network, graph_ops, value: float):
    """Add a one-element fp32 scale constant to the network."""
    return graph_ops.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
    """Add the FP8 Q/DQ matmul for one projection; Dequantize and matmul use the activation type."""
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
class T5QuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    graph_ops: Any
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Give each context its own list for buffers that must outlive the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        """Return True when this weight has a calibrated activation scale."""
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
        """Add the projection as an FP8 matmul, or as the unquantized matmul when it has no scale."""
        entry = self.scales.get(weight_name)
        if entry is None:
            return self.graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        return _wrap_fp8_matmul(
            network, lhs, entry, graph_ops=self.graph_ops, keep_alive=self.keep_alive
        )


def configure_fp8_build(trt_config, quant_ctx) -> None:
    """Apply the FP8 build options; a no-op for an unquantized build."""
    if quant_ctx is None:
        return
    # Keeps per-layer tactic names in the plan so FP8 lowering can be inspected.
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # TensorRT's fused dual-GEMM kernel fails to compile with scalar-scale FP8 inputs. The route
    # is set only when this TensorRT build lists the knob.
    if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes and ``scale = max(abs(weight)) / 448``; the same
    scale is the dequantize scale in the engine.
    """
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight_fp32 / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def load_activation_scales(config, weights) -> dict[str, float | None]:
    """Return the committed activation scales for this checkpoint, or raise when none exist.

    An entry of ``None`` marks a projection the calibration left unquantized on purpose.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"T5 FP8 build requires {path}, which was not found") from error
    key = scale_key(config, weights)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def reject_quantized_checkpoint(model_dir, config) -> None:
    """Raise when the checkpoint is already quantized; FP8 self-quantization needs 16-bit weights."""
    from .checkpoint_mapper import _open_safetensors

    if config.raw.get("quantization_config"):
        raise ValueError(
            "the checkpoint is already quantized; FP8 self-quantization needs fp32/bf16"
        )
    names = _open_safetensors(Path(model_dir)).tensor_map
    if any(str(name).endswith((".weight_scale", ".input_scale")) for name in names):
        raise ValueError(
            "the checkpoint is already quantized; FP8 self-quantization needs fp32/bf16"
        )


def calibrate_t5_fp8(config, weights, graph_ops) -> T5QuantContext:
    """Build the Q/DQ context from the loaded T5 weights (``[in, out]`` layout).

    Quantizes one projection at a time, so peak extra memory is one weight tensor. A projection
    with no stored activation scale, or a stored scale for a projection this model config does
    not quantize, is an error.
    """
    if any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights):
        raise ValueError(
            "the checkpoint is already quantized; FP8 self-quantization needs fp32/bf16"
        )
    activation_scales = load_activation_scales(config, weights)

    names = projection_names(config)
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
            continue  # massive-activation projection, left unquantized by the calibration
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        if name not in weights:
            raise ValueError(f"checkpoint has no {name}")
        transposed = np.asarray(weights[name])
        in_features, out_features = transposed.shape
        packed, weight_scale = quantize_fp8_weight(
            np.ascontiguousarray(transposed.T, dtype=np.float32)
        )
        scales[name] = _LayerScales(
            input_scale=float(input_scale),
            weight=_FP8Weight(packed, weight_scale, out_features, in_features),
        )
    if not scales:
        raise RuntimeError("T5 FP8 self-quantization matched no projection in the checkpoint")
    return T5QuantContext(scales=scales, graph_ops=graph_ops)
