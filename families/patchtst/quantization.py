# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTST-owned FP8 self-quantization and TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) quantizes the original checkpoint at build time, because no
FP8 checkpoint exists for PatchTST. Weights are quantized one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and the FP8
bytes are fed to the engine as a constant. FP8 weights are never stored. The activation
scale of each projection input needs real data, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``, max calibration, ETT training
windows) and is read from ``fp8_activation_scales.json``. Users do not run calibration.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize(fp16) -> MatMul -> + bias
    FP8 weight -> Dequantize(fp16) -----------------^

Quantized (recipe: the encoder linears, as for small transformer encoders): per encoder
layer the shared query/key/value input (``self_attn.qkv``: q, k and v read the same norm
output and share one input scale, each keeps its own weight scale) and the feed-forward
``ff.0`` and ``ff.3``. The attention output projection (``self_attn.out_proj``) stays in
fp16: on TensorRT 11.4 an FP8 Quantize right after the fused attention does not compile.
The patch embedding (12 inputs), the positional encoding, the norms, the attention, the
scaling and the head stay as in the fp16 build.

A projection whose calibrated input scale is more than 20 times the median of the same
projection in the other layers has a massive activation: its entry in the scale file is
``null`` and it stays in fp16 on purpose. A projection missing from the file is an error.
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
# Smallest normal fp16 value; a smaller Q/DQ scale would lose precision as an fp16 constant.
_FP16_MIN_NORMAL = 2.0**-14
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"

_SUPPORTED = ("fp8",)
# Quantized modules of one encoder layer. ``self_attn.qkv`` stands for q, k and v, which
# read the same input and share one input scale.
LAYER_PROJECTIONS = ("self_attn.qkv", "ff.0", "ff.3")
_GROUPS = {"self_attn.qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")}


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [in_features, out_features] raw fp8_e4m3 bytes
    weight_scale: float
    in_features: int
    out_features: int
    bias: np.ndarray | None


@dataclass(frozen=True)
class _ModuleScales:
    """Static activation scale and the FP8 weights (q, k, v for ``self_attn.qkv``)."""

    input_scale: float
    weights: tuple[_FP8Weight, ...]


def _scale_const(network, value: float):
    """One-element fp16 constant holding a Q/DQ scale (the Dequantize output type)."""
    data = np.asarray([value], dtype=np.float16)
    return network.add_constant((1,), trt.Weights(data)).get_output(0)


@dataclass
class PatchTSTQuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized.

    The channel rows of one layer share their weights, so the dequantized FP8 weight of a
    projection is added to the network once and reused by every channel.
    """

    scales: dict[str, _ModuleScales]
    keep_alive: list = field(default_factory=list)
    _weights: dict = field(default_factory=dict)

    def should_quantize(self, module: str) -> bool:
        """True when ``module`` (for example ``layers.0.ff.0``) is FP8."""
        return module in self.scales

    def quantize_input(self, network, activation, module: str):
        """``activation -> Quantize -> Dequantize`` with the static per-tensor scale."""
        if activation.dtype != trt.float16:
            raise ValueError(f"PatchTST FP8 needs an fp16 activation, got {activation.dtype}")
        act_scale = _scale_const(network, self.scales[module].input_scale)
        quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
        return network.add_dequantize(quant.get_output(0), act_scale, trt.float16).get_output(0)

    def _dequantized_weight(self, network, module: str, part: int, rank: int):
        key = (id(network), module, part, rank)
        if key not in self._weights:
            weight = self.scales[module].weights[part]
            shape = (1,) * max(0, rank - 2) + (weight.in_features, weight.out_features)
            # trt.Weights holds a raw pointer, so the buffer must outlive the build.
            self.keep_alive.append(weight.packed)
            w_fp8 = trt.Weights(
                trt.DataType.FP8,
                weight.packed.ctypes.data,
                weight.in_features * weight.out_features,
            )
            w_const = network.add_constant(shape, w_fp8).get_output(0)
            self._weights[key] = network.add_dequantize(
                w_const, _scale_const(network, weight.weight_scale), trt.float16
            ).get_output(0)
        return self._weights[key]

    def linear(self, network, dequantized_activation, module: str, part: int = 0):
        """``activation @ weight + bias`` for weight number ``part`` of ``module``."""
        rank = len(tuple(dequantized_activation.shape))
        out = network.add_matrix_multiply(
            dequantized_activation,
            trt.MatrixOperation.NONE,
            self._dequantized_weight(network, module, part, rank),
            trt.MatrixOperation.NONE,
        ).get_output(0)
        bias = self.scales[module].weights[part].bias
        if bias is None:
            return out
        shape = (1,) * (rank - 1) + (bias.size,)
        self.keep_alive.append(bias)
        bias_t = network.add_constant(shape, trt.Weights(bias.reshape(shape))).get_output(0)
        return network.add_elementwise(out, bias_t, trt.ElementWiseOperation.SUM).get_output(0)


def configure_fp8_build(trt_config, quant_ctx) -> None:
    """Keep per-layer tactic names in a quantized plan so FP8 lowering can be inspected."""
    if quant_ctx is None:
        return
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        # TensorRT's fused dual-GEMM kernel fails to compile with scalar-scale FP8.
        # The route is set only when this TensorRT build lists the knob.
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def _abs_sum(array) -> float:
    return float(np.abs(np.asarray(array, dtype=np.float32)).sum(dtype=np.float64))


def scale_key(raw: dict, weights) -> str:
    """Key of this model's entry in the activation-scale file.

    The key holds every config value that changes the activation ranges (model width,
    layers, feed-forward size, heads, context length, patch length and stride, input
    channels). Checkpoints of the same architecture differ in training, so the key also
    holds the numeric weight sums (sum of absolute values, 5 significant digits) of the
    patch embedding and of the first layer's query projection.
    """
    patch = int(raw.get("patch_length", 1))
    return (
        f"patchtst-d{int(raw['d_model'])}-l{int(raw['num_hidden_layers'])}"
        f"-f{int(raw['ffn_dim'])}-n{int(raw['num_attention_heads'])}"
        f"-c{int(raw['context_length'])}-p{patch}-s{int(raw.get('patch_stride', patch))}"
        f"-ch{int(raw['num_input_channels'])}"
        f"-w{_abs_sum(weights['model.encoder.embedder.input_embedding.weight']):.4e}"
        f"-{_abs_sum(weights['model.encoder.layers.0.self_attn.q_proj.weight']):.4e}"
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


def load_activation_scales(raw: dict, weights) -> dict[str, float | None]:
    """Return the committed activation scales for this model config, or raise when none exist.

    An entry of ``None`` marks a projection the calibration left unquantized on purpose.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"PatchTST FP8 build requires {path}, which was not found") from error
    key = scale_key(raw, weights)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run families/patchtst/calibrate_fp8_scales.py for this "
            "model config."
        )
    return table[key]


def projection_names(
    raw: dict, layer_projections: tuple[str, ...] = LAYER_PROJECTIONS
) -> list[str]:
    """Module names of every quantized projection of this model config."""
    return [
        f"layers.{layer}.{stem}"
        for layer in range(int(raw.get("num_hidden_layers", 0)))
        for stem in layer_projections
    ]


def weight_prefixes(name: str) -> list[str]:
    """Checkpoint prefixes (``<prefix>.weight``, HF ``[out, in]``) behind one module name."""
    layer, stem = name.split(".", 2)[1], name.split(".", 2)[2]
    return [f"model.encoder.layers.{layer}.{part}" for part in _GROUPS.get(stem, (stem,))]


def make_context(weights, activation_scales: dict, names: list[str]) -> PatchTSTQuantContext:
    """Quantize the weights of ``names`` and pair them with their activation scales."""
    unexpected = sorted(set(activation_scales) - set(names))
    if unexpected:
        raise ValueError(
            f"{ACTIVATION_SCALES_FILENAME} has scales for projections that are not quantized, "
            f"for example {unexpected[0]}; run calibrate_fp8_scales.py for this model config"
        )
    scales: dict[str, _ModuleScales] = {}
    for name in names:
        if name not in activation_scales:
            raise ValueError(
                f"{name} has no activation scale in {ACTIVATION_SCALES_FILENAME}; "
                "run families/patchtst/calibrate_fp8_scales.py"
            )
        input_scale = activation_scales[name]
        if input_scale is None:
            continue  # massive-activation projection, left in fp16 by the calibration
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        parts = []
        for prefix in weight_prefixes(name):
            if f"{prefix}.weight" not in weights:
                raise ValueError(f"checkpoint has no {prefix}.weight")
            # The family loader keeps HF [out, in]; the engine computes activation @ weight.
            matrix = np.asarray(weights[f"{prefix}.weight"], dtype=np.float32).T
            packed, weight_scale = quantize_fp8_weight(matrix)
            if min(input_scale, weight_scale) < _FP16_MIN_NORMAL:
                raise ValueError(f"FP8 scale of {name} is below the smallest normal fp16 value")
            bias = weights.get(f"{prefix}.bias")
            if bias is not None:
                bias = np.ascontiguousarray(bias, dtype=np.float16)
            parts.append(_FP8Weight(packed, weight_scale, matrix.shape[0], matrix.shape[1], bias))
        scales[name] = _ModuleScales(float(input_scale), tuple(parts))
    if not scales:
        raise RuntimeError("PatchTST FP8 self-quantization matched no projection")
    return PatchTSTQuantContext(scales=scales)


def is_quantized_checkpoint(raw: dict, model_dir: Path, weights) -> bool:
    """True for a checkpoint that already carries quantization metadata or scales."""
    return (
        "quantization_config" in raw
        or (Path(model_dir) / "hf_quant_config.json").is_file()
        or any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights)
    )


def calibrate_patchtst(raw: dict, weights, quantization: str, *, precision: str):
    """Quantize the fp32 weights one projection at a time and pair them with their scales.

    A projection with no stored scale, or a stored scale for a projection this model config
    does not quantize, is an error.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"PatchTST supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if precision != "fp16":
        raise ValueError(f"PatchTST FP8 needs precision fp16, got {precision!r}")
    return make_context(weights, load_activation_scales(raw, weights), projection_names(raw))
