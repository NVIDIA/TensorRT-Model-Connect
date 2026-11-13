# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-owned FP8 self-quantization and TensorRT Q/DQ graph context.

The original BF16 GLM checkpoint is used as-is; no FP8 checkpoint is needed.

Weights are quantized on the fly at build time, one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and
the FP8 bytes are fed to the engine as a constant. The weights are taken after
``_GlmModel.load_weights`` has absorbed the GLM multipliers, so the
quantized weight is the one the unquantized engine would use. The activation
scale of each projection input needs real data, so it was calibrated once
offline with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is
read from ``fp8_activation_scales.json``. Users do not run calibration.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize(fp32) -> MatMul(fp32) -> Cast
    FP8 weight -> Dequantize(fp32) ------------------^

The Dequantize and the matmul run in fp32: a fp16 Dequantize makes the fused
FP8 GEMM overflow to NaN when a projection output is large.

The attention output projection (``w_o``) stays unquantized: TensorRT 11.4
cannot compile an FP8 Quantize right after a fused attention. ``lm_head`` and
the embedding stay unquantized as well. In the FP8 build a projection with no scale entry in
the JSON file stays unquantized; the NVFP4 build refuses it with an error.

NVFP4 (``quantization="nvfp4"``) is also self-quantization of the BF16 checkpoint (no ModelOpt
NVFP4 checkpoint of this model exists). It needs a Blackwell GPU. The recipe
(``NVFP4_RECIPES``) is: q/k/v FP8 W8A8 as above, the gate, up and down projections NVFP4 W4A4,
except the whole MLP of the first two and last two layers and the down projection of layer 4
(its input has massive activations), which stay FP8 W8A8. Weights are quantized at build time:
``weight_scale_2 = amax / (6 * 448)``, one e4m3 block scale per 16 inputs
(``block_amax / 6 / weight_scale_2``, clamped to 448), values rounded to the E2M1 grid and packed
two per byte (first value in the low nibble). The activation goes through ``add_dynamic_quantize``
with a calibrated per-tensor ``input_scale = amax / (6 * 448)`` read from
``nvfp4_activation_scales.json`` (the FP8 projections use ``fp8_activation_scales.json``). The
weight reaches the matmul through a double Dequantize (block scale times ``weight_scale_2``, then
the packed FP4 constant) and TensorRT fuses the pattern into block-scaled FP4 GEMMs. The fused
dual-GEMM kernel (gate and up, SwiGLU and the FP4 quantization of the down input in one kernel)
stays enabled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import tensorrt as trt
import torch

FP8_E4M3_MAX = 448.0
E2M1_MAX = 6.0
NVFP4_BLOCK_SIZE = 16
NVFP4_MAX = E2M1_MAX * FP8_E4M3_MAX
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
NVFP4_ACTIVATION_SCALES_FILENAME = "nvfp4_activation_scales.json"

# HF projection module -> internal weight-dict stems used by graph_blocks. GLM fuses the
# gate and up projections into one HF module (mlp.gate_up_proj) with one input
# quantizer, so both internal weights share its activation scale.
PROJECTIONS = (
    ("self_attn.q_proj", ("w_q",)),
    ("self_attn.k_proj", ("w_k",)),
    ("self_attn.v_proj", ("w_v",)),
    ("mlp.gate_up_proj", ("w_gate", "w_up")),
    ("mlp.down_proj", ("w_down",)),
)
WEIGHT_STEMS = tuple(stem for _, stems in PROJECTIONS for stem in stems)
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"


def scale_key(config) -> str:
    """Key of this model's entry in the activation-scale file."""
    return (
        f"{str(config.model_type).lower()}"
        f"-h{int(config.hidden_size)}"
        f"-l{int(config.num_hidden_layers)}"
        f"-i{int(config.intermediate_size)}"
    )


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _FP4Weight:
    """One projection weight in NVFP4: packed E2M1 bytes, e4m3 block scales, one global scale."""

    packed: np.ndarray  # [out_features, in_features / 2] uint8, first value in the low nibble
    block_scale: np.ndarray  # [out_features, in_features / 16] raw e4m3 bytes
    weight_scale_2: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Static activation scale and quantized weight (FP8 or NVFP4) of one projection."""

    input_scale: float
    weight: _FP8Weight | _FP4Weight


def _scale_const(network, graph_ops, value: float):
    """Add a one-element fp32 scale constant to the network."""
    return graph_ops.add_constant(
        network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
    )


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
    """Add the FP8 Q/DQ matmul for one projection, with fp32 Dequantize and matmul."""
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


def _wrap_nvfp4_matmul(network, activation, scales: _LayerScales, *, graph_ops, keep_alive: list):
    """Add ``activation @ weight.T`` with packed FP4 weights and block-scaled FP4 activations.

    The activation is quantized at run time with ``add_dynamic_quantize`` (per-16 e4m3 block
    scales, global scale = the calibrated ``input_scale``). The weight is an FP4 constant that
    reaches the matmul through a double Dequantize (block scale times ``weight_scale_2``, then
    the weight). TensorRT fuses the pattern into a block-scaled FP4 GEMM.
    """
    output_dtype = activation.dtype
    w = scales.weight
    n, k = w.out_features, w.in_features

    # trt.Weights holds a raw pointer, so the buffers must outlive the build.
    keep_alive.append(w.packed)
    keep_alive.append(w.block_scale)

    w_fp4 = trt.Weights(trt.DataType.FP4, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp4).get_output(0)
    w_fp8_bs = trt.Weights(trt.DataType.FP8, w.block_scale.ctypes.data, w.block_scale.size)
    w_bs_const = network.add_constant((n, k // NVFP4_BLOCK_SIZE), w_fp8_bs).get_output(0)
    w_gs = _scale_const(network, graph_ops, w.weight_scale_2)
    w_effective_scale = network.add_dequantize(w_bs_const, w_gs, output_dtype)
    w_dequant = network.add_dequantize(w_const, w_effective_scale.get_output(0), output_dtype)
    w_dequant.axis = 1

    act_gs = _scale_const(network, graph_ops, scales.input_scale)
    a_dynq = network.add_dynamic_quantize(
        activation, 1, NVFP4_BLOCK_SIZE, trt.DataType.FP4, trt.DataType.FP8
    )
    a_dynq.set_input(1, act_gs)
    a_effective_scale = network.add_dequantize(a_dynq.get_output(1), act_gs, output_dtype)
    a_dequant = network.add_dequantize(
        a_dynq.get_output(0), a_effective_scale.get_output(0), output_dtype
    )
    a_dequant.axis = 1

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
class GlmQuantContext:
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

    @property
    def has_fp4_weights(self) -> bool:
        """True when any projection is NVFP4."""
        return any(isinstance(entry.weight, _FP4Weight) for entry in self.scales.values())

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
        wrap = _wrap_nvfp4_matmul if isinstance(entry.weight, _FP4Weight) else _wrap_fp8_matmul
        return wrap(network, lhs, entry, graph_ops=self.graph_ops, keep_alive=self.keep_alive)


def configure_fp8_build(trt_config, quant_ctx) -> None:
    """Apply the FP8 build options; a no-op for an unquantized build."""
    if quant_ctx is None:
        return
    # Keeps per-layer tactic names in the plan so FP8 lowering can be inspected.
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # NVFP4 builds keep the fused dual-GEMM (gate/up) kernel. With FP8 only, it fails to compile
    # with scalar-scale FP8 inputs, so the route is set only when this TensorRT build lists the knob.
    if getattr(quant_ctx, "has_fp4_weights", False):
        return
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


def load_activation_scales(config) -> dict[str, float]:
    """Return the committed activation scales for this model shape, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"GLM FP8 build requires {path}, which was not found") from error
    key = scale_key(config)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model shape."
        )
    return table[key]


def calibrate_glm_fp8(config, weights, graph_ops) -> GlmQuantContext:
    """Build the Q/DQ context from the loaded GLM weights (``[in, out]`` layout).

    Quantizes one projection at a time, so peak extra memory is one weight tensor.
    """
    if any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights):
        raise ValueError("the checkpoint is already quantized; FP8 self-quantization needs BF16")
    activation_scales = load_activation_scales(config)

    scales: dict[str, _LayerScales] = {}
    for layer in range(int(config.num_hidden_layers)):
        for weight_stem in WEIGHT_STEMS:
            name = f"layer.{layer}.{weight_stem}"
            input_scale = activation_scales.get(name)
            if input_scale is None:
                continue
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
        raise RuntimeError("GLM FP8 self-quantization matched no projection in the checkpoint")
    return GlmQuantContext(scales=scales, graph_ops=graph_ops)


# ---------------------------------------------------------------------------
# NVFP4
# ---------------------------------------------------------------------------

_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints between neighbouring E2M1 values. A value exactly on a midpoint goes to the
# neighbour with an even code (0, 2, 4, 6): the first, third, fifth and seventh midpoint round
# down and the others round up.
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_MIDPOINT_ROUNDS_UP = (False, True, False, True, False, True, False)


def nvfp4_supported(in_features: int) -> bool:
    """True when a projection with this input size can use 16-value blocks."""
    return in_features > 0 and in_features % NVFP4_BLOCK_SIZE == 0


def _e2m1_codes(scaled: torch.Tensor) -> torch.Tensor:
    """Sign-magnitude E2M1 codes (0..15) of already block-scaled values."""
    magnitude = scaled.abs()
    code = torch.zeros(scaled.shape, dtype=torch.uint8, device=scaled.device)
    for midpoint, up in zip(_MIDPOINTS, _MIDPOINT_ROUNDS_UP, strict=True):
        code += (magnitude >= midpoint if up else magnitude > midpoint).to(torch.uint8)
    return code | ((scaled < 0).to(torch.uint8) << 3)


def quantize_nvfp4_weight(
    weight: torch.Tensor | np.ndarray, weight_scale_2: float | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Quantize one ``[out, in]`` weight to NVFP4.

    Returns the packed bytes ``[out, in / 2]`` (first value in the low nibble), the raw e4m3
    block-scale bytes ``[out, in / 16]`` and ``weight_scale_2 = amax / (6 * 448)``. The block
    scale is ``block_amax / 6 / weight_scale_2`` clamped to 448 and cast to e4m3; an all-zero
    block (or one whose scale underflows e4m3) gets scale 1.
    """
    tensor = torch.as_tensor(weight).float()
    if tensor.ndim != 2:
        raise ValueError(f"expected a [out, in] weight, got shape {tuple(tensor.shape)}")
    out_features, in_features = tensor.shape
    if not nvfp4_supported(in_features):
        raise ValueError(f"in_features={in_features} is not a multiple of {NVFP4_BLOCK_SIZE}")
    if weight_scale_2 is None:
        amax = float(tensor.abs().max())
        if not np.isfinite(amax) or amax <= 0:
            raise ValueError("weight tensor has no finite positive amax to scale from")
        # ModelOpt computes this on the GPU as amax * fp32(1 / 2688) (a reciprocal multiply), which
        # differs in the last bit from the correctly rounded division. Same operation here, so the
        # scale is bit-identical to a ModelOpt export.
        weight_scale_2 = float(
            torch.tensor(amax, dtype=torch.float32)
            * torch.tensor(1.0 / NVFP4_MAX, dtype=torch.float32)
        )
    packed_rows, scale_rows = [], []
    step = 2048  # rows per chunk, bounds the temporary memory
    for start in range(0, out_features, step):
        rows = tensor[start : start + step]
        blocks = rows.reshape(rows.shape[0], in_features // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
        block_scale = (blocks.abs().amax(-1) / E2M1_MAX / weight_scale_2).clamp_(max=FP8_E4M3_MAX)
        block_scale = block_scale.to(torch.float8_e4m3fn)
        scale_float = block_scale.float()
        empty = scale_float == 0
        if bool(empty.any()):
            block_scale = torch.where(empty, torch.ones_like(scale_float), scale_float).to(
                torch.float8_e4m3fn
            )
            scale_float = block_scale.float()
        scaled = blocks / (scale_float.unsqueeze(-1) * weight_scale_2)
        code = _e2m1_codes(scaled).reshape(rows.shape[0], in_features)
        packed_rows.append((code[:, 0::2] | (code[:, 1::2] << 4)).contiguous().cpu())
        scale_rows.append(block_scale.view(torch.uint8).cpu())
    return (
        np.ascontiguousarray(torch.cat(packed_rows).numpy()),
        np.ascontiguousarray(torch.cat(scale_rows).numpy()),
        float(weight_scale_2),
    )


def dequantize_nvfp4_weight(
    packed: np.ndarray, block_scale: np.ndarray, weight_scale_2: float
) -> torch.Tensor:
    """The float32 values the NVFP4 bytes stand for, ``[out, in]`` (what the engine computes with)."""
    packed_t = torch.from_numpy(np.ascontiguousarray(packed))
    codes = torch.stack([packed_t & 0xF, packed_t >> 4], dim=-1).reshape(packed_t.shape[0], -1)
    values = torch.tensor(_E2M1_GRID)[(codes & 7).long()] * torch.where(codes >= 8, -1.0, 1.0)
    scale = torch.from_numpy(np.ascontiguousarray(block_scale)).view(torch.float8_e4m3fn).float()
    return values * scale.repeat_interleave(NVFP4_BLOCK_SIZE, dim=-1) * weight_scale_2


# NVFP4 recipe per calibrated model shape (see the module docstring): the layers whose whole MLP
# stays FP8 W8A8, and the layers whose down projection stays FP8 W8A8.
NVFP4_RECIPES = {
    "glm-h4096-l40-i13696": {"fp8_mlp_layers": (0, 1, 38, 39), "fp8_down_layers": (4,)},
}


def projection_format(config, layer: int, stem: str) -> str:
    """``"fp4"`` or ``"fp8"``: the format of one projection under the NVFP4 recipe."""
    key = scale_key(config)
    if key not in NVFP4_RECIPES:
        raise NotImplementedError(
            f"no NVFP4 recipe for {key!r}; available: {sorted(NVFP4_RECIPES)}"
        )
    recipe = NVFP4_RECIPES[key]
    if stem in ("w_q", "w_k", "w_v") or layer in recipe["fp8_mlp_layers"]:
        return "fp8"
    if stem == "w_down" and layer in recipe["fp8_down_layers"]:
        return "fp8"
    return "fp4"


def load_nvfp4_activation_scales(config) -> dict[str, float]:
    """Return the committed NVFP4 activation scales for this model shape, or raise."""
    path = Path(__file__).parent / NVFP4_ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"GLM NVFP4 build requires {path}, which was not found") from error
    key = scale_key(config)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated NVFP4 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_nvfp4_scales.py for this model shape."
        )
    return table[key]


def calibrate_glm_nvfp4(config, weights, graph_ops) -> GlmQuantContext:
    """Build the NVFP4 / FP8 Q/DQ context from the loaded GLM weights (``[in, out]`` layout).

    Quantizes one projection at a time, so peak extra memory is one weight tensor.
    """
    if any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights):
        raise ValueError("the checkpoint is already quantized; NVFP4 self-quantization needs BF16")
    fp8_scales = load_activation_scales(config)
    fp4_scales = load_nvfp4_activation_scales(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scales: dict[str, _LayerScales] = {}
    for layer in range(int(config.num_hidden_layers)):
        for weight_stem in WEIGHT_STEMS:
            name = f"layer.{layer}.{weight_stem}"
            fmt = projection_format(config, layer, weight_stem)
            table = fp4_scales if fmt == "fp4" else fp8_scales
            input_scale = table.get(name)
            if input_scale is None:
                where = (
                    "nvfp4_activation_scales.json" if fmt == "fp4" else "fp8_activation_scales.json"
                )
                raise ValueError(
                    f"{name} has no activation scale in {where}; "
                    "run calibrate_nvfp4_scales.py / calibrate_fp8_scales.py"
                )
            if not np.isfinite(input_scale) or input_scale <= 0:
                raise ValueError(f"activation scale of {name} must be finite and positive")
            if name not in weights:
                raise ValueError(f"checkpoint has no {name}")
            transposed = np.asarray(weights[name])
            in_features, out_features = transposed.shape
            if fmt == "fp4":
                weight = torch.from_numpy(np.ascontiguousarray(transposed.T)).to(device).float()
                packed, block_scale, weight_scale_2 = quantize_nvfp4_weight(weight)
                del weight
                entry: _FP8Weight | _FP4Weight = _FP4Weight(
                    packed, block_scale, weight_scale_2, out_features, in_features
                )
            else:
                packed, weight_scale = quantize_fp8_weight(
                    np.ascontiguousarray(transposed.T, dtype=np.float32)
                )
                entry = _FP8Weight(packed, weight_scale, out_features, in_features)
            scales[name] = _LayerScales(input_scale=float(input_scale), weight=entry)
    if not scales:
        raise RuntimeError("GLM NVFP4 self-quantization matched no projection in the checkpoint")
    return GlmQuantContext(scales=scales, graph_ops=graph_ops)
