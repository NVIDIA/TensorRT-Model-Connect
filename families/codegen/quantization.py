# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CodeGen-owned FP8 self-quantization and TensorRT Q/DQ graph context.

The original CodeGen checkpoint (``pytorch_model.bin``) is used as is; no FP8 checkpoint is needed.

Weights are quantized on the fly at build time, one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and the FP8 bytes
are fed to the engine as a constant. The weights are taken after ``_CodeGenModel.load_weights``
has split the fused ``qkv_proj`` matrix (``mp_num=4`` chunks of ``[Q, V, K]`` rows) into ``w_q``,
``w_k`` and ``w_v``, so the quantized weight is the one the unquantized engine uses. The
activation scale of each projection input needs real data, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is read from
``fp8_activation_scales.json``. Users do not run calibration. A projection with no entry in that
file is an error.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize(fp16) -> MatMul(fp16)
    FP8 weight -> Dequantize(fp16) -------------------^

Both Dequantize layers output the activation type (fp16), so the FP8 GEMM writes fp16 and the
bias and the activation function after it stay in the GEMM epilogue.

Quantized projections (internal weight names, ``N`` is the layer index): ``layer.N.w_q``,
``w_k``, ``w_v`` (``attn.qkv_proj`` of the checkpoint, one shared input scale), ``w_o``
(``attn.out_proj``), ``w_fc1`` (``mlp.fc_in``) and ``w_fc2`` (``mlp.fc_out``). A projection whose
input has a massive activation (a scale more than 20 times the median of the same projection, see
``calibrate_fp8_scales.split_outlier_scales``) stays in fp16; its entry in the scale file is
``null``. Not quantized: the biases, ``lm_head``, the embedding and the LayerNorms.

``w_o`` is quantized because this family's attention output projection compiles with an FP8
Quantize on TensorRT 11.1 and 11.4 (probed with tiny engines of both roles, and built at the real
size); other families cannot do this on 11.4 and keep it in fp16.

Scale-file key: the model config plus a short signature of the first query weight, because
CodeGen checkpoints of one size (``codegen-350M-mono``, ``-multi`` and ``-nl``) share a config but
not weights, so not activation ranges.
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

# HF projection stem -> internal weight-dict stems used by graph_blocks. The fused
# ``qkv_proj`` Linear has one input, so its three parts share one activation scale.
PROJECTIONS = (
    ("attn.qkv_proj", ("w_q", "w_k", "w_v")),
    ("attn.out_proj", ("w_o",)),
    ("mlp.fc_in", ("w_fc1",)),
    ("mlp.fc_out", ("w_fc2",)),
)
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"


def weights_signature(weights) -> str:
    """Return a short signature of the first query weight; it tells checkpoints of one config apart.

    It is the exact sum of the fp16 bit patterns modulo 2**32, so it is the same on every machine.
    """
    first = np.ascontiguousarray(np.asarray(weights["layer.0.w_q"]), dtype=np.float16)
    return f"{int(first.view(np.uint16).sum(dtype=np.uint64)) % 2**32:08x}"


def scale_key(config, weights) -> str:
    """Key of this checkpoint's entry in the activation-scale file."""
    return (
        f"{str(config.model_type).lower()}"
        f"-h{int(config.hidden_size)}"
        f"-l{int(config.num_hidden_layers)}"
        f"-i{int(config.intermediate_size)}"
        f"-{weights_signature(weights)}"
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
class CodeGenQuantContext:
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
        raise RuntimeError(f"CodeGen FP8 build requires {path}, which was not found") from error
    key = scale_key(config, weights)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def projection_names(config) -> list[str]:
    """Internal weight names of every quantized projection of this model config."""
    return [
        f"layer.{layer}.{weight_stem}"
        for layer in range(int(config.num_hidden_layers))
        for _, weight_stems in PROJECTIONS
        for weight_stem in weight_stems
    ]


def reject_quantized_weights(config, weights) -> None:
    """Raise when the checkpoint is already quantized; FP8 self-quantization needs 16-bit weights."""
    if (
        config.raw.get("quantization_config")
        or weights.get("_quantized_checkpoint")
        or any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights)
    ):
        raise ValueError("the checkpoint is already quantized; FP8 self-quantization needs BF16")


def calibrate_codegen_fp8(config, weights, graph_ops) -> CodeGenQuantContext:
    """Build the Q/DQ context from the loaded CodeGen weights (``[in, out]`` layout).

    Quantizes one projection at a time, so peak extra memory is one weight tensor. A projection
    with no stored activation scale, or a stored scale for a projection this model config does
    not quantize, is an error.
    """
    reject_quantized_weights(config, weights)
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
            continue  # massive-activation projection, left in fp16 by the calibration
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
        raise RuntimeError("CodeGen FP8 self-quantization matched no projection in the checkpoint")
    return CodeGenQuantContext(scales=scales, graph_ops=graph_ops)
