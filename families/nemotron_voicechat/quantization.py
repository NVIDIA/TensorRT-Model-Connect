# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VoiceChat-owned FP8 self-quantization and TensorRT Q/DQ graph context.

The original fp32 VoiceChat checkpoint is used as-is. No ModelOpt FP8 or NVFP4 export of
``nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`` exists on the Hub (community exports are
compressed-tensors, MLX or GGUF files, not ModelOpt).

Only the thinker (the Nemotron-H hybrid language model, 56 layers) is quantized. Weights
are quantized on the fly at build time, one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and the FP8
bytes are fed to the engine as a constant behind a Dequantize. The activation scale of
each projection input needs real data, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (running input maxima over real speech) and is read from
``fp8_activation_scales.json``, keyed by model config. Users do not run calibration.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize(fp32) -> MatMul(fp32)
    FP8 weight -> Dequantize(fp32) ----------------^

The engine is an fp32 engine, so the Dequantize and the matmul already run in the engine
dtype and no cast is added.

Quantized: Mamba-2 ``in_proj`` and ``out_proj``, MLP ``up_proj`` and ``down_proj``, and
attention ``q_proj``, ``k_proj`` and ``v_proj``. Not quantized: the attention output
projection (TensorRT 11.4 cannot compile an FP8 Quantize right after a fused attention),
``conv1d``, the SSM parameters, the norms, the embedding, ``lm_head`` and the function head.
The conformer encoder, the RNNT decoder, the TTS backbone and the codec are not touched.
A projection without an entry in the scales file is an error, never a silent fallback.
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
CALIBRATION_TOOL = "python -m families.nemotron_voicechat.calibrate_fp8_scales calibrate"

# HF projection stem (under ``stt_model.llm.layers.N``) -> weight-dict stem, per layer kind.
PROJECTIONS_BY_KIND = {
    "mamba2": (("mixer.in_proj", "mamba_in_proj"), ("mixer.out_proj", "mamba_out_proj")),
    "mlp": (("mixer.up_proj", "w_up"), ("mixer.down_proj", "w_down")),
    "attention": (
        ("mixer.q_proj", "w_q"),
        ("mixer.k_proj", "w_k"),
        ("mixer.v_proj", "w_v"),
    ),
}


def scale_key(hidden_size: int, num_layers: int, intermediate_size: int) -> str:
    """Key of a model config in the activation-scale file."""
    return f"nemotron_voicechat-h{hidden_size}-l{num_layers}-i{intermediate_size}"


def projection_names(layer_types: list[str]) -> list[str]:
    """Weight-dict names of every projection the recipe quantizes, in layer order."""
    names = []
    for layer, kind in enumerate(layer_types):
        for _, stem in PROJECTIONS_BY_KIND[kind]:
            names.append(f"layer.{layer}.{stem}")
    return names


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one ``[out, in]`` weight to FP8 e4m3 with a single scalar scale.

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


def _scale_const(network, value: float):
    """Add a one-element fp32 scale constant to the network."""
    array = np.asarray([value], dtype=np.float32)
    return network.add_constant((1,), trt.Weights(array)).get_output(0)


def _wrap_fp8_matmul(
    network, activation, weight: _FP8Weight, input_scale: float, *, keep_alive: list, name: str
):
    """Add the FP8 Q/DQ matmul for one projection, with fp32 Dequantize and matmul."""
    n, k = weight.out_features, weight.in_features

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(weight.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, weight.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, _scale_const(network, weight.weight_scale), trt.float32
    )

    act_scale = _scale_const(network, input_scale)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, trt.float32)

    layer = network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    )
    layer.name = name
    output = layer.get_output(0)
    if output.dtype == activation.dtype:
        return output
    return network.add_cast(output, activation.dtype).get_output(0)


@dataclass
class VoiceChatQuantContext:
    """FP8 weights and static activation scales of the quantized thinker projections.

    ``quantize_projection`` is called by the weight loader for each projection while the
    fp32 checkpoint is read, so a quantized projection is never held in fp32 twice.
    """

    activation_scales: dict[str, float]
    weights: dict[str, _FP8Weight] = field(default_factory=dict)
    keep_alive: list = field(default_factory=list)

    def wants(self, weight_name: str) -> bool:
        """True when the recipe quantizes this projection (it has an activation scale)."""
        return weight_name in self.activation_scales

    def quantize_projection(self, weight_name: str, weight_out_in: np.ndarray) -> None:
        """Quantize one ``[out, in]`` fp32 weight read from the checkpoint."""
        if not self.wants(weight_name):
            raise ValueError(f"{weight_name} is not a quantized projection")
        out_features, in_features = weight_out_in.shape
        packed, scale = quantize_fp8_weight(np.asarray(weight_out_in, dtype=np.float32))
        self.weights[weight_name] = _FP8Weight(packed, scale, out_features, in_features)

    def should_quantize(self, weight_name: str) -> bool:
        """True when this weight was quantized and runs as an FP8 matmul."""
        return weight_name in self.weights

    def project(
        self,
        network,
        lhs,
        lhs_width: int,
        rhs_width: int,
        rhs_weights: np.ndarray,
        weight_name: str,
        dtype: np.dtype = np.float32,
    ):
        """FP8 matmul output for a quantized projection, or None for an unquantized one."""
        weight = self.weights.get(weight_name)
        if weight is None:
            return None
        return _wrap_fp8_matmul(
            network,
            lhs,
            weight,
            self.activation_scales[weight_name],
            keep_alive=self.keep_alive,
            name=weight_name,
        )


def load_activation_scales(hidden_size: int, num_layers: int, intermediate_size: int):
    """Return the committed activation scales for this model config, or raise."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"VoiceChat FP8 build requires {path}, which was not found") from error
    key = scale_key(hidden_size, num_layers, intermediate_size)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run `{CALIBRATION_TOOL}` for this model."
        )
    return table[key]


def create_fp8_context(
    hidden_size: int,
    num_layers: int,
    intermediate_size: int,
    layer_types: list[str],
) -> VoiceChatQuantContext:
    """Build the quantization context of the thinker from the committed scales.

    Every projection of the recipe must have a finite positive scale; a missing or invalid
    entry raises ``ValueError`` naming the projection.
    """
    table = load_activation_scales(hidden_size, num_layers, intermediate_size)
    scales: dict[str, float] = {}
    for name in projection_names(layer_types):
        value = table.get(name)
        if value is None:
            raise ValueError(
                f"no FP8 activation scale for projection {name}; "
                f"run `{CALIBRATION_TOOL}` to calibrate it"
            )
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"FP8 activation scale of {name} must be finite and positive")
        scales[name] = float(value)
    return VoiceChatQuantContext(activation_scales=scales)


def reject_quantized_checkpoint(tensor_names) -> None:
    """Raise when the checkpoint already stores quantized tensors (self-quantization needs fp32)."""
    for name in tensor_names:
        if str(name).endswith((".weight_scale", ".input_scale", ".weight_scale_2")):
            raise ValueError(
                "the checkpoint is already quantized; FP8 self-quantization needs the "
                "original fp32 VoiceChat checkpoint"
            )


def configure_fp8_build(trt_config, quant_ctx) -> None:
    """Apply the FP8 build options; a no-op for an unquantized build."""
    if quant_ctx is None:
        return
    # Keeps per-layer tactic names in the plan so FP8 lowering can be inspected.
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
