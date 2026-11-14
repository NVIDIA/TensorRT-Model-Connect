# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clef-owned FP8 self-quantization of the Qwen3.5 backbone.

There is no FP8 export of the Clef checkpoints, so ``quantization="fp8"`` quantizes the
original BF16 backbone at build time, one projection at a time:
``weight_scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value). FP8
weights are never stored. The static per-tensor activation scale of each projection
input needs real data, so it was calibrated once offline with ``calibrate_fp8_scales.py``
(ModelOpt ``FP8_DEFAULT_CFG``, max calibration) and is read from
``fp8_activation_scales.json``. Users do not run calibration.

Quantized: ``in_proj_qkv``, ``in_proj_z`` and ``out_proj`` of the DeltaNet layers,
``q_proj``, ``k_proj``, ``v_proj`` and ``o_proj`` of the full-attention layers, and ``gate_proj``,
``up_proj`` and ``down_proj`` of every MLP. Each becomes
``activation -> Quantize -> Dequantize -> MatMul`` with ``FP8 weight -> Dequantize`` on
the other input; both Dequantize layers return BF16, the backbone's type.

Kept in BF16: the embeddings, the norms, the DeltaNet ``in_proj_a``/``in_proj_b``,
``conv1d`` and recurrent state, the attention matmuls, the vision encoder and the joint
schema head. The attention ``o_proj`` input is the attention output times its sigmoid
gate, so its Quantize does not follow the fused attention directly and it compiles on
TensorRT 11.4 and 11.1. A projection stored as ``null`` in the scale file is an outlier
(its scale is more than 20 times the median of its kind) and also stays in BF16.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
CALIBRATION_TOOL = "families/clef/calibrate_fp8_scales.py"
QUANTIZATIONS = (None, "none", "fp8")
LAYER_PREFIX = "model.language_model.layers"
DELTANET_PROJECTIONS = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")
ATTENTION_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
MLP_PROJECTIONS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
_FINGERPRINT_WEIGHT = f"{LAYER_PREFIX}.0.mlp.gate_proj.weight"
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"


def is_fp8(quantization) -> bool:
    """True for an FP8 build; raises for an unsupported value."""
    if quantization not in QUANTIZATIONS:
        raise ValueError(f"Clef quantization must be one of none or fp8, not {quantization!r}")
    return quantization == "fp8"


def projections(config, layer: int) -> tuple[str, ...]:
    """HF projection stems of ``layer`` that FP8 builds quantize."""
    if config["layer_types"][layer] == "linear_attention":
        mixer = DELTANET_PROJECTIONS
    else:
        mixer = ATTENTION_PROJECTIONS
    return (*mixer, *MLP_PROJECTIONS)


def weights_fingerprint(weight) -> str:
    """BF16 bit patterns of the first values of layer 0 ``gate_proj``.

    They tell apart checkpoints of one model config (for example a fine-tune of the
    same base model), which need their own activation scales.
    """
    import torch

    first = weight.reshape(-1)[:4].to(torch.bfloat16).contiguous().view(torch.int16)
    return "".join(f"{int(bits) & 0xFFFF:04x}" for bits in first.tolist())


def scale_key(config, fingerprint: str) -> str:
    """Key of a checkpoint's entry in the activation-scale file."""
    return (
        f"clef-h{int(config['hidden_size'])}"
        f"-l{int(config['num_hidden_layers'])}"
        f"-i{int(config['intermediate_size'])}"
        f"-{fingerprint}"
    )


def checkpoint_scale_key(checkpoint, config) -> str:
    """Scale key of a Clef checkpoint (reads one weight)."""
    weight = checkpoint.select(_FINGERPRINT_WEIGHT)[_FINGERPRINT_WEIGHT]
    return scale_key(config, weights_fingerprint(weight))


def load_activation_scales(key: str, path: Path | None = None) -> dict[str, float | None]:
    """Committed activation scales of one checkpoint; raises for an unknown model config."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME if path is None else path
    table = json.loads(path.read_text(encoding="utf-8"))
    if key not in table:
        raise NotImplementedError(
            f"no calibrated Clef FP8 activation scales for {key!r}; available: {sorted(table)}. "
            f"Run {CALIBRATION_TOOL} for this checkpoint and commit the result."
        )
    return table[key]


def layer_scales(scales: dict, config, layer: int) -> dict[str, float]:
    """HF projection prefix -> activation scale of the FP8 projections of ``layer``.

    Every quantized projection must have an entry; ``null`` keeps it in BF16.
    """
    result = {}
    for stem in projections(config, layer):
        name = f"{LAYER_PREFIX}.{layer}.{stem}"
        if name not in scales:
            raise KeyError(f"{name} has no FP8 activation scale; rerun {CALIBRATION_TOOL}")
        value = scales[name]
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} has an invalid FP8 activation scale {value!r}")
        result[name] = value
    return result


def quantize_fp8_weight(weight) -> tuple[np.ndarray, float]:
    """Raw e4m3 bytes of ``weight`` [out, in] and its scale ``max(abs(weight)) / 448``."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = weight.to(device=device, dtype=torch.float32)
    amax = float(w.abs().max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    packed = (w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return np.ascontiguousarray(packed.view(torch.uint8).cpu().numpy()), scale


def fp8_linear(g, x, weight, input_scale: float):
    """``x @ weight.T`` with an FP8 weight and a static FP8 activation scale."""
    import tensorrt as trt

    packed, weight_scale = quantize_fp8_weight(weight)
    g.keep.append(packed)  # trt.Weights holds a raw pointer.
    rows, columns = packed.shape
    w = g.n.add_constant(
        (rows, columns), trt.Weights(trt.DataType.FP8, packed.ctypes.data, packed.size)
    )
    w = g.n.add_dequantize(w.get_output(0), g.const([weight_scale]), x.dtype).get_output(0)
    if len(x.shape) > 2:
        w = g.reshape(w, (1,) * (len(x.shape) - 2) + (rows, columns))
    scale = g.const([input_scale])
    q = g.n.add_quantize(x, scale, trt.DataType.FP8).get_output(0)
    a = g.n.add_dequantize(q, scale, x.dtype).get_output(0)
    return g.mm(a, w, True)


def configure_fp8_build(settings) -> None:
    """FP8 build options: detailed layer names and no fused dual-GEMM when TensorRT lists it."""
    import tensorrt as trt

    settings.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if _DUAL_GEMM_KNOB in getattr(settings, "all_build_routes", ""):
        settings.build_route = f"-{_DUAL_GEMM_KNOB}=off"
