# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SeamlessM4T v2-owned FP8 self-quantization of the text encoder and text decoder.

No ModelOpt FP8 or NVFP4 checkpoint of SeamlessM4T v2 exists, so
``quantization="fp8"`` quantizes the original FP32 checkpoint while the engines
are built, one projection at a time. Weights are never stored in FP8.

Recipe (W8A8, per-tensor scales): every linear projection of the 24 text
encoder layers (self-attention q, k, v, out; ffn fc1, fc2) and of the 24 text
decoder layers (the same plus cross-attention q, k, v, out). The token
embedding, the sinusoidal positions, the language-model head, the layer norms
and the attention core keep the build precision (FP32). A projection whose
calibrated scale is ``null`` (an outlier input, more than 20 times the median
of its kind) stays FP32; the calibrated checkpoint has none.

Weight scale ``amax(W) / 448``. Projections that read one input (self-attention
q, k and v; cross-attention k and v) share one weight scale, the largest amax
of the group, as TensorRT runs them as one GEMM. The activation scale is
static, calibrated once offline with ModelOpt ``FP8_DEFAULT_CFG`` (max) by
``calibrate_fp8_scales.py`` and read from ``fp8_activation_scales.json``.
Users do not run calibration; a checkpoint without scales raises. Each
quantized projection becomes::

    activation -> Quantize -> Dequantize(fp32) -> MatMul (+ bias)
    FP8 weight -> Dequantize(fp32) ----------------^

The attention is explicit (FP32 scores, softmax, matmul), so the FP8 Quantize
in front of the attention output projections builds on TensorRT 11.4.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
QUANTIZATIONS = ("fp8",)
IDENTITY_VALUES = 8
ENCODER_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "ffn.fc1",
    "ffn.fc2",
)
DECODER_PROJECTIONS = (
    *ENCODER_PROJECTIONS[:4],
    "cross_attention.q_proj",
    "cross_attention.k_proj",
    "cross_attention.v_proj",
    "cross_attention.out_proj",
    *ENCODER_PROJECTIONS[4:],
)
# Projections that read the same input and therefore share one weight scale.
SHARED_INPUT = (
    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ("cross_attention.k_proj", "cross_attention.v_proj"),
)
_QUANTIZED_WEIGHT = re.compile(r"\.(weight_scale|input_scale|weight_scale_2)$")


def projection_names(cfg) -> list[str]:
    """Checkpoint stems of every projection the FP8 recipe covers."""
    names = []
    for prefix, stems, layers in (
        ("text_encoder", ENCODER_PROJECTIONS, cfg["encoder_layers"]),
        ("text_decoder", DECODER_PROJECTIONS, cfg["decoder_layers"]),
    ):
        for layer in range(int(layers)):
            names += [f"{prefix}.layers.{layer}.{stem}" for stem in stems]
    return names


def scale_key(cfg) -> str:
    """Key of one text encoder / decoder config in the activation-scale file."""
    return (
        f"seamless_m4t_v2-h{int(cfg['hidden_size'])}-e{int(cfg['encoder_layers'])}"
        f"-d{int(cfg['decoder_layers'])}-f{int(cfg['decoder_ffn_dim'])}"
    )


def final_norm_head(weights) -> list[float]:
    """The first 8 values of the final decoder norm weight, which tell checkpoints apart."""
    norm = np.asarray(weights["text_decoder.layer_norm.weight"], dtype=np.float32).reshape(-1)
    return [float(value) for value in norm[:IDENTITY_VALUES]]


def reject_quantized_checkpoint(cfg, weight_names) -> None:
    """Self-quantization starts from the original weights; refuse an already quantized checkpoint."""
    if cfg.get("quantization_config") or any(_QUANTIZED_WEIGHT.search(n) for n in weight_names):
        raise ValueError(
            "SeamlessM4T v2 FP8 quantizes the original checkpoint while building; "
            "this checkpoint is already quantized"
        )


def load_activation_scales(cfg, weights) -> dict[str, float | None]:
    """The committed FP8 activation scales of this checkpoint, or raise."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    key = scale_key(cfg)
    entry = table.get(key)
    if entry is None or entry["final_norm_head"] != final_norm_head(weights):
        raise NotImplementedError(
            f"SeamlessM4T v2 has no calibrated FP8 activation scales for this checkpoint ({key}); "
            f"calibrated configs: {sorted(table)}. Run calibrate_fp8_scales.py for it."
        )
    return entry["scales"]


def quantize_fp8_weight(weight, amax: float | None = None) -> tuple[np.ndarray, float]:
    """Raw e4m3 bytes of ``weight`` ``[out, in]`` and its per-tensor scale ``amax / 448``."""
    import torch

    tensor = torch.as_tensor(np.ascontiguousarray(weight)).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    if amax is None:
        amax = float(tensor.abs().max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = (tensor / torch.tensor(scale, dtype=torch.float32)).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()
    return np.ascontiguousarray(packed), scale


def _scalar(network, value: float, keep_alive: list):
    array = np.asarray([value], dtype=np.float32)
    keep_alive.append(array)
    return network.add_constant((1,), array).get_output(0)


@dataclass
class QuantContext:
    """Static FP8 input scale (and shared weight amax) of each quantized projection.

    Keys are checkpoint weight stems, ``text_decoder.layers.N.ffn.fc1``; a
    projection without a key (and the language-model head) keeps FP32.
    """

    input_scales: dict[str, float]
    weight_amax: dict[str, float] = field(default_factory=dict)
    keep_alive: list = field(default_factory=list)

    def configure(self, trt_config) -> None:
        """Keep per-layer names in the plan so the FP8 kernels can be inspected."""
        trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    def covers(self, key: str) -> bool:
        return key in self.input_scales

    def matmul(self, network, value, key: str, weight):
        """``value @ weight.T`` for the ``[out, in]`` FP32 ``weight`` of projection ``key``."""
        packed, weight_scale = quantize_fp8_weight(weight, self.weight_amax.get(key))
        rows, columns = packed.shape
        # trt.Weights and constants keep raw pointers: the buffers must outlive the build.
        self.keep_alive.append(packed)
        constant = network.add_constant(
            (rows, columns), trt.Weights(trt.DataType.FP8, packed.ctypes.data, packed.size)
        ).get_output(0)
        w = network.add_dequantize(
            constant, _scalar(network, weight_scale, self.keep_alive), value.dtype
        ).get_output(0)
        a_scale = _scalar(network, self.input_scales[key], self.keep_alive)
        q = network.add_quantize(value, a_scale, trt.DataType.FP8).get_output(0)
        x = network.add_dequantize(q, a_scale, value.dtype).get_output(0)
        return network.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)


def fp8_quantization(cfg, weights, quantization: str) -> QuantContext:
    """The FP8 recipe for this checkpoint: one activation scale per quantized projection."""
    if quantization not in QUANTIZATIONS:
        raise ValueError(f"SeamlessM4T v2 quantization must be none or fp8, not {quantization!r}")
    scales = load_activation_scales(cfg, weights)
    input_scales = {}
    for name in projection_names(cfg):
        if name not in scales:
            raise ValueError(f"SeamlessM4T v2 activation scale file has no entry for {name}")
        scale = scales[name]
        if scale is None:
            continue
        if isinstance(scale, bool) or not (np.isfinite(scale) and scale > 0):
            raise ValueError(
                f"SeamlessM4T v2 activation scale of {name} must be finite and positive"
            )
        input_scales[name] = float(scale)
    weight_amax = {}
    for name in projection_names(cfg):
        stem = name.split(".", 3)[3]
        for group in SHARED_INPUT:
            if stem != group[0]:
                continue
            base = name[: -len(stem)]
            names = [base + member for member in group]
            if all(member in input_scales for member in names):
                amax = max(float(np.abs(weights[member + ".weight"]).max()) for member in names)
                weight_amax.update(dict.fromkeys(names, amax))
    return QuantContext(input_scales, weight_amax)
