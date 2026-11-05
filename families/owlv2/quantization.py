# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OWLv2-owned FP8 self-quantization and TensorRT Q/DQ graph context.

FP8 (``quantization="fp8"``) quantizes the original checkpoint at build time; no FP8
checkpoint exists for OWLv2 and none is stored. Weights are quantized one projection at a
time: ``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and the
FP8 bytes are fed to the engine as a constant. The activation scale of each projection
input needs real data, so it was calibrated once offline with ``calibrate_fp8_scales.py``
(ModelOpt ``FP8_DEFAULT_CFG``, max calibration) on COCO images and is read from
``fp8_activation_scales.json`` under the model config key. Users do not run calibration.
A model config with no table, or a projection with no entry in the table, is an error.
An entry of ``null`` marks an outlier projection (activation scale above 20 times the
median of its kind), which stays in fp16.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize(work dtype) -> MatMul
    FP8 weight -> Dequantize(work dtype) --------------^

Both Dequantize layers output the activation type, so in an fp16 engine the FP8 GEMM
writes fp16 and the bias add stays in the epilogue. The risk of fp16 is an overflow above
65504 in a projection output; the NaN scan on real COCO images found none.

The projections form three parts, which are measured and can be quantized separately.
``quantization="fp8"`` quantizes only the part that keeps the detection output at the fp16
floor, ``DEFAULT_PARTS``:

* ``text`` (default): q/k/v/out_proj, fc1 and fc2 of the 12 text-tower layers. Detections
  stay at the fp16 floor; the gain is the plan size, not speed (a 16-token tower is launch
  bound).
* ``vision``: the same six projections of the 12 image-tower layers. Not enabled: every
  group of projections moves the detections far from the fp16 engine (see the PR), and the
  fused attention (about half of the engine time at 3601 tokens) stays fp16, so the
  ceiling is about 1.1x.
* ``heads``: the 768-wide ``box_head.dense0``, ``box_head.dense1`` and
  ``class_head.dense0``. The last box layer (4 outputs), ``logit_shift`` and
  ``logit_scale`` (1 output each) and the text projection are never quantized.

``out_proj`` is quantized: the attention here is explicit matmul, softmax, matmul, so no
fused attention feeds the Quantize (the TensorRT 11.4 limit does not apply; retest if the
attention is ever replaced by the fused layer). The patch convolution, embeddings, layer
norms and the box and class post-processing stay in the work precision.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import tensorrt as trt

from . import graph

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
# A projection whose calibrated scale is above this multiple of the median of its kind
# (same stem, same tower) stays in fp16.
OUTLIER_FACTOR = 20.0

_SUPPORTED = ("fp8",)
PARTS = ("vision", "text", "heads")
# Parts quantized by ``quantization="fp8"``.
DEFAULT_PARTS = ("text",)
LAYER_STEMS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
)
HEAD_PROJECTIONS = ("box_head.dense0", "box_head.dense1", "class_head.dense0")
_TOWER_PREFIX = {"vision": "owlv2.vision_model", "text": "owlv2.text_model"}


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


def _scale_const(network, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    return graph.add_constant(network, (1,), np.asarray([value], dtype=np.float32))


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, keep_alive: list):
    """Add ``activation @ weight.T``; Dequantize and MatMul run in the activation type."""
    dtype = activation.dtype
    if dtype not in (trt.float16, trt.float32):
        raise ValueError(f"OWLv2 FP8 supports fp16 or fp32 activations, got {dtype}")
    w = scales.weight
    n, k = w.out_features, w.in_features

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(w_const, _scale_const(network, w.weight_scale), dtype)

    act_scale = _scale_const(network, scales.input_scale)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, dtype)

    return network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)


@dataclass(frozen=True)
class Owlv2QuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    keep_alive: list = None  # type: ignore[assignment]

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection is FP8 (selects the FP8 build options)."""
        return bool(self.scales)

    def __post_init__(self):
        """Create the list that keeps weight buffers alive during the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, name: str) -> bool:
        """True when ``name`` (for example ``owlv2.vision_model.encoder.layers.3.mlp.fc1``)."""
        return name in self.scales

    def matmul(self, name: str) -> Any:
        """The FP8 matmul callable for ``name`` (``(network, input) -> output``), or None."""
        entry = self.scales.get(name)
        if entry is None:
            return None

        def run(network, activation):
            return _wrap_fp8_matmul(network, activation, entry, keep_alive=self.keep_alive)

        return run


def scale_key(cfg: dict) -> str:
    """Key of this model's table in the scale file, from the resolved model config."""
    vision, text = cfg["vision"], cfg["text"]
    return (
        f"owlv2-v{vision['hidden_size']}x{vision['num_layers']}"
        f"i{vision['intermediate_size']}p{vision['patch_size']}s{vision['image_size']}"
        f"-t{text['hidden_size']}x{text['num_layers']}"
        f"i{text['intermediate_size']}w{text['max_position_embeddings']}"
    )


def projections(cfg: dict, parts: tuple[str, ...] = PARTS) -> list[str]:
    """Checkpoint names (without ``.weight``) of every quantizable projection of ``parts``."""
    unknown = sorted(set(parts) - set(PARTS))
    if unknown:
        raise ValueError(f"unknown OWLv2 FP8 parts {unknown}; choose from {list(PARTS)}")
    names: list[str] = []
    for part in ("vision", "text"):
        if part in parts:
            prefix = _TOWER_PREFIX[part]
            for layer in range(cfg[part]["num_layers"]):
                names.extend(f"{prefix}.encoder.layers.{layer}.{stem}" for stem in LAYER_STEMS)
    if "heads" in parts:
        names.extend(HEAD_PROJECTIONS)
    return names


def split_outlier_scales(scales: dict[str, float]) -> dict[str, float | None]:
    """Set to ``None`` every scale above ``OUTLIER_FACTOR`` times the median of its kind.

    The kind of a projection is its tower and its stem (for example ``q_proj`` of the
    image tower), so a tower-wide shift does not hide an outlier layer.
    """
    kinds: dict[str, list[float]] = {}
    for name, value in scales.items():
        kinds.setdefault(_kind(name), []).append(value)
    medians = {kind: statistics.median(values) for kind, values in kinds.items()}
    return {
        name: (None if value > OUTLIER_FACTOR * medians[_kind(name)] else value)
        for name, value in scales.items()
    }


def _kind(name: str) -> str:
    return re.sub(r"\.layers\.\d+\.", ".layers.N.", name)


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


def load_activation_scales(cfg: dict) -> dict[str, float | None]:
    """Return the committed activation scales for this model config, or raise when none exist.

    An entry of ``None`` marks a projection the calibration left in fp16 on purpose.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"OWLv2 FP8 build requires {path}, which was not found") from error
    key = scale_key(cfg)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model config."
        )
    return table[key]


def calibrate_owlv2(
    weights: dict, cfg: dict, quantization: str, parts: tuple[str, ...] = DEFAULT_PARTS
) -> Owlv2QuantContext:
    """Quantize the checkpoint one projection at a time and pair it with the stored scales."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"OWLv2 supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    activation_scales = load_activation_scales(cfg)
    scales: dict[str, _LayerScales] = {}
    for name in projections(cfg, parts):
        if f"{name}.weight_scale" in weights:
            raise ValueError(
                f"{name} is already quantized; fp8 self-quantization needs the original checkpoint"
            )
        if name not in activation_scales:
            raise ValueError(
                f"{name} has no activation scale in {ACTIVATION_SCALES_FILENAME}; "
                "run calibrate_fp8_scales.py"
            )
        input_scale = activation_scales[name]
        if input_scale is None:
            continue  # outlier projection, stays in fp16
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        key = f"{name}.weight"
        if key not in weights:
            raise ValueError(f"checkpoint has no {key}")
        weight = np.asarray(weights[key], dtype=np.float32)
        packed, weight_scale = quantize_fp8_weight(weight)
        out_features, in_features = weight.shape
        scales[name] = _LayerScales(
            float(input_scale), _FP8Weight(packed, weight_scale, out_features, in_features)
        )
    if not scales:
        raise RuntimeError("OWLv2 FP8 self-quantization matched no projection in the checkpoint")
    return Owlv2QuantContext(scales=scales)
