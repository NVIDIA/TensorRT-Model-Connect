# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantization scale data structures.

LayerScales holds per-layer scale values. QuantScaleMap holds the complete
set of scales for a model, keyed by weight name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LayerScales:
    """Scales for one quantized matmul operation."""

    input_scale: float | np.ndarray = 1.0
    weight_scale: float | np.ndarray = 1.0
    output_scale: float | np.ndarray = 1.0
    block_size: int | None = None
    # Per-input-channel SmoothQuant factor (ModelOpt ``input_quantizer.
    # _pre_quant_scale``). When present, the format smooths activations by
    # ``*pre_quant_scale`` and weights by ``/pre_quant_scale`` so the calibrated
    # (smoothed-weight) scales match what runs. None for non-SmoothQuant formats.
    pre_quant_scale: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k in ("input_scale", "weight_scale", "output_scale"):
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        if self.block_size is not None:
            d["block_size"] = self.block_size
        if self.pre_quant_scale is not None:
            d["pre_quant_scale"] = np.asarray(self.pre_quant_scale).tolist()
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> LayerScales:
        kwargs: dict[str, Any] = {}
        for k in ("input_scale", "weight_scale", "output_scale"):
            if k in d:
                v = d[k]
                kwargs[k] = np.array(v) if isinstance(v, list) else v
        if "block_size" in d:
            kwargs["block_size"] = d["block_size"]
        if "pre_quant_scale" in d:
            kwargs["pre_quant_scale"] = np.asarray(d["pre_quant_scale"],
                                                   dtype=np.float32)
        return LayerScales(**kwargs)


@dataclass
class QuantScaleMap:
    """All scales for a model, keyed by weight name (e.g., 'layer.0.w_q')."""

    scales: dict[str, LayerScales] = field(default_factory=dict)
    dynamic: bool = False  # True for NVFP4/MXFP8 (runtime scales)

    def get(self, weight_name: str) -> LayerScales | None:
        exact = self.scales.get(weight_name)
        if exact is not None:
            return exact
        suffix = f"/{weight_name}"
        for key, scales in self.scales.items():
            if key.endswith(suffix):
                return scales
        return None

    def to_json(self) -> str:
        obj: dict[str, Any] = {
            "dynamic": self.dynamic,
            "scales": {k: v.to_dict() for k, v in self.scales.items()},
        }
        return json.dumps(obj, indent=2)

    @staticmethod
    def from_json(text: str) -> QuantScaleMap:
        obj = json.loads(text)
        scales = {
            k: LayerScales.from_dict(v)
            for k, v in obj.get("scales", {}).items()
        }
        return QuantScaleMap(scales=scales, dynamic=obj.get("dynamic", False))

    @staticmethod
    def load(path: str | Path) -> QuantScaleMap:
        return QuantScaleMap.from_json(Path(path).read_text())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())
