# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read PaliGemma safetensors under one canonical name per tensor.

Two layouts of ``PaliGemmaForConditionalGeneration`` are published: the
transformers 4.x save (``language_model.model.*``, the pinned checkpoint) and
the transformers 5.x save (``model.language_model.*``). A checkpoint must use
exactly one of them; canonical names are ``vision.*``, ``projector.*``,
``text.*`` and ``lm_head.*``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_LAYOUTS = (
    (("vision_tower.vision_model.", "vision."), ("multi_modal_projector.", "projector."),
     ("language_model.model.", "text."), ("language_model.lm_head.", "lm_head.")),
    (("model.vision_tower.vision_model.", "vision."), ("model.multi_modal_projector.", "projector."),
     ("model.language_model.", "text."), ("lm_head.", "lm_head.")),
)


def _canonical(stored: list[str]) -> dict[str, str]:
    for layout in _LAYOUTS:
        mapping = {}
        for key in stored:
            for prefix, replacement in layout:
                if key.startswith(prefix):
                    mapping[replacement + key[len(prefix):]] = key
                    break
            else:
                break
        else:
            return mapping
    raise ValueError("PaliGemma checkpoint tensors match neither published layout")


def _relative_shard(root: Path, shard: object) -> Path:
    if not isinstance(shard, str) or not shard or "\\" in shard or "\0" in shard:
        raise ValueError("PaliGemma weight_map names an invalid shard")
    relative = Path(shard)
    if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
        raise ValueError(f"PaliGemma shard must be a plain relative path: {shard}")
    if not (root / relative).is_file():
        raise ValueError(f"PaliGemma checkpoint shard is missing: {shard}")
    return relative


class CheckpointWeights:
    """Lazy tensor access; every tensor is returned as float32 converted exactly."""

    def __init__(self, model_dir: Path):
        from safetensors import safe_open

        self.root = Path(model_dir)
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("PaliGemma requires a nonempty safetensors weight_map")
            shards = {key: _relative_shard(self.root, shard) for key, shard in weight_map.items()}
        elif (self.root / "model.safetensors").is_file():
            with safe_open(self.root / "model.safetensors", framework="pt", device="cpu") as stream:
                shards = {key: Path("model.safetensors") for key in stream.keys()}
        else:
            raise ValueError("PaliGemma checkpoint has no safetensors weights")
        self._stored = _canonical(sorted(shards))
        self._shards = shards

    def __contains__(self, name: str) -> bool:
        return name in self._stored

    def names(self) -> list[str]:
        return sorted(self._stored)

    def shape(self, name: str) -> tuple[int, ...]:
        from safetensors import safe_open

        key = self._stored[name]
        with safe_open(self.root / self._shards[key], framework="pt", device="cpu") as stream:
            return tuple(stream.get_slice(key).get_shape())

    def __getitem__(self, name: str) -> np.ndarray:
        from safetensors import safe_open

        key = self._stored[name]
        # BF16 converts to FP32 exactly through torch; numpy has no BF16 type.
        with safe_open(self.root / self._shards[key], framework="pt", device="cpu") as stream:
            return stream.get_tensor(key).float().numpy()
