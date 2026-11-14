# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the reasoner's flat text and visual keys without loading policy tensors."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np



def _hf_cache_blob(root: Path, shard: Path, target: Path) -> bool:
    """Accept repo-local blobs or the marked Hub shared-Xet symlink chain."""
    if root.parent.name != "snapshots":
        return False
    repo = root.parent.parent
    blobs = repo / "blobs"
    if blobs.is_symlink():
        return False
    if target.is_relative_to(blobs):
        return True
    # Hub 1.33 can deduplicate Xet payloads across repositories. The snapshot
    # still links through its own repo blob, which links to the cache-wide
    # blobs/<two-hex-prefix>/<64-hex-Xet-hash> store. Do not accept arbitrary
    # siblings, direct external links, or symlinked store components.
    if not repo.name.startswith("models--") or not re.fullmatch(r"[0-9a-f]{40}", root.name):
        return False

    def link_target(link):
        value = link.readlink()
        return Path(os.path.abspath(value if value.is_absolute() else link.parent / value))

    try:
        if not shard.is_symlink() or not shard.parent.resolve().is_relative_to(root):
            return False
        blob = link_target(shard)
        if (blob.parent != blobs or not blob.is_symlink()
                or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", blob.name)):
            return False
        shared = link_target(blob)
        store = repo.parent / "blobs"
        marker = store / ".huggingface-shared-blobs"
        if (shared != target or shared.parent.parent != store
                or not re.fullmatch(r"[0-9a-f]{64}", shared.name)
                or shared.parent.name != shared.name[:2]
                or store.is_symlink() or shared.parent.is_symlink()
                or shared.is_symlink() or not shared.is_file()
                or marker.is_symlink() or not marker.is_file()):
            return False
        with marker.open("rb") as stream:
            return stream.read(3) == b"1\n"
    except (OSError, ValueError, RuntimeError):
        return False


class CheckpointWeights:
    """Lazy access through the checkpoint's reasoning safetensors index.

    Text tensors reside inside the diffusion transformer shards. Iterating every
    tensor in those shards would also load unrelated generation-policy weights.
    The root index explicitly names only reasoning tensors and their shard files.
    """

    def __init__(self, model_dir: Path):
        self.root = Path(model_dir).resolve()
        document = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.mapping = document.get("weight_map")
        if not isinstance(self.mapping, dict) or not self.mapping:
            raise ValueError("Cosmos3-Edge requires a nonempty reasoning weight_map")
        for key, shard in self.mapping.items():
            if not isinstance(key, str) or not isinstance(shard, str):
                raise ValueError("Cosmos3-Edge reasoning weight_map entries must be strings")
            relative = Path(shard)
            path = (self.root / relative).resolve()
            # Accept only recognized HF cache layouts, not arbitrary external links.
            cache_blob = _hf_cache_blob(self.root, self.root / relative, path)
            if (relative.is_absolute() or relative.as_posix() != shard
                    or any(part in {".", ".."} for part in relative.parts)
                    or "\\" in shard or "\0" in shard or not path.is_file()
                    or not (path.is_relative_to(self.root) or cache_blob)):
                raise ValueError(f"Cosmos3-Edge missing or non-contained checkpoint shard: {shard}")

    def __getitem__(self, key: str) -> np.ndarray:
        from safetensors import safe_open

        # Convert BF16 directly to FP32, never via FP16 (which has a smaller
        # exponent range). The graph owns the requested computation dtype.
        with safe_open(self.root / self.mapping[key], framework="pt", device="cpu") as stream:
            return stream.get_tensor(key).float().numpy()
