# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map the checkpoint file's tensor names onto the ones the builders use.

``transformers`` renames tensors while loading, so the keys a live model
reports from ``state_dict()`` are NOT the keys stored in ``model.safetensors``.
The bundle build reads the file, so the file's names are what matter. Verifying
a graph against a loaded model therefore proves the arithmetic and nothing
about whether a bundle can be built - which is exactly how these three renames
reached an end-to-end run.

The renames, measured by diffing the two key sets:

    model.backbone.model.*          <- model.backbone.conv_encoder.model.*
    <enc|dec>.layers.N.mlp.fc{1,2}  <- <enc|dec>.layers.N.fc{1,2}
    decoder.layers.N.self_attn.o_proj <- ...self_attn.out_proj
"""

from __future__ import annotations

import re

# (file pattern, builder replacement) applied in order.
_RENAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^model\.backbone\.conv_encoder\.model\."), "model.backbone.model."),
    (re.compile(r"^(model\.(?:encoder|decoder)\.layers\.\d+)\.(fc[12])\."), r"\1.mlp.\2."),
    (re.compile(r"^(model\.decoder\.layers\.\d+\.self_attn)\.out_proj\."), r"\1.o_proj."),
)


def canonical_name(name: str) -> str:
    """Return the builder-facing name for one checkpoint tensor."""
    for pattern, replacement in _RENAMES:
        new, count = pattern.subn(replacement, name)
        if count:
            return new
    return name


def normalise(weights: dict) -> dict:
    """Rename a checkpoint's tensors onto the builder vocabulary."""
    out: dict = {}
    for name, tensor in weights.items():
        key = canonical_name(name)
        if key in out:
            raise KeyError(f"deformable_detr checkpoint rename collided on {key}")
        out[key] = tensor
    return out
