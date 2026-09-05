# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize a reproducible prepared OpenFold3 build package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contracts import parse_query_json


CHECKPOINT = Path("of3-ob-2025-06-30-174k.pt")
QUERY = Path("query.json")
FEATURES = Path("openfold3_features.npz")
STRUCTURE_METADATA = Path("openfold3_structure.json")
COMPONENTS = Path("components.bcif")


def _regular_files(root: Path, relatives: tuple[Path, ...]) -> bool:
    try:
        resolved_root = root.resolve(strict=True)
    except OSError:
        return False
    if root.is_symlink() or not resolved_root.is_dir():
        return False
    for relative in relatives:
        try:
            resolved = (resolved_root / relative).resolve(strict=True)
        except OSError:
            return False
        if not resolved.is_file() or resolved_root not in resolved.parents:
            return False
    return True


def resolve_package_root(model_dir: str | Path) -> Path | None:
    """Return the package root only when all required files are regular files."""

    root = Path(model_dir)
    if not _regular_files(root, (CHECKPOINT, QUERY, FEATURES, STRUCTURE_METADATA, COMPONENTS)):
        return None
    return root.resolve(strict=True)


def config_from_dir(model_dir: str | Path) -> dict | None:
    """Return model metadata derived from a validated prepared package."""

    root = resolve_package_root(model_dir)
    if root is None:
        return None
    request = parse_query_json((root / QUERY).read_text(encoding="utf-8"))
    try:
        metadata = json.loads((root / STRUCTURE_METADATA).read_text(encoding="utf-8"))
        atom_count = int(metadata["atom_count"])
        with np.load(root / FEATURES, allow_pickle=False) as features:
            msa_depth = int(features["msa"].shape[1])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid OpenFold3 structure metadata") from error
    if atom_count <= 0:
        raise ValueError("OpenFold3 structure metadata atom_count must be positive")
    return {
        "model_type": "openfold3",
        "architectures": ["OpenFold3ForStructurePrediction"],
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": request.token_count,
        "token_count": request.token_count,
        "atom_count": atom_count,
        "msa_depth": msa_depth,
        "recycling_steps": 3,
        "sampling_steps": 200,
        "diffusion_samples": 1,
    }


def prefer_native_default(config: object) -> bool:
    """Select native dispatch for OpenFold3 configs."""

    return str(getattr(config, "model_type", "")).lower().replace("-", "_") == "openfold3"
