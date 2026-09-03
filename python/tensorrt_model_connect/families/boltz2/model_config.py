# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize a reproducible pinned Boltz-2 build package."""

from __future__ import annotations

from pathlib import Path


CHECKPOINT = Path("boltz2_conf.ckpt")
REQUEST = Path("protein_monomer.yaml")
MSA = Path("protein_monomer.a3m")
PROCESSED = Path("processed")
STRUCTURE = PROCESSED / "structures/protein_monomer.npz"
RECORD = PROCESSED / "records/protein_monomer.json"
MOLS = Path("mols")
MOLS_ARCHIVE = Path("mols.tar")


def resolve_package_root(model_dir: str | Path) -> Path | None:
    """Return a package root only when every pinned build input is present."""

    root = Path(model_dir)
    required = (CHECKPOINT, REQUEST, MSA, STRUCTURE, RECORD, MOLS_ARCHIVE)
    if root.is_symlink() or not root.is_dir():
        return None
    for relative in required:
        path = root / relative
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            return None
        if not resolved.is_file():
            return None
    if not (root / MOLS).is_dir():
        return None
    return root


def config_from_dir(model_dir: str | Path) -> dict | None:
    """Return the exact default static-profile native configuration."""

    if resolve_package_root(model_dir) is None:
        return None
    return {
        "model_type": "boltz2",
        "architectures": ["Boltz2ForStructurePrediction"],
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 64,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": 117,
        "token_count": 117,
        "atom_count": 928,
        "msa_depth": 1,
        "recycling_steps": 3,
        "sampling_steps": 200,
        "diffusion_samples": 1,
    }


def prefer_native_default(config: object) -> bool:
    return str(getattr(config, "model_type", "")).lower().replace("-", "_") == "boltz2"
