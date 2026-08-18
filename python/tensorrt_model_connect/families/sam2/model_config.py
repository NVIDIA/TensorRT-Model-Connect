# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize the supplied SAM2 package without loading its checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path


PACKAGE_DIRNAME = "sam2_nvidia_repro"
CONFIG_RELATIVE_PATH = Path("config/sam2.1_hiera_s_with_bbox_head.yaml")
CHECKPOINT_RELATIVE_PATH = Path("checkpoint/sam2.1_hiera_small_with_bbox_head.pt")
REFERENCE_CONFIG_SHA256 = "59488bb78c7cc48aaaebd966ea9d054014f683459d062b7a959a4aa501342656"
PUBLIC_CONFIG_RELATIVE_PATH = Path("sam2.1_hiera_s.yaml")
PUBLIC_CHECKPOINT_RELATIVE_PATH = Path("sam2.1_hiera_small.pt")
PUBLIC_CONFIG_SHA256 = "632e5cd0104f5ab6cd4f9d2dfd80a8e7240e481ad7960a13cad2ae3504b88dbd"


def _is_regular_descendant(root: Path, relative: Path) -> bool:
    candidate = root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            return False
    return candidate.is_file()


def resolve_package_root(model_dir: str | Path) -> Path | None:
    """Accept either the extracted package root or its containing directory."""

    path = Path(model_dir)
    for root in (path, path / PACKAGE_DIRNAME):
        if (
            root.is_dir()
            and not root.is_symlink()
            and _is_regular_descendant(root, CONFIG_RELATIVE_PATH)
            and _is_regular_descendant(root, CHECKPOINT_RELATIVE_PATH)
        ):
            return root
    return None


def resolve_public_package_root(model_dir: str | Path) -> Path | None:
    """Recognize the pinned public SAM2.1 small snapshot layout."""

    root = Path(model_dir)
    if (
        root.is_dir()
        and not root.is_symlink()
        and _resolved_regular_file(root / PUBLIC_CONFIG_RELATIVE_PATH) is not None
        and _resolved_regular_file(root / PUBLIC_CHECKPOINT_RELATIVE_PATH) is not None
    ):
        return root
    return None


def _resolved_regular_file(path: Path) -> Path | None:
    """Resolve one public HF-cache link to an authenticated regular blob."""

    try:
        target = path.resolve(strict=True)
    except OSError:
        return None
    return target if target.is_file() else None


def resolve_public_file(root: Path, relative: Path) -> Path:
    target = _resolved_regular_file(root / relative)
    if target is None:
        raise ValueError(f"unsupported SAM2 public snapshot file: {root / relative}")
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_from_dir(model_dir: str | Path) -> dict | None:
    """Return the fixed native-family config for the exact delivered YAML.

    Discovery hashes only the small YAML. The Python builder independently
    authenticates the checkpoint before building the TensorRT plans.
    """

    root = resolve_package_root(model_dir)
    expected = REFERENCE_CONFIG_SHA256
    relative = CONFIG_RELATIVE_PATH
    if root is None:
        root = resolve_public_package_root(model_dir)
        expected = PUBLIC_CONFIG_SHA256
        relative = PUBLIC_CONFIG_RELATIVE_PATH
    if root is None:
        return None
    config_path = (
        resolve_public_file(root, relative)
        if relative == PUBLIC_CONFIG_RELATIVE_PATH
        else root / relative
    )
    actual = _sha256(config_path)
    if actual != expected:
        raise ValueError(f"unsupported SAM2 config: expected SHA-256 {expected}, got {actual}")
    return {
        "model_type": "sam2",
        "architectures": ["Sam2BBoxVideoTracking"],
        "hidden_size": 256,
        "intermediate_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "max_position_embeddings": 1024,
    }


def prefer_native_default(config: object) -> bool:
    """Always use the family-owned Python builder for recognized SAM2 packages."""

    return str(getattr(config, "model_type", "")).lower().replace("-", "_") == "sam2"
