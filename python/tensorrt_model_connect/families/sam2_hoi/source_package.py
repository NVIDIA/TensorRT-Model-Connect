# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity and provenance contract for the supplied SAM2.1 HOI package."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


MODEL_TYPE = "sam2_hoi"
SOURCE_COMMIT = "79ab25d6bd5535bcb748de9f3b90e16b1a24e58d"
HOI_DETR_SOURCE_COMMIT = "a86ef1f24ae61df5d029cc51af4a362ff6f26781"
MMCV_SOURCE_COMMIT = "4c01b026f0afa5a91a5f54aea313788da1e40f95"
MMDET_PATCH_SHA256 = "6582ee3ed84be96bd1b2d2e3387e3853d06e93f75903808b59189ec14703dfeb"
MMCV_PATCH_SHA256 = "05460b8abf866d5c3ca5efa335c99c53526b8bbf00b94790838e3f70231e9506"
CHECKPOINT_SHA256 = "88849a8268a38ba66061093f90866af1d033d05d0f1de865534bf490e9880292"
CHECKPOINT_SIZE = 285_333_981

CHECKPOINT_RELATIVE_PATH = Path("checkpoint/sam2.1_hiera_small_with_hoi_ft_aug_e5_c4.pt")
CONFIG_RELATIVE_PATH = Path("sam2/configs/sam2.1/sam2.1_hiera_s_with_hoi_head_c4.yaml")
HOI_HEAD_RELATIVE_PATH = Path("sam2/modeling/backbones/hoi_head.py")
MMDET_PATCH_RELATIVE_PATH = Path("vendor/HOI-DETR/mmdet-custom.patch")
MMCV_PATCH_RELATIVE_PATH = Path("vendor/mmcv-custom.patch")


@dataclass(frozen=True)
class SourcePackage:
    root: Path
    checkpoint: Path
    config: Path
    mmdet_patch: Path
    mmcv_patch: Path
    source_commit: str
    hoi_detr_source_commit: str
    mmcv_source_commit: str


def _read_required_text(path: Path, description: str) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RuntimeError(f"Could not read SAM2 HOI {description}: {path}") from error
    if not value:
        raise RuntimeError(f"SAM2 HOI {description} is empty: {path}")
    return value


def _read_mmcv_source_commit(path: Path) -> str:
    text = _read_required_text(path, "MMCV source commit")
    prefix = "Public MMCV upstream commit:"
    first_line = text.splitlines()[0].strip()
    if not first_line.startswith(prefix):
        raise RuntimeError(f"Unsupported SAM2 HOI MMCV provenance format at {path}: {first_line!r}")
    commit = first_line.removeprefix(prefix).strip()
    if not commit:
        raise RuntimeError(f"SAM2 HOI MMCV source commit is empty: {path}")
    return commit


def _looks_like_source_package(root: Path) -> bool:
    return any(
        (root / relative).exists()
        for relative in (
            CHECKPOINT_RELATIVE_PATH,
            CONFIG_RELATIVE_PATH,
            HOI_HEAD_RELATIVE_PATH,
        )
    )


def inspect_source_package(model_dir: str | Path) -> SourcePackage | None:
    """Resolve the one reviewed package layout and reject partial lookalikes."""

    root = Path(model_dir)
    if not root.is_dir() or not _looks_like_source_package(root):
        return None

    required = {
        "checkpoint": root / CHECKPOINT_RELATIVE_PATH,
        "configuration": root / CONFIG_RELATIVE_PATH,
        "HOI head source": root / HOI_HEAD_RELATIVE_PATH,
        "source commit": root / "SOURCE_COMMIT",
        "HOI-DETR source commit": root / "vendor/HOI-DETR/SOURCE_COMMIT",
        "MMCV source commit": root / "vendor/MMCV_SOURCE_COMMIT",
        "MMDetection patch": root / MMDET_PATCH_RELATIVE_PATH,
        "MMCV patch": root / MMCV_PATCH_RELATIVE_PATH,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError("Incomplete SAM2 HOI source package; missing " + ", ".join(missing))

    source_commit = _read_required_text(required["source commit"], "source commit")
    hoi_detr_source_commit = _read_required_text(
        required["HOI-DETR source commit"], "HOI-DETR source commit"
    )
    mmcv_source_commit = _read_mmcv_source_commit(required["MMCV source commit"])
    expected = {
        "source commit": (source_commit, SOURCE_COMMIT),
        "HOI-DETR source commit": (hoi_detr_source_commit, HOI_DETR_SOURCE_COMMIT),
        "MMCV source commit": (mmcv_source_commit, MMCV_SOURCE_COMMIT),
    }
    mismatches = [
        f"{name} expected {want}, got {actual}"
        for name, (actual, want) in expected.items()
        if actual != want
    ]
    if mismatches:
        raise RuntimeError("Unsupported SAM2 HOI source package: " + "; ".join(mismatches))

    checkpoint = required["checkpoint"]
    size = checkpoint.stat().st_size
    if size != CHECKPOINT_SIZE:
        raise RuntimeError(
            "Unsupported SAM2 HOI checkpoint size: "
            f"expected {CHECKPOINT_SIZE}, got {size} at {checkpoint}"
        )

    return SourcePackage(
        root=root.resolve(),
        checkpoint=checkpoint.resolve(),
        config=required["configuration"].resolve(),
        mmdet_patch=required["MMDetection patch"].resolve(),
        mmcv_patch=required["MMCV patch"].resolve(),
        source_commit=source_commit,
        hoi_detr_source_commit=hoi_detr_source_commit,
        mmcv_source_commit=mmcv_source_commit,
    )


def checkpoint_sha256(checkpoint: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(checkpoint).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(package: SourcePackage) -> None:
    actual = checkpoint_sha256(package.checkpoint)
    if actual != CHECKPOINT_SHA256:
        raise RuntimeError(
            "Unsupported SAM2 HOI checkpoint digest: "
            f"expected {CHECKPOINT_SHA256}, got {actual} at {package.checkpoint}"
        )


def verify_source_patches(package: SourcePackage) -> None:
    """Verify both reviewed source patches before either one is consumed."""

    patches = (
        ("MMDetection patch", package.mmdet_patch, MMDET_PATCH_SHA256),
        ("MMCV patch", package.mmcv_patch, MMCV_PATCH_SHA256),
    )
    mismatches: list[str] = []
    for description, path, expected in patches:
        try:
            actual = checkpoint_sha256(path)
        except OSError as error:
            raise RuntimeError(f"Could not read reviewed SAM2 HOI {description}: {path}") from error
        if actual != expected:
            mismatches.append(f"{description} expected {expected}, got {actual} at {path}")
    if mismatches:
        raise RuntimeError("Unsupported SAM2 HOI source patches: " + "; ".join(mismatches))


def resolve_config(model_dir: Path) -> dict[str, object] | None:
    """Adapt the reviewed non-Hugging-Face source package into ModelConfig."""

    package = inspect_source_package(model_dir)
    if package is None:
        return None
    return {
        "model_type": MODEL_TYPE,
        "architectures": ["SAM2HoiVideoTracker"],
        "hidden_size": 256,
        "intermediate_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 1,
        "image_size": 1024,
        "source_commit": package.source_commit,
        "hoi_detr_source_commit": package.hoi_detr_source_commit,
        "mmcv_source_commit": package.mmcv_source_commit,
        "mmdet_patch_sha256": MMDET_PATCH_SHA256,
        "mmcv_patch_sha256": MMCV_PATCH_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "sam2_hoi": {
            "variant": "sam2.1_hiera_small_hoi_c4",
            "hiera_embed_dim": 96,
            "hiera_stages": [1, 2, 11, 2],
            "hiera_global_attention_blocks": [7, 10, 13],
            "fpn_hidden_size": 256,
            "hoi_num_queries": 1500,
            "hoi_num_classes": 4,
            "hoi_num_feature_levels": 3,
            "hoi_encoder_layers": 6,
            "hoi_decoder_layers": 6,
            "memory_attention_layers": 4,
            "memory_channels": 64,
            "num_mask_memory_frames": 7,
            "score_threshold": 0.35,
            "class_nms_threshold": 0.5,
            "global_nms_threshold": 0.75,
            "hand_nms_threshold": 0.25,
            "interaction_threshold": 0.5,
            "mask_logit_threshold": 0.01,
        },
    }
