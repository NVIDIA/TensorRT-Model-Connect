# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tensorrt_model_connect.families import find_plugin
from tensorrt_model_connect.families.sam2_hoi import source_package
from tensorrt_model_connect.families.sam2_hoi.source_package import (
    CHECKPOINT_RELATIVE_PATH,
    CHECKPOINT_SIZE,
    CONFIG_RELATIVE_PATH,
    HOI_DETR_SOURCE_COMMIT,
    HOI_HEAD_RELATIVE_PATH,
    MMCV_PATCH_RELATIVE_PATH,
    MMCV_PATCH_SHA256,
    MMCV_SOURCE_COMMIT,
    MMDET_PATCH_RELATIVE_PATH,
    MMDET_PATCH_SHA256,
    SOURCE_COMMIT,
    inspect_source_package,
    resolve_config,
    verify_source_patches,
)


MMDET_PATCH_FIXTURE = b"# fixture MMDetection patch\n"
MMCV_PATCH_FIXTURE = b"# fixture MMCV patch\n"


def _write_package(root: Path, *, source_commit: str = SOURCE_COMMIT) -> None:
    files = {
        CONFIG_RELATIVE_PATH: b"model: {}\n",
        HOI_HEAD_RELATIVE_PATH: b"# fixture\n",
        Path("SOURCE_COMMIT"): (source_commit + "\n").encode(),
        Path("vendor/HOI-DETR/SOURCE_COMMIT"): (HOI_DETR_SOURCE_COMMIT + "\n").encode(),
        Path("vendor/MMCV_SOURCE_COMMIT"): (
            f"Public MMCV upstream commit: {MMCV_SOURCE_COMMIT}\n"
            "Public version: 1.7.2\n"
            "The package includes the local compatibility delta.\n"
        ).encode(),
        MMDET_PATCH_RELATIVE_PATH: MMDET_PATCH_FIXTURE,
        MMCV_PATCH_RELATIVE_PATH: MMCV_PATCH_FIXTURE,
    }
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    checkpoint = root / CHECKPOINT_RELATIVE_PATH
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("wb") as stream:
        stream.truncate(CHECKPOINT_SIZE)


def _accept_fixture_patch_digests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        source_package,
        "MMDET_PATCH_SHA256",
        hashlib.sha256(MMDET_PATCH_FIXTURE).hexdigest(),
    )
    monkeypatch.setattr(
        source_package,
        "MMCV_PATCH_SHA256",
        hashlib.sha256(MMCV_PATCH_FIXTURE).hexdigest(),
    )


def test_family_selection_is_standalone() -> None:
    assert find_plugin("sam2_hoi").name == "sam2_hoi"
    assert find_plugin("sam2-hoi").name == "sam2_hoi"
    assert find_plugin("sam2.1-hoi").name == "sam2_hoi"
    sam = find_plugin("sam2")
    sam3 = find_plugin("sam3")
    assert sam is None or sam.name != "sam2_hoi"
    assert sam3 is None or sam3.name != "sam2_hoi"


def test_config_adapter_ignores_unrelated_directory(tmp_path: Path) -> None:
    assert resolve_config(tmp_path) is None


def test_config_adapter_returns_exact_architecture(tmp_path: Path) -> None:
    _write_package(tmp_path)
    config = resolve_config(tmp_path)
    assert config is not None
    assert config["model_type"] == "sam2_hoi"
    assert config["source_commit"] == SOURCE_COMMIT
    assert config["checkpoint_sha256"]
    assert config["mmdet_patch_sha256"] == MMDET_PATCH_SHA256
    assert config["mmcv_patch_sha256"] == MMCV_PATCH_SHA256
    assert config["sam2_hoi"] == {
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
    }


def test_config_adapter_rejects_partial_package(tmp_path: Path) -> None:
    checkpoint = tmp_path / CHECKPOINT_RELATIVE_PATH
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match="Incomplete SAM2 HOI source package"):
        resolve_config(tmp_path)


def test_config_adapter_requires_mmcv_patch(tmp_path: Path) -> None:
    _write_package(tmp_path)
    (tmp_path / MMCV_PATCH_RELATIVE_PATH).unlink()
    with pytest.raises(
        RuntimeError,
        match="Incomplete SAM2 HOI source package; missing MMCV patch",
    ):
        resolve_config(tmp_path)


def test_config_adapter_rejects_unreviewed_source_commit(tmp_path: Path) -> None:
    _write_package(tmp_path, source_commit="0" * 40)
    with pytest.raises(RuntimeError, match="Unsupported SAM2 HOI source package"):
        resolve_config(tmp_path)


def test_verify_source_patches_accepts_reviewed_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_package(tmp_path)
    _accept_fixture_patch_digests(monkeypatch)
    package = inspect_source_package(tmp_path)
    assert package is not None
    verify_source_patches(package)


@pytest.mark.parametrize(
    ("relative_path", "description"),
    [
        (MMDET_PATCH_RELATIVE_PATH, "MMDetection patch"),
        (MMCV_PATCH_RELATIVE_PATH, "MMCV patch"),
    ],
)
def test_verify_source_patches_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: Path,
    description: str,
) -> None:
    _write_package(tmp_path)
    _accept_fixture_patch_digests(monkeypatch)
    package = inspect_source_package(tmp_path)
    assert package is not None
    (tmp_path / relative_path).write_bytes(b"tampered patch\n")
    with pytest.raises(
        RuntimeError,
        match=rf"Unsupported SAM2 HOI source patches: {description} expected",
    ):
        verify_source_patches(package)
