# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused CPU contracts for the offline SAM2 reference package."""

from __future__ import annotations

import hashlib
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml

from tensorrt_model_connect.families import family_has_capability, find_plugin
from tensorrt_model_connect.families.sam2 import archive_contract
from tensorrt_model_connect.families.sam2.archive_contract import (
    Sam2ArchiveContractError,
)
from tensorrt_model_connect.families.sam2.model_config import (
    config_from_dir,
    require_reference_archive,
)


def _config_payload() -> dict:
    return {
        "model": {
            "_target_": "sam2.modeling.sam2_base.SAM2Base",
            "image_encoder": {
                "_target_": "sam2.modeling.backbones.image_encoder.ImageEncoderWithBBoxHead",
                "trunk": {
                    "_target_": "sam2.modeling.backbones.hieradet.Hiera",
                    "embed_dim": 96,
                    "num_heads": 1,
                    "stages": [1, 2, 11, 2],
                    "global_att_blocks": [7, 10, 13],
                },
                "neck": {
                    "d_model": 256,
                    "backbone_channel_list": [768, 384, 192, 96],
                },
                "bbox_head": {
                    "_target_": "sam2.modeling.backbones.bbox_head.RTMDetSepBNHeadModule",
                    "num_classes": 2,
                    "in_channels": 256,
                    "feat_channels": 256,
                    "stacked_convs": 2,
                    "featmap_strides": [8, 16, 32],
                    "featmap_sizes": [[128, 128], [64, 64], [32, 32]],
                    "share_conv": True,
                },
            },
            "memory_attention": {
                "d_model": 256,
                "num_layers": 4,
                "layer": {"dim_feedforward": 2048},
            },
            "memory_encoder": {"out_dim": 64},
            "num_maskmem": 7,
            "image_size": 1024,
            "use_high_res_features_in_sam": True,
            "use_obj_ptrs_in_encoder": True,
            "pred_obj_scores": True,
        }
    }


def _synthetic_inventory() -> dict:
    return {
        "format": "pytorch_zip",
        **dict(archive_contract._REFERENCE_CHECKPOINT_LAYOUT),
    }


@pytest.fixture
def compatible_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / archive_contract.PACKAGE_DIRNAME
    config = root / archive_contract.CONFIG_RELATIVE_PATH
    checkpoint = root / archive_contract.CHECKPOINT_RELATIVE_PATH
    config.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump(_config_payload()), encoding="utf-8")
    checkpoint.write_bytes(b"synthetic-checkpoint")
    (root / archive_contract.SHA256SUMS_RELATIVE_PATH).write_text(
        f"{archive_contract.sha256_file(config)}  {archive_contract.CONFIG_RELATIVE_PATH.as_posix()}\n"
        f"{archive_contract.sha256_file(checkpoint)}  "
        f"{archive_contract.CHECKPOINT_RELATIVE_PATH.as_posix()}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        archive_contract,
        "validate_checkpoint_inventory",
        lambda _path: _synthetic_inventory(),
    )
    monkeypatch.setattr(
        archive_contract,
        "REFERENCE_CONFIG_SHA256",
        archive_contract.sha256_file(config),
    )
    monkeypatch.setattr(
        archive_contract,
        "REFERENCE_CHECKPOINT_SHA256",
        archive_contract.sha256_file(checkpoint),
    )
    monkeypatch.setattr(
        archive_contract,
        "REFERENCE_SHA256SUMS_SHA256",
        archive_contract.sha256_file(root / archive_contract.SHA256SUMS_RELATIVE_PATH),
    )
    return root


def test_archive_contract_rejects_symlinked_package_root(
    compatible_package: Path,
    tmp_path: Path,
) -> None:
    linked = tmp_path / "linked-package"
    linked.symlink_to(compatible_package, target_is_directory=True)

    assert archive_contract.resolve_package_root(linked) is None
    with pytest.raises(Sam2ArchiveContractError, match="root must not be a symlink"):
        archive_contract.verify_declared_provenance(linked)


def test_config_contract_fails_closed_on_non_bbox_encoder(
    compatible_package: Path,
) -> None:
    config_path = compatible_package / archive_contract.CONFIG_RELATIVE_PATH
    payload = _config_payload()
    payload["model"]["image_encoder"]["_target_"] = (
        "sam2.modeling.backbones.image_encoder.ImageEncoder"
    )
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(Sam2ArchiveContractError, match="ImageEncoderWithBBoxHead"):
        archive_contract.validate_model_config(config_path)


def test_config_contract_rejects_adjacent_source_extra_fpn(
    compatible_package: Path,
) -> None:
    config_path = compatible_package / archive_contract.CONFIG_RELATIVE_PATH
    payload = _config_payload()
    payload["model"]["image_encoder"]["learnable_fpn_module"] = {
        "_target_": "sam2.modeling.backbones.image_encoder.CSPNeXtPAFPN",
        "in_channels": [256, 256, 256],
        "out_channels": 256,
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(
        Sam2ArchiveContractError,
        match=r"unsupported graph field model\.image_encoder\.learnable_fpn_module",
    ):
        archive_contract.validate_model_config(config_path)


def test_checkpoint_recognizer_never_unpickles_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    opaque_pickle = b"this is deliberately not a valid pickle"
    storage = b"opaque tensor bytes"
    with zipfile.ZipFile(checkpoint, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("checkpoint/data.pkl", opaque_pickle)
        archive.writestr("checkpoint/version", "3\n")
        archive.writestr("checkpoint/data/0", storage)
    monkeypatch.setattr(
        archive_contract,
        "_REFERENCE_CHECKPOINT_LAYOUT",
        {
            "serialization_version": "3",
            "archive_members": 3,
            "pickle_bytes": len(opaque_pickle),
            "zip_storage_members": 1,
            "stored_nbytes": len(storage),
        },
    )

    inventory = archive_contract.validate_checkpoint_inventory(checkpoint)

    assert inventory["pickle_bytes"] == len(opaque_pickle)
    assert inventory["stored_nbytes"] == len(storage)


def test_checkpoint_recognizer_rejects_unsafe_members(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unsafe.pt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("checkpoint/data.pkl", b"opaque")
        archive.writestr("checkpoint/version", "3\n")
        archive.writestr("checkpoint/data/0", b"tensor")
        archive.writestr("../escape", b"bad")

    with pytest.raises(Sam2ArchiveContractError, match="unsafe ZIP members"):
        archive_contract.validate_checkpoint_inventory(checkpoint)


def test_declared_provenance_is_optional_but_tamper_evident(tmp_path: Path) -> None:
    payload = b"checkpoint-values"
    (tmp_path / "weights.pt").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  weights.pt\n",
        encoding="utf-8",
    )

    report = archive_contract.verify_declared_provenance(tmp_path)
    assert report["files"] == {"weights.pt": digest}

    (tmp_path / "weights.pt").write_bytes(b"changed")
    with pytest.raises(Sam2ArchiveContractError, match="SHA256 mismatch"):
        archive_contract.verify_declared_provenance(tmp_path)


def test_manifest_rejects_unsafe_paths(tmp_path: Path) -> None:
    (tmp_path / "SHA256SUMS").write_text(
        f"{'0' * 64}  ../outside\n",
        encoding="utf-8",
    )
    with pytest.raises(Sam2ArchiveContractError, match="unsafe SHA256SUMS path"):
        archive_contract.read_declared_sha256s(tmp_path)


def test_build_and_runtime_are_registered() -> None:
    plugin = find_plugin("sam2")

    assert plugin is not None
    assert plugin.name == "sam2"
    assert plugin.runtime_strategy == "sam2_bbox_video_tracking"
    assert family_has_capability("sam2", "complete_bundle_builder")
    runtime_manifest = (
        Path(__file__).resolve().parents[5] / "src" / "runtime" / "models" / "sam2" / "MODEL.toml"
    )
    manifest = tomllib.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert manifest["id"] == "sam2"
    assert manifest["runtime_plugins"] == ["plugin.cpp|register_sam2_plugin"]
    assert manifest["runtime_strategies"] == ["sam2_bbox_video_tracking"]


def test_exact_supplied_package_has_a_family_config(compatible_package: Path) -> None:
    config = config_from_dir(compatible_package)

    assert config is not None
    assert config["model_type"] == "sam2"
    assert config["architectures"] == ["Sam2BBoxVideoTracking"]
    assert config["sam2_precision"] == "mixed_bf16_fp32"
    assert config["sam2_qualification"] == "unqualified"
    assert config["sam2_runtime_eligible"] is False


def test_family_discovery_does_not_hash_the_large_checkpoint(
    compatible_package: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = compatible_package / archive_contract.CHECKPOINT_RELATIVE_PATH
    real_sha256_file = archive_contract.sha256_file

    def guarded_sha256_file(path: Path) -> str:
        if path == checkpoint:
            raise AssertionError("discovery must defer full checkpoint hashing to build time")
        return real_sha256_file(path)

    monkeypatch.setattr(archive_contract, "sha256_file", guarded_sha256_file)

    assert config_from_dir(compatible_package)["model_type"] == "sam2"


@pytest.mark.parametrize("asset", ["config", "checkpoint", "SHA256SUMS"])
def test_adjacent_package_drift_is_rejected_before_native_builder_exec(
    compatible_package: Path,
    asset: str,
) -> None:
    if asset == "config":
        path = compatible_package / archive_contract.CONFIG_RELATIVE_PATH
        path.write_bytes(path.read_bytes() + b"\n")
    elif asset == "checkpoint":
        path = compatible_package / archive_contract.CHECKPOINT_RELATIVE_PATH
        path.write_bytes(path.read_bytes() + b"-adjacent")
    else:
        path = compatible_package / archive_contract.SHA256SUMS_RELATIVE_PATH
        path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(Sam2ArchiveContractError, match="SHA256|provenance"):
        require_reference_archive(compatible_package)


def test_python_package_contains_only_the_owned_build_and_evidence_surface() -> None:
    family = Path(__file__).resolve().parent.parent
    assert {path.name for path in family.glob("*.py")} == {
        "__init__.py",
        "archive_contract.py",
        "capture_golden.py",
        "golden_evidence.py",
        "model_config.py",
        "native_builder.py",
        "plugin.py",
    }


def test_native_production_has_only_direct_cpp_tensor_rt_dependencies() -> None:
    repo = Path(__file__).resolve().parents[5]
    production_roots = (
        repo / "tools" / "sam2_native_builder",
        repo / "src" / "runtime" / "models" / "sam2",
    )
    sources = sorted(
        path
        for root in production_roots
        for path in root.iterdir()
        if path.suffix in {".c", ".cc", ".cpp", ".cu", ".cuh", ".cxx", ".h", ".hpp"}
    )
    assert sources
    assert not any("onnx" in path.name.lower() for path in sources)

    forbidden = (
        "NvOnnxParser",
        "nvonnxparser",
        "onnxruntime",
        "onnx::",
        "<onnx",
        "torch::",
        "<torch/",
        '"torch/',
        "<ATen/",
        "<c10/",
        "libtorch",
        "Python.h",
        "pybind11",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "<NvInfer.h>" in combined
    for token in forbidden:
        assert token not in combined


def test_supplied_archive_contract_when_explicitly_available() -> None:
    import os

    raw_path = os.environ.get("TRTMC_SAM2_REPRO_DIR")
    if not raw_path:
        pytest.skip("set TRTMC_SAM2_REPRO_DIR to validate the supplied package")

    description = archive_contract.describe_archive(raw_path, verify_provenance=True)

    assert description.checkpoint_inventory["archive_members"] == 597
    assert description.checkpoint_inventory["zip_storage_members"] == 595
    assert description.checkpoint_inventory["stored_nbytes"] == 193_747_376
    assert description.provenance["matches_reference_manifest"] is True
    assert description.provenance["matches_reference_checkpoint"] is True
    assert description.provenance["matches_reference_config"] is True
