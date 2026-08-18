# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Essential contracts for the Python-owned SAM2 family."""

from __future__ import annotations

import hashlib
import importlib
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families import find_plugin
from tensorrt_model_connect.families.sam2.checkpoint_mapper import (
    Checkpoint,
    CheckpointError,
    PUBLIC_CHECKPOINT_SHA256,
    PublicCoreCheckpoint,
)
from tensorrt_model_connect.families.sam2.float_math import reciprocal_sqrtf
from tensorrt_model_connect.families.sam2.model_config import (
    CHECKPOINT_RELATIVE_PATH,
    CONFIG_RELATIVE_PATH,
    PACKAGE_DIRNAME,
    PUBLIC_CHECKPOINT_RELATIVE_PATH,
    PUBLIC_CONFIG_SHA256,
    PUBLIC_CONFIG_RELATIVE_PATH,
    config_from_dir,
    prefer_native_default,
    resolve_package_root,
    resolve_public_package_root,
)


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / PACKAGE_DIRNAME
    config = root / CONFIG_RELATIVE_PATH
    checkpoint = root / CHECKPOINT_RELATIVE_PATH
    config.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    config.write_bytes(b"exact-config")
    checkpoint.write_bytes(b"authenticated-checkpoint")
    from tensorrt_model_connect.families.sam2 import model_config

    monkeypatch.setattr(
        model_config,
        "REFERENCE_CONFIG_SHA256",
        hashlib.sha256(config.read_bytes()).hexdigest(),
    )
    return root


def test_family_loads_only_authenticated_checkpoint_bytes(
    package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_model_connect.families.sam2 import checkpoint_mapper

    checkpoint_path = package / CHECKPOINT_RELATIVE_PATH
    monkeypatch.setattr(
        checkpoint_mapper,
        "SUPPORTED_CHECKPOINT_SHA256",
        hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    )

    class FakeTensor:
        pass

    calls = []

    def load(stream, *, map_location, weights_only):
        calls.append((stream.read(), map_location, weights_only))
        return {"model": {"weight": FakeTensor()}}

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(Tensor=FakeTensor, load=load))
    plugin = find_plugin("sam2")
    assert plugin is not None
    weights = plugin.load_weights(str(package.parent), ModelConfig(model_type="sam2"))
    assert isinstance(weights["_sam2_checkpoint"], Checkpoint)
    assert calls == [(b"authenticated-checkpoint", "cpu", True)]

    checkpoint_path.write_bytes(b"changed")
    with pytest.raises(checkpoint_mapper.CheckpointError, match="SHA-256 mismatch"):
        checkpoint_mapper.load_checkpoint(checkpoint_path)


def test_family_maps_the_six_python_plans(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = find_plugin("sam2")
    assert plugin is not None
    module = importlib.import_module("tensorrt_model_connect.families.sam2.plugin")
    sections = []

    def serialize(_checkpoint, _populate, *, section, **_kwargs):
        sections.append(section)
        return section.encode()

    monkeypatch.setattr(module, "_serialize", serialize)
    weights = {"_sam2_checkpoint": Checkpoint({})}
    assert plugin.build_engine(ModelConfig(model_type="sam2"), weights, 1) == b"engine_plan"
    assert list(plugin.build_extra_engines(ModelConfig(model_type="sam2"), weights, 1)) == [
        "sam2_prompt_engine_plan",
        "sam2_recurrent_h1_engine_plan",
        "sam2_recurrent_h2_engine_plan",
        "sam2_recurrent_h3_engine_plan",
        "sam2_recurrent_h4_engine_plan",
    ]
    assert sections[0] == "engine_plan" and len(sections) == 6
    assert np.asarray(reciprocal_sqrtf(96), dtype=np.float32).view(np.uint32).item() == 0x3DD105EB


def test_public_core_synthesizes_only_the_fixed_bbox_overlay() -> None:
    checkpoint = PublicCoreCheckpoint({})
    assert checkpoint.tensor("image_encoder.bbox_head.rtm_cls.0.bias", (2,)).tolist() == [
        -8.0,
        -30.0,
    ]
    assert checkpoint.tensor("image_encoder.bbox_head.rtm_reg.0.bias", (4,)).tolist() == [
        -15.5,
        -15.5,
        111.5,
        111.5,
    ]
    classifier = checkpoint.tensor("image_encoder.bbox_head.rtm_cls.0.weight", (2, 256, 1, 1))
    assert classifier[1, 0, 0, 0] == 64.0 and np.count_nonzero(classifier) == 1
    spatial = checkpoint.tensor(
        "image_encoder.bbox_head.cls_convs.0.1.conv.weight", (256, 256, 3, 3)
    )
    assert spatial[0, 0, 0, 1] == spatial[0, 0, 1, 0] == -1.0
    assert np.count_nonzero(spatial) == 2
    constant = 1.0 / (1.0 + math.exp(-1.0))
    edge = 1.0 - constant
    assert 64.0 * constant - 30.0 > 8.0
    assert 64.0 * (edge / (1.0 + math.exp(-edge))) - 30.0 < -8.0
    assert np.all(
        checkpoint.tensor("image_encoder.bbox_head.cls_convs.0.0.bn.running_var", (256,)) == 1.0
    )
    with pytest.raises(CheckpointError, match="checkpoint tensor not found"):
        checkpoint.tensor("image_encoder.trunk.unexpected", (1,))
    with pytest.raises(CheckpointError, match="checkpoint tensor not found"):
        checkpoint.tensor("image_encoder.bbox_head.unexpected", (1,))
    with pytest.raises(CheckpointError, match="requested shape"):
        checkpoint.tensor("image_encoder.bbox_head.rtm_cls.0.bias", (3,))


def test_public_snapshot_links_resolve_to_authenticated_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshot"
    blobs.mkdir()
    snapshot.mkdir()
    config_blob = blobs / "config"
    checkpoint_blob = blobs / "checkpoint"
    config_blob.write_bytes(b"public-config")
    checkpoint_blob.write_bytes(b"public-checkpoint")
    (snapshot / PUBLIC_CONFIG_RELATIVE_PATH).symlink_to(config_blob)
    (snapshot / PUBLIC_CHECKPOINT_RELATIVE_PATH).symlink_to(checkpoint_blob)
    from tensorrt_model_connect.families.sam2 import model_config

    monkeypatch.setattr(
        model_config,
        "PUBLIC_CONFIG_SHA256",
        hashlib.sha256(config_blob.read_bytes()).hexdigest(),
    )
    assert resolve_public_package_root(snapshot) == snapshot
    assert config_from_dir(snapshot)["model_type"] == "sam2"


def test_public_snapshot_authentication_constants_are_pinned() -> None:
    assert PUBLIC_CONFIG_SHA256 == (
        "632e5cd0104f5ab6cd4f9d2dfd80a8e7240e481ad7960a13cad2ae3504b88dbd"
    )
    assert PUBLIC_CHECKPOINT_SHA256 == (
        "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
    )


def test_family_claims_only_the_pinned_package(package: Path, tmp_path: Path) -> None:
    assert resolve_package_root(package.parent) == package
    config = config_from_dir(package.parent)
    assert config is not None and config["model_type"] == "sam2"
    assert prefer_native_default(ModelConfig.from_dir(package.parent))

    link = tmp_path / "linked"
    link.symlink_to(package, target_is_directory=True)
    assert config_from_dir(link) is None
    (package / CONFIG_RELATIVE_PATH).write_bytes(b"changed")
    with pytest.raises(ValueError, match="unsupported SAM2 config"):
        config_from_dir(package)


@pytest.mark.parametrize(
    "raw",
    (
        {"_rtx_build_requested": True},
        {"_quantized_build_requested": True},
        {"_family_build_options": {"sam2": {"workspace_bytes": 1}}},
    ),
)
def test_family_rejects_unsupported_build_modes(package: Path, raw: dict) -> None:
    plugin = find_plugin("sam2")
    assert plugin is not None
    with pytest.raises(ValueError, match="SAM2 does not support"):
        plugin.load_weights(str(package), ModelConfig(model_type="sam2", raw=raw))


def test_builder_implementation_is_family_owned_python() -> None:
    repository = Path(__file__).resolve().parents[5]
    family = repository / "python/tensorrt_model_connect/families/sam2"
    for name in ("checkpoint_mapper.py", "graph_ops.py", "image_builder.py", "tracker_builder.py"):
        assert (family / name).is_file()
    assert not (repository / "tools/sam2_native_builder").exists()
    assert not (repository / "src/runtime/models/sam2/cmake/native.cmake").exists()
