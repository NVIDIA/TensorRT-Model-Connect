# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

family = importlib.import_module("tensorrt_model_connect.families.foundationpose.plugin")


def test_bundle_contract_is_pinned_and_explicit():
    metadata = family.plugin.get_bundle_config_overrides(SimpleNamespace(raw={}))
    assert metadata["foundationpose_source_revision"] == family.SOURCE_REVISION
    assert metadata["foundationpose_ngc_version"] == "1.0.1_onnx"
    assert metadata["pose_refiner_max_batch"] == 42
    assert metadata["pose_max_hypotheses"] == 252
    assert metadata["pose_crop_layout"] == "NHWC"
    assert metadata["includes_segmentation"] is False
    assert metadata["includes_cad_rendering"] is False
    assert metadata["robotics_safety_validated"] is False


def test_config_requires_both_pinned_artifacts(monkeypatch, tmp_path: Path):
    (tmp_path / family.REFINER_FILE).write_bytes(b"refiner")
    assert family.config_from_dir(tmp_path) is None
    (tmp_path / family.SCORER_FILE).write_bytes(b"scorer")
    digests = {
        family.REFINER_FILE: family.REFINER_SHA256,
        family.SCORER_FILE: family.SCORER_SHA256,
    }
    monkeypatch.setattr(family, "_sha256", lambda path: digests[path.name])
    config = family.config_from_dir(tmp_path)
    assert config is not None
    assert config["model_type"] == "foundationpose"
    assert config["runtime_strategy"] == "foundationpose_pose_refinement"


def test_digest_mismatch_fails_closed(monkeypatch, tmp_path: Path):
    for name in (family.REFINER_FILE, family.SCORER_FILE):
        (tmp_path / name).write_bytes(b"wrong")
    monkeypatch.setattr(family, "_sha256", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        family.config_from_dir(tmp_path)


def test_only_fp32_and_unquantized_builds_are_admitted(tmp_path: Path):
    config = SimpleNamespace(raw={})
    with pytest.raises(ValueError, match="fp32"):
        family.plugin.load_weights(str(tmp_path), config, precision="fp16")
    with pytest.raises(ValueError, match="quantized"):
        family.plugin.build_engine(
            config,
            {"refiner": "unused"},
            1,
            quant_ctx=object(),
        )


def test_reference_profile_version_check_survives_python_optimization():
    path = Path(family.__file__).with_name("python_profile_verify.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "raise RuntimeError" in source
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
