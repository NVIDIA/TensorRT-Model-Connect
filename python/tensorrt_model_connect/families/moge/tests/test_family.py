# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the native MoGe-2 ViT-L family."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from tensorrt_model_connect import engine_builder
from tensorrt_model_connect.families.moge import model as model_module


FAMILY_ROOT = Path(__file__).resolve().parents[1]
family_plugin = importlib.import_module("tensorrt_model_connect.families.moge.plugin")


def test_manifest_downloads_only_the_requested_checkpoint() -> None:
    manifest = tomllib.loads((FAMILY_ROOT / "MODEL.toml").read_text(encoding="utf-8"))

    assert manifest["id"] == "moge"
    assert manifest["module"] == "plugin"
    assert manifest["config_adapter"] == "plugin.py|config_from_dir"
    assert "model_dir_adapter" not in manifest
    assert manifest["hf_allow_patterns"] == ["model.pt"]
    assert manifest["hf_required_files"] == ["Ruicheng/moge-2-vitl|model.pt"]
    assert manifest["default_execution_profiles"] == ["reference|moge_reference"]


def test_config_adapter_claims_one_flat_checkpoint(tmp_path: Path) -> None:
    assert family_plugin.config_from_dir(tmp_path) is None
    (tmp_path / "model.pt").write_bytes(b"checkpoint")

    config = family_plugin.config_from_dir(tmp_path)

    assert config is not None
    assert config["model_type"] == "moge"
    assert config["architectures"] == ["MoGeModelV2"]
    assert config["runtime_strategy"] == "moge_monocular_geometry"
    assert config["max_position_embeddings"] == 1841
    assert config["requires_tokenizer"] is False
    assert family_plugin.plugin.matches("MoGeModelV2")
    assert not family_plugin.plugin.matches("MoGeModel")


def test_plugin_keeps_model_state_local_and_rejects_unimplemented_modes(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.pt").write_bytes(b"checkpoint")
    config = SimpleNamespace(raw={})

    assert family_plugin.plugin.load_weights(str(tmp_path), config) == {
        "model_dir": str(tmp_path.resolve())
    }
    assert config.raw == {}
    assert not engine_builder._call_supports_kwarg(
        family_plugin.plugin.build_engine, "parallel_config"
    )
    with pytest.raises(ValueError, match="does not support quantized"):
        family_plugin.plugin.build_engine(
            config,
            {"model_dir": str(tmp_path)},
            1,
            quant_ctx=object(),
        )


def test_checkpoint_loading_is_safe_and_memory_mapped() -> None:
    tree = ast.parse((FAMILY_ROOT / "model.py").read_text(encoding="utf-8"))
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    ]
    assert any(
        {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in call.keywords
            if keyword.arg in {"map_location", "weights_only", "mmap"}
        }
        == {"map_location": "cpu", "weights_only": True, "mmap": True}
        for call in loads
    )


def test_production_builder_is_fixed_and_tensor_rt_native_only() -> None:
    source = (FAMILY_ROOT / "model.py").read_text(encoding="utf-8")
    lowered = source.lower()

    for forbidden in (
        "torch.onnx",
        "torch.nn",
        "onnxparser",
        "onnx_path",
        "torch_tensorrt",
        "triton",
        "add_plugin",
        "get_plugin_registry",
        "trtmc_moge_",
    ):
        assert forbidden not in lowered
    for required in (
        "create_network",
        "build_serialized_network",
        "add_attention",
        "add_normalization_v2",
        "add_convolution_nd",
        "add_deconvolution_nd",
        "InterpolationMode.CUBIC",
        "ResizeCoordinateTransformation.HALF_PIXEL",
        "SampleMode.CLAMP",
        "GELU_ERF",
        "first_transpose=(0, 3, 1, 2)",
        "_NUM_TOKENS = 1800",
    ):
        assert required in source
    for output in ("points", "mask", "metric_scale"):
        assert f'("{output}",' in source


def test_build_rejects_unqualified_precision_and_wrong_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"wrong checkpoint")

    with pytest.raises(ValueError, match="supports precision='fp32' only"):
        model_module.build_moge_engine(str(tmp_path), precision="fp16")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        model_module.build_moge_engine(str(tmp_path), precision="fp32")
