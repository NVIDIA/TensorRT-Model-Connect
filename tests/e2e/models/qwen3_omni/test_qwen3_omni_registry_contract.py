# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned registry contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("tensorrt", reason="registry contract tests import plugin modules")

from tensorrt_model_connect.families import find_plugin
from tensorrt_model_connect.families.qwen3_omni.config import ModelConfig


def _plugin(model_type: str):
    plugin = find_plugin(model_type)
    assert plugin is not None
    return plugin


def test_vision_language_runtime_contract() -> None:
    plugin = _plugin("qwen3_omni")
    assert getattr(plugin, "runtime_strategy", None) == "qwen3_omni_multimodal"
    assert getattr(plugin, "embed_input", False) is True
    assert callable(getattr(plugin, "build_vision_engine", None))
    assert callable(getattr(plugin, "get_vl_config", None))

    assert callable(getattr(plugin, "build_extra_engines", None))


def test_native_kv_defaults_use_complete_official_context() -> None:
    plugin = _plugin("qwen3_omni")
    config = ModelConfig.create_tiny(
        "qwen3_omni", max_position_embeddings=65536)

    assert plugin.runtime_capabilities == {"decoder_kv"}
    assert plugin.supports_split_embed_input is True
    assert plugin.supports_split_decoder_roles(config) is True
    assert plugin.default_build_precision(config) == "bf16"
    assert plugin.default_max_cache_length(config) == 65536


def test_native_kv_builder_has_no_legacy_modules_or_concat_path() -> None:
    family_dir = (
        Path(__file__).resolve().parents[4]
        / "python/tensorrt_model_connect/families/qwen3_omni"
    )

    for module_name in (
        "default_decoder.py",
        "default_dual_profile_decoder.py",
        "standard_decoder_builder.py",
    ):
        assert not (family_dir / module_name).exists()

    graph_source = (family_dir / "graph_blocks.py").read_text(encoding="utf-8")
    assert "def add_attention_block(" not in graph_source
    assert "add_concatenation" not in graph_source
    assert "def add_native_kv_attention_block(" in graph_source
    assert "add_native_kv_cache_attention_from_rows" in graph_source


@pytest.mark.parametrize(
    "flag", ("_runtime_dynamic_kv_requested", "dynamic_kv_cache")
)
def test_legacy_dynamic_kv_build_flag_is_rejected_before_weight_loading(
    flag: str, tmp_path: Path
) -> None:
    plugin = _plugin("qwen3_omni")
    config = ModelConfig.create_tiny("qwen3_omni")
    config.raw[flag] = True

    with pytest.raises(ValueError, match="remove --dynamic-kv-cache"):
        plugin.load_weights(str(tmp_path), config)


def test_official_manifest_does_not_inject_build_time_kv_or_precision_flags() -> None:
    manifest_path = (
        Path(__file__).parent
        / "manifests"
        / "qwen3-omni-30b-a3b-instruct.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "max_cache_length" not in manifest
    assert "kv_cache_size_bytes" not in manifest
    assert "precision" not in manifest
