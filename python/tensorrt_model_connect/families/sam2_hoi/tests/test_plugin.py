# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.families.sam2_hoi import plugin, source_export
from tensorrt_model_connect.families.sam2_hoi.architecture import ARCHITECTURE


plugin_module = importlib.import_module("tensorrt_model_connect.families.sam2_hoi.plugin")


def _raw_config() -> dict[str, object]:
    return {
        "_model_dir": "/reviewed/hoi",
        "image_size": 1024,
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


def test_plugin_exports_distinct_video_tracking_family() -> None:
    assert plugin.name == "sam2_hoi"
    assert plugin.runtime_strategy == "sam2_hoi_video_tracking"
    assert not plugin.requires_tokenizer


def test_python_bundle_contract_matches_native_runtime_sections() -> None:
    repository_root = Path(__file__).resolve().parents[5]
    runtime_source = (repository_root / "src/runtime/models/sam2_hoi/plugin.cpp").read_text(
        encoding="utf-8"
    )
    runtime_plans = tuple(re.findall(r'load_legacy_module\(context, "([^"]+)"', runtime_source))
    assert runtime_plans == source_export.ENGINE_PLAN_SECTIONS
    assert 'header_has_section(context.bundle.info, "sam2_hoi_pafpn_manifest.json")' in (
        runtime_source
    )
    assert 'name << "sam2_hoi_pafpn_plan_"' in runtime_source
    assert "sam2_hoi::kPafpnPlanCount" in runtime_source
    assert (
        f'find_section(context.bundle, "{source_export.NATIVE_PLUGIN_SECTION}")' in runtime_source
    )


def test_build_wires_all_six_engine_sections(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = SimpleNamespace(raw=_raw_config())
    calls: list[tuple[str, object, str]] = []
    build_order: list[str] = []
    weights = {"reviewed": "weights"}
    native_plugin = tmp_path / "libsam2-hoi-native.so"
    native_plugin.write_bytes(b"native-plugin")
    monkeypatch.setattr(
        plugin_module,
        "ensure_native_plugin_loaded",
        lambda *, verbose: build_order.append("load_native_plugin") or native_plugin,
    )

    monkeypatch.setattr(
        plugin_module,
        "build_image_feature_engine",
        lambda model_weights, *, precision, verbose: (
            build_order.append("build_image")
            or calls.append(("image", model_weights, precision))
            or b"image"
        ),
    )
    monkeypatch.setattr(
        plugin_module,
        "build_hoi_detector_engine",
        lambda model_weights, *, precision, verbose: (
            calls.append(("detector", model_weights, precision)) or b"detector"
        ),
    )
    monkeypatch.setattr(
        plugin_module,
        "build_interaction_engine",
        lambda model_weights, *, precision, verbose: (
            calls.append(("interaction", model_weights, precision)) or b"interaction"
        ),
    )
    monkeypatch.setattr(
        plugin_module,
        "build_tracker_engines",
        lambda model_weights, *, precision, verbose: {
            "sam2_hoi_prompt_tracker_engine_plan": b"prompt",
            "sam2_hoi_recurrent_tracker_engine_plan": b"recurrent",
            "sam2_hoi_memory_encoder_engine_plan": b"memory",
        },
    )

    assert plugin.build_engine(config, weights, 32, precision="bf16") == b"image"
    assert plugin.build_engine(config, weights, 32, precision="fp32") == b"image"
    extras = plugin.build_extra_engines(config, weights, 32, precision="bf16")
    assert set(extras) == {
        "sam2_hoi_native_plugin_so",
        "sam2_hoi_detector_engine_plan",
        "sam2_hoi_interaction_engine_plan",
        "sam2_hoi_prompt_tracker_engine_plan",
        "sam2_hoi_recurrent_tracker_engine_plan",
        "sam2_hoi_memory_encoder_engine_plan",
    }
    assert extras["sam2_hoi_native_plugin_so"] == b"native-plugin"
    assert build_order[:2] == ["load_native_plugin", "build_image"]
    assert build_order.count("load_native_plugin") == 3
    assert calls == [
        ("image", weights, "bf16"),
        ("image", weights, "fp32"),
        ("detector", weights, "bf16"),
        ("interaction", weights, "bf16"),
    ]


def test_bf16_is_the_accuracy_qualified_default() -> None:
    assert plugin.default_build_precision == "bf16"


def test_bundle_overrides_publish_exact_runtime_contract() -> None:
    config = SimpleNamespace(raw=_raw_config())
    overrides = plugin.get_bundle_config_overrides(config)
    assert overrides["runtime_strategy"] == "sam2_hoi_video_tracking"
    assert overrides["sam2_hoi_fixture_frames"] == 5
    assert overrides["sam2_hoi_hoi_num_queries"] == 1500
    assert overrides["image_interpolation"] == "bicubic"
    assert overrides["sam2_hoi_mask_logit_threshold"] == 0.01
    assert ARCHITECTURE.bundle_config().items() <= overrides.items()
