# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ELF Flow bundle packaging contracts owned by the ELF Flow family."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from tensorrt_model_connect.engine_builder import (
    _tokenizer_json_bundle_override_from_plugin,
    build_bundle,
)
from tensorrt_model_connect.families.elf_flow.plugin import ELFPlugin


def _elf_plugin_mock() -> MagicMock:
    """Return an ELF-shaped mock without fabricating optional plugin hooks."""
    mock_plugin = MagicMock(spec=ELFPlugin)
    mock_plugin.name = "elf"
    mock_plugin.runtime_strategy = "elf_flow"
    mock_plugin.load_weights.return_value = {}
    mock_plugin.build_engine.return_value = b"PLAN"
    mock_plugin.get_bundle_config_overrides.return_value = {
        "runtime_strategy": "elf_flow",
        "model_type": "elf",
        "elf_max_length": 1024,
        "elf_max_input_length": 0,
        "elf_text_encoder_dim": 512,
        "elf_input_dim": 1024,
        "elf_denoiser_noise_scale": 2.0,
    }
    del mock_plugin.build_extra_engines
    return mock_plugin


def test_elf_mock_does_not_fabricate_optional_tokenizer_override(tmp_path):
    """ELF has no tokenizer override hook, so the builder takes its no-hook path."""
    mock_plugin = _elf_plugin_mock()

    assert not hasattr(mock_plugin, "tokenizer_json_bundle_override")
    assert _tokenizer_json_bundle_override_from_plugin(mock_plugin, tmp_path) is None


def test_yaml_only_elf_synthesizes_config_json_section(tmp_path):
    """GitHub ELF YAML-only directories still get runtime config.json."""
    (tmp_path / "train_owt_ELF-B.yml").write_text(
        "\n".join([
            "model: ELF-B",
            "max_length: 1024",
            "encoder_model_name: t5-small",
            "num_time_tokens: 4",
            "num_self_cond_cfg_tokens: 4",
            "num_model_mode_tokens: 4",
            "denoiser_p_mean: -1.5",
            "denoiser_p_std: 0.8",
            "denoiser_noise_scale: 2.0",
            "self_cond_prob: 0.5",
        ]),
        encoding="utf-8",
    )
    output_path = str(tmp_path / "output.trtfb")

    mock_plugin = _elf_plugin_mock()

    with patch(
        "tensorrt_model_connect.engine_builder.find_plugin",
        return_value=mock_plugin,
    ):
        with patch(
            "tensorrt_model_connect.engine_builder._get_trt_version",
            return_value="10.0",
        ):
            with patch(
                "tensorrt_model_connect.engine_builder._get_gpu_name",
                return_value="",
            ):
                with patch("tensorrt_model_connect.engine_builder._ensure_tokenizer_json"):
                    with patch(
                        "tensorrt_model_connect.engine_builder.write_bundle"
                    ) as mock_write:
                        build_bundle(str(tmp_path), output_path)

    sections = mock_write.call_args[0][2]
    section_map = {section.name: section.data for section in sections}
    cfg = json.loads(section_map["config.json"].decode("utf-8"))
    assert cfg["runtime_strategy"] == "elf_flow"
    assert cfg["model_type"] == "elf"
    assert cfg["model"] == "ELF-B"
    assert cfg["elf_max_length"] == 1024
    assert cfg["elf_max_input_length"] == 0
    assert cfg["elf_text_encoder_dim"] == 512
    assert cfg["elf_input_dim"] == 1024
    assert cfg["elf_denoiser_noise_scale"] == 2.0
