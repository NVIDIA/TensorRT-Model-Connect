# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for effective EOS defaults embedded in engine bundles."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_model_connect.engine_builder import build_bundle


@pytest.mark.parametrize(
    ("model_eos", "generation_eos"),
    [
        (5, [5, 7]),
        ([5, 7], 3),
    ],
)
def test_generation_config_eos_overrides_model_config_in_bundle(
    tmp_path,
    model_eos,
    generation_eos,
):
    """generation_config.json supplies the effective scalar/list EOS defaults."""
    model_config = {
        "model_type": "example_decoder",
        "architectures": ["ExampleDecoderForCausalLM"],
        "vocab_size": 100,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "eos_token_id": model_eos,
    }
    (tmp_path / "config.json").write_text(json.dumps(model_config))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": generation_eos}))

    mock_plugin = MagicMock()
    mock_plugin.name = "example_family"
    mock_plugin.runtime_strategy = ""
    mock_plugin.load_weights.return_value = {}
    mock_plugin.build_engine.return_value = b"PLAN"

    del mock_plugin.build_vision_engine
    del mock_plugin.build_extra_engines
    del mock_plugin.embed_input
    del mock_plugin.get_vl_config
    del mock_plugin.get_segmentation_config
    del mock_plugin.get_audio_config
    del mock_plugin.get_bundle_config_overrides
    del mock_plugin.tokenizer_json_bundle_override

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
                    with patch("tensorrt_model_connect.engine_builder.write_bundle") as mock_write:
                        build_bundle(
                            str(tmp_path),
                            str(tmp_path / "output.trtfb"),
                        )

    sections = mock_write.call_args[0][2]
    config_section = next(s for s in sections if s.name == "config.json")
    bundled_config = json.loads(config_section.data)
    assert bundled_config["eos_token_id"] == generation_eos
