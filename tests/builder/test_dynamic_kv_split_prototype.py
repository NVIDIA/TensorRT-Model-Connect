# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free orchestration coverage for dynamic-KV split decoder bundles."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tensorrt_model_connect import engine_builder
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.engine_builder import build_bundle

pytestmark = pytest.mark.dynamic_memory


class _DynamicSplitPlugin:
    name = "prototype_decoder"
    runtime_strategy = "prototype_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}
    supports_split_decoder_roles = True

    def load_weights(self, model_dir, config, *, precision="fp32"):
        return {}

    def build_engine(
        self,
        config,
        weights,
        max_cache_length,
        *,
        precision="fp32",
        verbose=False,
    ):
        del weights, max_cache_length, precision, verbose
        return config.raw["_decoder_engine_role"].encode()


@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
def test_opted_in_dynamic_kv_build_preserves_split_engines(
    tmp_path, model_type
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "architectures": ["PrototypeDecoderForCausalLM"],
                "vocab_size": 100,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 128,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }
        ),
        encoding="utf-8",
    )
    output_path = str(tmp_path / "output.trtfb")

    with patch(
        "tensorrt_model_connect.engine_builder.find_plugin",
        return_value=_DynamicSplitPlugin(),
    ):
        with patch(
            "tensorrt_model_connect.engine_builder._get_trt_version",
            return_value="11.2",
        ):
            with patch(
                "tensorrt_model_connect.engine_builder._get_gpu_name",
                return_value="",
            ):
                with patch(
                    "tensorrt_model_connect.engine_builder._ensure_tokenizer_json"
                ):
                    with patch(
                        "tensorrt_model_connect.engine_builder.write_bundle"
                    ) as mock_write:
                        build_bundle(
                            str(model_dir),
                            output_path,
                            dynamic_kv_cache=True,
                            dynamic_kv_profile_rows_override=[32, 64],
                        )

    sections = mock_write.call_args[0][2]
    assert [section.name for section in sections[:2]] == [
        "engine_plan",
        "prefill_engine_plan",
    ]
    assert sections[0].data == b"decode"
    assert sections[1].data == b"prefill"
    runtime_config = json.loads(
        next(section.data for section in sections if section.name == "config.json")
    )
    assert runtime_config["decoder_engine_layout"] == "split"
    assert runtime_config["dynamic_kv_cache"] is True
    assert runtime_config["dynamic_kv_profile_rows"] == [32, 64]


def test_dynamic_kv_split_requires_family_opt_in():
    config = ModelConfig(model_type="prototype_decoder")
    plugin = _DynamicSplitPlugin()

    assert not engine_builder._can_build_split_decoder_engines(
        plugin,
        config,
        plugin.runtime_strategy,
        dynamic_kv_cache=True,
        triattention_enabled=False,
    )
