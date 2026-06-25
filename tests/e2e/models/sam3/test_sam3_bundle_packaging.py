"""SAM3 bundle packaging contracts owned by the SAM3 family."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from tensorrt_model_connect.engine_builder import build_bundle


def _make_sam3_model_dir(tmp_path: Path) -> Path:
    config = {
        "model_type": "sam3",
        "architectures": ["Sam3ForCausalLM"],
        "vocab_size": 100,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def test_sam3_prompted_segmentation_packages_all_plans_and_tokenizer(tmp_path):
    """SAM3 prompted segmentation needs tokenizer provisioning and all TRT plans."""
    model_dir = _make_sam3_model_dir(tmp_path)
    output_path = str(tmp_path / "sam3.trtfb")

    class _Sam3Plugin:
        name = "sam3"
        runtime_strategy = "sam3_prompted_segmentation"
        requires_tokenizer = True

        def load_weights(self, model_dir, config):
            return {}

        def build_engine(self, config, weights, max_cache_length, *, verbose=False):
            return b"SAM3_TEXT_PLAN"

        def build_vision_engine(self, model_dir, config, weights, *, verbose=False):
            return b"SAM3_VISION_PLAN"

        def build_extra_engines(self, config, weights, max_cache_length, *, verbose=False):
            return {"sam3_core_engine_plan": b"SAM3_CORE_PLAN"}

        def get_segmentation_config(self, config):
            return {
                "prompted_segmentation_variant": "sam3_text_prompt_pcs",
                "sam3_text_max_position_embeddings": 32,
                "sam3_num_queries": 200,
            }

        def get_bundle_config_overrides(self, config):
            return {
                "model_type": "sam3",
                "runtime_strategy": "sam3_prompted_segmentation",
                "prompted_segmentation_variant": "sam3_text_prompt_pcs",
            }

    plugin = _Sam3Plugin()

    with patch(
        "tensorrt_model_connect.engine_builder.find_plugin",
        return_value=plugin,
    ):
        with patch(
            "tensorrt_model_connect.engine_builder._get_trt_version",
            return_value="10.0",
        ):
            with patch(
                "tensorrt_model_connect.engine_builder._get_gpu_name",
                return_value="",
            ):
                with patch(
                    "tensorrt_model_connect.engine_builder._ensure_tokenizer_json"
                ) as mock_ensure:
                    with patch(
                        "tensorrt_model_connect.engine_builder.write_bundle"
                    ) as mock_write:
                        build_bundle(str(model_dir), output_path)

    mock_ensure.assert_called_once_with(model_dir, plugin=plugin)
    sections = mock_write.call_args[0][2]
    section_map = {section.name: section.data for section in sections}
    assert section_map["engine_plan"] == b"SAM3_TEXT_PLAN"
    assert section_map["vision_engine_plan"] == b"SAM3_VISION_PLAN"
    assert section_map["sam3_core_engine_plan"] == b"SAM3_CORE_PLAN"

    cfg = json.loads(section_map["config.json"].decode("utf-8"))
    assert cfg["prompted_segmentation_variant"] == "sam3_text_prompt_pcs"
    assert cfg["has_vision_engine"] is True
    assert cfg["sam3_num_queries"] == 200
