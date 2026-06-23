"""ELF Flow bundle packaging contracts owned by the ELF Flow family."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from tensorrt_model_connect.engine_builder import build_bundle


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

    mock_plugin = MagicMock()
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

    del mock_plugin.build_vision_engine
    del mock_plugin.build_extra_engines
    del mock_plugin.embed_input
    del mock_plugin.get_vl_config
    del mock_plugin.get_segmentation_config
    del mock_plugin.get_audio_config

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
