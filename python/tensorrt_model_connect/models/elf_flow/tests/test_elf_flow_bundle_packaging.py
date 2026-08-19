# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ELF Flow bundle packaging contracts owned by the ELF Flow family."""

from __future__ import annotations

import json
from unittest.mock import patch

from tensorrt_model_connect.models.elf_flow import model as elf_model


def test_yaml_only_elf_model_build_synthesizes_config_json_section(tmp_path, monkeypatch):
    """GitHub ELF YAML-only directories still get runtime config.json."""
    (tmp_path / "train_owt_ELF-B.yml").write_text(
        "\n".join(
            [
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
            ]
        ),
        encoding="utf-8",
    )
    output_path = str(tmp_path / "output.bundle")

    monkeypatch.setattr(elf_model, "load_weights", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        elf_model,
        "_build_local_engine",
        lambda *_args, **_kwargs: (b"PLAN", "single"),
    )
    monkeypatch.setattr(elf_model, "build_extra_engines", lambda *_args, **_kwargs: {})

    with (
        patch(
            "tensorrt_model_connect.bundle_writer.tensorrt_version",
            return_value="10.0",
        ),
        patch("tensorrt_model_connect.bundle_writer.tensorrt_abi", return_value=""),
        patch("tensorrt_model_connect.bundle_writer.gpu_name", return_value=""),
        patch(
            "tensorrt_model_connect.tokenizer_conversion.prepare_tokenizer_special_frame",
            return_value=None,
        ),
        patch(
            "tensorrt_model_connect.tvm_ffi.graph_build.kernel_slots_section",
            return_value=None,
        ),
        patch("tensorrt_model_connect.bundle_writer.write_bundle") as mock_write,
    ):
        elf_model.build(str(tmp_path), output_path)

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
