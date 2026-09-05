# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from families.sam3.model import _RUNTIME_FIELDS, _load_sam3_processor_config, _resolve_sam3_config


def test_resolved_config_contains_the_native_image_contract() -> None:
    config = _resolve_sam3_config(
        {
            "detector_config": {
                "text_config": {"bos_token_id": 101, "eos_token_id": 102},
                "vision_config": {"backbone_config": {"image_size": 1008}},
            }
        }
    )

    assert not set(_RUNTIME_FIELDS).difference(config)
    runtime = {key: config[key] for key in _RUNTIME_FIELDS}
    assert runtime["tokenizer_add_special_tokens"] is False
    assert runtime["tokenizer_prefix_ids"] == [101]
    assert runtime["tokenizer_suffix_ids"] == [102]
    assert config["image_size"] == config["vision_image_size"] == 1008
    assert config["score_threshold"] == 0.5
    assert config["mask_threshold"] == 0.5
    assert config["image_mean"] == [0.5, 0.5, 0.5]
    assert config["image_std"] == [0.5, 0.5, 0.5]


def test_processor_values_update_both_build_and_runtime_names(tmp_path) -> None:
    (tmp_path / "processor_config.json").write_text(
        json.dumps(
            {
                "image_processor": {
                    "size": {"height": 896, "width": 896},
                    "image_mean": [0.1, 0.2, 0.3],
                    "image_std": [0.4, 0.5, 0.6],
                }
            }
        ),
        encoding="utf-8",
    )

    config = _load_sam3_processor_config(str(tmp_path))

    assert config["vision_image_size"] == config["image_size"] == 896
    assert config["processor_image_mean"] == config["image_mean"] == [0.1, 0.2, 0.3]
    assert config["processor_image_std"] == config["image_std"] == [0.4, 0.5, 0.6]
