# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generic builder packages all Python-built SAM2 plans."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from tensorrt_model_connect import engine_builder
from tensorrt_model_connect.models.sam2 import model as family_model
from tensorrt_model_connect.models.sam2 import model_config


def test_generic_builder_writes_the_sam2_runtime_sections(tmp_path: Path, monkeypatch) -> None:
    supplied = tmp_path / "models"
    package = supplied / model_config.PACKAGE_DIRNAME
    config = package / model_config.CONFIG_RELATIVE_PATH
    checkpoint = package / model_config.CHECKPOINT_RELATIVE_PATH
    config.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    config.write_bytes(b"config")
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        model_config, "REFERENCE_CONFIG_SHA256", hashlib.sha256(config.read_bytes()).hexdigest()
    )

    monkeypatch.setattr(
        family_model, "load_weights", lambda *_args, **_kwargs: {"ok": True}
    )
    monkeypatch.setattr(
        family_model, "build_engine", lambda *_args, **_kwargs: b"image"
    )
    monkeypatch.setattr(
        family_model,
        "build_extra_engines",
        lambda *_args, **_kwargs: {
            "sam2_prompt_engine_plan": b"prompt",
            **{f"sam2_recurrent_h{i}_engine_plan": bytes([i]) for i in range(1, 5)},
        },
    )
    monkeypatch.setattr(engine_builder, "_setup_trt_import", lambda _rtx: None)

    output = tmp_path / "sam2.bundle"
    engine_builder.build(str(supplied), str(output))
    with output.open("rb") as stream:
        assert stream.read(8) == b"BUNDLE\x01\x00"
        header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
        config_metadata = header["sections"]["config.json"]
        stream.seek(stream.tell() + config_metadata["offset"])
        runtime_config = json.loads(stream.read(config_metadata["size"]))

    assert list(header["sections"]) == [
        "engine_plan",
        "sam2_prompt_engine_plan",
        "sam2_recurrent_h1_engine_plan",
        "sam2_recurrent_h2_engine_plan",
        "sam2_recurrent_h3_engine_plan",
        "sam2_recurrent_h4_engine_plan",
        "config.json",
    ]
    assert (header["family"], header["precision"]) == ("sam2", "mixed_bf16_fp32")
    assert runtime_config["runtime_strategy"] == "sam2_bbox_video_tracking"
