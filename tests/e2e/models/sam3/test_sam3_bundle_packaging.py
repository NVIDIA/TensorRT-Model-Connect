# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3 bundle packaging contracts owned by the SAM3 family."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from unittest.mock import patch

from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC
from tensorrt_model_connect.families.sam3 import model as sam3_model


_EXPECTED_SAM3_SECTIONS = {
    "engine_plan",
    "vision_engine_plan",
    "sam3_core_engine_plan",
    "sam3_tracker_init_engine_plan",
    "sam3_tracker_step_engine_plan",
    "sam3_tracker_step_batch2_engine_plan",
    "sam3_tracker_memory_engine_plan",
    "sam3_tracker_memory_batch2_engine_plan",
    "sam3_tracker_hard_memory_engine_plan",
    "sam3_tracker_hard_memory_batch2_engine_plan",
    "sam3_hard_mask_resize_engine_plan",
    "sam3_hard_mask_resize_batch2_engine_plan",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "processor_config.json",
}


def _make_sam3_model_dir(tmp_path: Path) -> Path:
    config = {
        "model_type": "sam3",
        "architectures": ["Sam3ForCausalLM"],
        "vocab_size": 100,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "detector_config": {
            "text_config": {
                "bos_token_id": 101,
                "eos_token_id": 102,
                "pad_token_id": 0,
            }
        },
        "tracker_config": {
            "model_type": "sam3_tracker_video",
            "image_size": 1008,
            "num_maskmem": 7,
            "max_cond_frame_num": 4,
            "max_object_pointers_in_encoder": 16,
        },
    }
    assets = {
        "config.json": json.dumps(config),
        "tokenizer.json": json.dumps(
            {"version": "1.0", "model": {"type": "BPE", "vocab": {}, "merges": []}}
        ),
        "tokenizer_config.json": json.dumps({"add_bos_token": False, "add_eos_token": False}),
        "vocab.json": "{}",
        "merges.txt": "#version: 0.2\n",
        "special_tokens_map.json": json.dumps(
            {"bos_token": "<|startoftext|>", "eos_token": "<|endoftext|>"}
        ),
        "processor_config.json": json.dumps({"processor_class": "Sam3Processor"}),
    }
    for name, payload in assets.items():
        (tmp_path / name).write_text(payload, encoding="utf-8")
    return tmp_path


def _read_serialized_bundle(path: Path) -> tuple[dict, dict[str, bytes]]:
    payload = path.read_bytes()
    assert payload[: len(BUNDLE_MAGIC)] == BUNDLE_MAGIC
    header_size = struct.unpack_from("<Q", payload, len(BUNDLE_MAGIC))[0]
    header_start = len(BUNDLE_MAGIC) + struct.calcsize("<Q")
    header_end = header_start + header_size
    header = json.loads(payload[header_start:header_end])
    sections = {
        name: payload[
            header_end + int(metadata["offset"]) : header_end
            + int(metadata["offset"])
            + int(metadata["size"])
        ]
        for name, metadata in header["sections"].items()
    }
    return header, sections


def test_sam3_prompted_segmentation_packages_all_plans_and_tokenizer(tmp_path, monkeypatch):
    """SAM3 prompted segmentation needs tokenizer provisioning and all TRT plans."""
    model_dir = _make_sam3_model_dir(tmp_path)
    output_path = tmp_path / "sam3.bundle"

    monkeypatch.setattr(sam3_model, "load_weights", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        sam3_model,
        "_build_local_engine",
        lambda *_args, **_kwargs: (b"SAM3_TEXT_PLAN", "single"),
    )
    monkeypatch.setattr(
        sam3_model,
        "build_vision_engine",
        lambda *_args, **_kwargs: b"SAM3_VISION_PLAN",
    )
    monkeypatch.setattr(
        sam3_model,
        "build_extra_engines",
        lambda *_args, **_kwargs: {
            "sam3_core_engine_plan": b"SAM3_CORE_PLAN",
            "sam3_tracker_init_engine_plan": b"SAM3_TRACKER_INIT_PLAN",
            "sam3_tracker_step_engine_plan": b"SAM3_TRACKER_STEP_PLAN",
            "sam3_tracker_step_batch2_engine_plan": b"SAM3_TRACKER_STEP_BATCH2_PLAN",
            "sam3_tracker_memory_engine_plan": b"SAM3_TRACKER_MEMORY_PLAN",
            "sam3_tracker_memory_batch2_engine_plan": b"SAM3_TRACKER_MEMORY_BATCH2_PLAN",
            "sam3_tracker_hard_memory_engine_plan": b"SAM3_TRACKER_HARD_MEMORY_PLAN",
            "sam3_tracker_hard_memory_batch2_engine_plan": b"SAM3_TRACKER_HARD_MEMORY_BATCH2_PLAN",
            "sam3_hard_mask_resize_engine_plan": b"SAM3_HARD_MASK_RESIZE_PLAN",
            "sam3_hard_mask_resize_batch2_engine_plan": b"SAM3_HARD_MASK_RESIZE_BATCH2_PLAN",
        },
    )
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
    ):
        sam3_model.build(str(model_dir), str(output_path))

    header, section_map = _read_serialized_bundle(output_path)
    assert set(section_map) == _EXPECTED_SAM3_SECTIONS
    assert len(section_map) == 19
    for section_name in section_map:
        lowered = section_name.lower()
        assert "aoti" not in lowered
        assert "ffi" not in lowered
        assert not lowered.endswith((".pt2", ".so"))

    assert header["runtime_strategy"] == "sam3_prompted_segmentation"
    assert section_map["engine_plan"] == b"SAM3_TEXT_PLAN"
    assert section_map["vision_engine_plan"] == b"SAM3_VISION_PLAN"
    assert section_map["sam3_core_engine_plan"] == b"SAM3_CORE_PLAN"
    assert section_map["sam3_tracker_init_engine_plan"] == b"SAM3_TRACKER_INIT_PLAN"
    assert section_map["sam3_tracker_step_engine_plan"] == b"SAM3_TRACKER_STEP_PLAN"
    assert section_map["sam3_tracker_step_batch2_engine_plan"] == b"SAM3_TRACKER_STEP_BATCH2_PLAN"
    assert section_map["sam3_tracker_memory_engine_plan"] == b"SAM3_TRACKER_MEMORY_PLAN"
    assert (
        section_map["sam3_tracker_memory_batch2_engine_plan"] == b"SAM3_TRACKER_MEMORY_BATCH2_PLAN"
    )
    assert section_map["sam3_tracker_hard_memory_engine_plan"] == b"SAM3_TRACKER_HARD_MEMORY_PLAN"
    assert (
        section_map["sam3_tracker_hard_memory_batch2_engine_plan"]
        == b"SAM3_TRACKER_HARD_MEMORY_BATCH2_PLAN"
    )
    assert section_map["sam3_hard_mask_resize_engine_plan"] == b"SAM3_HARD_MASK_RESIZE_PLAN"
    assert (
        section_map["sam3_hard_mask_resize_batch2_engine_plan"]
        == b"SAM3_HARD_MASK_RESIZE_BATCH2_PLAN"
    )

    cfg = json.loads(section_map["config.json"].decode("utf-8"))
    assert cfg["prompted_segmentation_variant"] == "sam3_text_prompt_pcs"
    assert cfg["has_vision_engine"] is True
    assert cfg["sam3_num_queries"] == 200
    assert cfg["tokenizer_add_special_tokens"] == 1
    assert cfg["tokenizer_special_prefix_ids"] == [101]
    assert cfg["tokenizer_special_suffix_ids"] == [102]
    assert cfg["sam3_text_bos_token_id"] == 101
    assert cfg["sam3_text_eos_token_id"] == 102
