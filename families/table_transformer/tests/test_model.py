# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder option rejection and structure-checkpoint contract."""

from dataclasses import replace
import pytest
from tensorrt_model_connect import BuildRequest
from families.table_transformer.model import build, _resolve_config


@pytest.mark.parametrize(
    "option,value",
    [
        ("task", "object_detection"),
        ("backend", "trt_rtx"),
        ("dynamic_kv_cache", True),
        ("max_sequence_length", 2),
        ("max_batch_size", 2),
        ("tensor_parallel_size", 2),
        ("context_parallel_size", 2),
        ("video_num_frames", 2),
        ("quantization", "int8"),
        ("fp32_layers", (1,)),
        ("mtp_seq_len", 2),
    ],
)
def test_unsupported_options_fail_before_reading_weights(tmp_path, option, value):
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "model.bundle",
        family="table_transformer",
        task="image_to_boxes",
        precision="fp32",
    )
    with pytest.raises((ValueError, NotImplementedError)):
        build(replace(request, **{option: value}), None)


def test_structure_shape_and_full_resize_capacity():
    cfg = _resolve_config({"num_queries": 125, "id2label": {str(i): str(i) for i in range(6)}})
    assert cfg["num_queries"] == 125
    assert cfg["num_labels"] == 6
    assert (cfg["image_size_h"], cfg["image_size_w"]) == (1000, 1000)


@pytest.mark.parametrize("height,width", [(0, 1000), (1000, -1)])
def test_invalid_profile_capacity_rejected(height, width):
    with pytest.raises(ValueError, match="positive"):
        _resolve_config(
            {"_family_build_options": {"input_image_h": height, "input_image_w": width}}
        )
