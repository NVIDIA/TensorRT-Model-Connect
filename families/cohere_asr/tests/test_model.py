# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
import pytest

pytest.importorskip("tensorrt")
from families.cohere_asr.model import build
from tensorrt_model_connect import BuildRequest


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "transcription"),
        ("backend", "trt_rtx"),
        ("precision", "fp16"),
        ("precision", "bf16"),
        ("max_batch_size", 2),
        ("tensor_parallel_size", 2),
        ("context_parallel_size", 2),
        ("dynamic_kv_cache", True),
        ("quantization", "int8"),
        ("fp32_layers", (1,)),
        ("image_height", 32),
        ("image_width", 32),
        ("video_num_frames", 1),
        ("mtp_seq_len", 2),
        ("max_sequence_length", 15),
        ("max_sequence_length", 1025),
    ],
)
def test_unsupported_requests_fail_before_opening_checkpoint(tmp_path, field, value):
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "model.bundle",
        family="cohere_asr",
        task="speech_transcription",
        precision="fp32",
    )
    with pytest.raises((ValueError, NotImplementedError)):
        build(replace(request, **{field: value}), None)
