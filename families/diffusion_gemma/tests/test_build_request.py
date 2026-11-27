# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from families.diffusion_gemma.model import build
from tensorrt_model_connect.build import BuildRequest


@pytest.mark.parametrize("overrides", [
    {"precision": "bf16"}, {"precision": "fp8"}, {"backend": "trt_rtx"},
    {"task": "images_text_to_text"}, {"quantization": "fp8"},
    {"tensor_parallel_size": 2}, {"context_parallel_size": 2}, {"max_batch_size": 2},
    {"fp32_layers": (0,)}, {"dynamic_kv_cache": True}, {"mtp_seq_len": 2},
    {"image_height": 224}, {"video_num_frames": 2},
])
def test_build_rejects_unimplemented_options_before_compiling(tmp_path, overrides):
    request = BuildRequest(
        model_dir=tmp_path, output_path=tmp_path / "test.bundle", family="diffusion_gemma",
        task="text_continuation", precision="fp32",
    )
    expected = NotImplementedError if overrides.get("dynamic_kv_cache") else ValueError
    with pytest.raises(expected, match="diffusion_gemma"):
        build(replace(request, **overrides), None)
