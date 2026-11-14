# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare native FP32 rotary tables with the official vision reference."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


def test_native_vision_rotary_tables_match_reference(tmp_path):
    import torch
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeVisionConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeVisionModel

    root = Path(__file__).resolve().parents[3]
    runtime = Path(os.environ.get("TRTMC_RUNTIME_ROOT", root / "build"))
    binary = runtime / "test_qwen3_vl_moe_threads"
    if not binary.is_file():
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            pytest.skip("native preprocessing regression requires the C++ compiler")
        binary = tmp_path / "preprocessing"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O1",
                "-pthread",
                "-I",
                str(root),
                "-I",
                str(root / "third_party/stb"),
                str(Path(__file__).parent / "cpp/test_preprocess_threads.cpp"),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    config = Qwen3VLMoeVisionConfig(
        depth=0,
        hidden_size=72,
        intermediate_size=144,
        num_heads=1,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=4,
        out_hidden_size=8,
        deepstack_visual_indexes=[],
    )
    reference = Qwen3VLMoeVisionModel(config).eval()
    assert reference.rotary_pos_emb.inv_freq.dtype == torch.float32
    cases = [{"height": h, "width": w} for h, w in [(2, 2), (2, 6), (8, 8), (6, 4)]]
    result = subprocess.run(
        [str(binary), "--rotary"],
        input=json.dumps(cases),
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(result.stdout)
    assert len(actual) == len(cases)
    for case, tables in zip(cases, actual):
        height, width = case["height"], case["width"]
        grid = torch.tensor([[1, height, width]], dtype=torch.int64)
        with torch.inference_mode():
            rope = reference.rot_pos_emb(grid).reshape(height * width, -1)
        np.testing.assert_allclose(
            tables["cos"], rope.cos().numpy().flatten(), atol=5e-6, rtol=5e-6
        )
        np.testing.assert_allclose(
            tables["sin"], rope.sin().numpy().flatten(), atol=5e-6, rtol=5e-6
        )
