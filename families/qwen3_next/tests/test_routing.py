# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve official CPU expert selection, including equal routing scores."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("kind", "experts", "top_k"),
    [
        ("equal", 512, 10),
        ("cutoff_tie", 512, 10),
        ("quantized", 512, 10),
        ("quantized", 512, 8),
        ("quantized", 512, 9),
        ("unique", 512, 10),
        ("equal", 8, 1),
        ("equal", 512, 0),
        ("nonfinite", 512, 10),
    ],
)
def test_native_expert_selection(kind, experts, top_k, tmp_path):
    runtime = os.environ.get("TRTMC_RUNTIME_ROOT")
    if not runtime:
        pytest.skip("requires the native Qwen3-Next routing diagnostic")
    import torch

    random = np.random.default_rng(20261112)
    logits = torch.tensor(random.normal(size=experts), dtype=torch.float32)
    if kind == "equal":
        logits.zero_()
    elif kind == "cutoff_tie":
        logits.zero_()
        logits[:20] = 1
    elif kind == "quantized":
        logits = logits.to(torch.bfloat16).float()
    probabilities = logits.softmax(dim=0)
    if kind == "nonfinite":
        probabilities[:2] = float("nan")
        probabilities[3] = float("inf")
    filename = tmp_path / "probabilities.f32"
    probabilities.numpy().tofile(filename)
    result = subprocess.run(
        [str(Path(runtime) / "qwen3_next_routing_probe"), str(filename), str(top_k)],
        check=True,
        capture_output=True,
        text=True,
    )
    selected = json.loads(result.stdout)
    assert selected["indices"] == torch.topk(probabilities, top_k, sorted=True).indices.tolist()
