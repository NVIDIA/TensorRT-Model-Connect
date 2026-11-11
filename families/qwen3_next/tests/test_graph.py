# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real BF16 DeltaNet/MoE and gated-attention/MoE layer comparisons."""

import json
import os
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.parametrize("index", [0, 3])
def test_checkpoint_layer(index):
    location = os.environ.get("TRTMC_QWEN3_NEXT_CHECKPOINT")
    if not location:
        pytest.skip("requires the pinned Qwen3-Coder-Next checkpoint and CUDA")
    import ml_dtypes
    import torch
    from families.qwen3_next.config import Config
    from families.qwen3_next.weights import Weights
    from families.qwen3_next.model import engine
    from families.qwen3_next.tests.gpu_runner import run
    from families.qwen3_next.tests.reference import OfficialReference

    torch.set_num_threads(8)
    root = Path(location)
    config = Config.load(root)
    plan = engine(Weights(root), config, index, 1024)
    random = np.random.default_rng(20261111)
    hidden = random.normal(0, 0.2, (1, config.hidden_size)).astype(ml_dtypes.bfloat16)
    inputs = {"hidden": hidden}
    if config.layer_types[index] == "linear_attention":
        inputs["conv_state"] = random.normal(
            0, 0.1, (config.conv_dim, config.linear_conv_kernel_dim - 1)
        ).astype(ml_dtypes.bfloat16)
        inputs["recurrent_state"] = random.normal(
            0,
            0.001,
            (
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            ),
        ).astype(np.float32)
    else:
        shape = (config.num_key_value_heads, 4, config.head_dim)
        inputs["past_k"] = random.normal(0, 0.2, shape).astype(ml_dtypes.bfloat16)
        inputs["past_v"] = random.normal(0, 0.2, shape).astype(ml_dtypes.bfloat16)
        inputs["mask"] = np.zeros((1, 1, 5), dtype=np.float32)
        inputs["mask"][..., 0] = -1e30
        inputs["position"] = np.array([3], dtype=np.int32)
    actual = run(plan, inputs)

    def tensor(value):
        t = torch.from_numpy(value.astype(np.float32))
        return t.to(torch.bfloat16) if value.dtype == ml_dtypes.bfloat16 else t

    reference = OfficialReference(root).run_layer(
        index,
        tensor(hidden),
        {key: tensor(value) for key, value in inputs.items() if key not in {"hidden", "position"}},
        3,
    )
    metrics = {}
    for name, value in reference.items():
        predicted = actual[name].astype(np.float32).ravel()
        expected = value.float().numpy().ravel()
        relative = float(
            np.linalg.norm(predicted - expected) / max(np.linalg.norm(expected), 1e-12)
        )
        metrics[name] = relative
        print(f"Qwen3-Next layer {index} {name}: relative_l2={relative:.9g}", flush=True)
        assert np.isfinite(predicted).all()
        assert relative < 0.03
    artifacts = os.environ.get("TRTMC_QWEN3_NEXT_TEST_ARTIFACTS")
    if artifacts:
        destination = Path(artifacts)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"validated-layer-{index}.plan").write_bytes(plan)
        (destination / f"layer-{index}-comparison.json").write_text(
            json.dumps({"index": index, "relative_l2": metrics}, indent=2)
        )
