# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact-checkpoint checks for dense/full and MoE/sliding Flint layers."""

import os
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.parametrize("layer", [0, 2, 5])
def test_checkpoint_layer(layer):
    directory = os.environ.get("TRTMC_FLINT_CHECKPOINT")
    if not directory:
        pytest.skip("requires the pinned Flint checkpoint and a CUDA device")
    import ml_dtypes
    import torch
    from families.flint.config import Config
    from families.flint.model import engine
    from families.flint.weights import Weights
    from families.flint.tests.gpu_runner import run
    from families.flint.tests.reference import Reference

    torch.set_num_threads(8)
    root = Path(directory)
    config = Config.load(root)
    weights = Weights(root)
    plan = engine(weights, config, layer, 1024)
    rng = np.random.default_rng(20261110)
    hidden = rng.normal(0, 0.2, (1, config.hidden_size)).astype(ml_dtypes.bfloat16)
    cache_shape = (config.num_key_value_heads, 4, config.head_dim)
    k = rng.normal(0, 0.2, cache_shape).astype(ml_dtypes.bfloat16)
    v = rng.normal(0, 0.2, cache_shape).astype(ml_dtypes.bfloat16)
    mask = np.zeros((1, 1, 5), dtype=np.float32)
    mask[..., 0] = -1e30
    actual = run(
        plan,
        {
            "hidden": hidden,
            "past_k": k,
            "past_v": v,
            "mask": mask,
            "position": np.array([3], dtype=np.int32),
        },
    )

    def tensor(a):
        return torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16)

    expected = Reference(root).layer(
        layer, tensor(hidden), tensor(k), tensor(v), torch.from_numpy(mask), 3
    )
    for name, reference in zip(("hidden_out", "present_k", "present_v"), expected):
        predicted = actual[name].astype(np.float32).ravel()
        reference = reference.float().numpy().ravel()
        relative = float(
            np.linalg.norm(predicted - reference) / max(np.linalg.norm(reference), 1e-12)
        )
        print(f"Flint layer {layer} {name}: relative_l2={relative:.8g}", flush=True)
        assert np.isfinite(predicted).all()
        assert relative < 0.03
