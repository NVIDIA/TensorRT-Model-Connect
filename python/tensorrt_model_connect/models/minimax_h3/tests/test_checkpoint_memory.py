# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 checkpoint selection and host-memory contracts."""

from __future__ import annotations

import gc
import json
import ml_dtypes
import numpy as np
import pytest

from tensorrt_model_connect.models.minimax_h3.adaln_builder import (
    checkpoint_keys as adaln_checkpoint_keys,
)
from tensorrt_model_connect.models.minimax_h3.checkpoint import (
    numpy_state,
    validate_component_key_partition,
)
from tensorrt_model_connect.models.minimax_h3.config import SOL_ENGINE_1344X768_124F
from tensorrt_model_connect.models.minimax_h3.dit_builder import (
    checkpoint_keys as dit_checkpoint_keys,
)


torch = pytest.importorskip("torch")


def test_indexed_component_partition_fails_closed(tmp_path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {name: "model.safetensors" for name in "abc"}}))

    validate_component_key_partition(tmp_path, (("a",), ("b", "c")))
    with pytest.raises(ValueError, match="partitions overlap"):
        validate_component_key_partition(tmp_path, (("a", "b"), ("b", "c")))
    with pytest.raises(ValueError, match="not exhaustive"):
        validate_component_key_partition(tmp_path, (("a",), ("b", "missing")))


def test_transformer_component_keys_are_disjoint_and_exhaustive() -> None:
    profile = SOL_ENGINE_1344X768_124F
    adaln = set(adaln_checkpoint_keys(profile))
    dit = set(dit_checkpoint_keys(profile))

    assert len(adaln) == 106
    assert len(dit) == 532
    assert adaln.isdisjoint(dit)
    assert len(adaln | dit) == 638
    assert {name for name in adaln if name.startswith("time_embedder.")} == {
        "time_embedder.linear_1.weight",
        "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight",
        "time_embedder.linear_2.bias",
    }
    assert {"norm_out.linear.weight", "norm_out.linear.bias"} <= adaln
    assert "norm_out.norm.weight" in dit

    for index in range(profile.num_layers):
        prefix = f"transformer_blocks.{index}."
        assert len({name for name in adaln if name.startswith(prefix)}) == 2
        assert len({name for name in dit if name.startswith(prefix)}) == 10
    for index in range(profile.num_refiner_layers):
        prefix = f"token_refiner.refiner_blocks.{index}."
        assert len({name for name in dit if name.startswith(prefix)}) == 10


def test_numpy_state_preserves_bf16_storage_bits_without_fp32_expansion() -> None:
    bf16 = torch.tensor([1.5, -2.25, 3.125, 4.5], dtype=torch.bfloat16)
    fp32 = torch.tensor([0.125, -8.0], dtype=torch.float32)
    expected_bf16_bits = bf16.view(torch.uint16).numpy().copy()

    state = {"bf16": bf16, "fp32": fp32}
    arrays = numpy_state(state)

    assert arrays["bf16"].dtype == np.dtype(ml_dtypes.bfloat16)
    assert arrays["bf16"].nbytes == bf16.numel() * 2
    assert np.shares_memory(
        arrays["bf16"].view(np.uint16),
        bf16.view(torch.uint16).numpy(),
    )
    assert arrays["fp32"].dtype == np.float32
    np.testing.assert_array_equal(arrays["fp32"], fp32.numpy())

    del state, bf16, fp32
    gc.collect()
    assert getattr(arrays["bf16"], "_tensor_owner", None) is not None
    np.testing.assert_array_equal(arrays["bf16"].view(np.uint16), expected_bf16_bits)
