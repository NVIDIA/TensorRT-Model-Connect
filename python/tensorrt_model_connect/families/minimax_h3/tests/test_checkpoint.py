# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
import torch

from tensorrt_model_connect.families.minimax_h3 import checkpoint
from tensorrt_model_connect.families.minimax_h3.checkpoint import (
    load_selected_component_state_dict,
    merge_fast_h3_adapter_state,
    validate_fast_h3_adapter,
)
from tensorrt_model_connect.families.minimax_h3.config import (
    SOL_ENGINE_1344X768_124_TO_345F,
)
from tensorrt_model_connect.families.minimax_h3.dit_builder import (
    checkpoint_keys as dit_checkpoint_keys,
    vsa_segment_checkpoint_partitions,
)


def test_selective_loader_supports_unsharded_audio_vae_checkpoint(tmp_path: Path) -> None:
    save_file(
        {
            "decoder.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
            "encoder.weight": np.ones((4, 4), dtype=np.float32),
        },
        tmp_path / "diffusion_pytorch_model.safetensors",
    )

    selected = load_selected_component_state_dict(tmp_path, ("decoder.weight",))

    assert set(selected) == {"decoder.weight"}
    np.testing.assert_array_equal(
        selected["decoder.weight"].numpy(),
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )


def test_selective_loader_rejects_missing_unsharded_tensor(tmp_path: Path) -> None:
    save_file(
        {"decoder.weight": np.ones((1,), dtype=np.float32)},
        tmp_path / "diffusion_pytorch_model.safetensors",
    )

    with pytest.raises(ValueError, match="missing tensors"):
        load_selected_component_state_dict(tmp_path, ("decoder.bias",))


def test_selective_loader_rejects_ambiguous_unsharded_component(tmp_path: Path) -> None:
    save_file({"a": np.ones((1,), dtype=np.float32)}, tmp_path / "a.safetensors")
    save_file({"b": np.ones((1,), dtype=np.float32)}, tmp_path / "b.safetensors")

    with pytest.raises(ValueError, match="one unsharded file"):
        load_selected_component_state_dict(tmp_path, ("a",))


def test_fast_h3_hybrid_merge_applies_fp32_lora_diff_and_replacement(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter.safetensors"
    a = np.zeros((64, 3), dtype=np.float32)
    b = np.zeros((2, 64), dtype=np.float32)
    a[0] = (1.0, 2.0, 3.0)
    b[:, 0] = (2.0, -1.0)
    save_file(
        {
            "layer.lora_A.weight": a,
            "layer.lora_B.weight": b,
            "layer.diff": np.full((2, 3), 0.5, dtype=np.float32),
            "layer.diff_b": np.asarray((1.25, -0.75), dtype=np.float32),
            "transformer_blocks.0.attn.to_gate_compress.set_weight": np.full(
                (2, 3), 7.0, dtype=np.float32
            ),
        },
        adapter,
    )
    state = {
        "layer.weight": torch.zeros((2, 3), dtype=torch.bfloat16),
        "layer.bias": torch.zeros((2,), dtype=torch.bfloat16),
    }

    counts = merge_fast_h3_adapter_state(
        state,
        adapter,
        (
            "layer.weight",
            "layer.bias",
            "transformer_blocks.0.attn.to_gate_compress.weight",
        ),
    )

    expected = b @ a + 0.5
    np.testing.assert_allclose(state["layer.weight"].float().numpy(), expected)
    np.testing.assert_allclose(
        state["layer.bias"].float().numpy(), np.asarray((1.25, -0.75))
    )
    np.testing.assert_allclose(
        state["transformer_blocks.0.attn.to_gate_compress.weight"].numpy(),
        np.full((2, 3), 7.0),
    )
    assert counts == {"low_rank": 1, "diff": 2, "set_weight": 1, "tensors": 5}


def _tiny_strict_fast_h3_adapter(path: Path) -> tuple[dict[str, np.ndarray], set[str]]:
    tensors: dict[str, np.ndarray] = {}
    targets: set[str] = set()
    for index in range(362):
        prefix = f"module_{index}"
        tensors[f"{prefix}.lora_A.weight"] = np.zeros((64, 1), dtype=np.float32)
        tensors[f"{prefix}.lora_B.weight"] = np.zeros((1, 64), dtype=np.float32)
        targets.add(f"{prefix}.weight")
    for index in range(24):
        tensors[f"weight_delta_{index}.diff"] = np.zeros((1,), dtype=np.float32)
        targets.add(f"weight_delta_{index}.weight")
    for index in range(58):
        tensors[f"bias_delta_{index}.diff_b"] = np.zeros((1,), dtype=np.float32)
        targets.add(f"bias_delta_{index}.bias")
    for index in range(50):
        prefix = f"transformer_blocks.{index}.attn.to_gate_compress"
        tensors[f"{prefix}.set_weight"] = np.zeros((1,), dtype=np.float32)
        targets.add(f"{prefix}.weight")
    save_file(tensors, path, metadata=dict(checkpoint._FASTH3_ADAPTER_METADATA))
    return tensors, targets


def test_fast_h3_identity_requires_exhaustive_856_tensor_accounting(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter.safetensors"
    tensors, targets = _tiny_strict_fast_h3_adapter(adapter)
    payload = adapter.read_bytes()

    identity = validate_fast_h3_adapter(
        adapter,
        {"all": targets},
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
    )

    assert len(tensors) == identity.tensor_count == 856
    assert identity.low_rank_tensor_count == 724
    assert identity.diff_tensor_count == 82
    assert identity.set_weight_tensor_count == identity.gate_tensor_count == 50
    assert identity.partition_tensor_counts == {"all": 856}
    assert str(tmp_path) not in str(identity.bundle_metadata())

    with pytest.raises(ValueError, match="not exhaustive"):
        validate_fast_h3_adapter(
            adapter,
            {"incomplete": targets - {"module_0.weight"}},
            expected_sha256=identity.sha256,
            expected_size_bytes=identity.size_bytes,
        )


def test_segmented_vsa_weight_partition_is_exhaustive_and_non_overlapping() -> None:
    profile = SOL_ENGINE_1344X768_124_TO_345F
    partitions = vsa_segment_checkpoint_partitions(profile)
    flattened = tuple(key for keys in partitions.values() for key in keys)

    assert len(partitions) == profile.num_layers + 1 == 51
    assert tuple(partitions) == (
        "denoiser_entry",
        *(f"denoiser_transition_{index:02d}" for index in range(49)),
        "denoiser_finish",
    )
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(
        dit_checkpoint_keys(profile, include_vsa_gates=True)
    )
    assert sum(
        key.endswith(".attn.to_gate_compress.weight") for key in flattened
    ) == profile.num_layers
