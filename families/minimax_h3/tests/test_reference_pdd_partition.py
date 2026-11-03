# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Student index admission, without loading shards or reusing teacher weights."""

from copy import deepcopy

import pytest

from families.minimax_h3.reference_pdd import checkpoint_partition, validate_checkpoint_index


def fixture_index():
    groups = checkpoint_partition()
    keys = [key for group in groups.values() for key in group]
    return {
        "metadata": {"total_size": 70142174464},
        "weight_map": {
            key: f"diffusion_pytorch_model-{1 + index % 14:05}-of-00014.safetensors"
            for index, key in enumerate(keys)
        },
    }


def test_partition_requires_new_modulation_values_and_all_fifty_trained_gates():
    groups = checkpoint_partition()
    assert len(groups["adaln"]) == 106 and len(groups["denoiser"]) == 582
    assert "norm_out.linear.weight" in groups["adaln"]
    assert "transformer_blocks.49.adaln_proj.linear.weight" in groups["adaln"]
    assert "transformer_blocks.49.attn.to_gate_compress.weight" in groups["denoiser"]
    assert {
        "proj_out.weight",
        "proj_out.bias",
        "audio_proj_out.weight",
        "audio_proj_out.bias",
    } <= set(groups["denoiser"])
    validate_checkpoint_index(fixture_index())


@pytest.mark.parametrize(
    "problem",
    [
        "missing_gate",
        "missing_head",
        "extra",
        "redirected",
        "wrong_size",
        "float_size",
        "missing_shard",
    ],
)
def test_incomplete_student_or_teacher_substitution_is_rejected(problem):
    value = deepcopy(fixture_index())
    if problem == "missing_gate":
        value["weight_map"].pop("transformer_blocks.0.attn.to_gate_compress.weight")
    elif problem == "missing_head":
        value["weight_map"].pop("audio_proj_out.weight")
    elif problem == "extra":
        value["weight_map"]["unsupported.weight"] = (
            "diffusion_pytorch_model-00001-of-00014.safetensors"
        )
    elif problem == "redirected":
        value["weight_map"]["proj_out.weight"] = "../other.safetensors"
    elif problem == "wrong_size":
        value["metadata"]["total_size"] -= 1
    elif problem == "float_size":
        value["metadata"]["total_size"] = float(value["metadata"]["total_size"])
    else:
        value["weight_map"] = {
            key: name.replace("-00014-of-", "-00013-of-")
            for key, name in value["weight_map"].items()
        }
    with pytest.raises(ValueError):
        validate_checkpoint_index(value)
