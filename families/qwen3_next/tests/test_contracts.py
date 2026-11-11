# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family identity, command, and checkpoint-integrity contracts."""

import json
import struct
import pytest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family
from families.qwen3_next.support import describe
from families.qwen3_next.weights import Weights


def test_exact_family_ownership():
    metadata = ModelMetadata(
        config={"model_type": "qwen3_next", "architectures": ["Qwen3NextForCausalLM"]},
        model_index={},
    )
    family, support = resolve_family(metadata)
    assert family == "qwen3_next"
    assert support.tasks == ("text_generation",)
    assert support.default_task == "text_generation"
    assert support.default_precision == "bf16"
    assert describe(ModelMetadata(config={"model_type": "qwen3_moe"}, model_index={})) is None


def test_cli_routes_bf16_checkpoint(monkeypatch, tmp_path):
    from families.qwen3_next import cli

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_next"}))
    calls = []
    monkeypatch.setattr(
        cli, "build_bundle", lambda request, output: calls.append((request, output))
    )
    assert (
        cli.build(model=str(tmp_path), output=tmp_path / "out.bundle", max_sequence_length=600) == 0
    )
    assert len(calls) == 1
    assert calls[0][0].precision == "bf16"
    assert calls[0][0].max_sequence_length == 600
    assert calls[0][1] == tmp_path / "out.bundle"


def make_checkpoint(root, offsets=(0, 4), payload=b"\x80\x3f\x00\x40"):
    header = json.dumps(
        {"x": {"dtype": "BF16", "shape": [2], "data_offsets": list(offsets)}}
    ).encode()
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "part.safetensors"}})
    )
    (root / "part.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return Weights(root)


def test_weight_ranges_and_complete_mapping(tmp_path):
    weights = make_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="unmapped"):
        weights.finish()
    with weights.view("x") as view:
        assert bytes(view) == b"\x80\x3f\x00\x40"
    weights.finish()
    with (tmp_path / "copy").open("wb") as stream:
        weights.copy("x", stream)
    assert (tmp_path / "copy").read_bytes() == b"\x80\x3f\x00\x40"


def test_truncated_checkpoint_rejected(tmp_path):
    weights = make_checkpoint(tmp_path, payload=b"\x00\x00")
    with pytest.raises(ValueError, match="truncated"):
        weights.describe("x")


@pytest.mark.parametrize(
    "values", [{"precision": "fp16"}, {"tensor_parallel_size": 2}, {"max_sequence_length": 0}]
)
def test_unsupported_build_requests(tmp_path, values):
    from families.qwen3_next.cli import BuildRequest

    with pytest.raises(ValueError):
        BuildRequest(model_dir=tmp_path, **values)
