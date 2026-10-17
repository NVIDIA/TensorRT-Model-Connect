# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source precision layout tests independent of GPU build and numerical qualification."""

import pytest
from families.qwen3_5_moe.edge_llm.checkpoint import projection_layouts, validate_layout


@pytest.fixture(params=["int4", "nvfp4"])
def layout(request):
    tensors = projection_layouts(request.param)
    tensors["model.visual.blocks.0.attn.qkv.weight"] = ("BF16", (3456, 1152))
    tensors["model.language_model.layers.0.mlp.gate.weight"] = ("BF16", (256, 2048))
    return tensors, request.param


def test_complete_declared_layout(layout):
    tensors, kind = layout
    validate_layout(tensors, kind)


@pytest.mark.parametrize(
    "fault", ["missing", "dtype", "shape", "extra_quant", "vision_quant", "missing_visual"]
)
def test_source_layout_conflicts(layout, fault):
    tensors, kind = layout
    key = next(iter(tensors))
    if fault == "missing":
        del tensors[key]
    if fault == "dtype":
        tensors[key] = ("F32", tensors[key][1])
    if fault == "shape":
        tensors[key] = (tensors[key][0], (1,))
    if fault == "extra_quant":
        tensors["model.language_model.layers.0.mlp.gate.qweight"] = ("I32", (1, 1))
    if fault == "vision_quant":
        tensors["model.visual.blocks.0.attn.qkv.weight"] = ("F8_E4M3", (3456, 1152))
    if fault == "missing_visual":
        del tensors["model.visual.blocks.0.attn.qkv.weight"]
    with pytest.raises(ValueError):
        validate_layout(tensors, kind)


@pytest.mark.parametrize(
    "fault", ["valid", "missing", "wrong_shard", "extra", "traversal", "symlink"]
)
def test_index_matches_actual_container_headers(tmp_path, monkeypatch, fault):
    import json
    import numpy as np
    from safetensors.numpy import save_file
    from families.qwen3_5_moe.edge_llm import checkpoint

    root = tmp_path / "model"
    root.mkdir()
    save_file({"tensor": np.ones((2, 2), dtype=np.float16)}, root / "one.safetensors")
    mapping = {"tensor": "one.safetensors"}
    if fault == "missing":
        mapping["missing"] = "one.safetensors"
    if fault == "wrong_shard":
        save_file({"other": np.ones((1,), dtype=np.float16)}, root / "two.safetensors")
        mapping = {"tensor": "two.safetensors", "other": "one.safetensors"}
    if fault == "extra":
        mapping = {"other": "one.safetensors"}
    if fault == "traversal":
        mapping = {"tensor": "../one.safetensors"}
    if fault == "symlink":
        (root / "link.safetensors").symlink_to(root / "one.safetensors")
        mapping = {"tensor": "link.safetensors"}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    seen = []
    monkeypatch.setattr(
        checkpoint, "validate_layout", lambda tensors, kind: seen.append((tensors, kind))
    )
    if fault == "valid":
        assert checkpoint.validate_checkpoint(root, "int4") == {
            "tensor_count": 1,
            "shard_count": 1,
            "layout": "int4",
        }
        assert seen == [({"tensor": ("F16", (2, 2))}, "int4")]
    else:
        with pytest.raises(ValueError):
            checkpoint.validate_checkpoint(root, "int4")
