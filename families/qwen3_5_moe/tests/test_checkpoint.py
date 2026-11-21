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


@pytest.mark.parametrize("zero", [7, 8])
def test_native_gptq_preserves_every_signed_code(zero, tmp_path):
    import numpy as np
    from safetensors.numpy import save_file
    from families.qwen3_5_moe.checkpoint_mapper import _open_safetensors
    from families.qwen3_5_moe.quantization import load_projection

    original = (np.arange(128 * 8).reshape(128, 8) % 16).astype(np.uint32)
    qweight = np.zeros((16, 8), dtype=np.uint32)
    for bit in range(8):
        qweight |= original[bit::8] << (4 * bit)
    qzeros = np.full((1, 1), sum(zero << (4 * bit) for bit in range(8)), np.uint32)
    scales = np.arange(1, 9, dtype=np.float16).reshape(1, 8) / np.float16(32)
    scales[0, 1] = 0  # Original checkpoint zero groups must remain zero.
    save_file({"expert.qweight": qweight.view(np.int32), "expert.qzeros": qzeros.view(np.int32),
               "expert.scales": scales, "expert.g_idx": np.zeros(128, np.int32)},
              tmp_path / "model.safetensors")
    value = load_projection(_open_safetensors(tmp_path, prepared=True), "expert")
    packed = value.packed
    codes = np.empty((8, 128), dtype=np.int8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    decoded = np.where(codes >= 8, codes - 16, codes)
    np.testing.assert_array_equal(decoded, original.T.astype(np.int16) - 8)
    np.testing.assert_array_equal(value.scales, scales.T)
    assert value.shape == (128, 8)
    selected = value.select_outputs(np.array([7, 0]))
    np.testing.assert_array_equal(selected.packed, value.packed[[7, 0]])
    np.testing.assert_array_equal(selected.scales, value.scales[[7, 0]])


@pytest.mark.parametrize("fault", ["act_order", "asymmetric_zero", "scale_dtype", "negative_scale"])
def test_native_gptq_rejects_precision_changes(fault):
    import numpy as np
    from families.qwen3_5_moe.quantization import gptq_projection

    weights = np.zeros((16, 8), np.int32)
    zeros = np.full((1, 1), 0x77777777, np.int32)
    scales = np.ones((1, 8), np.float16)
    groups = np.zeros(128, np.int32)
    if fault == "act_order":
        groups[0] = 1
    elif fault == "asymmetric_zero":
        zeros[0, 0] = 0
    elif fault == "scale_dtype":
        scales = scales.astype(np.float32)
    else:
        scales[0, 0] = -1
    with pytest.raises(ValueError):
        gptq_projection(weights, zeros, scales, groups)


def test_native_nvfp4_retains_zero_blocks_without_activation_quantization():
    import ml_dtypes
    import numpy as np
    from families.qwen3_5_moe.quantization import nvfp4_projection

    packed = np.arange(32 * 64, dtype=np.uint16).astype(np.uint8).reshape(32, 64)
    scales = np.ones((32, 8), dtype=ml_dtypes.float8_e4m3fn)
    scales[1] = 0  # Original packed experts contain whole zero-valued groups.
    value = nvfp4_projection(packed, scales, np.float32(.000137))
    assert value.format == "nvfp4_a16" and value.input_scale is None
    assert value.shape == (128, 32)
    np.testing.assert_array_equal(value.packed, packed)
    np.testing.assert_array_equal(value.scales, scales.view(np.uint8))
