# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from dataclasses import replace
from pathlib import Path
import json

import ml_dtypes
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from families.minimax_h3 import turbo_checkpoint as checkpoint
from families.minimax_h3 import quantized_checkpoint as quantized
from families.minimax_h3.checkpoint import numpy_state
from families.minimax_h3.turbo_checkpoint import (
    TurboLoraWeight,
    load_selected_turbo_transformer_weights,
    pack_turbo_qkv,
    validate_turbo_transformer_checkpoint,
)


def _array(value) -> np.ndarray:
    return np.asarray(value, dtype=ml_dtypes.bfloat16)


def _tensor(value: np.ndarray) -> torch.Tensor:
    dtype = torch.bfloat16 if value.dtype == np.dtype(ml_dtypes.bfloat16) else torch.float32
    return torch.from_numpy(value.astype(np.float32)).to(dtype)


def _linear(x, weight: TurboLoraWeight, bias=None):
    """CPU oracle retaining all authored BF16 activation rounding boundaries."""
    return F.linear(x, _tensor(weight.base), bias) + F.linear(
        F.linear(x, _tensor(weight.lora_a)), _tensor(weight.lora_b)
    )


def _tiny_checkpoints(tmp_path: Path, monkeypatch):
    shapes = {
        "blocks.0.attn.qkv_proj.weight": (12, 4),
        "blocks.0.mlp.fc1.weight": (6, 4),
        "blocks.0.mlp.fc2.weight": (4, 3),
        "blocks.0.adaln_proj.linear.weight": (8, 4),
        "blocks.0.adaln_proj.linear.bias": (8,),
        "final_layer.adaln_proj.linear.weight": (8, 4),
        "final_layer.adaln_proj.linear.bias": (8,),
        "token_refiner.blocks.0.attn.qkv_proj.weight": (12, 4),
        "token_refiner.blocks.0.mlp.fc1.weight": (6, 4),
        "video_patch_proj.weight": (4, 4),
    }
    base = {
        name: (torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape) / 17 - 1).to(
            torch.float32 if name.startswith("video_patch_proj") else torch.bfloat16
        )
        for name, shape in shapes.items()
    }
    lora = {}
    for name, value in base.items():
        if value.ndim != 2 or name.startswith("video_patch_proj"):
            continue
        module = name.removesuffix(".weight")
        rank = 2
        lora[f"{module}.lora_A.weight"] = (
            torch.arange(rank * value.shape[1]).reshape(rank, -1) / 11 + 0.01
        ).to(torch.bfloat16)
        lora[f"{module}.lora_B.weight"] = (
            torch.arange(value.shape[0] * rank).reshape(-1, rank) / 23 - 0.3
        ).to(torch.bfloat16)
    base_file, lora_file = tmp_path / "base.safetensors", tmp_path / "lora.safetensors"
    save_file(base, base_file, metadata={"config": json.dumps(checkpoint._EXPECTED_CONFIG)})
    save_file(lora, lora_file, metadata=checkpoint._LORA_METADATA)
    monkeypatch.setattr(
        checkpoint,
        "_BASE_SPECS",
        {
            name: ("F32" if value.dtype == torch.float32 else "BF16", tuple(value.shape))
            for name, value in base.items()
        },
    )
    monkeypatch.setattr(
        checkpoint,
        "_LORA_SPECS",
        {name: ("BF16", tuple(value.shape)) for name, value in lora.items()},
    )
    monkeypatch.setattr(checkpoint, "BASE_BYTES", base_file.stat().st_size)
    monkeypatch.setattr(checkpoint, "LORA_BYTES", lora_file.stat().st_size)
    return base_file, lora_file, base, lora


def test_full_public_header_contract_preserves_native_precision_and_ranks():
    assert len(checkpoint._BASE_SPECS) == 535
    assert Counter(dtype for dtype, _shape in checkpoint._BASE_SPECS.values()) == {
        "BF16": 522,
        "F32": 13,
    }
    assert len(checkpoint._LORA_SPECS) == 518
    ranks = Counter(
        shape[0]
        for name, (_dtype, shape) in checkpoint._LORA_SPECS.items()
        if name.endswith("lora_A.weight")
    )
    assert ranks == {64: 208, 16: 51}
    assert checkpoint._BASE_SPECS["final_layer.video_out.weight"][0] == "F32"
    assert checkpoint._BASE_SPECS["time_embedder.proj_out.weight"][0] == "F32"


@pytest.mark.parametrize(
    "root,physical",
    (
        ("transformer_blocks.0", "blocks.0"),
        ("token_refiner.refiner_blocks.0", "token_refiner.blocks.0"),
    ),
)
def test_grouped_qkv_shares_unpermuted_parent_and_preserves_runtime_lora(
    tmp_path, monkeypatch, root, physical
):
    base_file, lora_file, base, lora = _tiny_checkpoints(tmp_path, monkeypatch)
    names = [f"{root}.attn.to_{name}.weight" for name in ("q", "k", "v")]
    loaded = load_selected_turbo_transformer_weights(base_file, lora_file, names)
    children = [loaded[name] for name in names]
    packed = pack_turbo_qkv(children)
    assert packed is children[0].packed_parent
    assert all(child.lora_a is packed.lora_a for child in children)
    torch.testing.assert_close(
        _tensor(packed.base), base[f"{physical}.attn.qkv_proj.weight"], rtol=0, atol=0
    )
    assert [child.row_slice for child in children] == [(0, 4), (4, 8), (8, 12)]
    x = torch.tensor([[0.3, -0.7, 1.4, 0.9]], dtype=torch.bfloat16)
    module = f"{physical}.attn.qkv_proj"
    expected = F.linear(x, base[f"{module}.weight"]) + F.linear(
        F.linear(x, lora[f"{module}.lora_A.weight"]), lora[f"{module}.lora_B.weight"]
    )
    torch.testing.assert_close(_linear(x, packed), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.cat([_linear(x, child) for child in children], -1), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "root,physical",
    (
        ("transformer_blocks.0", "blocks.0"),
        ("token_refiner.refiner_blocks.0", "token_refiner.blocks.0"),
    ),
)
def test_swiglu_swaps_base_and_b_rows_but_keeps_a_and_rounding(
    tmp_path, monkeypatch, root, physical
):
    base_file, lora_file, base, lora = _tiny_checkpoints(tmp_path, monkeypatch)
    name = f"{root}.ff.net.0.proj.weight"
    loaded = load_selected_turbo_transformer_weights(base_file, lora_file, [name])[name]
    x = torch.tensor([[0.3, -0.7, 1.4, 0.9]], dtype=torch.bfloat16)
    module = f"{physical}.mlp.fc1"
    authored = F.linear(x, base[f"{module}.weight"]) + F.linear(
        F.linear(x, lora[f"{module}.lora_A.weight"]), lora[f"{module}.lora_B.weight"]
    )
    gate, value = authored.chunk(2, dim=-1)
    converted_value, converted_gate = _linear(x, loaded).chunk(2, dim=-1)
    torch.testing.assert_close(
        F.silu(converted_gate) * converted_value, F.silu(gate) * value, rtol=0, atol=0
    )
    torch.testing.assert_close(
        _tensor(loaded.lora_a), lora[f"{module}.lora_A.weight"], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "logical,physical",
    (
        ("transformer_blocks.0.adaln_proj.linear", "blocks.0.adaln_proj.linear"),
        ("norm_out.linear", "final_layer.adaln_proj.linear"),
    ),
)
def test_adaln_and_final_bias_belong_to_base_before_low_rank_add(
    tmp_path, monkeypatch, logical, physical
):
    base_file, lora_file, base, lora = _tiny_checkpoints(tmp_path, monkeypatch)
    names = [f"{logical}.weight", f"{logical}.bias"]
    loaded = load_selected_turbo_transformer_weights(base_file, lora_file, names)
    weight, bias = loaded[names[0]], _tensor(loaded[names[1]])
    x = F.silu(torch.tensor([[0.3, -0.7, 1.4, 0.9]], dtype=torch.bfloat16))
    expected = F.linear(x, base[f"{physical}.weight"], base[f"{physical}.bias"]) + F.linear(
        F.linear(x, lora[f"{physical}.lora_A.weight"]), lora[f"{physical}.lora_B.weight"]
    )
    torch.testing.assert_close(_linear(x, weight, bias), expected, rtol=0, atol=0)


def test_small_lora_update_is_not_silently_rounded_away_by_weight_merge():
    x = torch.tensor([[1.0, -1.0]], dtype=torch.bfloat16)
    weight = TurboLoraWeight(_array([[1, 1]]), _array([[1, 0]]), _array([[0.001]]))
    authored = _linear(x, weight)
    merged = (
        _tensor(weight.base).float()
        + _tensor(weight.lora_b).float() @ _tensor(weight.lora_a).float()
    ).bfloat16()
    assert authored.item() != 0.0
    assert F.linear(x, merged).item() == 0.0


def test_unadapted_fp32_weight_is_unchanged_and_reads_only_requested_payload(tmp_path, monkeypatch):
    base_file, lora_file, base, _lora = _tiny_checkpoints(tmp_path, monkeypatch)
    import safetensors

    original = safetensors.safe_open
    reads = []

    class Reader:
        def __init__(self, *args, **kwargs):
            self.reader = original(*args, **kwargs)

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def get_tensor(self, name):
            reads.append(name)
            return self.reader.get_tensor(name)

    monkeypatch.setattr(safetensors, "safe_open", Reader)
    loaded = load_selected_turbo_transformer_weights(base_file, lora_file, ["proj_in.weight"])
    assert reads == ["video_patch_proj.weight"]
    assert loaded["proj_in.weight"].dtype == np.float32
    np.testing.assert_array_equal(loaded["proj_in.weight"], base["video_patch_proj.weight"].numpy())


def test_pack_rejects_mixed_factors_or_reordered_parent_children(tmp_path, monkeypatch):
    base_file, lora_file, _base, _lora = _tiny_checkpoints(tmp_path, monkeypatch)
    names = [f"transformer_blocks.0.attn.to_{name}.weight" for name in ("q", "k", "v")]
    children = list(load_selected_turbo_transformer_weights(base_file, lora_file, names).values())
    with pytest.raises(ValueError, match="Q,K,V order"):
        pack_turbo_qkv([children[1], children[0], children[2]])
    altered = TurboLoraWeight(children[1].base, children[1].lora_a.copy(), children[1].lora_b)
    with pytest.raises(ValueError, match="exact shared A"):
        pack_turbo_qkv([children[0], altered, children[2]])


def test_pack_independent_views_requires_shared_a_and_keeps_low_rank_branch():
    a = _array([[1, 0]])
    values = [
        TurboLoraWeight(_array([[index, 2]]), a, _array([[index / 10]])) for index in range(3)
    ]
    packed = pack_turbo_qkv(values)
    assert packed.is_full_fused_qkv
    assert packed.lora_a is a
    assert packed.shape == (3, 2)


def test_strict_headers_reject_missing_lora_or_wrong_rank(tmp_path, monkeypatch):
    base_file, lora_file, _base, lora = _tiny_checkpoints(tmp_path, monkeypatch)
    name = "blocks.0.attn.qkv_proj.lora_A.weight"
    lora[name] = torch.ones((3, 4), dtype=torch.bfloat16)
    save_file(lora, lora_file, metadata=checkpoint._LORA_METADATA)
    monkeypatch.setattr(checkpoint, "LORA_BYTES", lora_file.stat().st_size)
    with pytest.raises(ValueError, match="shape/dtype mismatch"):
        validate_turbo_transformer_checkpoint(base_file, lora_file)
    lora.pop(name)
    save_file(lora, lora_file, metadata=checkpoint._LORA_METADATA)
    monkeypatch.setattr(checkpoint, "LORA_BYTES", lora_file.stat().st_size)
    with pytest.raises(ValueError, match="tensor inventory"):
        validate_turbo_transformer_checkpoint(base_file, lora_file)


def test_loader_rejects_duplicate_unknown_names_and_wrong_ref_source(tmp_path, monkeypatch):
    base_file, lora_file, _base, _lora = _tiny_checkpoints(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="duplicate"):
        load_selected_turbo_transformer_weights(base_file, lora_file, ["proj_in.weight"] * 2)
    with pytest.raises(ValueError, match="Unsupported"):
        load_selected_turbo_transformer_weights(base_file, lora_file, ["unrecognized.weight"])
    with pytest.raises(ValueError, match="released diffusion_models filename"):
        validate_turbo_transformer_checkpoint(base_file, lora_file, workflow="ref2va")
    metadata = validate_turbo_transformer_checkpoint(base_file, lora_file, workflow="t2va")
    assert metadata["lora_merged"] is False
    assert metadata["lora_strength"] == 1.0


def test_numpy_bf16_owner_preserves_exact_checkpoint_bits():
    source = torch.tensor([[1.0, 0.001, -3.25]], dtype=torch.bfloat16)
    view = numpy_state({"weight": source})["weight"]
    np.testing.assert_array_equal(view.view(np.uint16), source.view(torch.uint16).numpy())


def _tiny_int8_checkpoints(tmp_path, monkeypatch):
    """A small full-inventory stand-in; keep the real pinned-source checks."""
    _dense_file, lora_file, base, lora = _tiny_checkpoints(tmp_path, monkeypatch)
    groups = {
        "blocks.0.attn.qkv_proj": 4,
        "blocks.0.mlp.fc1": 4,
        "blocks.0.adaln_proj.linear": 4,
    }
    payload = dict(base)
    for module, group_size in groups.items():
        original = payload[f"{module}.weight"]
        payload[f"{module}.weight"] = (
            torch.arange(original.numel()).reshape(original.shape) % 31 - 15
        ).to(torch.int8)
        payload[f"{module}.weight_scale"] = (
            torch.arange(original.shape[0], dtype=torch.float32).reshape(-1, 1) + 1
        ) * 0.0033
        marker = json.dumps(
            {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size}
        ).encode("utf-8")
        payload[f"{module}.comfy_quant"] = torch.tensor(list(marker), dtype=torch.uint8)
    path = tmp_path / "int8" / quantized.CHECKPOINT_FILENAME
    path.parent.mkdir(parents=True)
    save_file(payload, path, metadata={"config": json.dumps(checkpoint._EXPECTED_CONFIG)})
    source_metadata = (
        path.parents[1]
        / ".cache/huggingface/download"
        / f"{quantized.CHECKPOINT_FILENAME}.metadata"
    )
    source_metadata.parent.mkdir(parents=True)
    source_metadata.write_text(f"{quantized.CHECKPOINT_REVISION}\nfixture\n0\n", encoding="utf-8")
    dtype_names = {
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bfloat16: "BF16",
        torch.float32: "F32",
    }
    monkeypatch.setattr(
        quantized,
        "_PHYSICAL_SPECS",
        {
            name: quantized._TensorSpec(dtype_names[value.dtype], tuple(value.shape))
            for name, value in payload.items()
        },
    )
    monkeypatch.setattr(quantized, "_QUANT_GROUPS", groups)
    monkeypatch.setattr(quantized, "_NUM_HEADS", 1)
    monkeypatch.setattr(quantized, "_HEAD_DIM", 4)
    monkeypatch.setattr(
        quantized,
        "QUANTIZED_CHECKPOINT_IDENTITY",
        replace(
            quantized.QUANTIZED_CHECKPOINT_IDENTITY,
            size_bytes=path.stat().st_size,
            tensor_count=len(payload),
            quantized_weight_count=len(groups),
        ),
    )
    return path, lora_file, payload, lora, source_metadata


def _convrot_linear(x, weight, bias=None):
    """Tiny CPU oracle for the existing ConvRot BF16/W8A8 base contract."""
    assert weight.group_size == 4
    hadamard = (
        torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=torch.bfloat16,
        )
        / 2
    )
    rotated = (x.reshape(-1, 4) @ hadamard).reshape(x.shape)
    row_scale = (rotated.abs().amax(dim=-1, keepdim=True).float() / 127).clamp_min(1e-30)
    codes = (rotated / row_scale.bfloat16()).round().clamp(-128, 127).float()
    output = (codes @ torch.from_numpy(weight.qweight).float().T) * (
        row_scale * torch.from_numpy(weight.scale).reshape(1, -1)
    )
    if bias is not None:
        output = output + bias.float()
    return output.bfloat16()


def _int8_or_dense_linear(x, weight, bias=None):
    if isinstance(weight.base, quantized.ConvRotInt8Weight):
        base = _convrot_linear(x, weight.base, bias)
    else:
        base = F.linear(x, _tensor(weight.base), bias)
    return base + F.linear(F.linear(x, _tensor(weight.lora_a)), _tensor(weight.lora_b))


def _authored_int8_linear(x, payload, lora, module, bias=None):
    if payload[f"{module}.weight"].dtype == torch.int8:
        weight = quantized.ConvRotInt8Weight(
            payload[f"{module}.weight"].numpy(), payload[f"{module}.weight_scale"].numpy(), 4
        )
        base = _convrot_linear(x, weight, bias)
    else:
        base = F.linear(x, payload[f"{module}.weight"], bias)
    return base + F.linear(
        F.linear(x, lora[f"{module}.lora_A.weight"]), lora[f"{module}.lora_B.weight"]
    )


def test_int8_contract_keeps_all_fifty_unpruned_adaln_modules():
    assert quantized.CHECKPOINT_REVISION == "4cc1d817b6184899b41293954329f576cb5ae86b"
    assert quantized.CHECKPOINT_BYTES == 34_038_892_334
    assert len(quantized._PHYSICAL_SPECS) == 1035
    assert len(quantized._QUANT_GROUPS) == 250
    for index in range(50):
        module = f"blocks.{index}.adaln_proj.linear"
        assert quantized._PHYSICAL_SPECS[f"{module}.weight"].shape == (96768, 2688)
        assert quantized._QUANT_GROUPS[module] == 64
        assert checkpoint._LORA_SPECS[f"{module}.lora_B.weight"] == ("BF16", (96768, 16))


@pytest.mark.parametrize(
    "root,physical,is_quantized",
    [
        ("transformer_blocks.0", "blocks.0", True),
        ("token_refiner.refiner_blocks.0", "token_refiner.blocks.0", False),
    ],
)
def test_int8_source_qkv_retains_grouped_base_scales_and_original_lora_input(
    tmp_path, monkeypatch, root, physical, is_quantized
):
    path, lora_file, payload, lora, _metadata = _tiny_int8_checkpoints(tmp_path, monkeypatch)
    names = [f"{root}.attn.to_{name}.weight" for name in ("q", "k", "v")]
    loaded = load_selected_turbo_transformer_weights(path, lora_file, names, base_precision="int8")
    children = [loaded[name] for name in names]
    packed = pack_turbo_qkv(children)
    assert packed is children[0].packed_parent
    assert all(child.lora_a is packed.lora_a for child in children)
    assert isinstance(packed.base, quantized.ConvRotInt8Weight) == is_quantized
    module = f"{physical}.attn.qkv_proj"
    if is_quantized:
        np.testing.assert_array_equal(packed.base.qweight, payload[f"{module}.weight"].numpy())
        np.testing.assert_array_equal(packed.base.scale, payload[f"{module}.weight_scale"].numpy())
        assert packed.dtype == np.int8 and packed.shape == (12, 4)
        assert all(child.base.packed_parent is packed.base for child in children)
    x = torch.tensor([[0.3, -0.7, 1.4, 0.9], [0, 0, 0, 0]], dtype=torch.bfloat16)
    expected = _authored_int8_linear(x, payload, lora, module)
    torch.testing.assert_close(_int8_or_dense_linear(x, packed), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.cat([_int8_or_dense_linear(x, child) for child in children], dim=-1),
        expected,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "root,physical",
    [
        ("transformer_blocks.0", "blocks.0"),
        ("token_refiner.refiner_blocks.0", "token_refiner.blocks.0"),
    ],
)
def test_int8_source_swiglu_swaps_qweight_scale_and_b_not_a(tmp_path, monkeypatch, root, physical):
    path, lora_file, payload, lora, _metadata = _tiny_int8_checkpoints(tmp_path, monkeypatch)
    name = f"{root}.ff.net.0.proj.weight"
    weight = load_selected_turbo_transformer_weights(
        path, lora_file, [name], base_precision="int8"
    )[name]
    module = f"{physical}.mlp.fc1"
    if isinstance(weight.base, quantized.ConvRotInt8Weight):
        np.testing.assert_array_equal(
            weight.base.scale, torch.cat(payload[f"{module}.weight_scale"].chunk(2)[::-1]).numpy()
        )
    torch.testing.assert_close(
        _tensor(weight.lora_a), lora[f"{module}.lora_A.weight"], rtol=0, atol=0
    )
    x = torch.tensor([[0.3, -0.7, 1.4, 0.9]], dtype=torch.bfloat16)
    gate, value = _authored_int8_linear(x, payload, lora, module).chunk(2, dim=-1)
    actual_value, actual_gate = _int8_or_dense_linear(x, weight).chunk(2, dim=-1)
    torch.testing.assert_close(
        F.silu(actual_gate) * actual_value, F.silu(gate) * value, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "logical,physical,quantized_base",
    [
        ("transformer_blocks.0.adaln_proj.linear", "blocks.0.adaln_proj.linear", True),
        ("norm_out.linear", "final_layer.adaln_proj.linear", False),
    ],
)
def test_int8_source_adaln_and_dense_final_preserve_base_bias_then_lora(
    tmp_path, monkeypatch, logical, physical, quantized_base
):
    path, lora_file, payload, lora, _metadata = _tiny_int8_checkpoints(tmp_path, monkeypatch)
    names = [f"{logical}.weight", f"{logical}.bias"]
    loaded = load_selected_turbo_transformer_weights(path, lora_file, names, base_precision="int8")
    weight, bias = loaded[names[0]], _tensor(loaded[names[1]])
    assert isinstance(weight.base, quantized.ConvRotInt8Weight) == quantized_base
    x = F.silu(torch.tensor([[0.3, -0.7, 1.4, 0.9]], dtype=torch.bfloat16))
    expected = _authored_int8_linear(x, payload, lora, physical, payload[f"{physical}.bias"])
    torch.testing.assert_close(_int8_or_dense_linear(x, weight, bias), expected, rtol=0, atol=0)


def test_int8_validation_keeps_source_authentication_and_rejects_pruned_inventory(
    tmp_path, monkeypatch
):
    path, lora_file, payload, _lora, metadata = _tiny_int8_checkpoints(tmp_path, monkeypatch)
    receipt = validate_turbo_transformer_checkpoint(
        path, lora_file, workflow="t2va", base_precision="int8"
    )
    assert receipt["base_revision"] == quantized.CHECKPOINT_REVISION
    assert receipt["base_quantization"] == "int8_tensorwise_convrot"
    assert receipt["lora_strength"] == 1 and receipt["lora_merged"] is False
    assert not any("path" in key or "identity" in key for key in receipt)
    metadata.write_text("wrong-revision\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned revision"):
        validate_turbo_transformer_checkpoint(path, lora_file, base_precision="int8")
    metadata.write_text(f"{quantized.CHECKPOINT_REVISION}\nfixture\n0\n", encoding="utf-8")
    payload.pop("blocks.0.adaln_proj.linear.weight")
    save_file(payload, path, metadata={"config": json.dumps(checkpoint._EXPECTED_CONFIG)})
    monkeypatch.setattr(
        quantized,
        "QUANTIZED_CHECKPOINT_IDENTITY",
        replace(quantized.QUANTIZED_CHECKPOINT_IDENTITY, size_bytes=path.stat().st_size),
    )
    with pytest.raises(ValueError, match="tensor inventory mismatch"):
        validate_turbo_transformer_checkpoint(path, lora_file, base_precision="int8")


def test_int8_loader_reads_only_requested_weight_scale_and_lora_and_retains_fp32(
    tmp_path, monkeypatch
):
    path, lora_file, payload, _lora, _metadata = _tiny_int8_checkpoints(tmp_path, monkeypatch)
    import safetensors

    original = safetensors.safe_open
    reads = []

    class Reader:
        def __init__(self, *args, **kwargs):
            self.reader = original(*args, **kwargs)

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def get_tensor(self, name):
            reads.append(name)
            return self.reader.get_tensor(name)

    monkeypatch.setattr(safetensors, "safe_open", Reader)
    name = "transformer_blocks.0.attn.to_k.weight"
    loaded = load_selected_turbo_transformer_weights(
        path, lora_file, [name, "proj_in.weight"], base_precision="int8"
    )
    assert set(reads) == {
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.qkv_proj.weight_scale",
        "blocks.0.attn.qkv_proj.lora_A.weight",
        "blocks.0.attn.qkv_proj.lora_B.weight",
        "video_patch_proj.weight",
    }
    assert len(reads) == 5
    assert loaded[name].row_slice == (4, 8)
    assert loaded["proj_in.weight"].dtype == np.float32
    np.testing.assert_array_equal(
        loaded["proj_in.weight"], payload["video_patch_proj.weight"].numpy()
    )


def test_int8_qkv_rejects_detached_scale_views_and_reordered_children(tmp_path, monkeypatch):
    path, lora_file, _payload, _lora, _metadata = _tiny_int8_checkpoints(tmp_path, monkeypatch)
    names = [f"transformer_blocks.0.attn.to_{name}.weight" for name in ("q", "k", "v")]
    children = list(
        load_selected_turbo_transformer_weights(
            path, lora_file, names, base_precision="int8"
        ).values()
    )
    with pytest.raises(ValueError, match="Q,K,V order"):
        pack_turbo_qkv([children[1], children[0], children[2]])
    child = children[0]
    detached = replace(child.base, scale=child.base.scale.copy())
    with pytest.raises(ValueError, match="exact base/B row views"):
        replace(child, base=detached)


def test_independent_int8_qkv_pack_preserves_scales_and_rejects_different_rotations():
    a = _array(np.arange(32).reshape(2, 16) / 31)
    weights = [
        TurboLoraWeight(
            quantized.ConvRotInt8Weight(
                np.full((4, 16), index, np.int8), np.full((4, 1), index + 1, np.float32), 4
            ),
            a,
            _array(np.full((4, 2), index / 10)),
        )
        for index in range(3)
    ]
    packed = pack_turbo_qkv(weights)
    assert packed.base.is_full_fused_qkv and packed.lora_a is a
    np.testing.assert_array_equal(
        packed.base.qweight, np.concatenate([w.base.qweight for w in weights])
    )
    np.testing.assert_array_equal(
        packed.base.scale, np.concatenate([w.base.scale for w in weights])
    )
    weights[1] = replace(weights[1], base=replace(weights[1].base, group_size=16))
    with pytest.raises(ValueError, match="ConvRot group size"):
        pack_turbo_qkv(weights)


@pytest.mark.parametrize("precision", ["fp8", "INT8", "int8_pruned", None])
def test_turbo_base_precision_requires_explicit_supported_storage(precision):
    with pytest.raises(ValueError, match="base_precision"):
        validate_turbo_transformer_checkpoint("unused", "unused", base_precision=precision)


def test_bf16_ref_uses_distinct_source_and_preserves_same_unmerged_layout(tmp_path, monkeypatch):
    _fl_file, lora_file, base, _lora = _tiny_checkpoints(tmp_path, monkeypatch)
    root = tmp_path / "reference"
    path = root / checkpoint.REF2VA_BASE_FILENAME
    path.parent.mkdir(parents=True)
    base["blocks.0.attn.qkv_proj.weight"] += 1
    save_file(base, path, metadata={"config": json.dumps(checkpoint._EXPECTED_CONFIG)})
    monkeypatch.setattr(checkpoint, "REF2VA_BASE_BYTES", path.stat().st_size)
    metadata = root / ".cache/huggingface/download" / f"{checkpoint.REF2VA_BASE_FILENAME}.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        f"{checkpoint.BASE_REVISION}\n{checkpoint.REF2VA_BASE_ETAG}\n0\n", encoding="utf-8"
    )
    receipt = validate_turbo_transformer_checkpoint(path, lora_file, workflow="ref2va")
    assert receipt["base_filename"] == checkpoint.REF2VA_BASE_FILENAME
    assert receipt["base_workflow"] == "ref2va"
    assert receipt["adapter_compatibility"] == "experimental_not_author_certified"
    assert receipt["lora_merged"] is False
    names = [f"transformer_blocks.0.attn.to_{suffix}.weight" for suffix in ("q", "k", "v")]
    weights = load_selected_turbo_transformer_weights(path, lora_file, names, workflow="ref2va")
    packed = pack_turbo_qkv([weights[name] for name in names])
    torch.testing.assert_close(_tensor(packed.base), base["blocks.0.attn.qkv_proj.weight"])
    metadata.write_text(f"{checkpoint.BASE_REVISION}\nwrong-file\n0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned revision and file"):
        validate_turbo_transformer_checkpoint(path, lora_file, workflow="ref2va")


def test_int8_ref_selects_distinct_existing_source_validator(tmp_path, monkeypatch):
    fl_path, lora_file, payload, _lora, _metadata = _tiny_int8_checkpoints(tmp_path, monkeypatch)
    ref_path = fl_path.parent / Path(quantized.REF2VA_CHECKPOINT_FILENAME).name
    save_file(payload, ref_path, metadata={"config": json.dumps(checkpoint._EXPECTED_CONFIG)})
    metadata = (
        ref_path.parents[1]
        / ".cache/huggingface/download"
        / f"{quantized.REF2VA_CHECKPOINT_FILENAME}.metadata"
    )
    metadata.write_text(f"{quantized.CHECKPOINT_REVISION}\nfixture\n0\n", encoding="utf-8")
    monkeypatch.setattr(
        quantized,
        "QUANTIZED_REF2VA_CHECKPOINT_IDENTITY",
        replace(
            quantized.QUANTIZED_REF2VA_CHECKPOINT_IDENTITY,
            size_bytes=ref_path.stat().st_size,
            tensor_count=len(payload),
            quantized_weight_count=len(quantized._QUANT_GROUPS),
        ),
    )
    receipt = validate_turbo_transformer_checkpoint(
        ref_path, lora_file, workflow="ref2va", base_precision="int8"
    )
    assert receipt["base_filename"] == quantized.REF2VA_CHECKPOINT_FILENAME
    assert receipt["base_workflow"] == "ref2va"
    name = "transformer_blocks.0.attn.to_q.weight"
    weight = load_selected_turbo_transformer_weights(
        ref_path, lora_file, [name], workflow="ref2va", base_precision="int8"
    )[name]
    assert isinstance(weight.base, quantized.ConvRotInt8Weight)
    with pytest.raises(ValueError, match="filename must be minimax_h3_ref2va"):
        validate_turbo_transformer_checkpoint(
            fl_path, lora_file, workflow="ref2va", base_precision="int8"
        )


def test_turbo_rejects_unknown_workflow_before_checkpoint_io():
    with pytest.raises(ValueError, match="workflow"):
        validate_turbo_transformer_checkpoint("unused", "unused", workflow="image")
