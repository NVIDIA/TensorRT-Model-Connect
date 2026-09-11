# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from families.minimax_h3 import nvfp4_text_checkpoint as checkpoint


def _blocked(plain):
    """Independent index formula from Comfy's documented 128x4 scale tiles."""
    rows, cols = plain.shape
    padded_rows, padded_cols = ((rows + 127) // 128) * 128, ((cols + 3) // 4) * 4
    output = np.zeros(padded_rows * padded_cols, dtype=plain.dtype)
    for row in range(rows):
        for col in range(cols):
            index = ((row // 128) * (padded_cols // 4) + col // 4) * 512
            index += (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4
            output[index] = plain[row, col]
    return output.reshape(padded_rows, padded_cols)


def _tiny_checkpoint(tmp_path, monkeypatch, *, full_precision=True):
    base = "model.layers.0.mlp.down_proj"
    plain_scales = np.arange(1, 257, dtype=np.float32).reshape(128, 2).astype(ml_dtypes.float8_e4m3fn)
    arrays = {
        f"{base}.weight": np.tile(np.arange(16, dtype=np.uint8), (128, 1)),
        f"{base}.weight_scale": _blocked(plain_scales),
        f"{base}.weight_scale_2": np.asarray(0.0035109748132526875, dtype=np.float32),
        f"{base}.pre_quant_scale": np.linspace(.5, 1.5, 32).astype(ml_dtypes.bfloat16),
        f"{base}.comfy_quant": np.frombuffer(json.dumps({
            "format": "nvfp4", "full_precision_matrix_mult": full_precision,
        }).encode(), dtype=np.uint8),
        "model.embed_tokens.weight": np.arange(-128, 128, dtype=np.int8).reshape(8, 32),
        "model.embed_tokens.weight_scale": np.arange(1, 9, dtype=np.float32).reshape(8, 1) / 127,
        "model.embed_tokens.comfy_quant": np.frombuffer(b'{"format": "int8_tensorwise"}', np.uint8),
        "visual.patch_embed.proj.weight": np.arange(8).reshape(2, 4).astype(ml_dtypes.bfloat16),
    }
    dtype_names = {dtype: name for name, dtype in checkpoint._DTYPES.items()}
    header, payload = {}, bytearray()
    for name, value in arrays.items():
        header[name] = {"dtype": dtype_names[value.dtype], "shape": list(value.shape),
                        "data_offsets": [len(payload), len(payload) + value.nbytes]}
        payload.extend(value.tobytes())
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path = tmp_path / checkpoint.CHECKPOINT_FILENAME
    path.parent.mkdir(parents=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    metadata = tmp_path / ".cache/huggingface/download" / f"{checkpoint.CHECKPOINT_FILENAME}.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"{checkpoint.CHECKPOINT_REVISION}\netag\n0\n", encoding="utf-8")
    monkeypatch.setattr(checkpoint, "_PHYSICAL_SPECS", {
        name: (entry["dtype"], tuple(entry["shape"])) for name, entry in header.items()
    })
    monkeypatch.setattr(checkpoint, "QUANTIZED_TEXT_CHECKPOINT_IDENTITY", replace(
        checkpoint.QUANTIZED_TEXT_CHECKPOINT_IDENTITY,
        size_bytes=path.stat().st_size, tensor_count=len(header), quantized_weight_count=1,
    ))
    return path, arrays


def test_released_inventory_and_truthful_compute_metadata():
    specs = checkpoint._PHYSICAL_SPECS
    assert len(specs) == 2054
    assert Counter(dtype for dtype, _ in specs.values()) == {
        "U8": 701, "BF16": 651, "F32": 351, "F8_E4M3": 350, "I8": 1,
    }
    assert sum(name.endswith("pre_quant_scale") for name in specs) == 100
    assert not any(name.endswith("input_scale") for name in specs)
    metadata = checkpoint.QUANTIZED_TEXT_CHECKPOINT_IDENTITY.bundle_metadata()
    assert metadata["size_bytes"] == 15_687_142_551
    assert metadata["quantization"] == "nvfp4_awq"
    assert metadata["full_precision_matrix_mult"] is True
    assert metadata["engine_weights_dtype"] == metadata["matmul_dtype"] == "bfloat16"
    assert metadata["runtime_framework"] is None
    assert "source_file_identity" not in metadata


def test_inverse_swizzle_crosses_all_tile_boundaries():
    plain = np.arange(131 * 7, dtype=np.int32).reshape(131, 7)
    np.testing.assert_array_equal(checkpoint._unblock_scales(_blocked(plain), 131, 7), plain)


def test_nvfp4_decode_matches_cuda_reference_math_and_high_nibble_order():
    rng = np.random.default_rng(71)
    packed = rng.integers(0, 256, (131, 40), dtype=np.uint8)
    plain = rng.uniform(.02, 448, (131, 5)).astype(ml_dtypes.float8_e4m3fn)
    scale = np.asarray(0.0035109748132526875, np.float32)
    decoded = checkpoint._dequantize_nvfp4(packed, _blocked(plain), scale)
    # Independent scalar E2M1 interpretation, then CUDA's FP32 products.
    codes = np.stack((packed >> 4, packed & 15), axis=-1).reshape(131, 80)
    exponent = (codes >> 1) & 3
    mantissa = codes & 1
    magnitude = np.where(exponent == 0, mantissa * .5,
                         (1 + mantissa * .5) * np.exp2(exponent.astype(np.int32) - 1))
    value = (magnitude * np.where(codes & 8, -1, 1)).astype(np.float32)
    total = plain.astype(np.float32) * scale
    expected = (value * np.repeat(total, 16, axis=1)).astype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(decoded.view(np.uint16), expected.view(np.uint16))
    assert decoded.dtype == ml_dtypes.bfloat16
    assert decoded.flags.c_contiguous


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_nvfp4_rejects_invalid_global_scale(scale):
    with pytest.raises(ValueError, match="positive finite scalar"):
        checkpoint._dequantize_nvfp4(np.zeros((1, 8), np.uint8),
                                     np.ones((128, 4), dtype=ml_dtypes.float8_e4m3fn),
                                     np.asarray(scale, np.float32))


def test_selected_loader_preserves_awq_and_shared_vision(tmp_path, monkeypatch):
    path, arrays = _tiny_checkpoint(tmp_path, monkeypatch)
    identity = checkpoint.validate_quantized_text_checkpoint(path)
    assert identity.source_file_identity.size_bytes == path.stat().st_size
    logical = "model.language_model.layers.0.mlp.down_proj.weight"
    names = [logical, "model.language_model.embed_tokens.weight", "model.visual.patch_embed.proj.weight"]
    loaded = checkpoint.load_selected_quantized_text_weights(path, names)
    prescale = logical.removesuffix(".weight") + ".pre_quant_scale"
    assert set(loaded) == set(names) | {prescale}
    np.testing.assert_array_equal(loaded[prescale], arrays["model.layers.0.mlp.down_proj.pre_quant_scale"])
    np.testing.assert_array_equal(loaded[names[2]], arrays["visual.patch_embed.proj.weight"])
    expected_embedding = (arrays["model.embed_tokens.weight"].astype(np.float32)
                          * arrays["model.embed_tokens.weight_scale"]).astype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(loaded[names[1]].view(np.uint16), expected_embedding.view(np.uint16))
    assert loaded[logical].shape == (128, 32)
    assert all(value.dtype == ml_dtypes.bfloat16 for value in loaded.values())


def test_validator_rejects_changed_full_precision_marker(tmp_path, monkeypatch):
    path, _ = _tiny_checkpoint(tmp_path, monkeypatch, full_precision=False)
    with pytest.raises(ValueError, match="comfy_quant marker"):
        checkpoint.validate_quantized_text_checkpoint(path)


def test_loader_rejects_unknown_or_duplicate_names(tmp_path, monkeypatch):
    path, _ = _tiny_checkpoint(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="duplicates"):
        checkpoint.load_selected_quantized_text_weights(path, ["x", "x"])
    with pytest.raises(ValueError, match="logical tensor"):
        checkpoint.load_selected_quantized_text_weights(path, ["model.layers.0.mlp.down_proj.weight"])


def test_validator_rejects_changed_revision(tmp_path, monkeypatch):
    path, _ = _tiny_checkpoint(tmp_path, monkeypatch)
    metadata = tmp_path / ".cache/huggingface/download" / f"{checkpoint.CHECKPOINT_FILENAME}.metadata"
    metadata.write_text("wrong\netag\n0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned Hugging Face revision"):
        checkpoint.validate_quantized_text_checkpoint(path)


def test_language_graph_applies_awq_before_linear_at_activation_dtype():
    # Exercise the real graph helper with CPU tensors; importing TRT or a GPU
    # is unnecessary for this topology/rounding-order regression.
    source = Path(checkpoint.__file__).with_name("multimodal_text_encoder_builder.py")
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_linear")
    namespace = {"np": np, "trt": SimpleNamespace(ElementWiseOperation=SimpleNamespace(PROD="prod"))}
    calls = []
    namespace["op"] = SimpleNamespace(
        weight_constant=lambda network, value: value,
        cast=lambda network, value, dtype: value.astype(dtype),
        linear=lambda network, value, weight: calls.append((value.copy(), weight)) or value,
    )
    network = SimpleNamespace(add_elementwise=lambda x, y, kind:
                              SimpleNamespace(get_output=lambda index: (x * y).astype(x.dtype)))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    hidden = np.asarray([[.13, .37]], dtype=ml_dtypes.bfloat16)
    scale = np.asarray([1.13, 1.37], dtype=ml_dtypes.bfloat16)
    weights = {"p.weight": np.ones((2, 2), dtype=ml_dtypes.bfloat16), "p.pre_quant_scale": scale}
    namespace["_linear"](network, hidden, weights, "p")
    expected = (hidden.astype(np.float32) * scale.astype(np.float32)).astype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(calls[0][0].view(np.uint16), expected.view(np.uint16))
    assert calls[0][1] is weights["p.weight"]
