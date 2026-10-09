# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import ml_dtypes
import pytest
from safetensors.numpy import save_file

from families.modernbert.config import ModelConfig
from families.modernbert.weights import WeightDict, _has_tensor, _load_tensor, _open_safetensors


@pytest.fixture(scope="module")
def load_weights():
    # Note (Jiaxin Deng): Compile the loader alone to avoid TensorRT graph imports in CPU tests.
    path = Path(__file__).resolve().parents[1] / "model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_ModernBertModel"
    )
    loader = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
    )
    namespace = {
        "Path": Path,
        "np": np,
        "ModelConfig": ModelConfig,
        "WeightDict": WeightDict,
        "_has_tensor": _has_tensor,
        "_load_tensor": _load_tensor,
        "_open_safetensors": _open_safetensors,
    }
    exec(compile(ast.Module(body=[loader], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["load_weights"]


def _checkpoint(root: str, dtype, with_head: bool):
    config = ModelConfig(
        vocab_size=8, hidden_size=4, intermediate_size=6, num_hidden_layers=2, num_attention_heads=2
    )
    tensors = {}
    expected = {}

    def weight(hf_name, logical_name, shape, offset, *, transpose=False):
        value = (np.arange(np.prod(shape)).reshape(shape) + offset).astype(dtype)
        tensors[root + hf_name] = value
        expected[logical_name] = (value.T if transpose else value).astype(np.float32)
        return value

    weight("embeddings.tok_embeddings.weight", "embedding", (8, 4), 1)
    weight("embeddings.norm.weight", "embed_norm", (4,), 41)
    weight("final_norm.weight", "final_norm", (4,), 51)
    for layer in range(2):
        hf = f"layers.{layer}"
        logical = f"layer.{layer}"
        offset = 100 * (layer + 1)
        if layer == 1:
            weight(f"{hf}.attn_norm.weight", f"{logical}.attn_norm", (4,), offset)
        projections = []
        for index, name in enumerate(("q", "k", "v")):
            value = (np.arange(16).reshape(4, 4) + offset + index * 20).astype(dtype)
            projections.append(value)
            expected[f"{logical}.w_{name}"] = value.T.astype(np.float32)
        tensors[f"{root}{hf}.attn.Wqkv.weight"] = np.concatenate(projections)
        weight(f"{hf}.attn.Wo.weight", f"{logical}.w_o", (4, 4), offset + 60, transpose=True)
        weight(f"{hf}.mlp_norm.weight", f"{logical}.mlp_norm", (4,), offset + 80)
        input_weight = (np.arange(24).reshape(6, 4) + offset + 90).astype(dtype)
        gate_weight = (np.arange(24).reshape(6, 4) + offset + 120).astype(dtype)
        tensors[f"{root}{hf}.mlp.Wi.weight"] = np.concatenate((input_weight, gate_weight))
        expected[f"{logical}.w_mlp_input"] = input_weight.T.astype(np.float32)
        expected[f"{logical}.w_mlp_gate"] = gate_weight.T.astype(np.float32)
        weight(f"{hf}.mlp.Wo.weight", f"{logical}.w_down", (4, 6), offset + 150, transpose=True)
    if with_head:
        for hf_name, logical, shape, transpose in (
            ("head.dense.weight", "head_dense_w", (4, 4), True),
            ("head.norm.weight", "head_norm", (4,), False),
            ("decoder.bias", "decoder_bias", (8,), False),
        ):
            value = (np.arange(np.prod(shape)).reshape(shape) + 400).astype(dtype)
            tensors[hf_name] = value
            expected[logical] = (value.T if transpose else value).astype(np.float32)
    return config, tensors, expected


def _save_checkpoint(path, tensors, sharded):
    if not sharded:
        save_file(tensors, str(path / "model.safetensors"))
        return
    weight_map = {}
    items = sorted(tensors.items())
    for index in range(2):
        filename = f"model-{index + 1:05d}-of-00002.safetensors"
        shard = dict(items[index::2])
        save_file(shard, str(path / filename))
        weight_map.update(dict.fromkeys(shard, filename))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )


@pytest.mark.parametrize("root", ["", "model."], ids=["bare", "wrapped"])
@pytest.mark.parametrize("sharded", [False, True], ids=["single", "indexed-shards"])
@pytest.mark.parametrize(
    "dtype", [np.float32, np.float16, ml_dtypes.bfloat16], ids=["fp32", "fp16", "bf16"]
)
@pytest.mark.parametrize("with_head", [False, True], ids=["encoder", "mlm-head"])
def test_loads_encoder_roots_and_preserves_projection_layout(
    tmp_path, load_weights, root, sharded, dtype, with_head
):
    config, tensors, expected = _checkpoint(root, dtype, with_head)
    _save_checkpoint(tmp_path, tensors, sharded)

    actual = load_weights(None, str(tmp_path), config)

    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value, err_msg=name)
        assert actual[name].dtype == np.float32
        assert actual[name].flags.c_contiguous


@pytest.mark.parametrize("sharded", [False, True], ids=["single", "indexed-shards"])
def test_missing_bare_projection_is_not_silently_accepted(tmp_path, load_weights, sharded):
    config, tensors, _ = _checkpoint("", np.float32, False)
    del tensors["layers.1.attn.Wqkv.weight"]
    _save_checkpoint(tmp_path, tensors, sharded)

    with pytest.raises(KeyError, match=r"Tensor not found: layers\.1\.attn\.Wqkv\.weight"):
        load_weights(None, str(tmp_path), config)
