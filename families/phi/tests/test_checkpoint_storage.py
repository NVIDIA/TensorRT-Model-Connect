# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import save_file


FAMILY = Path(__file__).resolve().parents[1]


def _loader():
    spec = importlib.util.spec_from_file_location(
        "_phi_checkpoint_storage", FAMILY / "checkpoint_mapper.py"
    )
    mapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapper)
    tree = ast.parse((FAMILY / "model.py").read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_PhiModel"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
    )
    namespace = {**vars(mapper), "np": np, "Path": Path}
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(FAMILY / "model.py"), "exec"), namespace)
    return namespace["load_weights"]


def _checkpoint(tmp_path, dtype, tied):
    hidden, mlp, vocab = 8, 16, 12
    rng = np.random.default_rng(5)
    shapes = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "model.layers.0.input_layernorm.weight": (hidden,),
        "model.layers.0.post_attention_layernorm.weight": (hidden,),
        "model.layers.0.self_attn.qkv_proj.weight": (16, hidden),
        "model.layers.0.self_attn.o_proj.weight": (hidden, hidden),
        "model.layers.0.mlp.gate_up_proj.weight": (2 * mlp, hidden),
        "model.layers.0.mlp.down_proj.weight": (hidden, mlp),
    }
    if not tied:
        shapes["lm_head.weight"] = (vocab, hidden)
    tensors = {name: rng.normal(size=shape).astype(dtype) for name, shape in shapes.items()}
    save_file(tensors, tmp_path / "model.safetensors")
    config = SimpleNamespace(
        hidden_size=hidden,
        vocab_size=vocab,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    return config


@pytest.mark.parametrize("checkpoint_dtype", [np.float16, np.float32, ml_dtypes.bfloat16])
@pytest.mark.parametrize("tied", [False, True])
def test_fp16_projection_storage_matches_graph_rounding(tmp_path, checkpoint_dtype, tied):
    config = _checkpoint(tmp_path, checkpoint_dtype, tied)
    load = _loader()
    baseline = load(None, str(tmp_path), config)
    compact = load(None, str(tmp_path), config, precision="fp16")
    projections = [f"layer.0.w_{name}" for name in ("q", "k", "v", "o", "gate", "up", "down")]
    for name in projections:
        actual = compact[name]
        assert actual.dtype == np.float16
        assert actual.flags.c_contiguous
        np.testing.assert_array_equal(
            actual.view(np.uint16), baseline[name].astype(np.float16).view(np.uint16)
        )
        assert actual.nbytes * 2 == baseline[name].nbytes
    for name in set(baseline) - set(projections):
        if isinstance(baseline[name], np.ndarray):
            assert compact[name].dtype == np.float32
            np.testing.assert_array_equal(compact[name], baseline[name])
        else:
            assert compact[name] == baseline[name]
    assert sum(v.nbytes for v in compact.values() if isinstance(v, np.ndarray)) < sum(
        v.nbytes for v in baseline.values() if isinstance(v, np.ndarray)
    )


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_non_fp16_storage_preserves_fp32_checkpoint_values(tmp_path, precision):
    config = _checkpoint(tmp_path, np.float32, False)
    load = _loader()
    baseline = load(None, str(tmp_path), config)
    actual = load(None, str(tmp_path), config, precision=precision)
    for name, expected in baseline.items():
        if isinstance(expected, np.ndarray):
            assert actual[name].dtype == np.float32
            np.testing.assert_array_equal(actual[name], expected)
        else:
            assert actual[name] == expected


def test_fp16_projection_rounding_covers_extreme_checkpoint_values(tmp_path):
    config = _checkpoint(tmp_path, np.float32, False)
    from safetensors.numpy import load_file

    tensors = load_file(tmp_path / "model.safetensors")
    tensors["model.layers.0.mlp.gate_up_proj.weight"].flat[:8] = [
        0.0,
        -0.0,
        2**-24,
        -(2**-24),
        65504.0,
        70000.0,
        -70000.0,
        1.0001,
    ]
    tensors["lm_head.weight"].flat[0] = 1.0001
    save_file(tensors, tmp_path / "model.safetensors")
    load = _loader()
    baseline = load(None, str(tmp_path), config)
    with np.errstate(over="ignore"):
        compact = load(None, str(tmp_path), config, precision="fp16")
        expected = baseline["layer.0.w_gate"].astype(np.float16)
    np.testing.assert_array_equal(
        compact["layer.0.w_gate"].view(np.uint16), expected.view(np.uint16)
    )
    assert compact["w_out"][0, 0] == np.float32(1.0001)
    assert compact["w_out"][0, 0] != np.float16(1.0001)


def test_build_passes_validated_precision_to_checkpoint_loader(tmp_path):
    tree = ast.parse((FAMILY / "model.py").read_text(encoding="utf-8"))
    build = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build"
    )
    config = SimpleNamespace(model_type="phi3", max_position_embeddings=256, raw={})
    seen = []

    class StopAfterLoad(Exception):
        pass

    class Model:
        def load_weights(self, model_dir, config, **kwargs):
            seen.append(kwargs)
            raise StopAfterLoad

    namespace = {
        "Path": Path,
        "ModelConfig": SimpleNamespace(from_dir=lambda path: config),
        "ParallelConfig": lambda **kwargs: SimpleNamespace(validate=lambda: None),
        "_PhiModel": Model,
    }
    positive_int = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_positive_int"
    )
    module = ast.Module(body=[positive_int, build], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(FAMILY / "model.py"), "exec"), namespace)
    request = SimpleNamespace(
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="text_generation",
        model_dir=tmp_path,
        precision="FP16",
        max_sequence_length=128,
        quantization=None,
        fp32_layers=None,
        mtp_seq_len=None,
        tensor_parallel_size=1,
    )
    with pytest.raises(StopAfterLoad):
        namespace["build"](request, None)
    assert seen == [{"precision": "fp16"}]
