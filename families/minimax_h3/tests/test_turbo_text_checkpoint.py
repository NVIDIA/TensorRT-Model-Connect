# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from families.minimax_h3 import turbo_text_checkpoint as checkpoint


FAMILY = Path(checkpoint.__file__).parent


def _source_function(filename, function_name, namespace):
    """Exercise pure graph helpers without importing TensorRT or touching CUDA."""
    path = FAMILY / filename
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function_name]


def _tiny_checkpoint(tmp_path, monkeypatch):
    values = {
        "model.embed_tokens.weight": torch.arange(24).reshape(6, 4).bfloat16(),
        "model.layers.49.self_attn.q_proj.weight": (torch.arange(16).reshape(4, 4) / 9).bfloat16(),
        "visual.deepstack_merger_list.0.norm.weight": torch.tensor(
            [1.01, -0.003, 2.5, 1.0]
        ).bfloat16(),
    }
    path = tmp_path / "qwen3vl_32b_minimax_h3_bf16.safetensors"
    metadata = {
        "minimax_h3_te": json.dumps(
            {"num_hidden_layers": 50, "output": "unnormalized_hidden_after_layer_50"}
        )
    }
    save_file(values, path, metadata=metadata)
    monkeypatch.setattr(
        checkpoint,
        "_PHYSICAL_SPECS",
        {name: ("BF16", tuple(value.shape)) for name, value in values.items()},
    )
    monkeypatch.setattr(checkpoint, "CHECKPOINT_BYTES", path.stat().st_size)
    return path, values, metadata


def test_full_bf16_inventory_matches_both_native_builder_partitions():
    language = _source_function(
        "multimodal_text_encoder_builder.py", "checkpoint_keys", {"NUM_LAYERS": 50}
    )()
    standalone = _source_function(
        "text_encoder_builder.py", "checkpoint_keys", {"NUM_LAYERS": 50}
    )()
    vision = _source_function("multimodal_vision_builder.py", "checkpoint_keys", {"DEPTH": 27})()
    assert language == standalone
    assert len(language) == 551 and len(vision) == 351
    physical = {checkpoint._physical_name(name) for name in (*language, *vision)}
    assert physical == set(checkpoint._PHYSICAL_SPECS)
    assert Counter(dtype for dtype, _shape in checkpoint._PHYSICAL_SPECS.values()) == {"BF16": 902}
    assert checkpoint._PHYSICAL_SPECS["model.layers.49.input_layernorm.weight"] == ("BF16", (5120,))
    assert not any(".layers.50." in name or name.endswith(".pre_quant_scale") for name in physical)
    assert "model.norm.weight" not in physical and "lm_head.weight" not in physical


def test_selective_language_and_vision_renaming_is_bit_preserving(tmp_path, monkeypatch):
    path, values, _metadata = _tiny_checkpoint(tmp_path, monkeypatch)
    names = [
        "model.language_model.layers.49.self_attn.q_proj.weight",
        "model.visual.deepstack_merger_list.0.norm.weight",
    ]
    result = checkpoint.load_selected_turbo_text_weights(path, names)
    assert set(result) == set(names)
    for logical, loaded in result.items():
        physical = checkpoint._physical_name(logical)
        assert loaded.dtype == np.dtype(ml_dtypes.bfloat16)
        np.testing.assert_array_equal(
            loaded.view(np.uint16), values[physical].view(torch.uint16).numpy()
        )


def test_single_requested_payload_does_not_load_vision_or_other_layers(tmp_path, monkeypatch):
    path, _values, _metadata = _tiny_checkpoint(tmp_path, monkeypatch)
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
    result = checkpoint.load_selected_turbo_text_weights(
        path, ["model.language_model.embed_tokens.weight"]
    )
    assert reads == ["model.embed_tokens.weight"]
    assert len(result) == 1


def test_metadata_exposes_raw_prompt_and_fp32_compute_not_bf16_compute(tmp_path, monkeypatch):
    path, _values, _metadata = _tiny_checkpoint(tmp_path, monkeypatch)
    metadata = checkpoint.validate_turbo_text_checkpoint(path)
    contract = metadata["conditioning"]
    assert contract == {
        "num_hidden_layers": 50,
        "hidden_size": 5120,
        "presentation": "raw_prompt",
        "add_special_tokens": False,
        "chat_template": False,
        "final_normalization": False,
        "language_output_head": False,
        "compute_dtype": "float32",
        "conditioning_dtype": "bfloat16",
    }
    assert metadata["checkpoint_dtype"] == "bfloat16"
    assert metadata["quantization"] is None


@pytest.mark.parametrize(
    "bad",
    [
        "model.language_model.norm.weight",
        "model.language_model.layers.50.self_attn.q_proj.weight",
        "lm_head.weight",
        "model.embed_tokens.weight",
    ],
)
def test_rejects_extra_layers_final_norm_head_and_untranslated_names(tmp_path, monkeypatch, bad):
    path, _values, _metadata = _tiny_checkpoint(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Unsupported"):
        checkpoint.load_selected_turbo_text_weights(path, [bad])
    with pytest.raises(ValueError, match="duplicate"):
        checkpoint.load_selected_turbo_text_weights(
            path, ["model.language_model.embed_tokens.weight"] * 2
        )


@pytest.mark.parametrize("fault", ["precision", "extra_layer", "metadata"])
def test_header_rejects_quantization_or_wrong_layer_contract(tmp_path, monkeypatch, fault):
    path, values, metadata = _tiny_checkpoint(tmp_path, monkeypatch)
    if fault == "precision":
        values["model.embed_tokens.weight"] = values["model.embed_tokens.weight"].float()
    elif fault == "extra_layer":
        values["model.layers.50.input_layernorm.weight"] = torch.ones(4).bfloat16()
    else:
        metadata["minimax_h3_te"] = json.dumps({"num_hidden_layers": 64, "output": "normalized"})
    save_file(values, path, metadata=metadata)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_BYTES", path.stat().st_size)
    with pytest.raises(ValueError):
        checkpoint.validate_turbo_text_checkpoint(path)


@pytest.mark.parametrize(
    "filename", ["text_encoder_builder.py", "multimodal_text_encoder_builder.py"]
)
def test_linear_opt_in_changes_compute_dtype_without_touching_weights(filename):
    calls = []
    op = SimpleNamespace(linear=lambda *args, **kwargs: calls.append((args, kwargs)))
    namespace = {"op": op, "trt": SimpleNamespace(float32="float32", bfloat16="bfloat16")}
    linear = _source_function(filename, "_linear", namespace)
    weight = np.ones((2, 2), dtype=ml_dtypes.bfloat16)
    weights = {"layer.weight": weight}
    linear(None, "hidden", weights, "layer")
    linear(None, "hidden", weights, "layer", turbo_fp32=True)
    assert calls[0][1] == {}
    assert calls[1][1]["compute_dtype"] == "float32"
    assert calls[0][0][2] is weight and calls[1][0][2] is weight


class _Tensor:
    def __init__(self, value, dtype="float32"):
        self.value = np.asarray(value, dtype=np.float32)
        self.dtype = dtype


class _Layer:
    def __init__(self, tensor):
        self.tensor = tensor
        self.reshape_dims = None

    def get_output(self, index):
        if self.reshape_dims is not None:
            return _Tensor(self.tensor.value.reshape(self.reshape_dims), self.tensor.dtype)
        return self.tensor


class _Network:
    def add_shuffle(self, tensor):
        return _Layer(tensor)

    def add_elementwise(self, left, right, operation):
        value = left.value * right.value if operation == "prod" else left.value + right.value
        return _Layer(_Tensor(value, left.dtype))

    def add_unary(self, tensor, operation):
        return _Layer(
            _Tensor((np.cos if operation == "cos" else np.sin)(tensor.value), tensor.dtype)
        )


def _rope_namespace():
    def cast(_network, tensor, dtype):
        values = (
            tensor.value.astype(ml_dtypes.bfloat16).astype(np.float32)
            if dtype == "bfloat16"
            else tensor.value
        )
        return _Tensor(values, dtype)

    def sliced(_network, tensor, starts, sizes):
        slices = tuple(
            slice(start, None if size is None else start + size)
            for start, size in zip(starts, sizes)
        )
        return _Tensor(tensor.value[slices], tensor.dtype)

    tree = ast.parse((FAMILY / "multimodal_text_encoder_builder.py").read_text())
    bits = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ROPE_INV_FREQ_BITS"
            for target in node.targets
        )
    )
    return {
        "np": np,
        "HEAD_DIM": 128,
        "ROPE_THETA": 5_000_000.0,
        "ROPE_INV_FREQ_BITS": bits,
        "op": SimpleNamespace(
            cast=cast, constant=lambda _network, array: _Tensor(array), dynamic_slice=sliced
        ),
        "trt": SimpleNamespace(
            float32="float32",
            bfloat16="bfloat16",
            ElementWiseOperation=SimpleNamespace(PROD="prod", SUM="sum"),
            UnaryOperation=SimpleNamespace(COS="cos", SIN="sin"),
        ),
    }


@pytest.mark.parametrize(
    "filename,function_name",
    [
        ("text_encoder_builder.py", "_rope_cache"),
        ("multimodal_text_encoder_builder.py", "_mrope_cache"),
    ],
)
def test_turbo_rope_retains_fp32_coefficients(filename, function_name):
    namespace = _rope_namespace()
    function = _source_function(filename, function_name, namespace)
    positions = np.asarray([0, 1, 127, 866], dtype=np.float32)
    if function_name == "_mrope_cache":
        positions = np.tile(positions, (3, 1))
    default_cos, _ = function(_Network(), _Tensor(positions))
    turbo_cos, _ = function(_Network(), _Tensor(positions), turbo_fp32=True)
    assert default_cos.dtype == "bfloat16" and turbo_cos.dtype == "float32"
    assert np.max(np.abs(default_cos.value - turbo_cos.value)) > 0.0
    inverse = np.asarray(namespace["ROPE_INV_FREQ_BITS"], dtype=np.uint32).view(np.float32)
    expected = np.cos(np.asarray([0, 1, 127, 866], np.float32)[:, None] * inverse)
    np.testing.assert_array_equal(turbo_cos.value.reshape(4, 64), expected)


@pytest.mark.parametrize(
    "filename,function_name",
    [
        ("text_encoder_builder.py", "build_text_encoder_engine"),
        ("multimodal_text_encoder_builder.py", "build_multimodal_text_encoder_engine"),
    ],
)
def test_builder_fp32_switch_is_opt_in_and_controls_every_linear(filename, function_name):
    tree = ast.parse((FAMILY / filename).read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    defaults = dict(
        zip((argument.arg for argument in function.args.kwonlyargs), function.args.kw_defaults)
    )
    assert (
        isinstance(defaults["turbo_fp32"], ast.Constant) and defaults["turbo_fp32"].value is False
    )
    linears = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_linear"
    ]
    assert len(linears) == 7
    assert all(
        any(
            keyword.arg == "turbo_fp32"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "turbo_fp32"
            for keyword in call.keywords
        )
        for call in linears
    )
    decomposable = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "decomposable"
            for target in node.targets
        )
    ]
    assert len(decomposable) == 1 and isinstance(decomposable[0].value, ast.Name)
    assert decomposable[0].value.id == "turbo_fp32"
    switches = [
        node
        for node in function.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "turbo_fp32"
    ]
    assert any(
        "clear_flag" in ast.unparse(node) and "TF32" in ast.unparse(node) for node in switches
    )
    rounding = [
        node
        for node in function.body
        if isinstance(node, ast.If)
        and "hidden = op.cast(network, hidden, trt.bfloat16)" in ast.unparse(node)
    ]
    # The runtime owns the final BF16 conditioning round. All Turbo text
    # stages must retain actual FP32 output storage at the engine boundary.
    assert not rounding
    assert "output = op.cast(network, hidden, trt.float32)" in ast.unparse(function)
