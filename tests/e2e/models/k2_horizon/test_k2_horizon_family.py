# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families import find_plugin
from tensorrt_model_connect.families.k2_horizon.weights import (
    WeightDict,
    _copy_to_numpy,
    _expected_tensor_names,
    _target_np_dtype,
    _validate_checkpoint_tensor_names,
)
from tensorrt_model_connect.families.k2_horizon.config import validate_config


plugin_module = importlib.import_module("tensorrt_model_connect.families.k2_horizon.plugin")
debug_runner_module = importlib.import_module(
    "tensorrt_model_connect.families.k2_horizon.debug_runner"
)
plugin = plugin_module.plugin


def _config(**overrides) -> ModelConfig:
    raw = {
        "model_type": "k2_horizon",
        "architectures": ["K2HorizonForCausalLM"],
        "vocab_size": 64,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000_000.0,
        "max_position_embeddings": 4096,
        "hidden_act": "silu",
        "layernorm_num_groups": 4,
        "attention_bias": False,
        "mlp_bias": False,
        "query_key_norm": False,
        "attention_gate_func": None,
        "use_sliding_window": False,
        "num_experts": 0,
        "mova_num_experts": 0,
        "tie_word_embeddings": False,
    }
    raw.update(overrides)
    return ModelConfig.from_json(json.dumps(raw))


def test_registry_resolves_k2_horizon_to_its_own_family() -> None:
    selected = find_plugin(_config())

    assert selected is not None
    assert selected.name == "k2_horizon"
    assert selected.runtime_strategy == "k2_horizon_decoder_kv_cache"


def test_config_owns_the_grouped_rmsnorm_contract() -> None:
    resolved = validate_config(_config())

    assert resolved.layernorm_num_groups == 4
    assert resolved.head_dim == 128
    assert resolved.attention_size == 512
    assert resolved.kv_attention_size == 256
    assert plugin.default_build_precision(_config()) == "bf16"
    assert plugin.default_max_cache_length(_config()) == 256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("architectures", ["Qwen3ForCausalLM"]),
        ("hidden_act", "gelu"),
        ("attention_bias", True),
        ("query_key_norm", True),
        ("attention_gate_func", "silu"),
        ("use_sliding_window", True),
        ("dynamic_kv_cache", True),
        ("quantization_config", {"quant_method": "gptq"}),
        ("num_experts", 8),
        ("mova_num_experts", 8),
        ("rope_head_dim", 64),
        ("layernorm_num_groups", 3),
        ("layernorm_num_groups", 2),
    ],
)
def test_config_rejects_unqualified_graph_variants(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        validate_config(_config(**{field: value}))


def test_plugin_rejects_unqualified_build_modes() -> None:
    config = _config()
    weights = WeightDict()

    with pytest.raises(NotImplementedError, match="quantized"):
        plugin.build_engine(config, weights, 256, quant_ctx=object())
    with pytest.raises(NotImplementedError, match="debug"):
        plugin.build_engine(config, weights, 256, debug_layer_outputs=True)
    with pytest.raises(NotImplementedError, match="tensor-parallel"):
        plugin.build_engine(
            config,
            weights,
            256,
            parallel_config=SimpleNamespace(enabled=True),
        )


def test_plugin_rejects_a_false_dual_profile_claim_before_loading_weights() -> None:
    config = _config()
    config.raw["_decoder_engine_layout"] = "dual_profile"

    with pytest.raises(NotImplementedError, match="dual_profile"):
        plugin.validate_build_request(config)


def test_plugin_delegates_weight_loading_with_the_validated_source_config(
    monkeypatch, tmp_path: Path
) -> None:
    config = _config()
    captured = {}

    def fake_load(model_dir, delegated_config, *, precision):
        captured.update(
            model_dir=model_dir,
            config=delegated_config,
            precision=precision,
        )
        return WeightDict()

    monkeypatch.setattr(plugin_module, "load_standard_weights", fake_load)

    result = plugin.load_weights(str(tmp_path), config)

    assert isinstance(result, WeightDict)
    assert captured == {
        "model_dir": str(tmp_path),
        "config": config,
        "precision": "bf16",
    }


def test_checkpoint_tensor_inventory_fails_closed_on_architecture_drift() -> None:
    expected = _expected_tensor_names(2)
    readers = SimpleNamespace(
        tensor_map={name: object() for name in expected | {"model.layers.0.self_attn.q_proj.bias"}}
    )

    with pytest.raises(ValueError, match="unexpected=.*q_proj.bias"):
        _validate_checkpoint_tensor_names(readers, 2)


def test_bf16_checkpoint_storage_preserves_exact_bits_and_transpose() -> None:
    bits = np.array(
        [
            [0x0001, 0x3F80, 0x7F7F],
            [0x8001, 0xBF80, 0xFF7F],
        ],
        dtype=np.uint16,
    )

    copied = _copy_to_numpy(
        bits,
        _target_np_dtype("bf16"),
        transpose_name="projection",
    )

    assert copied.dtype == np.uint16
    assert copied.flags.c_contiguous
    np.testing.assert_array_equal(copied, bits.T)


def test_tensor_rt_constant_buffers_remain_alive_through_serialization() -> None:
    model_path = Path(__file__).parents[4] / "python" / "tensorrt_model_connect" / "families"
    model_path = model_path / "k2_horizon" / "model" / "model.py"
    source = model_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def has_array_call(function_name: str, owner: str, method: str) -> bool:
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == owner
            and node.func.attr == method
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "array"
            for node in ast.walk(functions[function_name])
        )

    assert has_array_call("_constant", "keepalive", "append")
    assert has_array_call("_work_constant", "constant_keepalive", "append")

    constant_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"_constant", "_work_constant", "_matmul"}:
            continue
        constant_calls.append(node)
        if node.func.id == "_constant":
            assert len(node.args) >= 4
        else:
            assert "constant_keepalive" in {keyword.arg for keyword in node.keywords}

    assert constant_calls
    serialization_guards = [
        node
        for node in ast.walk(functions["build_engine"])
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.Attribute) and child.attr == "build_serialized_network"
            for statement in node.body
            for child in ast.walk(statement)
        )
    ]
    assert len(serialization_guards) == 1
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "constant_keepalive"
        and node.func.attr == "clear"
        for statement in serialization_guards[0].finalbody
        for node in ast.walk(statement)
    )


def test_reference_profile_checks_survive_python_optimization() -> None:
    verifier = Path(__file__).parents[4] / "python" / "tensorrt_model_connect" / "families"
    verifier = verifier / "k2_horizon" / "python_profile_verify.py"
    tree = ast.parse(verifier.read_text(encoding="utf-8"))

    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def test_debug_runner_releases_partially_initialized_resources(monkeypatch) -> None:
    class FakeCuda:
        def __init__(self):
            self.freed = []
            self.destroyed = []

        def cudaFree(self, pointer):
            self.freed.append(pointer)

        def cudaStreamDestroy(self, stream):
            self.destroyed.append(stream)

    fake_cuda = FakeCuda()
    partial = {}

    def fail_after_allocations(self, *_args):
        partial["runner"] = self
        self.stream = 7
        self._device_scalars["token_id"] = 11
        self._cache_k.append(12)
        self._device_logits = 13
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(debug_runner_module, "cudart", fake_cuda)
    monkeypatch.setattr(
        debug_runner_module.K2HorizonTrtRunner,
        "_initialize",
        fail_after_allocations,
    )

    with pytest.raises(RuntimeError, match="initialization failed"):
        debug_runner_module.K2HorizonTrtRunner(b"plan", 8, 1)

    runner = partial["runner"]
    assert fake_cuda.freed == [11, 12, 13]
    assert fake_cuda.destroyed == [7]
    runner.close()
    assert fake_cuda.freed == [11, 12, 13]
    assert fake_cuda.destroyed == [7]


def test_plugin_records_the_native_kv_contract(monkeypatch) -> None:
    config = _config()

    monkeypatch.setattr(plugin_module, "_build_engine", lambda *_args, **_kwargs: b"plan")

    assert plugin.build_engine(config, WeightDict(), 256) == b"plan"
    assert plugin.get_bundle_config_overrides(config) == {
        "native_kv_cache": True,
        "native_kv_contract_version": 1,
    }
