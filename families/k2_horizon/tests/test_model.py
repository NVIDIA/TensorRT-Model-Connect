# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.model_support import ModelMetadata

from families.k2_horizon.checkpoint_mapper import (
    _copy_to_numpy,
    _expected_tensor_names,
    _target_np_dtype,
    _validate_checkpoint_tensor_names,
)
from families.k2_horizon.config import validate_config
from families.k2_horizon.support import describe
from families.k2_horizon.tests import debug_runner


def _model_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", ModuleType("tensorrt"))
    sys.modules.pop("families.k2_horizon.native_kv_attention_builder", None)
    sys.modules.pop("families.k2_horizon.model", None)
    return importlib.import_module("families.k2_horizon.model")


def _config(**overrides) -> SimpleNamespace:
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
        "bos_token_id": 1,
        "eos_token_id": [2, 3],
        "pad_token_id": 0,
    }
    raw.update(overrides)
    return SimpleNamespace(raw=raw, **raw)


def test_support_matches_only_the_independent_k2_identity() -> None:
    supported = ModelMetadata(
        config={
            "model_type": "k2_horizon",
            "architectures": ["K2HorizonForCausalLM"],
        },
        model_index={},
    )
    qwen = ModelMetadata(
        config={"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]},
        model_index={},
    )

    assert describe(supported).tasks == ("text_generation",)
    assert describe(qwen) is None


def test_config_owns_the_grouped_rmsnorm_contract() -> None:
    resolved = validate_config(_config())

    assert resolved.layernorm_num_groups == 4
    assert resolved.head_dim == 128
    assert resolved.attention_size == 512
    assert resolved.kv_attention_size == 256


def test_publisher_config_uses_nested_rope_and_no_pad_token(monkeypatch, tmp_path: Path) -> None:
    model = _model_module(monkeypatch)
    raw = _config().raw
    raw.pop("rope_theta")
    raw["rope_parameters"] = {"rope_type": "default", "rope_theta": 10_000_000.0}
    raw["pad_token_id"] = None
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")

    loaded = model._load_config(tmp_path)
    resolved = validate_config(loaded)
    runtime = model._runtime_config(tmp_path, loaded.raw, resolved, 256)

    assert resolved.rope_theta == 10_000_000.0
    assert runtime["rope_theta"] == 10_000_000.0
    assert runtime["pad_token_id"] == -1


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


def test_checkpoint_tensor_inventory_fails_closed_on_architecture_drift() -> None:
    expected = _expected_tensor_names(2)
    readers = SimpleNamespace(
        tensor_map={name: object() for name in expected | {"model.layers.0.self_attn.q_proj.bias"}}
    )

    with pytest.raises(ValueError, match="unexpected=.*q_proj.bias"):
        _validate_checkpoint_tensor_names(readers, 2)


def test_bf16_checkpoint_storage_preserves_exact_bits_and_transpose() -> None:
    bits = np.array(
        [[0x0001, 0x3F80, 0x7F7F], [0x8001, 0xBF80, 0xFF7F]],
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
    path = Path(__file__).parents[1] / "model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for function_name, owner in (
        ("_constant", "keepalive"),
        ("_work_constant", "constant_keepalive"),
    ):
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == owner
            and node.func.attr == "append"
            for node in ast.walk(functions[function_name])
        )
    guards = [
        node
        for node in ast.walk(functions["build_engine"])
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.Attribute) and child.attr == "build_serialized_network"
            for statement in node.body
            for child in ast.walk(statement)
        )
    ]
    assert len(guards) == 1
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "clear"
        for statement in guards[0].finalbody
        for node in ast.walk(statement)
    )


def test_build_writes_the_exact_native_kv_bundle(monkeypatch, tmp_path: Path) -> None:
    model = _model_module(monkeypatch)
    raw = _config().raw
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(model, "load_standard_weights", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(model, "build_engine", lambda *_args, **_kwargs: b"plan")

    class Writer:
        def __init__(self):
            self.header = None
            self.sections = {}

        def set_header(self, **value):
            self.header = value

        def add_bytes(self, name, value):
            self.sections[name] = value

        def add_json(self, name, value):
            self.sections[name] = value

    writer = Writer()
    request = SimpleNamespace(
        model_dir=tmp_path,
        backend="trt",
        dynamic_kv_cache=False,
        task="text_generation",
        precision="bf16",
        max_sequence_length=256,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        quantization=None,
        fp32_layers=(),
        verbose=False,
    )
    model.build(request, writer)

    assert writer.header == {
        "family": "k2_horizon",
        "task": "text_generation",
        "backend": "trt",
    }
    assert writer.sections["engine.plan"] == b"plan"
    runtime = writer.sections["runtime.json"]
    assert len(runtime) == 24
    assert runtime["native_kv_cache"] is True
    assert runtime["native_kv_contract_version"] == 1
    assert runtime["layernorm_num_groups"] == 4
    assert runtime["tensor_parallel_mode"] == "single"


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

    monkeypatch.setattr(debug_runner, "cudart", fake_cuda)
    monkeypatch.setattr(debug_runner.K2HorizonTrtRunner, "_initialize", fail_after_allocations)
    with pytest.raises(RuntimeError, match="initialization failed"):
        debug_runner.K2HorizonTrtRunner(b"plan", 8, 1)

    runner = partial["runner"]
    assert fake_cuda.freed == [11, 12, 13]
    assert fake_cuda.destroyed == [7]
    runner.close()
    assert fake_cuda.freed == [11, 12, 13]


def test_native_timeout_preserves_full_stderr(monkeypatch, tmp_path: Path) -> None:
    from families.k2_horizon.tests import test_e2e

    def timeout(*_args, **_kwargs):
        raise test_e2e.subprocess.TimeoutExpired(
            cmd=["trtmc"],
            timeout=600,
            stderr=b"complete native timeout evidence",
        )

    monkeypatch.setattr(test_e2e.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="native generation timed out"):
        test_e2e._run_native(
            Path("trtmc"),
            Path("runtime"),
            Path("model.bundle"),
            "prompt",
            {"max_new_tokens": 4},
            1,
            tmp_path,
        )

    assert (tmp_path / "native-timeout.stderr.log").read_text(
        encoding="utf-8"
    ) == "complete native timeout evidence"
