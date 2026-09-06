# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.model_support import ModelMetadata

from families.k2_horizon_uno.checkpoint_mapper import (
    _adapter_tensor_names,
    _base_tensor_names,
    _to_bf16_bits,
    _validate_inventory,
)
from families.k2_horizon_uno.config import (
    BASE_MODEL_ID,
    LORA_TARGET_MODULES,
    load_and_validate_adapter_config,
    validate_config,
)
from families.k2_horizon_uno.support import describe


def _adapter_recipe() -> dict:
    return {
        "base_model_name_or_path": BASE_MODEL_ID,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": 8192.0,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 128,
        "revision": None,
        "target_modules": list(LORA_TARGET_MODULES),
        "task_type": "CAUSAL_LM",
    }


def _base_config(**overrides) -> SimpleNamespace:
    raw = {
        "model_type": "k2_horizon",
        "architectures": ["K2HorizonForCausalLM"],
        "vocab_size": 250624,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10_000_000.0, "rope_type": "default"},
        "max_position_embeddings": 524288,
        "hidden_act": "silu",
        "dtype": "bfloat16",
        "layernorm_num_groups": 4,
    }
    raw.update(overrides)
    return SimpleNamespace(raw=raw, **raw)


def _prepare_tracked_reimport(monkeypatch, module_name: str) -> None:
    marker = ModuleType(f"{module_name}.__test_restore_marker__")
    monkeypatch.setitem(sys.modules, module_name, marker)
    monkeypatch.delitem(sys.modules, module_name)


def _model_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", ModuleType("tensorrt"))
    _prepare_tracked_reimport(monkeypatch, "families.k2_horizon_uno.native_kv_attention_builder")
    _prepare_tracked_reimport(monkeypatch, "families.k2_horizon_uno.model")
    return importlib.import_module("families.k2_horizon_uno.model")


def test_support_matches_only_the_adapter_identity() -> None:
    adapter = ModelMetadata({}, {}, ("adapter_config.json", "adapter_model.safetensors"))
    assert describe(adapter).tasks == ("text_generation",)
    assert describe(ModelMetadata({}, {}, ("adapter_config.json",))) is None
    assert describe(ModelMetadata({}, {}, (*adapter.files, "README.md"))) is None


def test_exact_base_config_and_adapter_recipe(tmp_path: Path) -> None:
    resolved = validate_config(_base_config())
    assert resolved.hidden_size == 4096
    assert resolved.attention_size == 4096
    assert resolved.kv_attention_size == 1024
    assert resolved.lora_rank == 128
    assert resolved.lora_scale == 64.0
    assert resolved.max_block_size == 8

    for field, value in (
        ("model_type", "qwen"),
        ("architectures", ["QwenForCausalLM"]),
        ("hidden_size", 512),
        ("num_hidden_layers", 35),
        ("dtype", "float16"),
        ("layernorm_num_groups", 1),
    ):
        with pytest.raises(ValueError):
            validate_config(_base_config(**{field: value}))

    adapter_config = tmp_path / "adapter_config.json"
    adapter_config.write_text(json.dumps(_adapter_recipe()), encoding="utf-8")
    assert load_and_validate_adapter_config(adapter_config)["target_modules"] == list(
        LORA_TARGET_MODULES
    )
    changed = _adapter_recipe()
    changed["r"] = 64
    adapter_config.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="changed fields: r"):
        load_and_validate_adapter_config(adapter_config)


def test_checkpoint_tensor_inventories_are_closed() -> None:
    assert len(_base_tensor_names(36)) == 327
    adapter_names = _adapter_tensor_names(36)
    assert len(adapter_names) == 504
    assert "model.layers.0.self_attn.q_proj.lora_A.weight" in adapter_names
    assert "model.layers.35.mlp.down_proj.lora_B.weight" in adapter_names
    with pytest.raises(ValueError, match="unexpected=drift.weight"):
        _validate_inventory(adapter_names | {"drift.weight"}, adapter_names, "adapter")


def test_bf16_adapter_conversion_rounds_and_transposes() -> None:
    source = np.array(
        [[1.0 / 3.0, 1.0, -1.0 / 3.0], [2.0 / 3.0, 2.0, -2.0 / 3.0]],
        dtype=np.float32,
    )
    bits = _to_bf16_bits(source, transpose=True)
    expected = np.array(
        [[0x3EAB, 0x3F2B], [0x3F80, 0x4000], [0xBEAB, 0xBF2B]],
        dtype=np.uint16,
    )
    assert bits.dtype == np.uint16
    assert bits.flags.c_contiguous
    np.testing.assert_array_equal(bits, expected)


def test_native_kv_constants_keep_the_exact_trt_buffer_alive(monkeypatch) -> None:
    fake_trt = ModuleType("tensorrt")
    fake_trt.Weights = lambda array: SimpleNamespace(array=array)
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    _prepare_tracked_reimport(monkeypatch, "families.k2_horizon_uno.native_kv_attention_builder")
    helper = importlib.import_module("families.k2_horizon_uno.native_kv_attention_builder")

    class Layer:
        def get_output(self, index):
            assert index == 0
            return "constant"

    class Network:
        def add_constant(self, shape, weights):
            assert shape == (2,)
            self.weights = weights
            return Layer()

    network = Network()
    keepalive = []
    assert (
        helper._constant(
            network,
            (2,),
            np.array([1, 2], dtype=np.int64),
            keepalive=keepalive,
            dtype=np.dtype(np.int32),
        )
        == "constant"
    )
    assert len(keepalive) == 1
    assert keepalive[0].dtype == np.int32
    assert keepalive[0].flags.c_contiguous
    assert network.weights.array is keepalive[0]


def _build_request(adapter: Path, output: Path, **overrides) -> BuildRequest:
    request = BuildRequest(
        model_dir=adapter,
        output_path=output,
        family="k2_horizon_uno",
        task="text_generation",
        precision="bf16",
        max_sequence_length=256,
    )
    return replace(request, **overrides)


def test_build_resolves_separate_base_and_adapter_and_writes_exact_sections(
    tmp_path: Path, monkeypatch
) -> None:
    model = _model_module(monkeypatch)
    adapter = tmp_path / "adapter"
    base = tmp_path / "base"
    adapter.mkdir()
    base.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps(_adapter_recipe()), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (base / "config.json").write_text(json.dumps(_base_config().raw), encoding="utf-8")
    (base / "tokenizer.json").write_text("{}", encoding="utf-8")
    (base / "chat_template.jinja").write_text("publisher template", encoding="utf-8")
    captured = {}

    def fake_weights(base_dir, adapter_dir, config):
        captured.update(
            base_dir=Path(base_dir),
            adapter_dir=Path(adapter_dir),
            config=config,
        )
        return {}

    monkeypatch.setattr(model, "_resolve_base_model", lambda: base)
    monkeypatch.setattr(model, "load_standard_weights", fake_weights)
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
    model.build(_build_request(adapter, tmp_path / "model.bundle"), writer)

    assert captured["base_dir"] == base
    assert captured["adapter_dir"] == adapter
    assert writer.header == {
        "family": "k2_horizon_uno",
        "task": "text_generation",
        "backend": "trt",
    }
    assert {"engine.plan", "runtime.json", "tokenizer.json", "chat_template.jinja"} <= set(
        writer.sections
    )
    runtime = writer.sections["runtime.json"]
    assert runtime == {"max_cache_length": 256}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dynamic_kv_cache", True),
        ("max_batch_size", 2),
        ("tensor_parallel_size", 2),
        ("context_parallel_size", 2),
        ("backend", "trt_rtx"),
        ("quantization", "fp8"),
        ("fp32_layers", (0,)),
    ],
)
def test_build_rejects_unsupported_request_fields(
    tmp_path: Path, monkeypatch, field: str, value: object
) -> None:
    model = _model_module(monkeypatch)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    request = _build_request(adapter, tmp_path / "model.bundle", **{field: value})
    with pytest.raises((ValueError, NotImplementedError)):
        model.build(request, SimpleNamespace())
