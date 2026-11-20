# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public build selection and source lifetime for Qwen's weightless path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt")

from tensorrt_model_connect.build import BuildRequest  # noqa: E402
from families.qwen import embedding, model, quantization, weightless  # noqa: E402
from families.qwen.config import ModelConfig  # noqa: E402


def _config(model_type="qwen3", **changes):
    values = dict(
        architectures=["Qwen3ForCausalLM" if model_type == "qwen3" else "Qwen2ForCausalLM"],
        hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128,
        vocab_size=151936, max_position_embeddings=40960, hidden_act="silu",
    )
    values.update(changes)
    return ModelConfig.create_tiny(model_type, **values)


def _weights(config):
    # The real shape validator runs, while the graph builder is stubbed. Views
    # avoid allocating model-sized buffers in a routing-only CPU test.
    def zeros(shape):
        return np.broadcast_to(np.asarray(0, dtype=np.float16), shape)

    hidden, attention = config.hidden_size, config.attention_size
    kv = config.num_key_value_heads * config.head_dim
    mlp, vocab = config.intermediate_size, config.vocab_size
    values = {"embedding": zeros((vocab, hidden)), "w_out": zeros((hidden, vocab)),
              "final_norm": zeros((hidden,))}
    for layer in range(config.num_hidden_layers):
        for suffix, shape in {
            "input_norm": (hidden,), "post_attn_norm": (hidden,),
            "q_norm": (attention,), "k_norm": (kv,),
            "w_q": (hidden, attention), "w_k": (hidden, kv),
            "w_v": (hidden, kv), "w_o": (attention, hidden),
            "w_gate": (hidden, mlp), "w_up": (hidden, mlp), "w_down": (mlp, hidden),
        }.items():
            values[f"layer.{layer}.{suffix}"] = zeros(shape)
    return values


@pytest.fixture
def build_harness(monkeypatch, tmp_path):
    calls, loads, parallel_calls, sections = [], [], [], {}
    checkpoint, output = tmp_path / "checkpoint", tmp_path / "output"
    checkpoint.mkdir()
    output.mkdir()

    def load(self, model_dir, config, **kwargs):
        loads.append(config.model_type)
        return _weights(config)

    def engine(config, weights, length, **kwargs):
        calls.append({
            "role": kwargs.get("profile_mode", config.raw.get("_decoder_engine_role")),
            "session": weightless._SESSION.get(),
            "native": kwargs.get("native_kv_cache", False),
            "rank": getattr(kwargs.get("parallel_config"), "rank", -1),
        })
        return b"plan"

    def parallel(model_dir, config, length, **kwargs):
        parallel_calls.append((config, length, kwargs))
        config.raw["_native_kv_cache_metadata"] = {"native_kv_cache": True, "native_kv_contract_version": 1}
        return b"prefill", b"decode"

    monkeypatch.setattr(model._QwenModel, "load_weights", load)
    monkeypatch.setattr(model, "build_dual_profile_decoder_engine", engine)
    monkeypatch.setattr(model, "build_standard_decoder_engine", engine)
    monkeypatch.setattr(model, "build_dual_profile_tp_decoder_engine", engine)
    monkeypatch.setattr(model, "_build_weightless_native_engines", parallel)
    monkeypatch.setattr(
        quantization, "calibrate_qwen_fp8",
        lambda *args: SimpleNamespace(profile=SimpleNamespace(format=SimpleNamespace(name="fp8"))),
    )
    writer = SimpleNamespace(set_header=lambda **kwargs: None,
                             add_bytes=sections.__setitem__, add_json=sections.__setitem__)

    def run(model_type="qwen3", config_changes=None, **kwargs):
        config = _config(model_type, **(config_changes or {}))
        monkeypatch.setattr(model.ModelConfig, "from_dir", lambda path: config)
        options = dict(model_dir=checkpoint, output_path=output / "model.bundle",
                       family="qwen", task="text_continuation", precision="bf16",
                       max_sequence_length=16384)
        options.update(kwargs)
        model.build(BuildRequest(**options), writer)

    return SimpleNamespace(run=run, calls=calls, loads=loads, parallel=parallel_calls,
                           sections=sections, output_dir=output)


def test_selected_bf16_topology_uses_workers_without_loading_parent_weights(build_harness):
    harness = build_harness
    harness.run()
    assert len(harness.parallel) == 1
    assert harness.parallel[0][2]["output_dir"] == harness.output_dir
    assert not harness.loads and not harness.calls
    assert harness.sections["prefill.plan"] == b"prefill"
    assert harness.sections["engine.plan"] == b"decode"
    assert harness.sections["runtime.json"]["decoder_engine_layout"] == "split"
    assert harness.sections["runtime.json"]["native_kv_cache"] is True
    assert weightless._SESSION.get() is None


@pytest.mark.parametrize("reason", ["transform", "rtx", "fp8", "fp16"])
def test_public_exclusions_keep_original_serial_build(build_harness, monkeypatch, reason):
    harness, options = build_harness, {}
    if reason == "transform":
        options["graph_transform"] = lambda network, index: None
    elif reason == "rtx":
        options["backend"] = "trt_rtx"
    elif reason == "fp16":
        options["precision"] = "fp16"
    else:
        options.update(quantization="fp8", precision="fp16")
    harness.run(**options)
    assert len(harness.calls) == (1 if reason == "fp8" else 2)
    assert all(call["session"] is None for call in harness.calls)
    assert len(harness.loads) == 1 and not harness.parallel


@pytest.mark.parametrize("field,value", [
    ("hidden_size", 2048), ("intermediate_size", 8192), ("num_hidden_layers", 32),
    ("num_attention_heads", 32), ("num_key_value_heads", 4), ("vocab_size", 152064),
])
def test_other_qwen3_topologies_keep_original_serial_build(build_harness, field, value):
    harness = build_harness
    harness.run(config_changes={field: value})
    assert len(harness.calls) == 2 and len(harness.loads) == 1
    assert all(call["native"] and call["session"] is None for call in harness.calls)
    assert not harness.parallel


@pytest.mark.parametrize("changes", [{"head_dim": 64}, {"hidden_act": "gelu"}])
def test_unsupported_architectures_are_not_promoted_to_workers(build_harness, changes):
    harness = build_harness
    with pytest.raises(ValueError, match="requires the fixed-capacity native KV path"):
        harness.run(config_changes=changes)
    assert not harness.calls and not harness.parallel


def test_fp32_keeps_existing_native_rejection(build_harness):
    with pytest.raises(ValueError, match="native Qwen3 requires FP16 or BF16"):
        build_harness.run(precision="fp32")
    assert not build_harness.parallel


@pytest.mark.parametrize("parallel", [1, 2])
def test_qwen2_keeps_its_existing_route(build_harness, parallel):
    harness = build_harness
    harness.run(model_type="qwen2", tensor_parallel_size=parallel)
    assert len(harness.calls) == 2 and not harness.parallel
    assert all(not call["native"] and call["session"] is None for call in harness.calls)


def test_qualified_fp8_tensor_parallel_route_keeps_ordinary_weights(build_harness):
    harness = build_harness
    harness.run(quantization="fp8", precision="bf16", tensor_parallel_size=4)
    assert [call["rank"] for call in harness.calls] == [0, 1, 2, 3]
    assert all(not call["native"] and call["session"] is None for call in harness.calls)
    assert not harness.parallel


def test_unqualified_native_tensor_parallel_still_fails_closed(build_harness):
    with pytest.raises(ValueError, match="does not support tensor parallel"):
        build_harness.run(tensor_parallel_size=2)
    assert not build_harness.calls and not build_harness.parallel


def test_embedding_keeps_its_separate_owner_path(build_harness, monkeypatch):
    observed = []
    monkeypatch.setattr(embedding, "build_embedding",
                        lambda request, writer: observed.append((request.task, weightless._SESSION.get())))
    build_harness.run(task="embedding")
    assert observed == [("embedding", None)]
    assert not build_harness.calls and not build_harness.loads and not build_harness.parallel


def test_internal_serial_engines_keep_distinct_placeholder_sessions(build_harness):
    config, owner = _config(), model._QwenModel()
    for role in ("prefill", "decode"):
        config.raw["_decoder_engine_role"] = role
        with weightless.weightless_build():
            owner.build_engine(config, _weights(config), 64, precision="bf16")
    calls = build_harness.calls
    assert [call["role"] for call in calls] == ["prefill", "decode"]
    assert all(call["session"] is not None for call in calls)
    assert calls[0]["session"] is not calls[1]["session"]
    assert weightless._SESSION.get() is None


def test_failed_internal_engine_restores_the_enclosing_scope(build_harness, monkeypatch):
    def fail(*args, **kwargs):
        assert weightless._SESSION.get() is not None
        assert weightless._SESSION.get() is not enclosing
        raise RuntimeError("intentional builder failure")

    monkeypatch.setattr(model, "build_dual_profile_decoder_engine", fail)
    config = _config()
    config.raw["_decoder_engine_role"] = "prefill"
    with weightless.weightless_build():
        enclosing = weightless._SESSION.get()
        with pytest.raises(RuntimeError, match="intentional builder failure"):
            with weightless.weightless_build():
                model._QwenModel().build_engine(
                    config, _weights(config), 64, precision="bf16")
        assert weightless._SESSION.get() is enclosing
    assert weightless._SESSION.get() is None


def test_empty_weightless_graph_cannot_silently_use_ordinary_compilation():
    builder = SimpleNamespace(build_serialized_network=lambda *args: pytest.fail("unexpected ordinary build"))
    with weightless.weightless_build():
        with pytest.raises(RuntimeError, match="no named projection weights"):
            weightless.build_serialized_network(builder, None, None, None)
    assert weightless._SESSION.get() is None
