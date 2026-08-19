# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel tests for PersonaPlex temporal decoder support."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


pytest.importorskip(
    "tensorrt_model_connect.config",
    reason="tensorrt_model_connect requires tensorrt",
)

from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.models.personaplex import model as PersonaPlexModel
from tensorrt_model_connect.parallel_config import ParallelConfig, shard_standard_decoder_weights

personaplex_plugin = importlib.import_module(
    "tensorrt_model_connect.models.personaplex.model")
personaplex_mimi_weights = importlib.import_module(
    "tensorrt_model_connect.models.personaplex.mimi_weights")


_LAYERS = 2
_HIDDEN = 16
_HEADS = 4
_MLP = 64
_VOCAB = 32


def test_personaplex_mimi_loader_requires_checkpoint_owned_weights(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint-owned Mimi codec"):
        personaplex_mimi_weights._load_mimi_weights(tmp_path)


def _personaplex_tp_builder_module():
    return pytest.importorskip(
        "tensorrt_model_connect.models.personaplex.decoder_tp_builder",
        reason="TensorRT is required for PersonaPlex TP builder tests",
    )


def _make_config(
    *,
    hidden: int = _HIDDEN,
    heads: int = _HEADS,
    layers: int = _LAYERS,
    mlp: int = _MLP,
) -> ModelConfig:
    return ModelConfig(
        model_type="personaplex",
        vocab_size=_VOCAB,
        hidden_size=hidden,
        intermediate_size=mlp,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        rms_norm_eps=1e-8,
        _head_dim=hidden // heads,
    )


def _make_temporal_weights(
    *,
    hidden: int = _HIDDEN,
    layers: int = _LAYERS,
    mlp: int = _MLP,
    vocab: int = _VOCAB,
) -> WeightDict:
    rng = np.random.RandomState(123)

    def rand(*shape: int) -> np.ndarray:
        return rng.randn(*shape).astype(np.float32)

    weights = WeightDict({
        "_attention_size": hidden,
        "_mlp_size": mlp,
        "embedding": rand(vocab, hidden),
        "position_embedding": np.zeros((128, hidden), dtype=np.float32),
        "final_norm": rand(hidden),
        "w_out": rand(hidden, vocab),
    })
    for i in range(layers):
        prefix = f"layer.{i}"
        weights[f"{prefix}.input_norm"] = rand(hidden)
        weights[f"{prefix}.post_attn_norm"] = rand(hidden)
        weights[f"{prefix}.w_q"] = rand(hidden, hidden)
        weights[f"{prefix}.w_k"] = rand(hidden, hidden)
        weights[f"{prefix}.w_v"] = rand(hidden, hidden)
        weights[f"{prefix}.w_o"] = rand(hidden, hidden)
        weights[f"{prefix}.w_gate"] = rand(hidden, mlp)
        weights[f"{prefix}.w_up"] = rand(hidden, mlp)
        weights[f"{prefix}.w_down"] = rand(mlp, hidden)
    return weights


def test_personaplex_tp_builder_rejects_single_device_mode():
    decoder_tp_builder = _personaplex_tp_builder_module()

    with pytest.raises(ValueError, match="requires an enabled parallel config"):
        decoder_tp_builder.build_personaplex_tp_decoder_engine(
            _make_config(),
            _make_temporal_weights(),
            max_cache_length=4,
            parallel_config=ParallelConfig(),
        )


def test_personaplex_tp_metadata_defaults_to_attention_size():
    decoder_tp_builder = _personaplex_tp_builder_module()
    config = _make_config()
    weights = _make_temporal_weights()

    with_metadata = decoder_tp_builder._ensure_tp_metadata(config, weights)

    assert with_metadata is not weights
    assert with_metadata["_kv_attention_size"] == _HIDDEN
    assert "_kv_attention_size" not in weights


def test_personaplex_tp_preserves_existing_kv_metadata():
    decoder_tp_builder = _personaplex_tp_builder_module()
    config = _make_config()
    weights = _make_temporal_weights()
    weights["_kv_attention_size"] = _HIDDEN

    assert decoder_tp_builder._ensure_tp_metadata(config, weights) is weights


def test_personaplex_tp_shards_rank_local_temporal_weights():
    decoder_tp_builder = _personaplex_tp_builder_module()
    weights = decoder_tp_builder._ensure_tp_metadata(
        _make_config(), _make_temporal_weights())

    shard = shard_standard_decoder_weights(
        _make_config(),
        weights,
        ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2),
    )

    assert isinstance(shard, WeightDict)
    assert shard["_tensor_parallel_size"] == 4
    assert shard["_tensor_parallel_rank"] == 2
    assert shard["_attention_size"] == _HIDDEN // 4
    assert shard["_kv_attention_size"] == _HIDDEN // 4
    assert shard["_mlp_size"] == _MLP // 4
    assert shard["embedding"] is weights["embedding"]
    assert shard["final_norm"] is weights["final_norm"]

    q_start = (_HIDDEN // 4) * 2
    q_end = (_HIDDEN // 4) * 3
    mlp_start = (_MLP // 4) * 2
    mlp_end = (_MLP // 4) * 3
    np.testing.assert_array_equal(
        shard["layer.0.w_q"],
        weights["layer.0.w_q"][:, q_start:q_end],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_k"],
        weights["layer.0.w_k"][:, q_start:q_end],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_o"],
        weights["layer.0.w_o"][q_start:q_end, :],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_gate"],
        weights["layer.0.w_gate"][:, mlp_start:mlp_end],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_down"],
        weights["layer.0.w_down"][mlp_start:mlp_end, :],
    )


def test_personaplex_plugin_routes_temporal_tp_build(monkeypatch):
    decoder_tp_builder = _personaplex_tp_builder_module()
    captured = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured["config"] = config
        captured["weights"] = weights
        captured["max_cache_length"] = max_cache_length
        captured["kwargs"] = kwargs
        return b"tp-plan"

    monkeypatch.setattr(
        personaplex_plugin,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )
    monkeypatch.setattr(
        decoder_tp_builder,
        "build_personaplex_tp_decoder_engine",
        fake_build,
    )

    weights = WeightDict({
        "_hidden_size": _HIDDEN,
        "_num_hidden_layers": _LAYERS,
        "_num_attention_heads": _HEADS,
        "_head_dim": _HIDDEN // _HEADS,
        "_intermediate_size": _MLP,
        "_text_vocab": _VOCAB,
        "temporal._attention_size": _HIDDEN,
        "temporal._mlp_size": _MLP,
        "temporal.embedding": np.zeros((_VOCAB, _HIDDEN), dtype=np.float32),
        "temporal.w_out": np.zeros((_HIDDEN, _VOCAB), dtype=np.float32),
    })

    plan = PersonaPlexModel.build_engine(
        _make_config(),
        weights,
        max_cache_length=8,
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1),
    )

    assert plan == b"tp-plan"
    assert captured["config"].num_attention_heads == _HEADS
    assert captured["weights"]["_kv_attention_size"] == _HIDDEN
    assert captured["max_cache_length"] == 8
    assert captured["kwargs"]["mlp_type"] == "swiglu"
    assert captured["kwargs"]["position_type"] == "rope"
    assert captured["kwargs"]["embed_input"] is True
    assert captured["kwargs"]["hidden_state_output"] is True
    assert captured["kwargs"]["parallel_config"].tp_size == 4


def test_personaplex_plugin_forwards_fp16_to_temporal_builder(monkeypatch):
    captured = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured["kwargs"] = kwargs
        return b"plan"

    monkeypatch.setattr(
        personaplex_plugin, "build_standard_decoder_engine", fake_build)

    decoder_weights = _make_temporal_weights()
    weights = WeightDict({
        "_hidden_size": _HIDDEN,
        "_num_hidden_layers": _LAYERS,
        "_num_attention_heads": _HEADS,
        "_head_dim": _HIDDEN // _HEADS,
        "_intermediate_size": _MLP,
        "_text_vocab": _VOCAB,
    })
    for key, value in decoder_weights.items():
        weights[f"temporal.{key}"] = value

    plan = PersonaPlexModel.build_engine(
        _make_config(), weights, max_cache_length=8, precision="fp16")

    assert plan == b"plan"
    assert captured["kwargs"]["precision"] == "fp16"
    assert captured["kwargs"]["fp32_layers"] == ()


def test_personaplex_plugin_can_keep_temporal_component_in_fp32(monkeypatch):
    captured = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured["kwargs"] = kwargs
        return b"plan"

    monkeypatch.setattr(
        personaplex_plugin, "build_standard_decoder_engine", fake_build)
    config = _make_config()
    config.raw["_fp32_layers"] = [0]
    weights = WeightDict({
        "_hidden_size": _HIDDEN,
        "_num_hidden_layers": _LAYERS,
        "_num_attention_heads": _HEADS,
        "_head_dim": _HIDDEN // _HEADS,
        "_intermediate_size": _MLP,
        "_text_vocab": _VOCAB,
    })
    for key, value in _make_temporal_weights().items():
        weights[f"temporal.{key}"] = value

    plan = PersonaPlexModel.build_engine(
        config, weights, max_cache_length=8, precision="fp16")

    assert plan == b"plan"
    assert captured["kwargs"]["precision"] == "fp32"
    assert captured["kwargs"]["fp32_layers"] == ()


def test_personaplex_plugin_routes_temporal_block_selectors(monkeypatch):
    captured = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured["kwargs"] = kwargs
        return b"plan"

    monkeypatch.setattr(
        personaplex_plugin, "build_standard_decoder_engine", fake_build)
    config = _make_config()
    config.raw["_fp32_layers"] = [4, 5]
    weights = WeightDict({
        "_hidden_size": _HIDDEN,
        "_num_hidden_layers": _LAYERS,
        "_num_attention_heads": _HEADS,
        "_head_dim": _HIDDEN // _HEADS,
        "_intermediate_size": _MLP,
        "_text_vocab": _VOCAB,
    })
    for key, value in _make_temporal_weights().items():
        weights[f"temporal.{key}"] = value

    plan = PersonaPlexModel.build_engine(
        config, weights, max_cache_length=8, precision="fp16")

    assert plan == b"plan"
    assert captured["kwargs"]["precision"] == "fp16"
    assert captured["kwargs"]["fp32_layers"] == (0, 1)


def test_personaplex_plugin_routes_final_temporal_block_selector(monkeypatch):
    captured = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured["kwargs"] = kwargs
        return b"plan"

    monkeypatch.setattr(
        personaplex_plugin, "build_standard_decoder_engine", fake_build)
    config = _make_config()
    config.raw["_fp32_layers"] = [4 + _LAYERS - 1]
    weights = WeightDict({
        "_hidden_size": _HIDDEN,
        "_num_hidden_layers": _LAYERS,
        "_num_attention_heads": _HEADS,
        "_head_dim": _HIDDEN // _HEADS,
        "_intermediate_size": _MLP,
        "_text_vocab": _VOCAB,
    })
    for key, value in _make_temporal_weights().items():
        weights[f"temporal.{key}"] = value

    plan = PersonaPlexModel.build_engine(
        config, weights, max_cache_length=8, precision="fp16")

    assert plan == b"plan"
    assert captured["kwargs"]["precision"] == "fp16"
    assert captured["kwargs"]["fp32_layers"] == (_LAYERS - 1,)


def test_personaplex_depth_mapping_does_not_add_a_final_norm():
    class Reader:
        def __init__(self, tensors):
            self.tensors = tensors

        def keys(self):
            return self.tensors.keys()

        def get_tensor(self, name):
            return self.tensors[name]

    hidden = 4
    intermediate = 3
    codebooks = 2
    tensors = {
        "depformer_emb.0.weight": np.zeros((5, hidden), dtype=np.float32),
    }
    for cb in range(codebooks):
        tensors[f"linears.{cb}.weight"] = np.zeros(
            (5, hidden), dtype=np.float32)
        tensors[f"depformer.layers.0.gating.{cb}.linear_in.weight"] = (
            np.zeros((2 * intermediate, hidden), dtype=np.float32))
        tensors[f"depformer.layers.0.gating.{cb}.linear_out.weight"] = (
            np.zeros((hidden, intermediate), dtype=np.float32))
    tensors.update({
        "depformer.layers.0.norm1.alpha": np.ones(
            (1, 1, hidden), dtype=np.float32),
        "depformer.layers.0.norm2.alpha": np.ones(
            (1, 1, hidden), dtype=np.float32),
        "depformer.layers.0.self_attn.in_proj_weight": np.zeros(
            (codebooks * 3 * hidden, hidden), dtype=np.float32),
        "depformer.layers.0.self_attn.out_proj.weight": np.zeros(
            (codebooks * hidden, hidden), dtype=np.float32),
    })
    weights = WeightDict()

    personaplex_plugin._load_depth_weights(
        weights,
        [Reader(tensors)],
        {
            "depth_hidden": hidden,
            "depth_num_layers": 1,
            "depth_intermediate": intermediate,
            "num_codebooks": codebooks,
            "num_depformer_emb": 1,
        },
    )

    assert "depth_cb0.final_norm" not in weights
    assert "depth_cb1.final_norm" not in weights
    assert "depth_cb0.w_out" in weights
    assert "depth_cb1.w_out" in weights


def test_personaplex_mimi_mapping_uses_checkpoint_owned_layout():
    hidden = 4
    source = {
        "encoder.model.0.conv.conv.weight": np.arange(
            12, dtype=np.float32).reshape(4, 1, 3),
        "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum": np.ones(
            (5, 2), dtype=np.float32),
        "encoder_transformer.transformer.layers.0.norm1.weight": np.ones(
            hidden, dtype=np.float32),
        "encoder_transformer.transformer.layers.0.norm1.bias": np.zeros(
            hidden, dtype=np.float32),
        "encoder_transformer.transformer.layers.0.norm2.weight": np.ones(
            hidden, dtype=np.float32),
        "encoder_transformer.transformer.layers.0.norm2.bias": np.zeros(
            hidden, dtype=np.float32),
        "encoder_transformer.transformer.layers.0.linear1.weight": np.zeros(
            (8, hidden), dtype=np.float32),
        "encoder_transformer.transformer.layers.0.linear2.weight": np.zeros(
            (hidden, 8), dtype=np.float32),
        "encoder_transformer.transformer.layers.0.layer_scale_1.scale": np.ones(
            hidden, dtype=np.float32),
        "encoder_transformer.transformer.layers.0.layer_scale_2.scale": np.ones(
            hidden, dtype=np.float32),
        "encoder_transformer.transformer.layers.0.self_attn.in_proj_weight": (
            np.arange(3 * hidden * hidden, dtype=np.float32).reshape(
                3 * hidden, hidden)),
        "encoder_transformer.transformer.layers.0.self_attn.out_proj.weight": (
            np.eye(hidden, dtype=np.float32)),
    }

    mapped = personaplex_mimi_weights._translate_personaplex_mimi_weights(source)

    np.testing.assert_array_equal(
        mapped["encoder.layers.0.conv.weight"],
        source["encoder.model.0.conv.conv.weight"],
    )
    assert (
        "quantizer.semantic_residual_vector_quantizer.layers.0."
        "codebook.embed_sum"
    ) in mapped
    fused = source[
        "encoder_transformer.transformer.layers.0.self_attn.in_proj_weight"]
    np.testing.assert_array_equal(
        mapped["encoder_transformer.layers.0.self_attn.q_proj.weight"],
        fused[:hidden],
    )
    np.testing.assert_array_equal(
        mapped["encoder_transformer.layers.0.self_attn.v_proj.weight"],
        fused[2 * hidden:],
    )


def test_personaplex_mimi_transformer_uses_causal_attention(monkeypatch):
    class Tensor:
        pass

    class Layer:
        def __init__(self, output):
            self.output = output

        def get_output(self, _index):
            return self.output

    class Network:
        def add_elementwise(self, _lhs, _rhs, _operation):
            return Layer(Tensor())

    captured = {}

    monkeypatch.setattr(personaplex_plugin.graph_ops, "add_constant", lambda *a, **k: Tensor())
    monkeypatch.setattr(personaplex_plugin.graph_ops, "add_layer_norm", lambda *a, **k: Tensor())
    monkeypatch.setattr(
        personaplex_plugin.graph_ops,
        "add_matmul_rhs_constant",
        lambda *a, **k: Tensor(),
    )
    monkeypatch.setattr(personaplex_plugin.graph_ops, "add_gelu_erf", lambda *a, **k: Tensor())

    def fake_attention(*args, **kwargs):
        captured.update(kwargs)
        return Tensor()

    monkeypatch.setattr(
        personaplex_plugin.graph_ops,
        "add_self_attention_block_with_rope",
        fake_attention,
    )
    hidden = 4
    weights = {
        "input_layernorm.weight": np.ones(hidden, dtype=np.float32),
        "input_layernorm.bias": np.zeros(hidden, dtype=np.float32),
        "post_attention_layernorm.weight": np.ones(hidden, dtype=np.float32),
        "post_attention_layernorm.bias": np.zeros(hidden, dtype=np.float32),
        "self_attn.q_proj.weight": np.eye(hidden, dtype=np.float32),
        "self_attn.k_proj.weight": np.eye(hidden, dtype=np.float32),
        "self_attn.v_proj.weight": np.eye(hidden, dtype=np.float32),
        "self_attn.o_proj.weight": np.eye(hidden, dtype=np.float32),
        "self_attn_layer_scale.scale": np.ones(hidden, dtype=np.float32),
        "mlp.fc1.weight": np.zeros((hidden, 8), dtype=np.float32),
        "mlp.fc2.weight": np.zeros((8, hidden), dtype=np.float32),
        "mlp_layer_scale.scale": np.ones(hidden, dtype=np.float32),
        "_cos_table": np.ones((2, hidden), dtype=np.float32),
        "_sin_table": np.zeros((2, hidden), dtype=np.float32),
    }

    personaplex_plugin._add_mimi_transformer_layer(
        Network(), Tensor(), 2, hidden, 1, hidden, 8, 1e-5, weights)

    assert captured["causal"] is True
    assert captured["interleaved_rope"] is True


def test_personaplex_plugin_rejects_quantized_tp(monkeypatch):
    monkeypatch.setattr(
        personaplex_plugin,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )

    with pytest.raises(ValueError, match="do not support quantization"):
        PersonaPlexModel.build_engine(
            _make_config(),
            WeightDict({
                "_hidden_size": _HIDDEN,
                "_num_hidden_layers": _LAYERS,
                "_num_attention_heads": _HEADS,
                "_head_dim": _HIDDEN // _HEADS,
                "_intermediate_size": _MLP,
                "_text_vocab": _VOCAB,
            }),
            max_cache_length=8,
            quant_ctx=object(),
            parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )
