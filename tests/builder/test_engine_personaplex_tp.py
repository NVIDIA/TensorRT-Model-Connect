"""Tensor-parallel tests for PersonaPlex temporal decoder support."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip(
    "tensorrt_model_connect.config",
    reason="tensorrt_model_connect requires tensorrt",
)

from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families.personaplex.plugin import PersonaPlexPlugin
from tensorrt_model_connect.parallel_config import ParallelConfig, shard_standard_decoder_weights

personaplex_plugin = importlib.import_module(
    "tensorrt_model_connect.families.personaplex.plugin")


_LAYERS = 2
_HIDDEN = 16
_HEADS = 4
_MLP = 64
_VOCAB = 32


def _personaplex_tp_builder_module():
    return pytest.importorskip(
        "tensorrt_model_connect.families.personaplex.decoder_tp_builder",
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

    plan = PersonaPlexPlugin().build_engine(
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


def test_personaplex_plugin_rejects_quantized_tp(monkeypatch):
    monkeypatch.setattr(
        personaplex_plugin,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )

    with pytest.raises(ValueError, match="do not support quantization"):
        PersonaPlexPlugin().build_engine(
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
