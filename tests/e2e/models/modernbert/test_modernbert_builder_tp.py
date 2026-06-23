"""Tensor-parallel tests for ModernBERT encoder support."""

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
from tensorrt_model_connect.families.modernbert.plugin import ModernbertPlugin
from tensorrt_model_connect.parallel_config import ParallelConfig

modernbert_plugin = importlib.import_module(
    "tensorrt_model_connect.families.modernbert.plugin")

_LAYERS = 1
_HIDDEN = 16
_HEADS = 4
_MLP = 32
_VOCAB = 24


def _modernbert_tp_builder_module():
    return pytest.importorskip(
        "tensorrt_model_connect.families.modernbert.tp_builder",
        reason="TensorRT is required for ModernBERT TP builder tests",
    )


def _make_config(
    *,
    hidden: int = _HIDDEN,
    heads: int = _HEADS,
    layers: int = _LAYERS,
    mlp: int = _MLP,
) -> ModelConfig:
    return ModelConfig(
        model_type="modernbert",
        vocab_size=_VOCAB,
        hidden_size=hidden,
        intermediate_size=mlp,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        _head_dim=hidden // heads,
        raw={
            "norm_eps": 1e-5,
            "layer_types": ["full_attention"],
            "rope_parameters": {},
        },
    )


def _matrix(rows: int, cols: int, offset: int = 0) -> np.ndarray:
    return np.arange(rows * cols, dtype=np.float32).reshape(rows, cols) + offset


def _make_encoder_weights(
    *,
    hidden: int = _HIDDEN,
    layers: int = _LAYERS,
    mlp: int = _MLP,
    vocab: int = _VOCAB,
) -> WeightDict:
    weights = WeightDict({
        "embedding": _matrix(vocab, hidden),
        "embed_norm": np.ones(hidden, dtype=np.float32),
        "final_norm": np.ones(hidden, dtype=np.float32),
    })
    for layer_idx in range(layers):
        prefix = f"layer.{layer_idx}"
        weights[f"{prefix}.attn_norm"] = np.ones(hidden, dtype=np.float32)
        weights[f"{prefix}.w_q"] = _matrix(hidden, hidden, 10)
        weights[f"{prefix}.w_k"] = _matrix(hidden, hidden, 20)
        weights[f"{prefix}.w_v"] = _matrix(hidden, hidden, 30)
        weights[f"{prefix}.w_o"] = _matrix(hidden, hidden, 40)
        weights[f"{prefix}.mlp_norm"] = np.ones(hidden, dtype=np.float32)
        weights[f"{prefix}.w_mlp_input"] = _matrix(hidden, mlp, 50)
        weights[f"{prefix}.w_mlp_gate"] = _matrix(hidden, mlp, 60)
        weights[f"{prefix}.w_down"] = _matrix(mlp, hidden, 70)
    return weights


def test_modernbert_tp_builder_rejects_single_device_mode():
    tp_builder = _modernbert_tp_builder_module()

    with pytest.raises(ValueError, match="requires tensor_parallel mode"):
        tp_builder.build_tp_modernbert_engine(
            _make_config(),
            _make_encoder_weights(),
            max_seq_length=8,
            parallel_config=ParallelConfig(),
        )


@pytest.mark.parametrize(
    ("parallel", "overrides", "message"),
    [
        (
            ParallelConfig(mode="tensor_parallel", tp_size=4, rank=-1),
            {},
            "concrete rank",
        ),
        (
            ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
            {"heads": _HEADS + 2},
            "num_attention_heads divisible",
        ),
        (
            ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
            {"mlp": _MLP + 2},
            "intermediate_size divisible",
        ),
    ],
)
def test_modernbert_tp_validation_rejects_bad_config_dimensions(
    parallel,
    overrides,
    message,
):
    tp_builder = _modernbert_tp_builder_module()
    config_kwargs = {
        "hidden": _HIDDEN,
        "heads": _HEADS,
        "layers": _LAYERS,
        "mlp": _MLP,
    }
    config_kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        tp_builder._validate_modernbert_tp(
            _make_config(**config_kwargs),
            _make_encoder_weights(mlp=config_kwargs["mlp"]),
            parallel,
        )


def test_modernbert_tp_shards_rank_local_encoder_weights():
    tp_builder = _modernbert_tp_builder_module()
    config = _make_config()
    weights = _make_encoder_weights()
    shard = tp_builder.shard_modernbert_weights(
        config,
        weights,
        parallel=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2),
    )

    hidden_start = (_HIDDEN // 4) * 2
    hidden_end = (_HIDDEN // 4) * 3
    mlp_start = (_MLP // 4) * 2
    mlp_end = (_MLP // 4) * 3

    assert isinstance(shard, WeightDict)
    assert shard["_tensor_parallel_size"] == 4
    assert shard["_tensor_parallel_rank"] == 2
    assert shard["_attention_size"] == _HIDDEN // 4
    assert shard["_intermediate_size"] == _MLP // 4
    assert shard["embedding"] is weights["embedding"]

    np.testing.assert_array_equal(
        shard["layer.0.w_q"],
        weights["layer.0.w_q"][:, hidden_start:hidden_end],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_o"],
        weights["layer.0.w_o"][hidden_start:hidden_end, :],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_mlp_input"],
        weights["layer.0.w_mlp_input"][:, mlp_start:mlp_end],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_mlp_gate"],
        weights["layer.0.w_mlp_gate"][:, mlp_start:mlp_end],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_down"],
        weights["layer.0.w_down"][mlp_start:mlp_end, :],
    )


def test_modernbert_plugin_routes_tp_build(monkeypatch):
    tp_builder = _modernbert_tp_builder_module()
    captured = {}

    def fake_build(config, weights, max_seq_length, **kwargs):
        captured["config"] = config
        captured["weights"] = weights
        captured["max_seq_length"] = max_seq_length
        captured["kwargs"] = kwargs
        return b"modernbert-tp-plan"

    monkeypatch.setattr(
        modernbert_plugin,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )
    monkeypatch.setattr(tp_builder, "build_tp_modernbert_engine", fake_build)

    plan = ModernbertPlugin().build_engine(
        _make_config(),
        _make_encoder_weights(),
        max_cache_length=8,
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1),
    )

    assert plan == b"modernbert-tp-plan"
    assert captured["config"].model_type == "modernbert"
    assert captured["max_seq_length"] == 8
    assert captured["kwargs"]["parallel_config"].tp_size == 4
    assert captured["kwargs"]["parallel_config"].rank == 1


def test_modernbert_plugin_rejects_quantized_tp(monkeypatch):
    monkeypatch.setattr(
        modernbert_plugin,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )

    with pytest.raises(ValueError, match="do not support quantization"):
        ModernbertPlugin().build_engine(
            _make_config(),
            _make_encoder_weights(),
            max_cache_length=8,
            quant_ctx=object(),
            parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )
