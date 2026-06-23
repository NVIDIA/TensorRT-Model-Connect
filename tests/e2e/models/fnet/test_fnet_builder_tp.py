"""Tensor-parallel tests for FNet encoder support."""

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
from tensorrt_model_connect.families.fnet.plugin import FNetPlugin
from tensorrt_model_connect.parallel_config import ParallelConfig

fnet_plugin = importlib.import_module(
    "tensorrt_model_connect.families.fnet.plugin")

_LAYERS = 1
_HIDDEN = 16
_MLP = 32
_VOCAB = 24


def _fnet_tp_builder_module():
    return pytest.importorskip(
        "tensorrt_model_connect.families.fnet.tp_builder",
        reason="TensorRT is required for FNet TP builder tests",
    )


def _make_config(
    *,
    hidden: int = _HIDDEN,
    layers: int = _LAYERS,
    mlp: int = _MLP,
) -> ModelConfig:
    return ModelConfig(
        model_type="fnet",
        vocab_size=_VOCAB,
        hidden_size=hidden,
        intermediate_size=mlp,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        _head_dim=hidden // 4,
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
        "position_embedding": _matrix(32, hidden, 1000),
        "token_type_embedding": np.zeros((4, hidden), dtype=np.float32),
        "embed_norm": np.ones(hidden, dtype=np.float32),
        "embed_norm_beta": np.zeros(hidden, dtype=np.float32),
        "embed_projection": _matrix(hidden, hidden, 2000),
        "embed_projection_bias": np.zeros(hidden, dtype=np.float32),
    })
    for layer_idx in range(layers):
        prefix = f"layer.{layer_idx}"
        weights[f"{prefix}.post_attn_norm"] = np.ones(hidden, dtype=np.float32)
        weights[f"{prefix}.post_attn_norm_beta"] = np.zeros(hidden, dtype=np.float32)
        weights[f"{prefix}.w_fc1"] = _matrix(hidden, mlp, 50)
        weights[f"{prefix}.fc1_bias"] = np.arange(mlp, dtype=np.float32)
        weights[f"{prefix}.w_fc2"] = _matrix(mlp, hidden, 60)
        weights[f"{prefix}.fc2_bias"] = np.arange(hidden, dtype=np.float32) + 40
        weights[f"{prefix}.output_norm"] = np.ones(hidden, dtype=np.float32)
        weights[f"{prefix}.output_norm_beta"] = np.zeros(hidden, dtype=np.float32)
    return weights


def test_fnet_tp_builder_rejects_single_device_mode():
    tp_builder = _fnet_tp_builder_module()

    with pytest.raises(ValueError, match="requires tensor_parallel mode"):
        tp_builder.build_tp_fnet_encoder_engine(
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
            {"mlp": _MLP + 2},
            "intermediate_size divisible",
        ),
    ],
)
def test_fnet_tp_validation_rejects_bad_config_dimensions(
    parallel,
    overrides,
    message,
):
    tp_builder = _fnet_tp_builder_module()
    config_kwargs = {
        "hidden": _HIDDEN,
        "layers": _LAYERS,
        "mlp": _MLP,
    }
    config_kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        tp_builder._validate_fnet_tp(
            _make_config(**config_kwargs),
            _make_encoder_weights(mlp=config_kwargs["mlp"]),
            parallel,
        )


def test_fnet_tp_shards_rank_local_ffn_weights():
    tp_builder = _fnet_tp_builder_module()
    config = _make_config()
    weights = _make_encoder_weights()
    shard = tp_builder.shard_fnet_encoder_weights(
        config,
        weights,
        parallel=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2),
    )

    mlp_start = (_MLP // 4) * 2
    mlp_end = (_MLP // 4) * 3

    assert isinstance(shard, WeightDict)
    assert shard["_tensor_parallel_size"] == 4
    assert shard["_tensor_parallel_rank"] == 2
    assert shard["_intermediate_size"] == _MLP // 4
    assert shard["embedding"] is weights["embedding"]

    np.testing.assert_array_equal(
        shard["layer.0.w_fc1"],
        weights["layer.0.w_fc1"][:, mlp_start:mlp_end],
    )
    np.testing.assert_array_equal(
        shard["layer.0.fc1_bias"],
        weights["layer.0.fc1_bias"][mlp_start:mlp_end],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_fc2"],
        weights["layer.0.w_fc2"][mlp_start:mlp_end, :],
    )


def test_fnet_plugin_routes_tp_build(monkeypatch):
    tp_builder = _fnet_tp_builder_module()
    captured = {}

    def fake_build(config, weights, max_seq_length, **kwargs):
        captured["config"] = config
        captured["weights"] = weights
        captured["max_seq_length"] = max_seq_length
        captured["kwargs"] = kwargs
        return b"fnet-tp-plan"

    monkeypatch.setattr(
        fnet_plugin,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )
    monkeypatch.setattr(tp_builder, "build_tp_fnet_encoder_engine", fake_build)

    plan = FNetPlugin().build_engine(
        _make_config(),
        _make_encoder_weights(),
        max_cache_length=8,
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1),
    )

    assert plan == b"fnet-tp-plan"
    assert captured["config"].model_type == "fnet"
    assert captured["max_seq_length"] == 8
    assert captured["kwargs"]["parallel_config"].tp_size == 4
    assert captured["kwargs"]["parallel_config"].rank == 1


def test_fnet_plugin_rejects_quantized_tp(monkeypatch):
    monkeypatch.setattr(
        fnet_plugin,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )

    with pytest.raises(ValueError, match="do not support quantization"):
        FNetPlugin().build_engine(
            _make_config(),
            _make_encoder_weights(),
            max_cache_length=8,
            quant_ctx=object(),
            parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )
