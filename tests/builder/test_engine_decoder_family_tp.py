"""Tensor-parallel routing tests for decoder families with local TP builders."""

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
from tensorrt_model_connect.parallel_config import (
    ParallelConfig,
    shard_standard_decoder_weights,
)


_FAMILIES = [
    pytest.param(
        "codegen",
        "CodeGenPlugin",
        "codegen",
        4,
        {"rotary_dim": 2},
        {
            "norm_type": "layernorm",
            "mlp_type": "gelu_fc",
            "position_type": "rope",
            "activation": "gelu_new",
            "partial_rotary_factor": 0.5,
            "interleaved_rope": True,
            "parallel_residual": True,
        },
        id="codegen-tp4",
    ),
    pytest.param(
        "glm",
        "GlmPlugin",
        "glm",
        2,
        {"partial_rotary_factor": 0.5},
        {"partial_rotary_factor": 0.5, "interleaved_rope": True},
        id="glm-tp2",
    ),
    pytest.param(
        "internlm",
        "InternLMPlugin",
        "internlm2",
        4,
        {},
        {},
        id="internlm-tp4",
    ),
    pytest.param(
        "phi",
        "PhiPlugin",
        "phi3",
        4,
        {},
        {},
        id="phi-tp4",
    ),
    pytest.param(
        "stablelm",
        "StableLMPlugin",
        "stablelm",
        4,
        {"partial_rotary_factor": 0.75},
        {
            "norm_type": "layernorm",
            "mlp_type": "swiglu",
            "position_type": "rope",
            "partial_rotary_factor": 0.75,
        },
        id="stablelm-tp4",
    ),
    pytest.param(
        "starcoder2",
        "StarCoder2Plugin",
        "starcoder2",
        2,
        {},
        {
            "norm_type": "layernorm",
            "mlp_type": "gelu_fc",
            "position_type": "rope",
            "activation": "gelu_new",
        },
        id="starcoder2-tp2",
    ),
    pytest.param(
        "xglm",
        "XGLMPlugin",
        "xglm",
        4,
        {},
        {
            "norm_type": "layernorm",
            "mlp_type": "gelu_fc",
            "position_type": "learned",
            "activation": "gelu",
        },
        id="xglm-tp4",
    ),
]


def _config(model_type: str, tp_size: int, raw: dict[str, object]) -> ModelConfig:
    kv_heads = 2 if tp_size == 2 else 4
    return ModelConfig(
        model_type=model_type,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        raw=raw,
    )


@pytest.mark.parametrize(
    "family,plugin_class,model_type,tp_size,raw,expected_kwargs",
    _FAMILIES,
)
def test_decoder_family_plugin_routes_tp_build(
    monkeypatch,
    family: str,
    plugin_class: str,
    model_type: str,
    tp_size: int,
    raw: dict[str, object],
    expected_kwargs: dict[str, object],
) -> None:
    plugin_mod = importlib.import_module(
        f"tensorrt_model_connect.families.{family}.plugin")
    captured: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured["config"] = config
        captured["weights"] = weights
        captured["max_cache_length"] = max_cache_length
        captured["kwargs"] = kwargs
        return b"tp-plan"

    monkeypatch.setattr(
        plugin_mod,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )
    monkeypatch.setattr(
        plugin_mod,
        "build_dual_profile_tp_decoder_engine",
        fake_build,
    )

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=tp_size, rank=1)
    plan = getattr(plugin_mod, plugin_class)().build_engine(
        _config(model_type, tp_size, raw),
        WeightDict(),
        max_cache_length=17,
        precision="fp16",
        verbose=True,
        parallel_config=parallel,
    )

    assert plan == b"tp-plan"
    assert captured["max_cache_length"] == 17
    kwargs = captured["kwargs"]
    assert kwargs["precision"] == "fp16"
    assert kwargs["quant_ctx"] is None
    assert kwargs["verbose"] is True
    assert kwargs["parallel_config"] == parallel
    for key, expected in expected_kwargs.items():
        assert kwargs[key] == expected


@pytest.mark.parametrize(
    "family,plugin_class,model_type,tp_size,raw,expected_kwargs",
    _FAMILIES,
)
def test_decoder_family_plugin_rejects_quantized_tp(
    monkeypatch,
    family: str,
    plugin_class: str,
    model_type: str,
    tp_size: int,
    raw: dict[str, object],
    expected_kwargs: dict[str, object],
) -> None:
    del expected_kwargs
    plugin_mod = importlib.import_module(
        f"tensorrt_model_connect.families.{family}.plugin")
    monkeypatch.setattr(
        plugin_mod,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )

    with pytest.raises(ValueError, match="do not support quantization"):
        getattr(plugin_mod, plugin_class)().build_engine(
            _config(model_type, tp_size, raw),
            WeightDict(),
            max_cache_length=17,
            quant_ctx=object(),
            parallel_config=ParallelConfig(
                mode="tensor_parallel", tp_size=tp_size, rank=0),
        )


def test_standard_decoder_tp_shards_gelu_fc_input_bias_only() -> None:
    weights = WeightDict({
        "_attention_size": 16,
        "_kv_attention_size": 16,
        "_mlp_size": 32,
        "layer.0.w_fc1": np.zeros((16, 32), dtype=np.float32),
        "layer.0.fc1_bias": np.arange(32, dtype=np.float32),
        "layer.0.w_fc2": np.zeros((32, 16), dtype=np.float32),
        "layer.0.fc2_bias": np.arange(16, dtype=np.float32),
    })

    shard = shard_standard_decoder_weights(
        _config("codegen", 4, {}),
        weights,
        ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2),
    )

    np.testing.assert_array_equal(
        shard["layer.0.fc1_bias"],
        np.arange(16, 24, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        shard["layer.0.fc2_bias"],
        weights["layer.0.fc2_bias"],
    )
