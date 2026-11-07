# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer-selective debug outputs on the Qwen3 native KV path."""

from __future__ import annotations

import json

import numpy as np
import pytest

from families.qwen.build_routing import native_kv_build_capability
from families.qwen.config import ModelConfig

# Qwen3-0.6B architecture fields; routing decisions need no weights.
_QWEN3_0_6B = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "vocab_size": 151936,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 40960,
}


def _qwen3(**updates: object) -> ModelConfig:
    return ModelConfig.from_json(json.dumps({**_QWEN3_0_6B, **updates}))


def _builder():
    pytest.importorskip("tensorrt")
    from families.qwen import dual_profile_decoder_builder

    return dual_profile_decoder_builder


@pytest.mark.parametrize("debug_layer_outputs", [True, (0, 13)])
def test_debug_layer_outputs_keep_the_native_kv_path_eligible(debug_layer_outputs) -> None:
    capability = native_kv_build_capability(
        _qwen3(),
        precision="fp16",
        max_cache_length=1024,
        debug_layer_outputs=debug_layer_outputs,
    )

    assert capability.eligible, capability.reason


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (False, ()),
        (None, ()),
        ((), ()),
        (True, tuple(range(28))),
        ((13, 0), (0, 13)),
        ([27], (27,)),
    ],
)
def test_debug_layer_selection_is_normalized(requested, expected) -> None:
    assert _builder()._resolve_debug_layers(requested, 28) == expected


@pytest.mark.parametrize(
    ("requested", "message"),
    [
        ((28,), "outside"),
        ((-1,), "outside"),
        ((1, 1), "distinct"),
        ((True,), "must be an integer"),
        ((1.0,), "must be an integer"),
        ("", "sequence of layer indices"),
        (b"\x00", "sequence of layer indices"),
        ("0,13", "sequence of layer indices"),
        (13, "sequence of layer indices"),
    ],
)
def test_debug_layer_selection_rejects_invalid_requests(requested, message) -> None:
    with pytest.raises(ValueError, match=message):
        _builder()._resolve_debug_layers(requested, 28)


@pytest.mark.parametrize("selection", [(0, 13), True])
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_native_kv_build_forwards_the_layer_selection(monkeypatch, role, selection) -> None:
    _builder()
    from families.qwen import model

    calls = []
    monkeypatch.setattr(model, "validate_native_kv_weights", lambda config, weights: None)
    monkeypatch.setattr(
        model,
        "build_dual_profile_decoder_engine",
        lambda *args, **kwargs: calls.append(kwargs) or b"plan",
    )
    config = _qwen3()
    config.raw["_decoder_engine_role"] = role

    plan = model._QwenModel().build_engine(
        config, {}, 1024, precision="fp16", debug_layer_outputs=selection
    )

    assert plan == b"plan"
    assert calls[0]["debug_layer_outputs"] == selection
    assert calls[0]["profile_mode"] == role
    assert calls[0]["native_kv_cache"] is True


@pytest.mark.parametrize(
    ("config", "precision", "message"),
    [
        (ModelConfig.create_tiny("qwen2"), "fp16", r"native KV path$"),
        (_qwen3(), "fp32", "native Qwen3 requires FP16 or BF16"),
    ],
)
def test_layer_selection_off_the_native_path_fails_closed(config, precision, message) -> None:
    _builder()
    from families.qwen.model import _QwenModel

    with pytest.raises(NotImplementedError, match=message):
        _QwenModel().build_engine(config, {}, 64, precision=precision, debug_layer_outputs=(0,))


@pytest.mark.gpu
@pytest.mark.trt
def test_native_kv_engines_mark_only_the_selected_layers() -> None:
    builder = _builder()
    import tensorrt as trt

    hidden, vocab, mlp, heads, kv_heads, head_dim = 256, 64, 512, 2, 1, 128
    config = _qwen3(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=mlp,
        num_hidden_layers=2,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=64,
    )
    rng = np.random.default_rng(0)

    def weight(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    ones = np.ones(hidden, dtype=np.float32)
    weights = {
        "embedding": weight(vocab, hidden),
        "final_norm": ones,
        "w_out": weight(hidden, vocab),
    }
    for layer in range(2):
        prefix = f"layer.{layer}"
        weights |= {
            f"{prefix}.input_norm": ones,
            f"{prefix}.post_attn_norm": ones,
            f"{prefix}.w_q": weight(hidden, heads * head_dim),
            f"{prefix}.w_k": weight(hidden, kv_heads * head_dim),
            f"{prefix}.w_v": weight(hidden, kv_heads * head_dim),
            f"{prefix}.w_o": weight(heads * head_dim, hidden),
            f"{prefix}.w_gate": weight(hidden, mlp),
            f"{prefix}.w_up": weight(hidden, mlp),
            f"{prefix}.w_down": weight(mlp, hidden),
            f"{prefix}.q_norm": np.ones(heads * head_dim, dtype=np.float32),
            f"{prefix}.k_norm": np.ones(kv_heads * head_dim, dtype=np.float32),
        }
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

    def debug_outputs(role: str, selection) -> dict[str, tuple]:
        engine = runtime.deserialize_cuda_engine(
            builder.build_dual_profile_decoder_engine(
                config,
                weights,
                64,
                precision="fp16",
                profile_mode=role,
                native_kv_cache=True,
                debug_layer_outputs=selection,
            )
        )
        names = (engine.get_tensor_name(index) for index in range(engine.num_io_tensors))
        return {
            name: (engine.get_tensor_dtype(name), tuple(engine.get_tensor_shape(name)))
            for name in names
            if name.startswith("debug_")
        }

    assert debug_outputs("prefill", (1,)) == {"debug_hidden_1": (trt.float32, (-1, hidden))}
    assert debug_outputs("decode", (1,)) == {"debug_hidden_1": (trt.float32, (1, hidden))}
    assert debug_outputs("prefill", False) == {}
