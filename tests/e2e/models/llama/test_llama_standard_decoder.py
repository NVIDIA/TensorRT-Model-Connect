# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for standard_decoder_builder.py — tensor naming contract.

Builds tiny engines and verifies all I/O tensor names match C++ expectations.
Requires TRT + GPU.

Trace: ARCH-GRP-001, UD-GRP-DECODER
Intent: Validate standard decoder builder I/O tensor naming contract against C++ runtime expectations
Preconditions: TRT and CUDA GPU are available; synthetic weight dicts match builder requirements
Postconditions: Built engine I/O tensor names exactly match the naming convention expected by C++ runtime
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")
from tests.builder.conftest import requires_trt


_MINITRON_LLAMA3_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
}


def _make_weights(hidden: int, vocab: int, num_layers: int,
                  attention_size: int, mlp_size: int,
                  *, mlp_type: str = "swiglu",
                  position_type: str = "rope",
                  has_bias: bool = False,
                  kv_attention_size: int | None = None) -> dict:
    """Create a minimal synthetic weight dict for the standard decoder builder."""
    from tensorrt_model_connect.checkpoint_mapper import WeightDict
    rng = np.random.RandomState(42)
    w = WeightDict()
    w["embedding"] = rng.randn(vocab, hidden).astype(np.float32)
    kv_attention_size = kv_attention_size or attention_size

    for i in range(num_layers):
        p = f"layer.{i}"
        w[f"{p}.input_norm"] = rng.randn(hidden).astype(np.float32)
        w[f"{p}.post_attn_norm"] = rng.randn(hidden).astype(np.float32)
        w[f"{p}.w_q"] = rng.randn(hidden, attention_size).astype(np.float32)
        w[f"{p}.w_k"] = rng.randn(hidden, kv_attention_size).astype(np.float32)
        w[f"{p}.w_v"] = rng.randn(hidden, kv_attention_size).astype(np.float32)
        w[f"{p}.w_o"] = rng.randn(attention_size, hidden).astype(np.float32)

        if mlp_type == "swiglu":
            w[f"{p}.w_gate"] = rng.randn(hidden, mlp_size).astype(np.float32)
            w[f"{p}.w_up"] = rng.randn(hidden, mlp_size).astype(np.float32)
            w[f"{p}.w_down"] = rng.randn(mlp_size, hidden).astype(np.float32)
        else:  # gelu_fc
            w[f"{p}.w_fc1"] = rng.randn(hidden, mlp_size).astype(np.float32)
            w[f"{p}.w_fc2"] = rng.randn(mlp_size, hidden).astype(np.float32)

    w["final_norm"] = rng.randn(hidden).astype(np.float32)
    w["w_out"] = rng.randn(hidden, vocab).astype(np.float32)
    w["_attention_size"] = attention_size
    w["_mlp_size"] = mlp_size

    if position_type == "learned":
        max_pos = 64
        w["position_embedding"] = rng.randn(max_pos, hidden).astype(np.float32)

    return w


def _get_io_names(engine_plan: bytes) -> tuple[list[str], list[str]]:
    """Deserialize engine plan and return (input_names, output_names)."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_plan)
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inputs.append(name)
        else:
            outputs.append(name)
    return inputs, outputs


def _deserialize_engine(engine_plan: bytes):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(engine_plan)


def test_llama3_rope_table_matches_hf_reference():
    """Minitron's 131K RoPE cache follows the HF Llama 3 frequency formula."""
    from tensorrt_model_connect.families.llama import graph_ops

    head_dim = 128
    rope_theta = 500000.0
    table_rows = 131072
    sample_positions = np.asarray(
        [0, 1, 2048, 8192, 32768, 65536, 131071],
        dtype=np.int64,
    )

    parameters = graph_ops.resolve_llama3_rope_parameters({
        "rope_scaling": _MINITRON_LLAMA3_ROPE_SCALING,
    })
    assert parameters == {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
    }

    # Independent NumPy transcription of transformers'
    # _compute_llama3_parameters. Keep this reference in the test so a change
    # to the production helper cannot update both sides of the comparison.
    dims = np.arange(0, head_dim, 2, dtype=np.float32)
    inv_freq = np.float32(1.0) / np.power(
        np.float32(rope_theta), dims / np.float32(head_dim))
    wavelengths = np.float32(2.0 * np.pi) / inv_freq
    low_wavelength = np.float32(8192.0)
    high_wavelength = np.float32(8192.0 / 4.0)

    reference_inv_freq = np.where(
        wavelengths > low_wavelength,
        inv_freq / np.float32(8.0),
        inv_freq,
    )
    smooth = (
        np.float32(8192.0) / wavelengths - np.float32(1.0)
    ) / np.float32(3.0)
    smoothed_inv_freq = (
        (np.float32(1.0) - smooth) * inv_freq / np.float32(8.0)
        + smooth * inv_freq
    )
    medium = (
        (wavelengths >= high_wavelength)
        & (wavelengths <= low_wavelength)
    )
    reference_inv_freq = np.where(
        medium, smoothed_inv_freq, reference_inv_freq)
    reference_angles = (
        sample_positions.astype(np.float32)[:, None]
        * reference_inv_freq[None, :]
    )

    actual_cos = graph_ops.make_llama3_rope_table_half_dim(
        table_rows,
        head_dim,
        rope_theta,
        True,
        **parameters,
    )
    actual_sin = graph_ops.make_llama3_rope_table_half_dim(
        table_rows,
        head_dim,
        rope_theta,
        False,
        **parameters,
    )

    assert np.any(wavelengths < high_wavelength)
    assert np.any(medium)
    assert np.any(wavelengths > low_wavelength)
    np.testing.assert_allclose(
        actual_cos[sample_positions],
        np.cos(reference_angles),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        actual_sin[sample_positions],
        np.sin(reference_angles),
        rtol=1e-6,
        atol=1e-6,
    )


def test_llama_plugin_selects_native_kv_cache(monkeypatch):
    """The native graph is a model-owned default, not a user build flag."""
    import importlib

    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.llama.plugin")

    captured = {}

    def _fake_build(config, weights, max_cache_length, **kwargs):
        captured.update(kwargs)
        return b"plan"

    monkeypatch.setattr(
        plugin_module, "build_standard_decoder_engine", _fake_build)
    from tensorrt_model_connect.config import ModelConfig
    config = ModelConfig(
        model_type="llama",
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        raw={
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 8.0,
                "original_max_position_embeddings": 8192,
            }
        },
    )
    result = plugin_module.LlamaPlugin().build_engine(
        config, {}, 131072, precision="bf16")

    assert result == b"plan"
    assert captured["native_kv_cache"] is True

    generic = ModelConfig.create_tiny("llama")
    plugin_module.LlamaPlugin().build_engine(
        generic, {}, 256, precision="bf16")
    assert captured["native_kv_cache"] is False


@requires_trt
class TestTensorNamingContract:
    """Verify that built engines have the exact I/O tensor names the C++ runtime expects."""

    def _build_engine(self, **kwargs):
        from tensorrt_model_connect.config import ModelConfig
        from tensorrt_model_connect.families.llama.standard_decoder_builder import build_standard_decoder_engine

        hidden, vocab, num_layers = 16, 32, 2
        num_heads = 4
        attention_size = hidden
        mlp_size = 32
        max_cache = 4

        config = ModelConfig(
            hidden_size=hidden,
            vocab_size=vocab,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
        )
        mlp_type = kwargs.get("mlp_type", "swiglu")
        position_type = kwargs.get("position_type", "rope")
        weights = _make_weights(
            hidden, vocab, num_layers, attention_size, mlp_size,
            mlp_type=mlp_type, position_type=position_type)

        return build_standard_decoder_engine(
            config, weights, max_cache, **kwargs)

    def test_default_rope_swiglu(self):
        """Default: RoPE + SwiGLU, standard I/O names."""
        plan = self._build_engine()
        inputs, outputs = _get_io_names(plan)

        assert "token_id" in inputs
        assert "position_id" in inputs
        assert "attention_mask" in inputs
        assert "cache_k_0" in inputs
        assert "cache_k_1" in inputs
        assert "cache_v_0" in inputs
        assert "cache_v_1" in inputs

        assert "logits" in outputs
        assert "present_k_0" in outputs
        assert "present_k_1" in outputs
        assert "present_v_0" in outputs
        assert "present_v_1" in outputs

    def test_layernorm_gelu_fc(self):
        """LayerNorm + gelu_fc MLP, same I/O names."""
        plan = self._build_engine(
            norm_type="layernorm", mlp_type="gelu_fc", activation="gelu_new")
        inputs, outputs = _get_io_names(plan)

        assert "token_id" in inputs
        assert "logits" in outputs
        assert "present_k_0" in outputs

    def test_learned_positions(self):
        """Learned position embeddings, same I/O names."""
        plan = self._build_engine(position_type="learned")
        inputs, outputs = _get_io_names(plan)

        assert "token_id" in inputs
        assert "position_id" in inputs
        assert "logits" in outputs

    def test_alibi_positions(self):
        """ALiBi positions, same I/O names."""
        plan = self._build_engine(position_type="alibi")
        inputs, outputs = _get_io_names(plan)

        assert "token_id" in inputs
        assert "position_id" in inputs
        assert "logits" in outputs

    def test_embed_input(self):
        """With embed_input=True, extra VL inputs appear."""
        plan = self._build_engine(embed_input=True)
        inputs, outputs = _get_io_names(plan)

        assert "input_embed" in inputs
        assert "use_input_embed" in inputs
        assert "token_id" in inputs
        assert "logits" in outputs

    def test_bf16_embed_input_keeps_external_features_fp32(self):
        """VL image features stay fp32 while reduced-precision cache uses bf16."""
        import tensorrt as trt
        from tensorrt_model_connect.config import ModelConfig
        from tensorrt_model_connect.families.llama.standard_decoder_builder import (
            build_standard_decoder_engine,
        )

        hidden, vocab, num_layers = 16, 32, 2
        num_heads = 4
        max_cache = 4
        config = ModelConfig(
            hidden_size=hidden,
            vocab_size=vocab,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
        )
        weights = _make_weights(hidden, vocab, num_layers, hidden, 32)

        plan = build_standard_decoder_engine(
            config, weights, max_cache, embed_input=True, precision="bf16")
        engine = _deserialize_engine(plan)

        assert engine.get_tensor_dtype("input_embed") == trt.float32
        assert engine.get_tensor_dtype("use_input_embed") == trt.float32
        assert engine.get_tensor_dtype("cache_k_0") == trt.bfloat16

    def test_debug_layer_outputs(self):
        """With debug_layer_outputs=True, per-layer debug outputs appear."""
        plan = self._build_engine(debug_layer_outputs=True)
        inputs, outputs = _get_io_names(plan)

        assert "debug_embed" in outputs
        assert "debug_hidden_0" in outputs
        assert "debug_hidden_1" in outputs
        assert "debug_post_attn_0" in outputs
        assert "debug_post_attn_1" in outputs
        assert "logits" in outputs


    def test_interleaved_rope(self):
        plan = self._build_engine(interleaved_rope=True)
        inputs, outputs = _get_io_names(plan)
        assert "logits" in outputs

    def test_partial_rotary(self):
        plan = self._build_engine(partial_rotary_factor=0.5)
        inputs, outputs = _get_io_names(plan)
        assert "logits" in outputs

    @pytest.mark.parametrize(
        ("engine_role", "expected_profiles"),
        [("dual_profile", 2), ("prefill", 1), ("decode", 1)],
    )
    def test_native_kv_cache_contract(self, engine_role, expected_profiles):
        """Llama native cache uses fixed model capacity and engine aliases."""
        from tensorrt_model_connect.config import ModelConfig
        from tensorrt_model_connect.families.llama.standard_decoder_builder import (
            build_standard_decoder_engine,
        )

        # Use a production-supported fused-attention geometry. The native KV
        # path is deliberately non-decomposable, so toy head_dim=4 shapes must
        # fail rather than silently falling back to primitive attention.
        hidden, vocab, num_layers = 512, 32, 1
        num_heads = 4
        num_kv_heads = 2
        head_dim = hidden // num_heads
        kv_attention_size = num_kv_heads * head_dim
        capacity = 8
        config = ModelConfig(
            model_type="llama",
            hidden_size=hidden,
            vocab_size=vocab,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            max_position_embeddings=131072,
        )
        config.raw["rope_scaling"] = dict(_MINITRON_LLAMA3_ROPE_SCALING)
        config.raw["_decoder_engine_role"] = engine_role
        weights = _make_weights(
            hidden, vocab, num_layers, hidden, 1024,
            kv_attention_size=kv_attention_size)

        plan = build_standard_decoder_engine(
            config,
            weights,
            capacity,
            precision="bf16",
            native_kv_cache=True,
        )
        engine = _deserialize_engine(plan)
        inputs, outputs = _get_io_names(plan)

        assert engine is not None
        assert engine.num_optimization_profiles == expected_profiles
        assert "cache_write_indices" in inputs
        assert "key_value_lengths" in inputs
        assert "attention_mask" not in inputs
        assert tuple(engine.get_tensor_shape("cache_k_0")) == (
            1, num_kv_heads, capacity, head_dim)
        assert tuple(engine.get_tensor_shape("cache_v_0")) == (
            1, num_kv_heads, capacity, head_dim)
        assert tuple(engine.get_tensor_shape("present_k_0")) == (
            1, num_kv_heads, capacity, head_dim)
        assert tuple(engine.get_tensor_shape("present_v_0")) == (
            1, num_kv_heads, capacity, head_dim)
        assert "present_k_0" in outputs
        assert "present_v_0" in outputs
        assert (
            engine.get_aliased_input_tensor("present_k_0")
            == "cache_k_0"
        )
        assert (
            engine.get_aliased_input_tensor("present_v_0")
            == "cache_v_0"
        )

    def test_dynamic_kv_cache_shapes(self):
        from tensorrt_model_connect.config import ModelConfig
        from tensorrt_model_connect.families.llama.standard_decoder_builder import build_standard_decoder_engine

        hidden, vocab, num_layers = 16, 32, 2
        num_heads = 4
        attention_size = hidden
        mlp_size = 32
        max_cache = 4

        config = ModelConfig(
            hidden_size=hidden,
            vocab_size=vocab,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
        )
        config.raw["dynamic_kv_cache"] = True
        config.raw["_dynamic_kv_opt_length"] = 2
        weights = _make_weights(hidden, vocab, num_layers, attention_size, mlp_size)

        plan = build_standard_decoder_engine(config, weights, max_cache)
        engine = _deserialize_engine(plan)

        assert engine is not None
        assert engine.num_optimization_profiles == 1
        assert tuple(engine.get_tensor_shape("attention_mask")) == (1, -1)
        assert tuple(engine.get_tensor_shape("cache_k_0")) == (-1, attention_size)
        assert tuple(engine.get_tensor_shape("cache_v_0")) == (-1, attention_size)

    def test_dynamic_kv_cache_multiple_profiles(self):
        from tensorrt_model_connect.config import ModelConfig
        from tensorrt_model_connect.families.llama.standard_decoder_builder import build_standard_decoder_engine

        hidden, vocab, num_layers = 16, 32, 2
        num_heads = 4
        attention_size = hidden
        mlp_size = 32
        max_cache = 4

        config = ModelConfig(
            hidden_size=hidden,
            vocab_size=vocab,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
        )
        config.raw["dynamic_kv_cache"] = True
        config.raw["_dynamic_kv_profile_rows"] = [4, 2, 3]
        weights = _make_weights(hidden, vocab, num_layers, attention_size, mlp_size)

        plan = build_standard_decoder_engine(config, weights, max_cache)
        engine = _deserialize_engine(plan)

        assert engine is not None
        assert engine.num_optimization_profiles == 3
        assert engine.get_tensor_profile_shape("attention_mask", 0) == [(1, 2), (1, 3), (1, 3)]
        assert engine.get_tensor_profile_shape("attention_mask", 1) == [(1, 2), (1, 4), (1, 4)]
        assert engine.get_tensor_profile_shape("attention_mask", 2) == [(1, 2), (1, 5), (1, 5)]
        assert engine.get_tensor_profile_shape("cache_k_0", 0) == [(1, attention_size),
                                                                    (2, attention_size),
                                                                    (2, attention_size)]
        assert engine.get_tensor_profile_shape("cache_k_0", 1) == [(1, attention_size),
                                                                    (3, attention_size),
                                                                    (3, attention_size)]
        assert engine.get_tensor_profile_shape("cache_k_0", 2) == [(1, attention_size),
                                                                    (4, attention_size),
                                                                    (4, attention_size)]

    def test_dynamic_kv_cache_rejects_alibi(self):
        from tensorrt_model_connect.config import ModelConfig
        from tensorrt_model_connect.families.llama.standard_decoder_builder import build_standard_decoder_engine

        hidden, vocab, num_layers = 16, 32, 2
        num_heads = 4
        attention_size = hidden
        mlp_size = 32
        max_cache = 4

        config = ModelConfig(
            hidden_size=hidden,
            vocab_size=vocab,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
        )
        config.raw["dynamic_kv_cache"] = True
        weights = _make_weights(hidden, vocab, num_layers, attention_size, mlp_size)

        with pytest.raises(ValueError, match="ALiBi"):
            build_standard_decoder_engine(config, weights, max_cache, position_type="alibi")
