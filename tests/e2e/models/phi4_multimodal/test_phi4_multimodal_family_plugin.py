# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Phi-4-multimodal family plugin — weight loading.

Creates synthetic model directories with the Phi-4-multimodal weight naming
convention (fused QKV + gate_up with LoRA base_layer infix), then verifies
the plugin correctly splits/transposes weights.

No GPU or TRT needed.

Trace: ARCH-FAM-001, UD-FAM-PHI4MM
Intent: Validate Phi-4-multimodal fused QKV split, gate_up split, and LoRA base_layer weight loading
Preconditions: Synthetic safetensors with Phi-4-multimodal weight naming (fused QKV + gate_up) are available
Postconditions: Plugin correctly splits fused weights and produces expected per-head Q/K/V and gate/up projections
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires safetensors", allow_module_level=True)

# ---- helpers ----

RNG = np.random.RandomState(42)


def test_phi4_multimodal_dispatches_native_split_builder(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.phi4_multimodal.default_dual_profile_decoder")
    family = importlib.import_module(
        "tensorrt_model_connect.families.phi4_multimodal.plugin")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"phi4-multimodal-dual-profile-plan"

    monkeypatch.setattr(module, "build_dual_profile_decoder_engine", fake_build)
    config = type("Config", (), {
        "model_type": "phi4mm",
        "max_position_embeddings": 32,
        "head_dim": 8,
        "raw": {
            "_decoder_engine_role": "decode",
            "partial_rotary_factor": 0.75,
        },
    })()
    result = family.plugin.build_engine(
        config, {}, 32, precision="fp16")

    assert result == b"phi4-multimodal-dual-profile-plan"
    kwargs = calls["build"][3]
    assert kwargs["partial_rotary_factor"] == 0.75
    assert kwargs["profile_mode"] == "decode"
    assert config.raw["_native_kv_cache_metadata"] == {
        "native_kv_contract_version": 1,
        "native_kv_cache": True,
    }


def test_phi4_fp16_matmul_can_request_fp32_accumulation(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.phi4_multimodal.graph_ops")
    matrix_inputs: list[tuple[object, object]] = []

    class FakeTensor:
        def __init__(self, dtype: object) -> None:
            self.dtype = dtype
            self.shape = (1, 4)

    class FakeLayer:
        def __init__(self, output: object) -> None:
            self.output = output

        def get_output(self, index: int) -> object:
            assert index == 0
            return self.output

    class FakeNetwork:
        def add_cast(self, tensor: FakeTensor, dtype: object) -> FakeLayer:
            return FakeLayer(FakeTensor(dtype))

        def add_matrix_multiply(
            self,
            lhs: FakeTensor,
            lhs_op: object,
            rhs: FakeTensor,
            rhs_op: object,
        ) -> FakeLayer:
            del lhs_op, rhs_op
            matrix_inputs.append((lhs.dtype, rhs.dtype))
            return FakeLayer(FakeTensor(lhs.dtype))

    monkeypatch.setattr(
        module,
        "add_constant",
        lambda network, shape, values, dtype: FakeTensor(module.trt.float16),
    )
    output = module.add_matmul_rhs_constant(
        FakeNetwork(),
        FakeTensor(module.trt.float16),
        4,
        4,
        np.ones((4, 4), dtype=np.float16),
        dtype=np.float16,
        fp32_accumulation=True,
    )

    assert matrix_inputs == [(module.trt.float32, module.trt.float32)]
    assert output.dtype == module.trt.float16


def test_phi4_vision_linear_requests_fp32_accumulation(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.phi4_multimodal.phi4mm_vision_builder")
    calls: list[dict[str, object]] = []

    def fake_matmul(*args, **kwargs):
        del args
        calls.append(kwargs)
        return "matmul"

    monkeypatch.setattr(module.graph_ops, "add_matmul_rhs_constant", fake_matmul)
    monkeypatch.setattr(
        module.graph_ops, "add_bias_sum",
        lambda network, result, width, bias, dtype: result,
    )

    result = module._linear(
        "network",
        "input",
        np.ones((3, 4), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.float16,
    )

    assert result == "matmul"
    assert calls == [{"dtype": np.float16, "fp32_accumulation": True}]


def test_phi4_siglip_attention_preserves_fp16_score_boundaries(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.phi4_multimodal.graph_ops")
    accumulation_calls: list[tuple[object, object]] = []
    elementwise_dtypes: list[tuple[object, object]] = []

    class FakeTensor:
        def __init__(self, dtype: object) -> None:
            self.dtype = dtype

    class FakeLayer:
        def __init__(self, output: FakeTensor) -> None:
            self.output = output
            self.axes = 0

        def get_output(self, index: int) -> FakeTensor:
            assert index == 0
            return self.output

    class FakeNetwork:
        def add_cast(self, tensor: FakeTensor, dtype: object) -> FakeLayer:
            return FakeLayer(FakeTensor(dtype))

        def add_matrix_multiply(
            self,
            lhs: FakeTensor,
            lhs_op: object,
            rhs: FakeTensor,
            rhs_op: object,
        ) -> FakeLayer:
            del lhs_op, rhs, rhs_op
            return FakeLayer(FakeTensor(lhs.dtype))

        def add_elementwise(
            self, lhs: FakeTensor, rhs: FakeTensor, operation: object,
        ) -> FakeLayer:
            del operation
            elementwise_dtypes.append((lhs.dtype, rhs.dtype))
            return FakeLayer(FakeTensor(lhs.dtype))

        def add_softmax(self, tensor: FakeTensor) -> FakeLayer:
            assert tensor.dtype == module.trt.float32
            return FakeLayer(FakeTensor(tensor.dtype))

    monkeypatch.setattr(
        module,
        "_scalar_constant_for_trt_dtype",
        lambda network, shape, value, dtype: FakeTensor(dtype),
    )

    def fake_accumulation(network, lhs, lhs_op, rhs, rhs_op):
        del network, lhs_op, rhs_op
        accumulation_calls.append((lhs.dtype, rhs.dtype))
        return FakeTensor(lhs.dtype)

    monkeypatch.setattr(
        module,
        "_add_matrix_multiply_with_fp32_accumulation",
        fake_accumulation,
    )
    half = module.trt.float16
    output = module.add_siglip_attention_core(
        FakeNetwork(),
        FakeTensor(half),
        FakeTensor(half),
        FakeTensor(half),
        mask=FakeTensor(half),
        scale=8 ** -0.5,
    )

    assert accumulation_calls == [(half, half), (half, half)]
    assert elementwise_dtypes[0] == (half, half)
    assert output.dtype == half


def test_phi4_vision_norm_and_gelu_compute_in_fp32(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.phi4_multimodal.graph_ops")
    normalization_dtypes: list[tuple[object, object, object]] = []
    unary_dtypes: list[object] = []

    class FakeTensor:
        def __init__(self, dtype: object) -> None:
            self.dtype = dtype
            self.shape = (1, 4)

    class FakeLayer:
        def __init__(self, output: FakeTensor) -> None:
            self.output = output
            self.epsilon = 0.0

        def get_output(self, index: int) -> FakeTensor:
            assert index == 0
            return self.output

    class FakeNetwork:
        def add_cast(self, tensor: FakeTensor, dtype: object) -> FakeLayer:
            return FakeLayer(FakeTensor(dtype))

        def add_elementwise(
            self, lhs: FakeTensor, rhs: FakeTensor, operation: object,
        ) -> FakeLayer:
            del rhs, operation
            return FakeLayer(FakeTensor(lhs.dtype))

        def add_unary(self, tensor: FakeTensor, operation: object) -> FakeLayer:
            del operation
            unary_dtypes.append(tensor.dtype)
            return FakeLayer(FakeTensor(tensor.dtype))

        def add_normalization_v2(
            self, inp: FakeTensor, gamma: FakeTensor, beta: FakeTensor, axes: int,
        ) -> FakeLayer:
            del axes
            normalization_dtypes.append((inp.dtype, gamma.dtype, beta.dtype))
            return FakeLayer(FakeTensor(inp.dtype))

    monkeypatch.setattr(
        module,
        "add_constant",
        lambda network, shape, values, dtype: FakeTensor(
            module.trt.float32 if dtype is np.float32 else module.trt.float16),
    )
    half = module.trt.float16
    network = FakeNetwork()
    norm = module.add_layer_norm_native(
        network,
        FakeTensor(half),
        4,
        np.ones(4, dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        1.0e-6,
        dtype=np.float16,
        fp32_compute=True,
    )
    gelu = module.add_gelu_erf(
        network, FakeTensor(half), dtype=np.float16)

    assert normalization_dtypes == [
        (module.trt.float32, module.trt.float32, module.trt.float32)]
    assert unary_dtypes == [module.trt.float32]
    assert norm.dtype == half
    assert gelu.dtype == half


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))


def test_longrope_table_applies_frequency_and_attention_factors() -> None:
    from tensorrt_model_connect.families.phi4_multimodal.graph_ops import (
        make_rope_table_half_dim,
    )

    table = make_rope_table_half_dim(
        2,
        head_dim=4,
        rope_theta=1.0,
        cosine=True,
        frequency_factors=[2.0, 4.0],
        attention_factor=1.5,
    )

    np.testing.assert_allclose(table[0], [1.5, 1.5])
    np.testing.assert_allclose(
        table[1], 1.5 * np.cos(np.array([0.5, 0.25])), rtol=1e-6)


def test_native_defaults_use_complete_model_context() -> None:
    from tensorrt_model_connect.families.phi4_multimodal import plugin

    config = type("Config", (), {
        "model_type": "phi4mm",
        "max_position_embeddings": 131072,
    })()
    assert plugin.default_build_precision(config) == "fp16"
    assert inspect.signature(plugin.build_engine).parameters[
        "precision"].default == "fp16"
    assert plugin.default_max_cache_length(config) == 131072
    assert plugin.supports_split_decoder_roles(config)
    assert plugin.supports_split_embed_input is True


def test_native_split_builder_has_no_legacy_kv_switch() -> None:
    from tensorrt_model_connect.families.phi4_multimodal.default_dual_profile_decoder import (
        build_dual_profile_decoder_engine,
    )

    signature = inspect.signature(build_dual_profile_decoder_engine)
    assert "native_kv_cache" not in signature.parameters
    assert "dynamic_kv_profile_rows" not in signature.parameters

    config = type("Config", (), {"model_type": "phi4mm"})()
    weights = {"embedding": object(), "final_norm": object()}
    with pytest.raises(ValueError, match="prefill.*decode"):
        build_dual_profile_decoder_engine(
            config, weights, 128, profile_mode="dual_profile")


def test_native_build_rejects_hidden_capacity_override() -> None:
    from tensorrt_model_connect.families.phi4_multimodal import plugin

    config = type("Config", (), {
        "model_type": "phi4mm",
        "max_position_embeddings": 131072,
        "head_dim": 128,
        "raw": {
            "_decoder_engine_role": "decode",
            "partial_rotary_factor": 0.75,
        },
    })()
    with pytest.raises(ValueError, match="must equal the model context"):
        plugin.build_engine(config, {}, 768, precision="fp16")


def test_native_build_requires_split_role() -> None:
    from tensorrt_model_connect.families.phi4_multimodal import plugin

    config = type("Config", (), {
        "model_type": "phi4mm",
        "max_position_embeddings": 128,
        "head_dim": 8,
        "raw": {"partial_rotary_factor": 0.75},
    })()
    with pytest.raises(ValueError, match="explicit split decoder role"):
        plugin.build_engine(config, {}, 128, precision="fp16")


def test_native_build_rejects_generic_dynamic_kv_request() -> None:
    from tensorrt_model_connect.families.phi4_multimodal import plugin

    config = type("Config", (), {
        "model_type": "phi4mm",
        "max_position_embeddings": 128,
        "head_dim": 8,
        "raw": {
            "_decoder_engine_role": "decode",
            "dynamic_kv_cache": True,
            "partial_rotary_factor": 0.75,
        },
    })()
    with pytest.raises(ValueError, match="dynamic KV bucket profiles"):
        plugin.build_engine(config, {}, 128, precision="fp16")


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_native_builder_engine_contract(role: str) -> None:
    import tensorrt as trt

    from tensorrt_model_connect.families.phi4_multimodal import plugin

    config = ModelConfig.from_json(json.dumps({
        "model_type": "phi4mm",
        "vocab_size": 32,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.75,
    }))
    config.raw["_decoder_engine_role"] = role
    rng = np.random.RandomState(7)
    weights = {
        "embedding": rng.randn(32, 128).astype(np.float32),
        "final_norm": np.ones(128, dtype=np.float32),
        "w_out": rng.randn(128, 32).astype(np.float32),
        "_attention_size": 128,
        "_mlp_size": 256,
        "layer.0.input_norm": np.ones(128, dtype=np.float32),
        "layer.0.post_attn_norm": np.ones(128, dtype=np.float32),
        "layer.0.w_q": rng.randn(128, 128).astype(np.float32),
        "layer.0.w_k": rng.randn(128, 128).astype(np.float32),
        "layer.0.w_v": rng.randn(128, 128).astype(np.float32),
        "layer.0.w_o": rng.randn(128, 128).astype(np.float32),
        "layer.0.w_gate": rng.randn(128, 256).astype(np.float32),
        "layer.0.w_up": rng.randn(128, 256).astype(np.float32),
        "layer.0.w_down": rng.randn(256, 128).astype(np.float32),
    }

    plan = plugin.build_engine(
        config, weights, max_cache_length=128, precision="fp16")
    logger = trt.Logger(trt.Logger.ERROR)
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    assert engine is not None
    assert engine.get_tensor_shape("cache_k_0") == (1, 1, 128, 128)
    assert engine.get_tensor_shape("present_k_0") == (1, 1, 128, 128)
    assert engine.get_aliased_input_tensor("present_k_0") == "cache_k_0"
    assert engine.get_aliased_input_tensor("present_v_0") == "cache_v_0"
    assert engine.get_tensor_shape("cache_write_indices") == (1,)
    assert engine.get_tensor_shape("key_value_lengths") == (1,)
    assert "attention_mask" not in {
        engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
    }


# =========================================================================
# Phi-4-multimodal text decoder weights
# =========================================================================

class TestPhi4MultimodalPlugin:
    """Phi-4-multimodal plugin: fused QKV split, gate_up split."""

    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 64, 32, 2, 4, 4, 64

    @classmethod
    def _make_text_tensors(cls):
        """Create synthetic text decoder tensors with fused QKV/gate_up."""
        vocab = cls.VOCAB
        hidden = cls.HIDDEN
        layers = cls.LAYERS
        heads = cls.HEADS
        kv_heads = cls.KV_HEADS
        mlp = cls.MLP
        head_dim = hidden // heads
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim

        t = {}
        t["model.embed_tokens.weight"] = _rand(vocab, hidden)

        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)

            # Fused QKV: [q_dim + 2*kv_dim, hidden] (base_layer for LoRA)
            qkv = _rand(q_dim + 2 * kv_dim, hidden)
            t[f"{p}.self_attn.qkv_proj.base_layer.weight"] = qkv

            t[f"{p}.self_attn.o_proj.base_layer.weight"] = _rand(hidden, hidden)

            # Fused gate_up: [2 * mlp, hidden] (base_layer for LoRA)
            t[f"{p}.mlp.gate_up_proj.base_layer.weight"] = _rand(2 * mlp, hidden)
            t[f"{p}.mlp.down_proj.base_layer.weight"] = _rand(hidden, mlp)

        t["model.norm.weight"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    @classmethod
    def _make_config(cls):
        return {
            "model_type": "phi4mm",
            "vocab_size": cls.VOCAB,
            "hidden_size": cls.HIDDEN,
            "num_hidden_layers": cls.LAYERS,
            "num_attention_heads": cls.HEADS,
            "num_key_value_heads": cls.KV_HEADS,
            "intermediate_size": cls.MLP,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "img_processor": {
                "image_size": 336,
                "patch_size": 14,
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "image_token_id": 200011,
            },
        }

    def test_matches(self):
        from tensorrt_model_connect.families.phi4_multimodal import plugin

        assert plugin.matches("phi4mm")
        assert plugin.matches("phi4_multimodal")
        assert not plugin.matches("phi")
        assert not plugin.matches("phimoe")
        assert not plugin.matches("qwen3")

    def test_load_weights_keys(self, tmp_path):
        from tensorrt_model_connect.families.phi4_multimodal import plugin

        config = self._make_config()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make_text_tensors())

        mc = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), mc)

        # Check that all expected keys exist
        expected_keys = {"embedding", "final_norm", "w_out",
                         "_attention_size", "_mlp_size"}
        for i in range(self.LAYERS):
            expected_keys.update({
                f"layer.{i}.input_norm",
                f"layer.{i}.post_attn_norm",
                f"layer.{i}.w_q",
                f"layer.{i}.w_k",
                f"layer.{i}.w_v",
                f"layer.{i}.w_o",
                f"layer.{i}.w_gate",
                f"layer.{i}.w_up",
                f"layer.{i}.w_down",
            })

        for key in expected_keys:
            assert key in weights, f"Missing weight key: {key}"
        assert "_explicit_attention" not in weights

    def test_fused_qkv_split(self, tmp_path):
        """Verify fused QKV is correctly split into Q, K, V."""
        from tensorrt_model_connect.families.phi4_multimodal import plugin

        config = self._make_config()
        _write_config(tmp_path, config)
        tensors = self._make_text_tensors()
        _write_safetensors(tmp_path, tensors)

        mc = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), mc)

        hidden = self.HIDDEN
        heads = self.HEADS
        head_dim = hidden // heads
        q_dim = heads * head_dim
        kv_dim = self.KV_HEADS * head_dim

        # Q should be [hidden, q_dim] after transpose
        assert weights["layer.0.w_q"].shape == (hidden, q_dim)
        assert weights["layer.0.w_k"].shape == (hidden, kv_dim)
        assert weights["layer.0.w_v"].shape == (hidden, kv_dim)

    def test_vision_lora_is_merged_into_decoder_projection(self, tmp_path):
        from tensorrt_model_connect.families.phi4_multimodal import plugin

        config = self._make_config()
        config["vision_lora"] = {"r": 2, "lora_alpha": 4}
        tensors = self._make_text_tensors()
        prefix = "model.layers.0.self_attn.qkv_proj"
        base = tensors[f"{prefix}.base_layer.weight"].copy()
        lora_a = _rand(2, self.HIDDEN)
        lora_b = _rand(base.shape[0], 2)
        tensors[f"{prefix}.lora_A.vision.weight"] = lora_a
        tensors[f"{prefix}.lora_B.vision.weight"] = lora_b
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        mc = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), mc)

        merged = base + 2.0 * (lora_b @ lora_a)
        q_rows = self.HEADS * (self.HIDDEN // self.HEADS)
        np.testing.assert_allclose(
            weights["layer.0.w_q"], merged[:q_rows].T, rtol=1e-6, atol=1e-6)

    def test_fused_gate_up_split(self, tmp_path):
        """Verify fused gate_up is correctly split into gate and up."""
        from tensorrt_model_connect.families.phi4_multimodal import plugin

        config = self._make_config()
        _write_config(tmp_path, config)
        tensors = self._make_text_tensors()
        _write_safetensors(tmp_path, tensors)

        mc = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), mc)

        hidden = self.HIDDEN
        mlp = self.MLP

        # gate: [hidden, mlp], up: [hidden, mlp] (transposed)
        assert weights["layer.0.w_gate"].shape == (hidden, mlp)
        assert weights["layer.0.w_up"].shape == (hidden, mlp)
        assert weights["layer.0.w_down"].shape == (mlp, hidden)

    def test_embedding_shape(self, tmp_path):
        from tensorrt_model_connect.families.phi4_multimodal import plugin

        config = self._make_config()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make_text_tensors())

        mc = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), mc)

        assert weights["embedding"].shape == (self.VOCAB, self.HIDDEN)
        assert weights["w_out"].shape == (self.HIDDEN, self.VOCAB)

    def test_tied_embeddings(self, tmp_path):
        """When lm_head.weight is missing, w_out should be tied to embedding."""
        from tensorrt_model_connect.families.phi4_multimodal import plugin

        config = self._make_config()
        _write_config(tmp_path, config)
        tensors = self._make_text_tensors()
        del tensors["lm_head.weight"]
        _write_safetensors(tmp_path, tensors)

        mc = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), mc)

        # w_out should be the transpose of the embedding
        embedding = weights["embedding"]
        w_out = weights["w_out"]
        np.testing.assert_allclose(w_out, embedding.T, atol=1e-6)

    def test_selected_fp32_decoder_layer_is_rejected(self, tmp_path):
        from tensorrt_model_connect.families.phi4_multimodal import plugin

        config = self._make_config()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make_text_tensors())

        mc = ModelConfig.from_dir(tmp_path)
        mc.raw["_fp32_layers"] = [1]
        mc.raw["_decoder_engine_role"] = "decode"
        weights = plugin.load_weights(str(tmp_path), mc)

        with pytest.raises(ValueError, match="FP32 layer overrides"):
            plugin.build_engine(
                mc, weights,
                max_cache_length=mc.max_position_embeddings,
                precision="fp16")

    def test_vl_config_matches_dynamic_hd_contract(self, tmp_path):
        from tensorrt_model_connect.families.phi4_multimodal import plugin

        config = self._make_config()
        _write_config(tmp_path, config)
        mc = ModelConfig.from_dir(tmp_path)
        vl_config = plugin.get_vl_config(mc)

        assert plugin.embed_input is True
        assert vl_config["preprocessor_type"] == "phi4_hd_chw"
        assert vl_config["image_token_id"] == 200010
        assert vl_config["num_image_pad_tokens"] == 721
        assert vl_config["image_token_str"] == "<|endoftext10|>"
        assert vl_config["vision_output_dim"] == self.HIDDEN

    def test_vision_weight_prefix_is_canonicalized(self, tmp_path):
        from tensorrt_model_connect.families.phi4_multimodal.plugin import (
            _load_vision_weights,
        )

        tensor = _rand(4, 3)
        _write_safetensors(tmp_path, {
            "model.embed_tokens_extend.image_embed.img_projection.0.weight": tensor,
            "model.embed_tokens.weight": _rand(self.VOCAB, self.HIDDEN),
        })

        weights = _load_vision_weights(str(tmp_path))
        assert set(weights) == {"img_projection.0.weight"}
        np.testing.assert_array_equal(weights["img_projection.0.weight"], tensor)

    def test_dynamic_hd_preprocessor_shape(self):
        from tensorrt_model_connect.families.phi4_multimodal.vl_debug_runner import (
            _preprocess_phi4_hd_chw,
        )

        image_path = Path(__file__).parent / "data" / "test_img.jpeg"
        pixel_values = _preprocess_phi4_hd_chw(str(image_path))

        assert pixel_values.shape == (9, 448, 448)
        assert pixel_values.dtype == np.float32


# =========================================================================
# Compact GQA/MQA K/V
# =========================================================================

class TestPhi4MultimodalGQA:
    """Test compact GQA when num_kv_heads != num_heads."""

    VOCAB, HIDDEN, LAYERS = 64, 32, 1
    HEADS, KV_HEADS, MLP = 8, 4, 64

    def test_gqa_kv_stays_compact(self, tmp_path):
        from tensorrt_model_connect.families.phi4_multimodal import plugin

        hidden = self.HIDDEN
        heads = self.HEADS
        kv_heads = self.KV_HEADS
        head_dim = hidden // heads
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim

        config = {
            "model_type": "phi4mm",
            "vocab_size": self.VOCAB,
            "hidden_size": hidden,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "intermediate_size": self.MLP,
        }
        _write_config(tmp_path, config)

        t = {}
        t["model.embed_tokens.weight"] = _rand(self.VOCAB, hidden)
        p = "model.layers.0"
        t[f"{p}.input_layernorm.weight"] = _rand(hidden)
        t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)
        t[f"{p}.self_attn.qkv_proj.base_layer.weight"] = _rand(
            q_dim + 2 * kv_dim, hidden)
        t[f"{p}.self_attn.o_proj.base_layer.weight"] = _rand(hidden, hidden)
        t[f"{p}.mlp.gate_up_proj.base_layer.weight"] = _rand(
            2 * self.MLP, hidden)
        t[f"{p}.mlp.down_proj.base_layer.weight"] = _rand(hidden, self.MLP)
        t["model.norm.weight"] = _rand(hidden)
        t["lm_head.weight"] = _rand(self.VOCAB, hidden)

        _write_safetensors(tmp_path, t)

        mc = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), mc)

        assert weights["layer.0.w_k"].shape == (hidden, kv_dim)
        assert weights["layer.0.w_v"].shape == (hidden, kv_dim)


# =========================================================================
# Plugin auto-discovery
# =========================================================================

class TestPhi4MultimodalDiscovery:
    """Verify the plugin is auto-discovered by the families package."""

    def test_find_plugin(self):
        from tensorrt_model_connect.families import find_plugin
        p = find_plugin("phi4mm")
        assert p is not None
        assert p.name == "phi4_multimodal"

    def test_find_plugin_alternate_name(self):
        from tensorrt_model_connect.families import find_plugin
        p = find_plugin("phi4_multimodal")
        assert p is not None
        assert p.name == "phi4_multimodal"

    def test_no_conflict_with_phi(self):
        """phi4mm should not match the regular phi plugin."""
        from tensorrt_model_connect.families import find_plugin
        p = find_plugin("phi4mm")
        assert p is not None
        assert p.name == "phi4_multimodal"
