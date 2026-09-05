# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current Phi-4 Multimodal builder and checkpoint contracts."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

try:
    from safetensors.numpy import save_file
except (ImportError, ModuleNotFoundError):
    pytest.skip("Phi-4 Multimodal tests require safetensors", allow_module_level=True)

from families.phi4_multimodal import model as family_model
from families.phi4_multimodal.config import ModelConfig


RNG = np.random.RandomState(42)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_safetensors(
    model_dir: Path,
    tensors: dict[str, np.ndarray],
    filename: str = "model.safetensors",
) -> None:
    save_file(tensors, str(model_dir / filename))


def _load_weights(model_dir: Path, config: ModelConfig):
    return family_model._Phi4MultimodalModel().load_weights(str(model_dir), config)


def test_phi4_multimodal_prefill_dispatches_to_dual_profile_builder(monkeypatch) -> None:
    module = importlib.import_module("families.phi4_multimodal.default_decoder")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"phi4-multimodal-prefill-plan"

    monkeypatch.setattr(module, "build_dual_profile_decoder_engine", fake_build)
    config = type("Config", (), {"raw": {"_decoder_engine_role": "prefill"}})()
    result = module.build_standard_decoder_engine(
        config,
        {},
        31,
        precision="fp16",
        embed_input=True,
        partial_rotary_factor=0.75,
    )

    assert result == b"phi4-multimodal-prefill-plan"
    kwargs = calls["build"][3]
    assert kwargs["embed_input"] is True
    assert kwargs["partial_rotary_factor"] == 0.75
    assert kwargs["profile_mode"] == "prefill"


def test_phi4_fp16_matmul_can_request_fp32_accumulation(monkeypatch) -> None:
    module = importlib.import_module("families.phi4_multimodal.graph_ops")
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
    module = importlib.import_module("families.phi4_multimodal.phi4mm_vision_builder")
    calls: list[dict[str, object]] = []

    def fake_matmul(*args, **kwargs):
        del args
        calls.append(kwargs)
        return "matmul"

    monkeypatch.setattr(module.graph_ops, "add_matmul_rhs_constant", fake_matmul)
    monkeypatch.setattr(
        module.graph_ops,
        "add_bias_sum",
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
    module = importlib.import_module("families.phi4_multimodal.graph_ops")
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

        def add_elementwise(
            self,
            lhs: FakeTensor,
            rhs: FakeTensor,
            operation: object,
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
        scale=8**-0.5,
    )

    assert accumulation_calls == [(half, half), (half, half)]
    assert elementwise_dtypes[0] == (half, half)
    assert output.dtype == half


def test_phi4_vision_norm_and_gelu_compute_in_fp32(monkeypatch) -> None:
    module = importlib.import_module("families.phi4_multimodal.graph_ops")
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
            self,
            lhs: FakeTensor,
            rhs: FakeTensor,
            operation: object,
        ) -> FakeLayer:
            del rhs, operation
            return FakeLayer(FakeTensor(lhs.dtype))

        def add_unary(self, tensor: FakeTensor, operation: object) -> FakeLayer:
            del operation
            unary_dtypes.append(tensor.dtype)
            return FakeLayer(FakeTensor(tensor.dtype))

        def add_normalization_v2(
            self,
            inp: FakeTensor,
            gamma: FakeTensor,
            beta: FakeTensor,
            axes: int,
        ) -> FakeLayer:
            del axes
            normalization_dtypes.append((inp.dtype, gamma.dtype, beta.dtype))
            return FakeLayer(FakeTensor(inp.dtype))

    monkeypatch.setattr(
        module,
        "add_constant",
        lambda network, shape, values, dtype: FakeTensor(
            module.trt.float32 if dtype is np.float32 else module.trt.float16
        ),
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
    gelu = module.add_gelu_erf(network, FakeTensor(half), dtype=np.float16)

    assert normalization_dtypes == [(module.trt.float32, module.trt.float32, module.trt.float32)]
    assert unary_dtypes == [module.trt.float32]
    assert norm.dtype == half
    assert gelu.dtype == half


def test_longrope_table_applies_frequency_and_attention_factors() -> None:
    from families.phi4_multimodal.graph_ops import make_rope_table_half_dim

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
        table[1],
        1.5 * np.cos(np.array([0.5, 0.25])),
        rtol=1e-6,
    )


class TestPhi4MultimodalModel:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 64, 32, 2, 4, 4, 64

    @classmethod
    def _make_text_tensors(cls):
        vocab = cls.VOCAB
        hidden = cls.HIDDEN
        head_dim = hidden // cls.HEADS
        q_dim = cls.HEADS * head_dim
        kv_dim = cls.KV_HEADS * head_dim
        tensors = {"model.embed_tokens.weight": _rand(vocab, hidden)}

        for index in range(cls.LAYERS):
            prefix = f"model.layers.{index}"
            tensors[f"{prefix}.input_layernorm.weight"] = _rand(hidden)
            tensors[f"{prefix}.post_attention_layernorm.weight"] = _rand(hidden)
            tensors[f"{prefix}.self_attn.qkv_proj.base_layer.weight"] = _rand(
                q_dim + 2 * kv_dim, hidden
            )
            tensors[f"{prefix}.self_attn.o_proj.base_layer.weight"] = _rand(hidden, hidden)
            tensors[f"{prefix}.mlp.gate_up_proj.base_layer.weight"] = _rand(2 * cls.MLP, hidden)
            tensors[f"{prefix}.mlp.down_proj.base_layer.weight"] = _rand(hidden, cls.MLP)

        tensors["model.norm.weight"] = _rand(hidden)
        tensors["lm_head.weight"] = _rand(vocab, hidden)
        return tensors

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

    def test_load_weights_keys(self, tmp_path):
        config = self._make_config()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make_text_tensors())

        weights = _load_weights(tmp_path, ModelConfig.from_dir(tmp_path))

        expected_keys = {
            "embedding",
            "final_norm",
            "w_out",
            "_attention_size",
            "_mlp_size",
            "_explicit_attention",
        }
        for index in range(self.LAYERS):
            expected_keys.update(
                {
                    f"layer.{index}.input_norm",
                    f"layer.{index}.post_attn_norm",
                    f"layer.{index}.w_q",
                    f"layer.{index}.w_k",
                    f"layer.{index}.w_v",
                    f"layer.{index}.w_o",
                    f"layer.{index}.w_gate",
                    f"layer.{index}.w_up",
                    f"layer.{index}.w_down",
                }
            )

        for key in expected_keys:
            assert key in weights, f"Missing weight key: {key}"
        assert weights["_explicit_attention"] is True

    def test_fused_qkv_split(self, tmp_path):
        config = self._make_config()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make_text_tensors())

        weights = _load_weights(tmp_path, ModelConfig.from_dir(tmp_path))

        head_dim = self.HIDDEN // self.HEADS
        q_dim = self.HEADS * head_dim
        kv_dim = self.KV_HEADS * head_dim
        assert weights["layer.0.w_q"].shape == (self.HIDDEN, q_dim)
        assert weights["layer.0.w_k"].shape == (self.HIDDEN, kv_dim)
        assert weights["layer.0.w_v"].shape == (self.HIDDEN, kv_dim)

    def test_vision_lora_is_merged_into_decoder_projection(self, tmp_path):
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

        weights = _load_weights(tmp_path, ModelConfig.from_dir(tmp_path))

        merged = base + 2.0 * (lora_b @ lora_a)
        q_rows = self.HEADS * (self.HIDDEN // self.HEADS)
        np.testing.assert_allclose(
            weights["layer.0.w_q"],
            merged[:q_rows].T,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_fused_gate_up_split(self, tmp_path):
        config = self._make_config()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make_text_tensors())

        weights = _load_weights(tmp_path, ModelConfig.from_dir(tmp_path))

        assert weights["layer.0.w_gate"].shape == (self.HIDDEN, self.MLP)
        assert weights["layer.0.w_up"].shape == (self.HIDDEN, self.MLP)
        assert weights["layer.0.w_down"].shape == (self.MLP, self.HIDDEN)

    def test_embedding_shape(self, tmp_path):
        config = self._make_config()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make_text_tensors())

        weights = _load_weights(tmp_path, ModelConfig.from_dir(tmp_path))

        assert weights["embedding"].shape == (self.VOCAB, self.HIDDEN)
        assert weights["w_out"].shape == (self.HIDDEN, self.VOCAB)

    def test_tied_embeddings(self, tmp_path):
        config = self._make_config()
        _write_config(tmp_path, config)
        tensors = self._make_text_tensors()
        del tensors["lm_head.weight"]
        _write_safetensors(tmp_path, tensors)

        weights = _load_weights(tmp_path, ModelConfig.from_dir(tmp_path))

        np.testing.assert_allclose(weights["w_out"], weights["embedding"].T, atol=1e-6)

    def test_vision_weight_prefix_is_canonicalized(self, tmp_path):
        tensor = _rand(4, 3)
        _write_safetensors(
            tmp_path,
            {
                "model.embed_tokens_extend.image_embed.img_projection.0.weight": tensor,
                "model.embed_tokens.weight": _rand(self.VOCAB, self.HIDDEN),
            },
        )

        weights = family_model._load_vision_weights(str(tmp_path))

        assert set(weights) == {"img_projection.0.weight"}
        np.testing.assert_array_equal(weights["img_projection.0.weight"], tensor)


def test_gqa_kv_stays_compact(tmp_path):
    vocab, hidden, layers = 64, 32, 1
    heads, kv_heads, mlp = 8, 4, 64
    head_dim = hidden // heads
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    config = {
        "model_type": "phi4mm",
        "vocab_size": vocab,
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "intermediate_size": mlp,
    }
    _write_config(tmp_path, config)
    prefix = "model.layers.0"
    tensors = {
        "model.embed_tokens.weight": _rand(vocab, hidden),
        f"{prefix}.input_layernorm.weight": _rand(hidden),
        f"{prefix}.post_attention_layernorm.weight": _rand(hidden),
        f"{prefix}.self_attn.qkv_proj.base_layer.weight": _rand(q_dim + 2 * kv_dim, hidden),
        f"{prefix}.self_attn.o_proj.base_layer.weight": _rand(hidden, hidden),
        f"{prefix}.mlp.gate_up_proj.base_layer.weight": _rand(2 * mlp, hidden),
        f"{prefix}.mlp.down_proj.base_layer.weight": _rand(hidden, mlp),
        "model.norm.weight": _rand(hidden),
        "lm_head.weight": _rand(vocab, hidden),
    }
    _write_safetensors(tmp_path, tensors)

    weights = _load_weights(tmp_path, ModelConfig.from_dir(tmp_path))

    assert weights["layer.0.w_k"].shape == (hidden, kv_dim)
    assert weights["layer.0.w_v"].shape == (hidden, kv_dim)
