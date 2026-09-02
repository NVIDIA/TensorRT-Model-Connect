# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Branch-focused tests for the Qwen3.8 family plugin.

Trace: ARCH-FAM-001, UD-FAM-QWEN3-8
Intent: Validate Qwen3.8 DeltaNet/attention layer routing normalization and weight loading branches
Preconditions: Mixed DeltaNet/attention layer types and synthetic tensors with partial keys are provided
Postconditions: Layer type aliases are normalized correctly and branch-specific weights load with fallback behavior
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

trt = pytest.importorskip(
    "tensorrt", reason="TensorRT is required for family builder tests"
)


try:
    from tensorrt_model_connect.config import ModelConfig
    import tensorrt_model_connect.families.qwen3_8 as qwen3_8
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _seq(*shape: int, start: int = 0) -> np.ndarray:
    size = int(np.prod(shape))
    return np.arange(start, start + size, dtype=np.float32).reshape(shape)


def _patch_tensor_io(monkeypatch: pytest.MonkeyPatch,
                     tensor_map: dict[str, np.ndarray]) -> None:
    monkeypatch.setattr(qwen3_8, "_open_safetensors", lambda _: ["reader"])
    monkeypatch.setattr(
        qwen3_8, "_has_tensor", lambda _readers, name: name in tensor_map)

    def _load(_readers, name: str):
        if name not in tensor_map:
            raise KeyError(name)
        return tensor_map[name]

    monkeypatch.setattr(qwen3_8, "_load_tensor", _load)


def test_parse_layer_types_normalizes_aliases():
    """Intent: validate routing normalization for mixed aliases.
    Preconditions: layer type strings include canonical, alias, and unknown values.
    Postconditions: aliases map to deltanet/attention and unknowns are lower-cased.
    """
    parsed = qwen3_8._parse_layer_types(
        ["linear", "FULL", "linear_attention", "full_attention", "Custom"])
    assert parsed == ["deltanet", "attention", "deltanet", "attention", "custom"]


def test_fp16_runtime_inputs_preserve_fp32_recurrent_state():
    """FP16 storage must not quantize the persistent DeltaNet state."""

    class _Tensor:
        def __init__(self, name: str, dtype):
            self.name = name
            self.dtype = dtype

    class _Layer:
        def __init__(self, output: _Tensor):
            self.output = output

        def get_output(self, index: int) -> _Tensor:
            assert index == 0
            return self.output

    class _Network:
        def __init__(self):
            self.cast_inputs: list[_Tensor] = []

        def add_cast(self, tensor: _Tensor, dtype):
            self.cast_inputs.append(tensor)
            return _Layer(_Tensor(f"{tensor.name}_cast", dtype))

    network = _Network()
    attention_mask = _Tensor("attention_mask", trt.float32)
    conv_state = _Tensor("conv_state", trt.float32)
    ssm_state = _Tensor("ssm_state", trt.float32)
    cache_k = _Tensor("cache_k", trt.float16)
    cache_v = _Tensor("cache_v", trt.float16)

    (
        prepared_mask,
        prepared_conv,
        prepared_ssm,
        prepared_cache_k,
        prepared_cache_v,
    ) = qwen3_8._prepare_runtime_inputs(
        network,
        trt.float16,
        attention_mask,
        [conv_state],
        [ssm_state],
        [cache_k],
        [cache_v],
    )

    assert prepared_mask.dtype == trt.float16
    assert prepared_conv[0].dtype == trt.float16
    assert prepared_cache_k[0].dtype == trt.float16
    assert prepared_cache_v[0].dtype == trt.float16
    assert prepared_ssm == [ssm_state]
    assert prepared_ssm[0].dtype == trt.float32
    assert ssm_state not in network.cast_inputs


def test_load_weights_mixed_branches_and_fallbacks(monkeypatch: pytest.MonkeyPatch):
    """Intent: execute DeltaNet + attention branches and optional-key fallbacks.
    Preconditions: one layer is DeltaNet, one layer is attention, with partial tensors missing.
    Postconditions: normalized weights/metadata are emitted with correct fallback behavior.
    """
    raw = {
        "text_config": {
            "layer_types": ["linear", "FULL"],
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 3,
            "rope_parameters": {
                "partial_rotary_factor": 0.5,
                "rope_theta": 321.0,
            },
        }
    }
    cfg = ModelConfig(
        model_type="qwen3_8",
        vocab_size=5,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_theta=9999.0,
        raw=raw,
    )

    tensors: dict[str, np.ndarray] = {
        # Embedding only under fallback key.
        "model.embed_tokens.weight": _seq(5, 8, start=0),
        # Layer 0 DeltaNet path.
        "model.language_model.layers.0.input_layernorm.weight": _seq(8, start=100),
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": _seq(
            8, 8, start=200
        ),
        "model.language_model.layers.0.linear_attn.in_proj_z.weight": _seq(
            4, 8, start=400
        ),
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": _seq(
            2, 8, start=500
        ),
        "model.language_model.layers.0.linear_attn.in_proj_b.weight": _seq(
            2, 8, start=600
        ),
        "model.language_model.layers.0.linear_attn.A_log": np.log(
            np.array([1.0, 2.0], dtype=np.float32)
        ),
        "model.language_model.layers.0.linear_attn.dt_bias": np.array(
            [0.25, -0.5], dtype=np.float32
        ),
        "model.language_model.layers.0.linear_attn.conv1d.weight": _seq(
            8, 1, 3, start=700
        ),
        # conv1d.bias intentionally omitted to exercise zero fallback.
        "model.language_model.layers.0.linear_attn.norm.weight": np.array(
            [0.5, 1.5], dtype=np.float32
        ),
        "model.language_model.layers.0.linear_attn.out_proj.weight": _seq(
            8, 4, start=800
        ),
        "model.language_model.layers.0.mlp.gate_proj.weight": _seq(6, 8, start=900),
        "model.language_model.layers.0.mlp.up_proj.weight": _seq(6, 8, start=1000),
        "model.language_model.layers.0.mlp.down_proj.weight": _seq(8, 6, start=1100),
        # Layer 1 attention path.
        # input_layernorm intentionally omitted to exercise ones fallback.
        "model.language_model.layers.1.post_attention_layernorm.weight": _seq(
            8, start=1200
        ),
        "model.language_model.layers.1.self_attn.q_proj.weight": _seq(
            16, 8, start=1300
        ),
        "model.language_model.layers.1.self_attn.k_proj.weight": _seq(4, 8, start=1500),
        "model.language_model.layers.1.self_attn.v_proj.weight": _seq(4, 8, start=1600),
        "model.language_model.layers.1.self_attn.o_proj.weight": _seq(8, 8, start=1700),
        "model.language_model.layers.1.self_attn.q_norm.weight": np.array(
            [0.1, 0.2, 0.3, 0.4], dtype=np.float32
        ),
        # k_norm intentionally omitted.
        # Layer 1 MLP intentionally omitted (gate check should skip all MLP loads).
    }
    _patch_tensor_io(monkeypatch, tensors)

    weights = qwen3_8.plugin.load_weights("/unused", cfg)

    np.testing.assert_allclose(weights["embedding"], tensors["model.embed_tokens.weight"])

    np.testing.assert_allclose(
        weights["layer.0.input_norm"],
        1.0 + tensors["model.language_model.layers.0.input_layernorm.weight"],
    )
    np.testing.assert_allclose(weights["layer.1.input_norm"], np.ones(8, dtype=np.float32))

    np.testing.assert_allclose(
        weights["layer.0.post_attn_norm"], np.ones(8, dtype=np.float32))
    np.testing.assert_allclose(
        weights["layer.1.post_attn_norm"],
        1.0 + tensors["model.language_model.layers.1.post_attention_layernorm.weight"],
    )

    np.testing.assert_allclose(
        weights["layer.0.conv1d_bias"], np.zeros(8, dtype=np.float32))
    np.testing.assert_allclose(
        weights["layer.0.deltanet_norm"],
        np.array([0.5, 1.5, 0.5, 1.5], dtype=np.float32),
    )
    np.testing.assert_allclose(
        weights["layer.0.A"], np.array([-1.0, -2.0], dtype=np.float32))

    q_raw = tensors["model.language_model.layers.1.self_attn.q_proj.weight"]
    q_reshaped = q_raw.reshape(2, 8, 8)
    expected_q = q_reshaped[:, :4, :].reshape(8, 8).T.astype(np.float32)
    expected_gate = q_reshaped[:, 4:, :].reshape(8, 8).T.astype(np.float32)
    np.testing.assert_allclose(weights["layer.1.w_q"], expected_q)
    np.testing.assert_allclose(weights["layer.1.w_gate_attn"], expected_gate)
    np.testing.assert_allclose(
        weights["layer.1.q_norm"],
        np.tile(
            1.0 + tensors["model.language_model.layers.1.self_attn.q_norm.weight"],
            2,
        ),
    )
    assert "layer.1.k_norm" not in weights

    assert "layer.0.w_gate" in weights
    assert "layer.0.w_up" in weights
    assert "layer.0.w_down" in weights
    assert "layer.1.w_gate" not in weights
    assert "layer.1.w_up" not in weights
    assert "layer.1.w_down" not in weights

    np.testing.assert_allclose(weights["final_norm"], np.ones(8, dtype=np.float32))
    np.testing.assert_allclose(
        weights["w_lm_head"], tensors["model.embed_tokens.weight"].T.astype(np.float32)
    )

    assert weights["_layer_types"] == ["deltanet", "attention"]
    assert weights["_num_mamba_layers"] == 1
    assert weights["_num_attention_layers"] == 1
    assert weights["_partial_rotary_factor"] == 0.5
    assert weights["_rope_theta"] == 321.0


def test_get_bundle_config_overrides_normalizes_hybrid_fields():
    """Intent: validate bundle-config normalization for hybrid runtime fields.
    Preconditions: text_config provides aliased layer types and linear dimensions.
    Postconditions: output reports normalized layer types, counts, and derived dims.
    """
    raw = {
        "text_config": {
            "layer_types": ["linear_attention", "full_attention", "unknown"],
            "linear_num_value_heads": 3,
            "linear_num_key_heads": 1,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 5,
        }
    }
    cfg = ModelConfig(
        model_type="qwen3_8",
        vocab_size=5,
        hidden_size=12,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=3,
        num_key_value_heads=1,
        raw=raw,
    )

    overrides = qwen3_8.plugin.get_bundle_config_overrides(cfg)
    assert overrides["layer_types"] == ["deltanet", "attention", "unknown"]
    assert overrides["num_mamba_layers"] == 1
    assert overrides["num_attention_layers"] == 1
    assert overrides["d_inner"] == 12
    assert overrides["mamba_d_state"] == 4
    assert overrides["mamba_d_conv"] == 5
    assert overrides["mamba_nheads"] == 3
    assert overrides["mamba_head_dim"] == 4
    assert overrides["conv_dim"] == 20


def _config_from_raw(raw: dict) -> ModelConfig:
    text_cfg = raw.get("text_config", {})
    return ModelConfig(
        model_type=raw.get("model_type", ""),
        architectures=raw.get("architectures", []),
        hidden_size=text_cfg.get("hidden_size", 8),
        num_hidden_layers=text_cfg.get("num_hidden_layers", 2),
        num_attention_heads=text_cfg.get("num_attention_heads", 2),
        raw=raw,
    )


_QWEN38_RAW = {
    "model_type": "qwen3_5",
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "output_gate_type": "swish",
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    },
}

_QWEN35_RAW = {
    "model_type": "qwen3_5",
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "mlp_only_layers": [],
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    },
}


def test_matches_config_claims_qwen38_and_releases_qwen35():
    """Qwen3.8 ships Qwen3.5's model_type and architecture strings verbatim.

    Intent: the config body, not the checkpoint strings, decides ownership.
    Preconditions: two configs identical in model_type/architectures, differing
      only by the Qwen3.8 `output_gate_type` marker and the Qwen3.5
      `mlp_only_layers` marker.
    Postconditions: this family claims only the Qwen3.8 config, so a genuine
      Qwen3.5 checkpoint stays free to fall through to the qwen3_5 family.
    """
    assert qwen3_8.plugin.matches_config(_config_from_raw(_QWEN38_RAW)) is True
    assert qwen3_8.plugin.matches_config(_config_from_raw(_QWEN35_RAW)) is False


def test_matches_config_rejects_other_qwen38_architectures():
    """The dense family must not claim its MoE or qwen4_exp siblings.

    Qwen3.8-2.4T-A95B and Qwen3.8-Flash-Next both carry `output_gate_type`, so
    the marker alone is not sufficient -- the architecture/model_type gate is
    what keeps them out.
    """
    moe = {
        "model_type": "qwen3_5_moe_text",
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "text_config": {"output_gate_type": "swish"},
    }
    flash_next = {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {"output_gate_type": "sigmoid"},
    }
    assert qwen3_8.plugin.matches_config(_config_from_raw(moe)) is False
    assert qwen3_8.plugin.matches_config(_config_from_raw(flash_next)) is False


def test_matches_by_id_does_not_claim_qwen35_strings():
    assert qwen3_8.plugin.matches("qwen3_8") is True
    assert qwen3_8.plugin.matches("qwen3.8") is True
    assert qwen3_8.plugin.matches("qwen3_5") is False


def test_bundle_overrides_publish_flat_decoder_dims():
    """The C++ runtime reads bundle config with a top-level nlohmann lookup.

    Intent: `hidden_size`/`num_attention_heads`/`num_key_value_heads`/`head_dim`
      must appear at the top level of the bundle config, not only under
      `text_config`.
    Postconditions: without these, `compute_kv_dim()` returns 0 and the KV cache
      allocates zero-sized tensors, so pipeline construction fails.
    """
    overrides = qwen3_8.plugin.get_bundle_config_overrides(
        _config_from_raw(_QWEN38_RAW))

    assert overrides["hidden_size"] == 5120
    assert overrides["num_attention_heads"] == 24
    assert overrides["num_key_value_heads"] == 4
    assert overrides["head_dim"] == 256
    assert overrides["num_hidden_layers"] == 64
    # kv_dim as the runtime computes it: num_key_value_heads * head_dim.
    assert overrides["num_key_value_heads"] * overrides["head_dim"] == 1024
    # eos_token_id stays with the builder, which sources the full stop-id list
    # from generation_config.json rather than the single text_config value.
    assert "eos_token_id" not in overrides


def test_mock_bundle_serializes_decoder_and_hybrid_config(tmp_path):
    """Exercise Qwen3.8's nested producer contract with mocked engines."""
    from tensorrt_model_connect.engine_builder import build_bundle

    layer_types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(64)
    ]
    # Qwen/Qwen3.8-27B at 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
    source_config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen3_5",
        "text_config": {
            "bos_token_id": 248044,
            "eos_token_id": 248044,
            "head_dim": 256,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "layer_types": layer_types,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "num_attention_heads": 24,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "output_gate_type": "swish",
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
            },
            "vocab_size": 248320,
        },
    }
    (tmp_path / "config.json").write_text(
        json.dumps(source_config),
        encoding="utf-8",
    )
    # Qwen3.8 terminates on 248046, which appears only here; text_config carries
    # the single id 248044.
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"bos_token_id": 248044, "eos_token_id": [248046, 248044]}),
        encoding="utf-8",
    )

    class MockQwen38Plugin:
        name = "qwen3_8"
        runtime_strategy = "qwen3_8_hybrid_mamba_attention"
        requires_tokenizer = False

        @staticmethod
        def load_weights(_model_dir, _config):
            return {}

        @staticmethod
        def build_engine(_config, _weights, _max_cache_length, **_kwargs):
            return b"MOCK_HYBRID_PLAN"

        @staticmethod
        def get_bundle_config_overrides(config):
            return qwen3_8.plugin.get_bundle_config_overrides(config)

    with (
        patch(
            "tensorrt_model_connect.engine_builder.find_plugin",
            return_value=MockQwen38Plugin(),
        ),
        patch(
            "tensorrt_model_connect.engine_builder._get_trt_version",
            return_value="11.1.0",
        ),
        patch(
            "tensorrt_model_connect.engine_builder._get_gpu_name",
            return_value="CPU unit mock",
        ),
        patch("tensorrt_model_connect.engine_builder.write_bundle") as write_bundle,
    ):
        build_bundle(
            str(tmp_path),
            str(tmp_path / "qwen38-27b.bundle"),
            max_cache_length=256,
        )

    sections = {
        section.name: section.data for section in write_bundle.call_args.args[2]
    }
    runtime_config = json.loads(sections["config.json"])

    # text_config survives untouched for the Python side.
    assert runtime_config["text_config"] == source_config["text_config"]

    # The flat decoder contract the strict C++ parser reads. None of these keys
    # exist at the top level of the source config.
    decoder_contract = {
        "vocab_size": 248320,
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "bos_token_id": 248044,
    }
    assert all(key not in source_config for key in decoder_contract)
    assert {key: runtime_config[key] for key in decoder_contract} == decoder_contract
    # compute_kv_dim() reads these two; zero here means a zero-sized KV cache.
    assert runtime_config["num_key_value_heads"] * runtime_config["head_dim"] == 1024

    # The divergence from qwen3_5: eos_token_id must NOT be republished as an
    # override. Overrides are merged last, so doing so would collapse the
    # generation_config list to the single text_config id and leave 248046
    # unmatched, running generation to max_new_tokens.
    assert runtime_config["eos_token_id"] == [248046, 248044]

    assert runtime_config["layer_types"] == [
        "attention" if layer_type == "full_attention" else "deltanet"
        for layer_type in layer_types
    ]
    assert runtime_config["num_mamba_layers"] == 48
    assert runtime_config["num_attention_layers"] == 16
    assert runtime_config["d_inner"] == 6144
    assert runtime_config["mamba_d_state"] == 128
    assert runtime_config["mamba_d_conv"] == 4
    assert runtime_config["mamba_nheads"] == 48
    assert runtime_config["mamba_head_dim"] == 128
    assert runtime_config["conv_dim"] == 10240


class _FakeFp8Tensor:
    """Stands in for a safetensors float8 tensor without needing torch."""

    def __init__(self, values: np.ndarray):
        self._values = values.astype(np.float32)
        self.dtype = "torch.float8_e4m3fn"

    def float(self):
        return self

    def numpy(self):
        return self._values


def test_fp8_weights_are_dequantized_with_their_block_scales(monkeypatch):
    """FP8 projections are stored with one scale per weight_block_size block.

    Returning the raw float8 values would look like a plausible weight tensor
    and corrupt the engine silently, so the loader must resolve the companion
    `_scale_inv` and apply it.
    """
    from tensorrt_model_connect.families.qwen3_8 import checkpoint_mapper as cm

    # 4x4 weight, 2x2 blocks -> one scale per 2x2 quadrant
    values = np.ones((4, 4), dtype=np.float32)
    scale_inv = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    store = {
        "w": _FakeFp8Tensor(values),
        "w_scale_inv": scale_inv,
    }

    monkeypatch.setattr(cm, "_has_tensor", lambda _r, name: name in store)
    monkeypatch.setattr(cm, "_get_raw_tensor", lambda _r, name: store[name])

    out = cm._load_tensor(["reader"], "w")
    expected = np.array([
        [2.0, 2.0, 3.0, 3.0],
        [2.0, 2.0, 3.0, 3.0],
        [4.0, 4.0, 5.0, 5.0],
        [4.0, 4.0, 5.0, 5.0],
    ], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


def test_fp8_weight_without_scales_is_rejected(monkeypatch):
    """An FP8 tensor missing its scales must fail loudly, not load unscaled."""
    from tensorrt_model_connect.families.qwen3_8 import checkpoint_mapper as cm

    store = {"w": _FakeFp8Tensor(np.ones((4, 4), dtype=np.float32))}
    monkeypatch.setattr(cm, "_has_tensor", lambda _r, name: name in store)
    monkeypatch.setattr(cm, "_get_raw_tensor", lambda _r, name: store[name])

    with pytest.raises(KeyError, match="no companion"):
        cm._load_tensor(["reader"], "w")


def test_non_fp8_weights_are_untouched(monkeypatch):
    """A bf16/fp32 checkpoint must not gain any scaling behaviour."""
    from tensorrt_model_connect.families.qwen3_8 import checkpoint_mapper as cm

    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    store = {"w": values}
    monkeypatch.setattr(cm, "_has_tensor", lambda _r, name: name in store)
    monkeypatch.setattr(cm, "_get_raw_tensor", lambda _r, name: store[name])

    np.testing.assert_allclose(cm._load_tensor(["reader"], "w"), values)


def test_block_scales_handle_a_trailing_partial_block():
    """The last block is clipped when the shape is not a multiple of the block."""
    from tensorrt_model_connect.families.qwen3_8 import checkpoint_mapper as cm

    values = np.ones((3, 3), dtype=np.float32)
    scale_inv = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    out = cm._apply_block_scales(values, scale_inv)
    expected = np.array([
        [2.0, 2.0, 3.0],
        [2.0, 2.0, 3.0],
        [4.0, 4.0, 5.0],
    ], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


class _FakeTensor:
    """Reader stand-in returning a fixed payload for any requested name."""

    def __init__(self, values, dtype: str = ""):
        self._values = values
        self.dtype = dtype

    def float(self):
        return self

    def numpy(self):
        return self._values


def _nvfp4_store(name, packed, group_scale, global_scale):
    from tensorrt_model_connect.families.qwen3_8 import checkpoint_mapper as cm
    return {
        name: _FakeTensor(packed),
        cm._scale_key(name, ".weight_scale"): group_scale,
        cm._scale_key(name, ".weight_scale_2"): np.array([global_scale], dtype=np.float32),
    }


def test_nvfp4_weights_are_unpacked_and_double_scaled(monkeypatch):
    """NVFP4 packs two E2M1 values per byte, low nibble first.

    A wrong nibble order or magnitude table still yields plausible numbers, so
    the expected values are spelled out rather than recomputed.
    """
    from tensorrt_model_connect.families.qwen3_8 import checkpoint_mapper as cm

    # Low nibble is the even column: 0x10 -> (0x0, 0x1) = (0.0, 0.5),
    # 0x32 -> (0x2, 0x3) = (1.0, 1.5), 0x9E -> (0xE, 0x9) = (-4.0, -0.5).
    packed = np.array([[0x10, 0x32], [0x9E, 0x00]], dtype=np.uint8)
    group_scale = np.array([[2.0], [10.0]], dtype=np.float32)  # one group of 4
    store = _nvfp4_store("m.weight", packed, group_scale, 3.0)

    monkeypatch.setattr(cm, "_has_tensor", lambda _r, n: n in store)
    monkeypatch.setattr(cm, "_get_raw_tensor", lambda _r, n: store[n])

    out = cm._load_tensor(["r"], "m.weight")
    expected = np.array([
        [0.0, 0.5, 1.0, 1.5],
        [-4.0, -0.5, 0.0, 0.0],
    ], dtype=np.float32) * np.array([[2.0], [10.0]], dtype=np.float32) * 3.0
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_nvfp4_without_group_scale_is_rejected(monkeypatch):
    from tensorrt_model_connect.families.qwen3_8 import checkpoint_mapper as cm

    name = "m.weight"
    store = {
        name: _FakeTensor(np.zeros((2, 2), dtype=np.uint8)),
        cm._scale_key(name, ".weight_scale_2"): np.array([1.0], dtype=np.float32),
    }
    monkeypatch.setattr(cm, "_has_tensor", lambda _r, n: n in store)
    monkeypatch.setattr(cm, "_get_raw_tensor", lambda _r, n: store[n])

    with pytest.raises(KeyError, match="no .*weight_scale"):
        cm._load_tensor(["r"], name)


def test_modelopt_per_tensor_fp8_scale_is_applied(monkeypatch):
    """ModelOpt FP8 uses one scalar scale, unlike Qwen's per-block scale_inv."""
    from tensorrt_model_connect.families.qwen3_8 import checkpoint_mapper as cm

    name = "m.weight"
    store = {
        name: _FakeFp8Tensor(np.full((2, 2), 3.0, dtype=np.float32)),
        cm._scale_key(name, ".weight_scale"): np.array([0.5], dtype=np.float32),
    }
    monkeypatch.setattr(cm, "_has_tensor", lambda _r, n: n in store)
    monkeypatch.setattr(cm, "_get_raw_tensor", lambda _r, n: store[n])

    np.testing.assert_allclose(
        cm._load_tensor(["r"], name), np.full((2, 2), 1.5, dtype=np.float32))


def test_e2m1_magnitudes_match_the_format():
    """0 and 0.5 are subnormal; 1, 1.5, 2, 3, 4, 6 are the normals."""
    from tensorrt_model_connect.families.qwen3_8 import checkpoint_mapper as cm

    nibbles = np.arange(16, dtype=np.uint8)
    decoded = cm._decode_e2m1(nibbles)
    expected = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    np.testing.assert_allclose(decoded[:8], expected)
    np.testing.assert_allclose(decoded[8:], -expected)
