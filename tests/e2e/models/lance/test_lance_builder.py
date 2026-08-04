# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the LANCE decoder builder."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


def _native_config(*, role: str = "decode") -> SimpleNamespace:
    return SimpleNamespace(
        model_type="lance",
        max_position_embeddings=128_000,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=128,
        raw={
            "_decoder_engine_role": role,
            "rope_scaling": {
                "type": "mrope",
                "mrope_section": [16, 24, 24],
            },
        },
    )


def test_lance_native_build_uses_full_context_split_role(
    monkeypatch,
) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.lance.plugin")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["kwargs"] = kwargs
        return b"lance-bf16-plan"

    monkeypatch.setattr(module, "build_dual_profile_decoder_engine", fake_build)
    config = _native_config(role="prefill")
    result = module.LancePlugin().build_engine(config, {}, 128_000)

    assert result == b"lance-bf16-plan"
    assert calls["kwargs"]["precision"] == "bf16"
    assert calls["kwargs"]["profile_mode"] == "prefill"
    assert calls["kwargs"]["max_prefill_length"] == 4096
    assert "native_kv_cache" not in calls["kwargs"]
    assert config.raw["_native_kv_cache_metadata"] == {
        "native_kv_contract_version": 1,
        "native_kv_cache": True,
    }


def test_lance_production_builder_has_no_legacy_kv_selector_or_concat_path() -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.lance.default_dual_profile_decoder")
    builder = module.build_dual_profile_decoder_engine
    source = inspect.getsource(builder)

    assert set(inspect.signature(builder).parameters) == {
        "config",
        "weights",
        "max_cache_length",
        "precision",
        "opt_prefill_length",
        "max_prefill_length",
        "verbose",
        "profile_mode",
    }
    assert "if native_kv_cache" not in source
    assert "add_concatenation" not in source
    assert "graph_ops.add_attention_from_rows" not in source
    assert "all_k_cat" not in source
    assert "all_v_cat" not in source
    assert "add_native_kv_cache_attention_from_rows" in source


def test_lance_defaults_hide_cache_build_flags() -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.lance.plugin")
    plugin = module.LancePlugin()
    config = _native_config()

    assert plugin.default_build_precision(config) == "bf16"
    assert plugin.default_max_cache_length(config) == 128_000
    assert plugin.supports_split_decoder_roles(config)


def test_lance_native_build_rejects_a_hidden_context_cap() -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.lance.plugin")
    with pytest.raises(ValueError, match="must equal the model context"):
        module.LancePlugin().build_engine(
            _native_config(), {}, 384, precision="bf16")


def test_lance_native_build_rejects_generic_dynamic_kv() -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.lance.plugin")
    config = _native_config()
    config.raw["_runtime_dynamic_kv_requested"] = True
    with pytest.raises(ValueError, match="dynamic KV bucket profiles"):
        module.LancePlugin().build_engine(config, {}, 128_000, precision="bf16")


def test_lance_runtime_reads_boolean_vision_contract() -> None:
    source = (
        Path(__file__).resolve().parents[4]
        / "src/runtime/models/lance/plugin.cpp"
    ).read_text(encoding="utf-8")
    assert 'extract_json_bool(ctx.config_json, "has_vision_engine", false)' in source
    assert 'extract_json_int(ctx.config_json, "has_vision_engine"' not in source
    assert "declared_in_config || plan != nullptr" in source


def test_lance_rope_table_can_match_bf16_inv_freq_buffer() -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.lance.graph_ops")
    regular = module.make_rope_table_half_dim(
        388, 128, 1_000_000.0, True)
    official_bf16 = module.make_rope_table_half_dim(
        388,
        128,
        1_000_000.0,
        True,
        round_inv_freq_to_bf16=True,
    )

    # Position zero is invariant; later positions expose the BF16 frequency
    # quantization performed by the official reference's model.to(bfloat16).
    np.testing.assert_array_equal(official_bf16[0], regular[0])
    assert np.max(np.abs(official_bf16[387] - regular[387])) > 0.25
    np.testing.assert_allclose(
        official_bf16[387, :4],
        np.array(
            [-0.83420676, -0.92246085, 0.92788374, 0.06237314],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_lance_vl_config_matches_official_x2t_image_framing() -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.lance.plugin")
    config = SimpleNamespace(
        raw={
            "vision_config": {
                "patch_size": 14,
                "spatial_merge_size": 2,
            },
            "image_token_id": 151655,
            "video_token_id": 151656,
        },
        hidden_size=2048,
    )

    vl_config = module.LancePlugin().get_vl_config(config)

    assert vl_config is not None
    assert vl_config["fixed_image_size"] == 448
    assert vl_config["num_image_pad_tokens"] == 256
    assert vl_config["image_token_id"] == 151656
    assert vl_config["image_token_str"] == "<|video_pad|>"
    assert vl_config["vl_prompt_template"] == (
        "<|im_start|>system\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|>{image_pads}<|vision_end|>"
        "{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_lance_native_prefill_slices_cache_to_mask_key_width(
    monkeypatch,
) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.lance.graph_ops")
    trt = pytest.importorskip("tensorrt")

    class FakeTensor:
        def __init__(self, name, dtype, shape=(), values=None):
            self.name = name
            self.dtype = dtype
            self.shape = shape
            self.values = values

    class FakeLayer:
        def __init__(self, output):
            self.output = output
            self.name = ""
            self.axis = 0
            self.inputs = {}

        def get_output(self, index):
            assert index == 0
            return self.output

        def set_input(self, index, value):
            self.inputs[index] = value

    class FakeNetwork:
        def __init__(self):
            self.mask_shape = None
            self.gathers = []

        def add_kv_cache_update(self, cache, update, indices, mode):
            del update, indices, mode
            return FakeLayer(FakeTensor("updated", cache.dtype, cache.shape))

        def add_attention_v2(self, *args):
            del args
            raise AssertionError("masked prefill must use decomposed attention")

        def add_shape(self, tensor):
            output = FakeTensor("mask_shape", trt.int64, (4,))
            self.mask_shape = output
            return FakeLayer(output)

        def add_gather(self, tensor, indices, axis):
            self.gathers.append((tensor, indices, axis))
            return FakeLayer(FakeTensor("gather", trt.int64, (1,)))

        def add_concatenation(self, tensors):
            del tensors
            return FakeLayer(FakeTensor("active_shape", trt.int64, (4,)))

        def add_slice(self, tensor, start, shape, stride):
            del start, shape, stride
            return FakeLayer(FakeTensor("active_cache", tensor.dtype, tensor.shape))

    network = FakeNetwork()

    def fake_constant(_network, shape, values, dtype=np.float32):
        return FakeTensor(
            "constant", dtype, shape, np.asarray(values).copy())

    monkeypatch.setattr(module, "add_constant", fake_constant)
    monkeypatch.setattr(
        module,
        "reshape_rows_to_heads_4d",
        lambda _network, tensor, *_args, **_kwargs: tensor,
    )
    monkeypatch.setattr(
        module,
        "reshape_heads_4d_to_rows",
        lambda _network, tensor, *_args, **_kwargs: tensor,
    )
    monkeypatch.setattr(
        module,
        "_repeat_kv_heads_4d",
        lambda _network, tensor, **_kwargs: tensor,
    )
    monkeypatch.setattr(
        module,
        "_add_decomposed_attention_core",
        lambda _network, q, _k, _v, **_kwargs: q,
    )

    bf16 = trt.bfloat16
    q = FakeTensor("q", bf16, (1, 16, -1, 128))
    update = FakeTensor("update", bf16, (1, 2, -1, 128))
    cache = FakeTensor("cache", bf16, (1, 2, 128_000, 128))
    write_indices = FakeTensor("write_indices", trt.int32, (1,))
    lengths = FakeTensor("lengths", trt.int32, (1,))
    mask = FakeTensor("mask", bf16, (1, 1, -1, -1))

    module.add_native_kv_cache_attention_from_rows(
        network,
        q,
        update,
        update,
        cache,
        cache,
        write_indices,
        lengths,
        mask,
        num_heads=16,
        num_kv_heads=2,
        head_dim=128,
        q_seq=None,
    )

    mask_width_gathers = [
        gather for gather in network.gathers if gather[0] is network.mask_shape
    ]
    assert len(mask_width_gathers) == 1
    _, width_index, axis = mask_width_gathers[0]
    assert axis == 0
    np.testing.assert_array_equal(width_index.values, np.array([3], np.int32))
