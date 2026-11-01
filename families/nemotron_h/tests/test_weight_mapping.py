# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H hybrid layer and checkpoint mapping contracts."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for Nemotron-H mapping tests")

from families.nemotron_h import model  # noqa: E402
from families.nemotron_h.config import ModelConfig  # noqa: E402


def _seq(*shape: int, start: int = 0) -> np.ndarray:
    size = int(np.prod(shape))
    return np.arange(start, start + size, dtype=np.float32).reshape(shape)


def _patch_tensor_io(monkeypatch: pytest.MonkeyPatch, tensor_map: dict[str, np.ndarray]) -> None:
    monkeypatch.setattr(model, "_open_safetensors", lambda _: ["reader"])
    monkeypatch.setattr(model, "_has_tensor", lambda _readers, name: name in tensor_map)

    def _load(_readers, name: str):
        if name not in tensor_map:
            raise KeyError(name)
        return tensor_map[name]

    monkeypatch.setattr(model, "_load_tensor", _load)


def test_parse_layer_types_maps_valid_pattern_chars():
    """Intent: validate pattern-to-layer-type conversion.
    Preconditions: pattern includes only supported markers.
    Postconditions: every marker is mapped to its canonical layer name.
    """
    parsed = model._parse_layer_types("M-*-M")
    assert parsed == ["mamba2", "mlp", "attention", "mlp", "mamba2"]


@pytest.mark.parametrize("fp8_cache", [False, True])
def test_load_weights_mixed_layer_routing_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch, fp8_cache: bool,
):
    """Intent: execute mamba2/mlp/attention branches plus fallback code paths.
    Preconditions: three-layer hybrid pattern with missing optional mamba norm/final/lm_head keys.
    Postconditions: branch-specific keys are loaded and fallback values are synthesized correctly.
    """
    raw = {
        "hybrid_override_pattern": "M-*",
        "mamba_num_heads": 2,
        "mamba_head_dim": 3,
        "n_groups": 1,
        "ssm_state_size": 2,
        "conv_kernel": 2,
    }
    cfg = ModelConfig(
        model_type="nemotron_h",
        vocab_size=5,
        hidden_size=8,
        intermediate_size=10,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        raw=raw,
    )

    tensors: dict[str, np.ndarray] = {
        "backbone.embeddings.weight": _seq(5, 8, start=0),
        "backbone.layers.0.norm.weight": _seq(8, start=100),
        "backbone.layers.1.norm.weight": _seq(8, start=200),
        "backbone.layers.2.norm.weight": _seq(8, start=300),
        # Layer 0 (mamba2)
        "backbone.layers.0.mixer.in_proj.weight": _seq(18, 8, start=400),
        "backbone.layers.0.mixer.conv1d.weight": _seq(10, 1, 2, start=600),
        "backbone.layers.0.mixer.conv1d.bias": _seq(10, start=700),
        "backbone.layers.0.mixer.out_proj.weight": _seq(8, 6, start=800),
        "backbone.layers.0.mixer.A_log": np.log(np.array([1.0, 3.0], dtype=np.float32)),
        "backbone.layers.0.mixer.D": np.array([0.25, 0.5], dtype=np.float32),
        "backbone.layers.0.mixer.dt_bias": np.array([-0.1, 0.2], dtype=np.float32),
        # mixer.norm.weight intentionally omitted to exercise ones fallback.
        # Layer 1 (mlp)
        "backbone.layers.1.mixer.up_proj.weight": _seq(10, 8, start=900),
        "backbone.layers.1.mixer.down_proj.weight": _seq(8, 10, start=1000),
        # Layer 2 (attention)
        "backbone.layers.2.mixer.q_proj.weight": _seq(8, 8, start=1100),
        "backbone.layers.2.mixer.k_proj.weight": _seq(4, 8, start=1200),
        "backbone.layers.2.mixer.v_proj.weight": _seq(4, 8, start=1300),
        "backbone.layers.2.mixer.o_proj.weight": _seq(8, 8, start=1400),
    }
    if fp8_cache:
        tensors["backbone.layers.2.mixer.k_proj.k_scale"] = np.array(0.125, dtype=np.float32)
        tensors["backbone.layers.2.mixer.v_proj.v_scale"] = np.array(0.25, dtype=np.float32)
    _patch_tensor_io(monkeypatch, tensors)

    weights = model._NemotronHModel().load_weights("/unused", cfg)

    np.testing.assert_allclose(weights["embedding"], tensors["backbone.embeddings.weight"])
    np.testing.assert_allclose(
        weights["layer.0.input_norm"], tensors["backbone.layers.0.norm.weight"]
    )
    np.testing.assert_allclose(
        weights["layer.1.input_norm"], tensors["backbone.layers.1.norm.weight"]
    )
    np.testing.assert_allclose(
        weights["layer.2.input_norm"], tensors["backbone.layers.2.norm.weight"]
    )

    np.testing.assert_allclose(weights["layer.0.mamba_norm"], np.ones(6, dtype=np.float32))
    np.testing.assert_allclose(weights["layer.0.A"], np.array([-1.0, -3.0], dtype=np.float32))
    np.testing.assert_allclose(
        weights["layer.0.conv1d_weight"],
        tensors["backbone.layers.0.mixer.conv1d.weight"].reshape(10, 2),
    )

    assert "layer.1.w_up" in weights
    assert "layer.1.w_down" in weights
    assert "layer.1.w_q" not in weights

    assert weights["layer.2.w_q"].shape == (8, 8)
    assert weights["layer.2.w_k"].shape == (8, 4)
    assert weights["layer.2.w_v"].shape == (8, 4)
    assert weights["layer.2.w_o"].shape == (8, 8)

    np.testing.assert_allclose(weights["final_norm"], np.ones(8, dtype=np.float32))
    np.testing.assert_allclose(
        weights["w_lm_head"], tensors["backbone.embeddings.weight"].T.astype(np.float32)
    )

    assert weights["_layer_types"] == ["mamba2", "mlp", "attention"]
    assert weights["_num_mamba_layers"] == 1
    assert weights["_num_attention_layers"] == 1
    assert weights["_fp8_kv_cache"] is fp8_cache
    if fp8_cache:
        assert weights["layer.2.kv_scales"] == (1.0, 0.125, 0.25)
    else:
        assert "layer.2.kv_scales" not in weights
    assert weights["_d_inner"] == 6
    assert weights["_conv_dim"] == 10
    assert weights["_d_conv"] == 2

    # Only ordinary projection storage narrows, exactly as the FP16 graph
    # already does. Norm/recurrent values and explicit FP32 layers stay FP32.
    projections = {
        "layer.0.mamba_in_proj", "layer.0.mamba_out_proj",
        "layer.1.w_up", "layer.1.w_down",
        "layer.2.w_q", "layer.2.w_k", "layer.2.w_v", "layer.2.w_o",
    }
    for fp32_layers in ((), (1,)):
        cfg.raw["_fp32_layers"] = fp32_layers
        compact = model._NemotronHModel().load_weights("/unused", cfg, precision="fp16")
        for key, value in weights.items():
            if not isinstance(value, np.ndarray):
                assert compact[key] == value
                continue
            narrowed = key in projections and int(key.split(".")[1]) not in fp32_layers
            dtype = np.float16 if narrowed else value.dtype
            assert compact[key].dtype == dtype
            np.testing.assert_array_equal(compact[key], value.astype(dtype))
            if narrowed:
                assert compact[key].nbytes * 2 == value.nbytes


def test_load_weights_raises_for_pattern_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
):
    """Intent: ensure malformed hybrid patterns fail fast.
    Preconditions: pattern maps to fewer layer markers than num_hidden_layers.
    Postconditions: load_weights raises ValueError before tensor mapping proceeds.
    """
    cfg = ModelConfig(
        model_type="nemotron_h",
        vocab_size=5,
        hidden_size=8,
        intermediate_size=10,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        raw={"hybrid_override_pattern": "M-"},
    )
    monkeypatch.setattr(model, "_open_safetensors", lambda _: ["reader"])

    with pytest.raises(ValueError, match="Pattern length"):
        model._NemotronHModel().load_weights("/unused", cfg)


@pytest.mark.parametrize("algorithm", ["FP8", "NVFP4", "W4A16_NVFP4", "MIXED_PRECISION"])
def test_prepared_checkpoint_keeps_packed_linear_and_scales(tmp_path, monkeypatch, algorithm):
    import json
    import ml_dtypes
    from families.nemotron_h.checkpoint_mapper import _open_safetensors, _load_tensor, _transpose_2d
    from families.nemotron_h.prepared_quantization import PreparedLinear

    linear_algorithm = "NVFP4" if algorithm == "MIXED_PRECISION" else algorithm
    quantization = {"quant_algo": algorithm, "group_size": 16}
    if algorithm == "MIXED_PRECISION":
        quantization["quantized_layers"] = {"p": {"quant_algo": "NVFP4", "group_size": 16}}
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({
        "producer": {"name": "modelopt"},
        "quantization": quantization,
    }))
    # Minimal valid safetensors bytes exercise the actual training-framework-free FP8 reader.
    if algorithm == "FP8":
        values = {"p.weight": np.ones((32, 32), dtype=ml_dtypes.float8_e4m3fn),
                  "p.weight_scale": np.array([.5], np.float32)}
    else:
        values = {"p.weight": np.full((32, 16), 0x21, np.uint8),
                  "p.weight_scale": np.full((32, 2), 2, dtype=ml_dtypes.float8_e4m3fn),
                  "p.weight_scale_2": np.array([.25], np.float32)}
    if algorithm != "W4A16_NVFP4":
        values["p.input_scale"] = np.array([.125], np.float32)
    values["bf16_sample"] = np.array([.25, -2.0], dtype=ml_dtypes.bfloat16)
    values["backbone.embeddings.weight"] = np.ones((32, 32), dtype=np.float32)
    header, payload = {}, bytearray()
    for name, value in values.items():
        offset = len(payload)
        payload.extend(value.tobytes())
        dtype = ("F8_E4M3" if value.dtype == np.dtype(ml_dtypes.float8_e4m3fn)
                 else "BF16" if value.dtype == np.dtype(ml_dtypes.bfloat16)
                 else "U8" if value.dtype == np.uint8 else "F32")
        header[name] = {"dtype": dtype, "shape": list(value.shape),
                        "data_offsets": [offset, len(payload)]}
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (tmp_path / "model.safetensors").write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
    readers = _open_safetensors(tmp_path)
    for name, value in values.items():
        if header[name]["dtype"] not in {"F8_E4M3", "BF16"}:
            continue
        reader = readers.tensor_map[name]
        first, second = reader.get_tensor(name), reader.get_tensor(name)
        assert first.base is second.base is reader.storage
        assert not first.flags.writeable
        np.testing.assert_array_equal(first.view(np.uint8), value.view(np.uint8))
    weight = _load_tensor(readers, "p.weight")
    assert isinstance(weight, PreparedLinear)
    assert weight.shape == (32, 32)
    assert weight.format == linear_algorithm.lower()
    np.testing.assert_array_equal(weight.packed, values["p.weight"].view(np.uint8))
    assert weight.input_scale is None if algorithm == "W4A16_NVFP4" else weight.input_scale[0] == .125
    transposed = _transpose_2d(weight, "p")
    assert transposed.transposed and transposed.packed is weight.packed
    compact = model._load_projection(readers, "p.weight", "p", None, 32, 32, precision="fp16")
    assert isinstance(compact, PreparedLinear) and compact.transposed
    assert compact.format == weight.format
    np.testing.assert_array_equal(compact.packed, weight.packed)
    with pytest.raises(ValueError, match="already transposed"):
        transposed.transpose()

    from types import SimpleNamespace
    from families.nemotron_h.prepared_quantization import PreparedQuantContext

    calls = []
    dense_context = object()
    def dense_quantization(source, ops, requested, *, layer_types):
        calls.append((source, requested, layer_types))
        return dense_context
    monkeypatch.setattr(model, "calibrate_nemotron_h", dense_quantization)
    config = SimpleNamespace(raw={"hybrid_override_pattern": "M"}, num_hidden_layers=1)
    request = SimpleNamespace(quantization=None, tensor_parallel_size=1)
    policy, context = model._native_quantization(request, config, tmp_path, "fp16")
    assert policy.request_format == ("fp8" if algorithm == "FP8" else "nvfp4")
    assert isinstance(context, PreparedQuantContext)
    assert not calls
    request.quantization = "fp8" if algorithm == "FP8" else "nvfp4"
    policy, context = model._native_quantization(request, config, tmp_path, "fp16")
    assert policy.request_format == request.quantization
    if algorithm in {"W4A16_NVFP4", "MIXED_PRECISION"}:
        assert isinstance(context, PreparedQuantContext)
        assert not calls
    else:
        assert context is dense_context
        assert calls == [(tmp_path, request.quantization, ["mamba2"])]

    # Explicit legacy admission must reach the same actual reader without a
    # second, stricter metadata decision. Automatic admission stays strict.
    sidecar = tmp_path / "hf_quant_config.json"
    metadata = json.loads(sidecar.read_text())
    del metadata["producer"]
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="ModelOpt metadata"):
        _open_safetensors(tmp_path)
    request.quantization = None
    with pytest.raises(ValueError, match="ModelOpt metadata"):
        model._native_quantization(request, config, tmp_path, "fp16")
    request.quantization = "fp8" if algorithm == "FP8" else "nvfp4"
    config.raw["hybrid_override_pattern"] = "E"
    policy, context = model._native_quantization(request, config, tmp_path, "fp16")
    assert isinstance(context, PreparedQuantContext)
    from families.nemotron_h import checkpoint_mapper
    def no_second_policy_read(*args, **kwargs):
        pytest.fail("validated prepared policy must not be re-derived during loading")
    monkeypatch.setattr(checkpoint_mapper, "checkpoint_policy", no_second_policy_read)
    readers = _open_safetensors(tmp_path, policy=policy)
    assert readers.tensor_map["p.weight"].policy is policy
    np.testing.assert_array_equal(_load_tensor(readers, "p.weight").packed, weight.packed)
    minimal = ModelConfig(model_type="nemotron_h", vocab_size=32, hidden_size=32,
                          num_hidden_layers=0, raw={})
    loaded = model._NemotronHModel().load_weights(
        str(tmp_path), minimal, context, prepared_policy=policy
    )
    np.testing.assert_array_equal(loaded["embedding"], values["backbone.embeddings.weight"])

    # Paired admission must use that same explicit-legacy policy and reader.
    # Stop after loading: this checks metadata/bytes, not engine qualification.
    from families.nemotron_h import paired
    from families.nemotron_h.build_request import BuildRequest

    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text("{}")
    monkeypatch.setattr(ModelConfig, "from_dir", staticmethod(lambda _: minimal))
    monkeypatch.setattr(paired.DraftConfig, "from_dict", staticmethod(
        lambda *_: SimpleNamespace(query_rows=4)))
    monkeypatch.setattr(paired, "native_tokenizer_assets", lambda *_: {})
    original_load = model._NemotronHModel.load_weights

    class PairedWeightsLoaded(Exception):
        pass

    def load_paired_weights(self, source, config, quant_ctx=None, *, prepared_policy=None):
        assert prepared_policy is not None and prepared_policy.algorithm == algorithm
        mapped = original_load(self, source, config, quant_ctx, prepared_policy=prepared_policy)
        np.testing.assert_array_equal(mapped["embedding"], values["backbone.embeddings.weight"])
        raise PairedWeightsLoaded

    monkeypatch.setattr(model._NemotronHModel, "load_weights", load_paired_weights)
    for selection in (None, "none", "nvfp4"):
        paired_request = BuildRequest(tmp_path, tmp_path / "out", "nemotron_h",
                                      "text_generation", "fp16", quantization=selection)
        if selection != "nvfp4":
            with pytest.raises(ValueError, match="ModelOpt metadata"):
                paired.build(paired_request, None, draft, "dflash")
        elif algorithm == "FP8":
            with pytest.raises(ValueError, match="requires an NVFP4 target"):
                paired.build(paired_request, None, draft, "dflash")
        else:
            with pytest.raises(PairedWeightsLoaded):
                paired.build(paired_request, None, draft, "dflash")


def test_prepared_mixed_policy_preserves_activation_precision(tmp_path):
    import json
    from families.nemotron_h.prepared_quantization import checkpoint_policy

    policies = {"backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
                "backbone.layers.1.mixer.experts.0.up_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16}}
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({
        "producer": {"name": "modelopt"}, "quantization": {
            "quant_algo": "MIXED_PRECISION", "quantized_layers": policies,
            "kv_cache_quant_algo": "FP8",
        },
    }))
    policy = checkpoint_policy(tmp_path)
    assert policy.request_format == "nvfp4"
    assert policy.linear_format("backbone.layers.0.mixer.in_proj.weight") == "fp8"
    assert policy.linear_format("backbone.layers.1.mixer.experts.0.up_proj.weight") == "w4a16_nvfp4"
    assert policy.linear_format("lm_head.weight") is None


@pytest.mark.parametrize("problem", ["algorithm", "group", "conflict", "producer"])
def test_prepared_source_policy_rejects_inconsistent_metadata(tmp_path, problem):
    import json
    from families.nemotron_h.prepared_quantization import checkpoint_policy

    sidecar = {"producer": {"name": "modelopt"}, "quantization": {
        "quant_algo": "NVFP4", "group_size": 16,
    }}
    if problem == "algorithm":
        sidecar["quantization"]["quant_algo"] = "INT4"
    elif problem == "group":
        sidecar["quantization"]["group_size"] = 32
    elif problem == "producer":
        sidecar["producer"]["name"] = "unknown"
    else:
        (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {
            "quant_method": "modelopt", "quant_algo": "FP8",
        }}))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError):
        checkpoint_policy(tmp_path)
    with pytest.raises(ValueError):
        checkpoint_policy(tmp_path, allow_legacy_producer=True)


def test_load_moe_explicit_layers_and_latent_projections(monkeypatch):
    raw = {"layers_block_type": ["moe"], "n_routed_experts": 2, "moe_latent_size": 4}
    cfg = ModelConfig(model_type="nemotron_h", vocab_size=5, hidden_size=8,
                      num_hidden_layers=1, raw=raw)
    tensors = {"backbone.embeddings.weight": _seq(5, 8),
               "backbone.layers.0.norm.weight": _seq(8),
               "backbone.layers.0.mixer.gate.weight": _seq(2, 8),
               "backbone.layers.0.mixer.gate.e_score_correction_bias": _seq(2)}
    for i in range(2):
        tensors[f"backbone.layers.0.mixer.experts.{i}.up_proj.weight"] = _seq(6, 4, start=i)
        tensors[f"backbone.layers.0.mixer.experts.{i}.down_proj.weight"] = _seq(4, 6, start=i)
    tensors["backbone.layers.0.mixer.shared_experts.up_proj.weight"] = _seq(12, 8)
    tensors["backbone.layers.0.mixer.shared_experts.down_proj.weight"] = _seq(8, 12)
    tensors["backbone.layers.0.mixer.fc1_latent_proj.weight"] = _seq(4, 8)
    tensors["backbone.layers.0.mixer.fc2_latent_proj.weight"] = _seq(8, 4)
    _patch_tensor_io(monkeypatch, tensors)
    weights = model._NemotronHModel().load_weights("/unused", cfg)
    assert weights["_layer_types"] == ["moe"]
    assert weights["layer.0.expert.1.up"].shape == (4, 6)
    assert weights["layer.0.shared.down"].shape == (12, 8)
    np.testing.assert_array_equal(weights["layer.0.router"], tensors["backbone.layers.0.mixer.gate.weight"].T)
    np.testing.assert_array_equal(weights["layer.0.latent_in"], tensors["backbone.layers.0.mixer.fc1_latent_proj.weight"].T)
    np.testing.assert_array_equal(weights["layer.0.latent_out"], tensors["backbone.layers.0.mixer.fc2_latent_proj.weight"].T)


def test_native_layer_specs_do_not_drop_moe_or_unknown_layers():
    assert model._parse_layer_types("ME*") == ["mamba2", "moe", "attention"]
    assert model._config_layer_types({"layer_types": ["mamba", "moe", "attention"]}, 3) == ["mamba2", "moe", "attention"]
    with pytest.raises(ValueError, match="Unsupported"):
        model._config_layer_types({"hybrid_override_pattern": "Mx"}, 2)
    with pytest.raises(ValueError, match="Conflicting"):
        model._config_layer_types({"hybrid_override_pattern": "ME", "layer_types": ["mamba", "mlp"]}, 2)
