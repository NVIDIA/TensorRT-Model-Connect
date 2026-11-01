# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared payload bytes and scales are preserved by the owning reader."""

import numpy as np
import pytest

from families.nemotron_omni.checkpoint_mapper import _load_projection


@pytest.mark.parametrize("algorithm", ["FP8", "NVFP4", "W4A16_NVFP4", "MIXED_PRECISION"])
def test_prepared_checkpoint_keeps_packed_linear_and_scales(tmp_path, algorithm):
    import json
    import ml_dtypes
    from families.nemotron_omni.checkpoint_mapper import (
        _open_safetensors,
        _load_tensor,
        _transpose_2d,
    )
    from families.nemotron_omni.prepared_quantization import PreparedLinear

    linear_algorithm = "NVFP4" if algorithm == "MIXED_PRECISION" else algorithm
    quantization = {"quant_algo": algorithm, "group_size": 16}
    if algorithm == "MIXED_PRECISION":
        quantization["quantized_layers"] = {"p": {"quant_algo": "NVFP4", "group_size": 16}}
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "NemotronH_Nano_Omni_Reasoning_V3"})
    )
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": "modelopt"},
                "quantization": quantization,
            }
        )
    )
    # Minimal valid safetensors bytes exercise the actual training-framework-free FP8 reader.
    if algorithm == "FP8":
        values = {
            "p.weight": np.ones((32, 32), dtype=ml_dtypes.float8_e4m3fn),
            "p.weight_scale": np.array([0.5], np.float32),
        }
    else:
        values = {
            "p.weight": np.full((32, 16), 0x21, np.uint8),
            "p.weight_scale": np.full((32, 2), 2, dtype=ml_dtypes.float8_e4m3fn),
            "p.weight_scale_2": np.array([0.25], np.float32),
        }
    if algorithm != "W4A16_NVFP4":
        values["p.input_scale"] = np.array([0.125], np.float32)
    values["bf16_sample"] = np.array([0.25, -2.0], dtype=ml_dtypes.bfloat16)
    values["language_model.backbone.embeddings.weight"] = np.ones((32, 32), dtype=np.float32)
    header, payload = {}, bytearray()
    for name, value in values.items():
        offset = len(payload)
        payload.extend(value.tobytes())
        dtype = (
            "F8_E4M3"
            if value.dtype == np.dtype(ml_dtypes.float8_e4m3fn)
            else "BF16"
            if value.dtype == np.dtype(ml_dtypes.bfloat16)
            else "U8"
            if value.dtype == np.uint8
            else "F32"
        )
        header[name] = {
            "dtype": dtype,
            "shape": list(value.shape),
            "data_offsets": [offset, len(payload)],
        }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (tmp_path / "model.safetensors").write_bytes(
        len(encoded).to_bytes(8, "little") + encoded + payload
    )
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
    assert (
        weight.input_scale is None if algorithm == "W4A16_NVFP4" else weight.input_scale[0] == 0.125
    )
    transposed = _transpose_2d(weight, "p", precision="fp16")
    assert transposed.transposed and transposed.packed is weight.packed
    compact = _load_projection(readers, "p.weight", "p", None, 32, 32, precision="fp16")
    assert isinstance(compact, PreparedLinear) and compact.transposed
    assert compact.format == weight.format
    np.testing.assert_array_equal(compact.packed, weight.packed)
    with pytest.raises(ValueError, match="already transposed"):
        transposed.transpose()


def test_native_composition_threads_prepared_context(tmp_path, monkeypatch):
    """Reject the full-build regression before allocating or loading large weights."""
    import json
    from types import SimpleNamespace
    from tensorrt_model_connect import BuildRequest
    from families.nemotron_omni import native_builder
    from families.nemotron_omni.prepared_quantization import PreparedQuantContext
    from .test_config import fixture

    (tmp_path / "config.json").write_text(json.dumps(fixture()))
    monkeypatch.setattr(
        native_builder, "checkpoint_policy", lambda _: SimpleNamespace(request_format="nvfp4")
    )
    monkeypatch.setattr(native_builder, "_open_safetensors", lambda *a, **kw: object())
    context = []

    def load(self, *args, quant_ctx=None, **kwargs):
        assert isinstance(quant_ctx, PreparedQuantContext)
        context.append(quant_ctx)
        return {"_fp8_kv_cache": False}

    class ReachedPreparedGraph(Exception):
        pass

    def engine(self, *args, quant_ctx=None, **kwargs):
        assert quant_ctx is context[0]
        raise ReachedPreparedGraph

    monkeypatch.setattr(native_builder.OmniDecoder, "load_weights", load)
    monkeypatch.setattr(native_builder.OmniDecoder, "get_bundle_config_overrides", lambda *args: {})
    monkeypatch.setattr(native_builder.OmniDecoder, "build_engine", engine)
    writer = SimpleNamespace(set_header=lambda **kwargs: None, add_bytes=lambda *args: None)
    request = BuildRequest(
        tmp_path,
        tmp_path / "model.bundle",
        "nemotron_omni",
        "text_generation",
        "fp16",
        max_sequence_length=32,
        quantization="nvfp4",
    )
    with pytest.raises(ReachedPreparedGraph):
        native_builder.build_native(request, writer)


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("fp32_layers", [[], [0], [1], [0, 1]])
def test_decoder_dense_weights_match_graph_precision(tmp_path, precision, fp32_layers):
    from safetensors.numpy import save_file
    from families.nemotron_omni.config import DecoderConfig
    from families.nemotron_omni.decoder import OmniDecoder

    config = DecoderConfig(
        raw={
            "hybrid_override_pattern": "E",
            "n_routed_experts": 2,
            "moe_latent_size": 2,
            "_fp32_layers": fp32_layers,
        },
        vocab_size=8,
        hidden_size=4,
        intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        rms_norm_eps=1e-5,
        max_position_embeddings=16,
    )
    prefix = "language_model.backbone.layers.0"
    mixer = f"{prefix}.mixer"
    shapes = {
        "language_model.backbone.embeddings.weight": (8, 4),
        "language_model.backbone.norm_f.weight": (4,),
        "language_model.lm_head.weight": (8, 4),
        f"{prefix}.norm.weight": (4,),
        f"{mixer}.gate.weight": (2, 4),
        f"{mixer}.gate.e_score_correction_bias": (2,),
        f"{mixer}.fc1_latent_proj.weight": (2, 4),
        f"{mixer}.fc2_latent_proj.weight": (4, 2),
        f"{mixer}.shared_experts.up_proj.weight": (6, 4),
        f"{mixer}.shared_experts.down_proj.weight": (4, 6),
    }
    for expert in range(2):
        shapes[f"{mixer}.experts.{expert}.up_proj.weight"] = (6, 2)
        shapes[f"{mixer}.experts.{expert}.down_proj.weight"] = (2, 6)
    values = {key: np.full(shape, 0.12345, np.float32) for key, shape in shapes.items()}
    save_file(values, str(tmp_path / "model.safetensors"))
    weights = OmniDecoder().load_weights(tmp_path, config, precision=precision)
    dtype = np.float32 if precision == "fp32" or 0 in fp32_layers else np.float16
    for key in (
        "expert.0.up",
        "expert.0.down",
        "expert.1.up",
        "expert.1.down",
        "shared.up",
        "shared.down",
        "latent_in",
        "latent_out",
    ):
        value = weights[f"layer.0.{key}"]
        assert value.dtype == dtype
        np.testing.assert_array_equal(value, np.full(value.shape, 0.12345, dtype))
    assert weights["layer.0.router"].dtype == np.float32
    head_dtype = np.float32 if precision == "fp32" or 1 in fp32_layers else np.float16
    assert weights["w_lm_head"].dtype == head_dtype
    np.testing.assert_array_equal(
        weights["w_lm_head"], values["language_model.lm_head.weight"].T.astype(head_dtype)
    )


@pytest.mark.parametrize(
    "scales,missing",
    [
        ({}, None),
        ({"0.q": 0.125}, None),
        ({"0.k": 0.25, "0.v": 0.5, "1.k": 0.25, "1.v": 0.5}, None),
        ({"0.v": 0.5}, "0.mixer.k_proj.k_scale"),
        ({"0.k": 0.25}, "0.mixer.v_proj.v_scale"),
        ({"0.k": 0.25, "0.v": 0.5}, "1.mixer.k_proj.k_scale"),
        ({"0.k": 0.25, "0.v": 0.5, "1.k": 0.25}, "1.mixer.v_proj.v_scale"),
    ],
)
def test_decoder_rejects_partial_fp8_kv_scales(tmp_path, scales, missing):
    from safetensors.numpy import save_file
    from families.nemotron_omni.config import DecoderConfig
    from families.nemotron_omni.decoder import OmniDecoder

    config = DecoderConfig(
        raw={"hybrid_override_pattern": "**"},
        vocab_size=8,
        hidden_size=4,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        rms_norm_eps=1e-5,
        max_position_embeddings=16,
    )
    values = {
        "language_model.backbone.embeddings.weight": np.ones((8, 4), np.float32),
        "language_model.backbone.norm_f.weight": np.ones(4, np.float32),
        "language_model.lm_head.weight": np.ones((8, 4), np.float32),
    }
    for layer in range(2):
        prefix = f"language_model.backbone.layers.{layer}"
        values[f"{prefix}.norm.weight"] = np.ones(4, np.float32)
        for projection in ("q", "k", "v", "o"):
            rows = 2 if projection in {"k", "v"} else 4
            values[f"{prefix}.mixer.{projection}_proj.weight"] = np.ones((rows, 4), np.float32)
    for suffix, scale in scales.items():
        layer, projection = suffix.split(".")
        key = f"language_model.backbone.layers.{layer}.mixer.{projection}_proj.{projection}_scale"
        values[key] = np.array([scale], np.float32)
    save_file(values, str(tmp_path / "model.safetensors"))
    if missing:
        with pytest.raises(
            ValueError, match=f"Missing Nemotron Omni FP8 attention scale: .*{missing}"
        ):
            OmniDecoder().load_weights(tmp_path, config)
    else:
        weights = OmniDecoder().load_weights(tmp_path, config)
        assert weights["_fp8_kv_cache"] == ("0.k" in scales)
        if weights["_fp8_kv_cache"]:
            for layer in range(2):
                assert weights[f"layer.{layer}.kv_scales"] == (1.0, 0.25, 0.5)
