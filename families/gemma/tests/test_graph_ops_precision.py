# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision-contract tests for Gemma graph operations."""

from __future__ import annotations

import pytest


trt = pytest.importorskip("tensorrt")

from families.gemma import graph_ops  # noqa: E402


@pytest.mark.parametrize(("algorithm", "moe", "hidden", "layers", "experts", "top_k", "expected"), [
    (None, False, 5376, 60, None, None, -1),
    ("FP8", False, 5376, 60, None, None, -1),
    ("NVFP4", False, 5376, 60, None, None, 0),
    ("NVFP4", True, 2816, 30, 128, 8, -1),
    ("NVFP4", True, 1024, 12, 64, 4, -1),
    ("NVFP4", True, 2816, 30, 64, 8, -1),
])
def test_nvfp4_aux_stream_default_changes_only_dense_gemma4(
    algorithm, moe, hidden, layers, experts, top_k, expected
):
    from types import SimpleNamespace

    from families.gemma.gemma4_quant import configure_aux_streams
    from families.gemma.gemma4_text import text_config

    raw = {"text_config": {"enable_moe_block": moe, "hidden_size": hidden,
                           "num_hidden_layers": layers, "num_experts": experts,
                           "top_k_experts": top_k}}
    if algorithm is not None:
        raw["quantization_config"] = {"quant_algo": algorithm}
    config = SimpleNamespace(max_aux_streams=-1)

    configure_aux_streams(config, text_config(SimpleNamespace(raw=raw)))

    assert config.max_aux_streams == expected


class _Tensor:
    def __init__(self, dtype, shape=(1, 8)):
        self.dtype = dtype
        self.shape = shape


class _Layer:
    def __init__(self, output):
        self._output = output

    def get_output(self, index):
        assert index == 0
        return self._output


class _Network:
    def __init__(self):
        self.activation_type = None
        self.activation_dtype = None
        self.casts = []

    def add_cast(self, tensor, dtype):
        self.casts.append((tensor.dtype, dtype))
        return _Layer(_Tensor(dtype, tensor.shape))

    def add_activation(self, tensor, activation_type):
        self.activation_type = activation_type
        self.activation_dtype = tensor.dtype
        return _Layer(_Tensor(tensor.dtype, tensor.shape))


def test_gelu_tanh_uses_fused_trt_activation_for_fp16() -> None:
    network = _Network()

    output = graph_ops.add_gelu_new(network, _Tensor(trt.float16))

    assert network.activation_type == trt.ActivationType.GELU_TANH
    assert network.activation_dtype == trt.float32
    assert network.casts == [
        (trt.float16, trt.float32),
        (trt.float32, trt.float16),
    ]
    assert output.dtype == trt.float16


def test_gelu_tanh_uses_fused_trt_activation_for_fp32() -> None:
    network = _Network()

    output = graph_ops.add_gelu_new(network, _Tensor(trt.float32))

    assert network.activation_type == trt.ActivationType.GELU_TANH
    assert network.activation_dtype == trt.float32
    assert network.casts == []
    assert output.dtype == trt.float32


def test_gemma4_cache_owners_preserve_attention_geometry(monkeypatch):
    from families.gemma.gemma4_text import cache_geometry

    raw = {"num_hidden_layers": 6, "num_kv_shared_layers": 3,
           "num_attention_heads": 4, "num_key_value_heads": 1,
           "head_dim": 8, "global_head_dim": 16,
           "layer_types": ["sliding_attention", "full_attention", "sliding_attention",
                           "sliding_attention", "full_attention", "sliding_attention"]}
    from families.gemma.gemma4_moe import expert_geometry

    assert expert_geometry({"num_experts": 8, "top_k_experts": 2,
                            "moe_intermediate_size": 16}) == (8, 2, 16)
    for count, selected, width in ((0, 1, 4), (4, 5, 4), (4, 0, 4),
                                   (4, 1, 0), (True, 1, 4)):
        with pytest.raises(ValueError, match="expert geometry"):
            expert_geometry({"num_experts": count, "top_k_experts": selected,
                             "moe_intermediate_size": width})
    owners, widths = cache_geometry(raw)
    assert owners == [0, 1, 2, 2, 1, 2]
    assert widths == [8, 16, 8]
    from families.gemma.gemma4_assistant import shared_target_layers

    assert shared_target_layers(raw) == {"sliding_attention": 2, "full_attention": 1}
    import numpy as np
    from families.gemma import gemma4_assistant

    target = {**raw, "hidden_size": 16, "vocab_size": 8, "max_position_embeddings": 32}
    assistant = {**target, "num_kv_shared_layers": 6}
    config = {"text_config": assistant, "backbone_hidden_size": 16,
              "use_ordered_embeddings": True, "num_centroids": 4,
              "centroid_intermediate_top_k": 2}
    ordering = np.array([7, 0, 4, 3, 5, 1, 6, 2])
    monkeypatch.setattr(gemma4_assistant, "_has_tensor", lambda *args: True)
    monkeypatch.setattr(gemma4_assistant, "_load_tensor", lambda *args: ordering)
    graph = gemma4_assistant.AssistantGraph(None, config, [], target, [], "fp16", 16)
    assert graph.target_layers == {"sliding_attention": 2, "full_attention": 1}
    ordering[0] = 0
    with pytest.raises(ValueError, match="vocabulary permutation"):
        gemma4_assistant.AssistantGraph(None, config, [], target, [], "fp16", 16)
    for centroids, top_k in ((0, 1), (3, 2), (4, 5), (4, 0), (4, True)):
        bad = {**config, "num_centroids": centroids, "centroid_intermediate_top_k": top_k}
        with pytest.raises(ValueError, match="centroid geometry"):
            gemma4_assistant.AssistantGraph(None, bad, [], target, [], "fp16", 16)
    raw["layer_types"] = ["sliding_attention"] * 3 + ["full_attention"] * 3
    with pytest.raises(ValueError, match="no preceding KV owner"):
        cache_geometry(raw)


def test_gemma4_proportional_rope_rotates_across_the_full_head():
    import numpy as np
    from families.gemma.gemma4_text import rope_tables

    raw = {"num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 8,
           "global_head_dim": 16, "rope_parameters": {"full_attention": {
               "rope_type": "proportional", "rope_theta": 10000.0,
               "partial_rotary_factor": 0.25}}}
    cos, sin = rope_tables(raw, "full_attention", 4)
    assert cos.shape == sin.shape == (4, 8)
    frequencies = 10000.0 ** (-np.arange(0, 4, 2, dtype=np.float32) / 16)
    np.testing.assert_allclose(cos[:, :2], np.cos(np.arange(4)[:, None] * frequencies), atol=1e-7)
    np.testing.assert_allclose(sin[:, :2], np.sin(np.arange(4)[:, None] * frequencies), atol=1e-7)
    np.testing.assert_array_equal(cos[:, 2:], 1)
    np.testing.assert_array_equal(sin[:, 2:], 0)


def test_gemma4_per_layer_tables_preserve_original_token_rows(tmp_path):
    import numpy as np
    from families.gemma.gemma4_text import per_layer_embedding

    # Splitting the layer axis must not split or renumber vocabulary rows.
    packed = np.arange(7 * 3 * 4, dtype=np.float32).reshape(7, 12)
    ids = np.array([6, 0, 5], np.int32)
    original = packed[ids].reshape(3, 3, 4)
    for index in range(3):
        table = per_layer_embedding(packed, index, 4)
        assert table.shape == (7, 4)
        assert table.flags.c_contiguous
        np.testing.assert_array_equal(table[ids], original[:, index])
    # Source NVFP4 calibration scales must preserve their original FP8 bytes.
    import json
    import ml_dtypes
    from types import SimpleNamespace
    from families.gemma.checkpoint_mapper import _open_safetensors
    from families.gemma.gemma4_quant import validate_source

    scales = np.array([[0.5, 1], [2, 4]], dtype=ml_dtypes.float8_e4m3fn)
    header = json.dumps({"scale": {"dtype": "F8_E4M3", "shape": [2, 2],
                                  "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    (tmp_path / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + scales.tobytes())
    readers = _open_safetensors(tmp_path)
    loaded = readers.tensor_map["scale"].get_tensor("scale")
    assert loaded.dtype == scales.dtype
    np.testing.assert_array_equal(loaded.view(np.uint8), scales.view(np.uint8))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({
        "quantization": {"quant_algo": "NVFP4", "group_size": 16, "kv_cache_quant_algo": "FP8"}}))
    raw = {"enable_moe_block": True,
           "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}
    validate_source(tmp_path, raw, readers)  # A sidecar alone does not quantize KV.
    for name in ("model.layers.0.self_attn.k_proj.k_scale",
                 "model.layers.0.self_attn.q_proj.weight_scale"):
        with pytest.raises(NotImplementedError, match="attention quantization|routed experts only"):
            validate_source(tmp_path, raw, SimpleNamespace(tensor_map={name: None}))
    for index, width in ((3, 4), (-1, 4), (0, 5), (0, 0)):
        with pytest.raises(ValueError, match="dimensions"):
            per_layer_embedding(packed, index, width)


def test_gemma4_vision_spatial_rope_keeps_axes_independent():
    import numpy as np
    from families.gemma.gemma4_vision import spatial_rope_tables

    raw = {"head_dim": 8, "rope_parameters": {"rope_type": "default", "rope_theta": 100.0}}
    cos, sin = spatial_rope_tables(raw, 4)
    angles = np.arange(-1, 4)[:, None] * np.array([1, 0.1])
    expected = np.concatenate((angles, angles), axis=-1)
    np.testing.assert_allclose(cos, np.cos(expected), atol=1e-7)
    np.testing.assert_allclose(sin, np.sin(expected), atol=1e-7)
    np.testing.assert_array_equal(cos[1], 1)
    np.testing.assert_array_equal(sin[1], 0)
    from families.gemma.gemma4_unified import media_weight_name, validate_media_weights

    keys = {"model.vision_embedder.patch_ln1.weight",
            "model.embed_vision.embedding_projection.weight"}
    assert media_weight_name("model.embed_vision.patch_ln1.weight", keys) == (
        "model.vision_embedder.patch_ln1.weight")
    assert media_weight_name(
        "model.embed_vision.multimodal_embedder.embedding_projection.weight", keys
    ) == "model.embed_vision.embedding_projection.weight"
    assert media_weight_name("model.vision_embedder.patch_ln1.weight", keys) in keys
    with pytest.raises(KeyError, match="media weight is missing"):
        validate_media_weights([dict.fromkeys(keys)], vision=True, audio=False)
    from families.gemma.gemma4_unified import UnifiedMediaGraph

    class ProjectionProbe(UnifiedMediaGraph):
        def __init__(self):
            self.dtype = trt.float16
            self.boundaries = []

        def cast(self, value, dtype):
            return _Tensor(dtype)

        def norm(self, value, epsilon, name=None):
            self.boundaries.append((name, value.dtype))
            return value

        def linear(self, value, name, *, bias=False):
            self.boundaries.append((name, value.dtype))
            assert bias
            return value

    probe = ProjectionProbe()
    assert probe.patch_embeddings(_Tensor(trt.float32), "image").dtype == trt.float16
    assert probe.boundaries == [("image.patch_ln1", trt.float16),
                                ("image.patch_dense", trt.float32),
                                ("image.patch_ln2", trt.float32)]
    raw["head_dim"] = 6
    with pytest.raises(NotImplementedError, match="spatial RoPE"):
        spatial_rope_tables(raw, 4)


def test_gemma4_audio_frontend_uses_checkpoint_metadata(tmp_path):
    import json
    from families.gemma.gemma4_audio_frontend import frontend_constants

    raw = {"audio_config": {"subsampling_conv_channels": [128, 32]},
           "text_config": {"hidden_size": 2560}, "audio_token_id": 8,
           "boa_token_id": 9, "eoa_token_id": 10}
    (tmp_path / "processor_config.json").write_text(json.dumps({"feature_extractor": {
        "preemphasis": 0.97, "input_scale_factor": 2.0, "per_bin_mean": [0.5] * 128,
        "per_bin_stddev": [2.0] * 128}}))
    config, window, filters = frontend_constants(tmp_path, raw, 2048)
    assert config["max_frames"] == 8192
    assert config["preemphasis"] == 0.97 and config["input_scale"] == 2
    assert config["mean"] == [0.5] * 128 and config["std"] == [2.0] * 128
    assert window.shape == (320,) and str(window.dtype) == "float32"
    assert filters.shape == (257, 128) and str(filters.dtype) == "float64"

    # Unified models consume raw waveform chunks, not a mel frontend.
    raw.update(model_type="gemma4_unified", audio_config={"audio_embed_dim": 640},
               eoa_token_index=raw.pop("eoa_token_id"))
    (tmp_path / "processor_config.json").write_text(json.dumps({"feature_extractor": {
        "feature_size": 640, "audio_samples_per_token": 640, "sampling_rate": 16000}}))
    config, window, filters = frontend_constants(tmp_path, raw, 2048)
    assert config["waveform"] is True
    assert config["max_frames"] == 2048 and config["mel_bins"] == 640
    assert window.size == filters.size == 0
    for invalid in ({"feature_size": 128}, {"audio_samples_per_token": 320},
                    {"padding_value": 1}, {"sampling_rate": 48000}):
        (tmp_path / "processor_config.json").write_text(
            json.dumps({"feature_extractor": invalid}))
        with pytest.raises(ValueError, match="waveform metadata"):
            frontend_constants(tmp_path, raw, 2048)



@pytest.mark.parametrize("metadata", [{"sampling_rate": 48000}, {"fft_length": 500},
                                     {"mel_floor": 0}, {"per_bin_stddev": [-1] * 128},
                                     {"dither": 0.1}])
def test_gemma4_audio_frontend_rejects_unmapped_configuration(tmp_path, metadata):
    import json
    from families.gemma.gemma4_audio_frontend import frontend_constants

    raw = {"audio_config": {"subsampling_conv_channels": [128, 32]},
           "text_config": {"hidden_size": 2560}, "audio_token_id": 8,
           "boa_token_id": 9, "eoa_token_id": 10}
    (tmp_path / "processor_config.json").write_text(json.dumps({"feature_extractor": metadata}))
    with pytest.raises((ValueError, NotImplementedError)):
        frontend_constants(tmp_path, raw, 2048)


def test_dense_nvfp4_source_keeps_activation_scales_and_rejects_unmapped_cache(tmp_path):
    import json
    from types import SimpleNamespace
    from families.gemma.gemma4_quant import validate_source

    layout = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}
    raw = {"num_hidden_layers": 2, "enable_moe_block": False,
           "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4",
               "config_groups": {"group_0": {"weights": layout, "input_activations": layout,
                                               "targets": ["Linear"]}}}}
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({"quantization": {
        "quant_algo": "NVFP4", "group_size": 16, "kv_cache_quant_algo": "FP8"}}))
    prefix = "model.language_model"
    tensors = {f"{prefix}.layers.{i}.mlp.{projection}.{suffix}": None
               for i in range(2) for projection in ("gate_proj", "up_proj", "down_proj")
               for suffix in ("weight_scale", "weight_scale_2", "input_scale")}
    tensors[prefix + ".embed_tokens.weight"] = None
    readers = SimpleNamespace(tensor_map=tensors)
    validate_source(tmp_path, raw, readers)
    raw["quantization_config"]["config_groups"]["unmapped"] = {}
    with pytest.raises(NotImplementedError, match="W4A4 contract"):
        validate_source(tmp_path, raw, readers)
    del raw["quantization_config"]["config_groups"]["unmapped"]
    (tmp_path / "hf_quant_config.json").unlink()
    with pytest.raises(ValueError, match="explicit quantization metadata"):
        validate_source(tmp_path, {"num_hidden_layers": 2}, readers)
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({"quantization": {
        "quant_algo": "NVFP4", "group_size": 16, "kv_cache_quant_algo": "FP8"}}))
    key = prefix + ".layers.0.mlp.up_proj.input_scale"
    del tensors[key]
    with pytest.raises(NotImplementedError, match="all and only"):
        validate_source(tmp_path, raw, readers)
    tensors[key] = None
    tensors[prefix + ".layers.0.self_attn.k_scale"] = None
    with pytest.raises(NotImplementedError, match="calibrated attention"):
        validate_source(tmp_path, raw, readers)


@pytest.mark.parametrize("invalid,message", [
    ("weight_dtype", "packed rank-two uint8"),
    ("block_shape", "block-scale geometry"),
    ("block_dtype", "block-scale geometry"),
    ("global_zero", "global scales"),
    ("input_zero", "global scales"),
])
def test_dense_nvfp4_rejects_invalid_packing_and_scales_before_graph_build(invalid, message):
    from types import SimpleNamespace
    import ml_dtypes
    import numpy as np
    from families.gemma.gemma4_quant import dense_linear

    stem = "model.language_model.layers.0.mlp.up_proj"
    values = {stem + ".weight": np.zeros((8, 16), np.uint8),
              stem + ".weight_scale": np.ones((8, 2), ml_dtypes.float8_e4m3fn),
              stem + ".weight_scale_2": np.array([1.0], np.float32),
              stem + ".input_scale": np.array([1.0], np.float32)}
    if invalid == "weight_dtype":
        values[stem + ".weight"] = values[stem + ".weight"].astype(np.float16)
    elif invalid == "block_shape":
        values[stem + ".weight_scale"] = values[stem + ".weight_scale"][:, :1]
    elif invalid == "block_dtype":
        values[stem + ".weight_scale"] = values[stem + ".weight_scale"].astype(np.float32)
    else:
        values[stem + (".input_scale" if invalid == "input_zero" else ".weight_scale_2")][:] = 0
    reader = SimpleNamespace(get_tensor=lambda name: values[name])
    graph = SimpleNamespace(prefix="model.language_model", readers=SimpleNamespace(
        tensor_map={key: reader for key in values}))
    with pytest.raises(ValueError, match=message):
        dense_linear(graph, _Tensor(trt.float16, (1, 32)), "layers.0.mlp.up_proj")


@pytest.mark.parametrize("fp32_scores,softcap", [(True, None), (False, 4.0)])
def test_explicit_attention_preserves_mask_scale_and_geometry(monkeypatch, fp32_scores, softcap):
    tensors = [_Tensor(trt.bfloat16, (3, 32)), _Tensor(trt.bfloat16, (7, 8)),
               _Tensor(trt.bfloat16, (7, 8))]
    mask = _Tensor(trt.bfloat16, (1, 1, 3, 7))
    seen = {}
    monkeypatch.setattr(graph_ops, "reshape_rows_to_heads_4d", lambda net, x, *args, **kw: x)
    monkeypatch.setattr(graph_ops, "reshape_heads_4d_to_rows", lambda net, x, *args, **kw: x)

    def explicit(network, q, k, v, **kwargs):
        assert [q, k, v] == tensors
        seen.update(kwargs)
        return q

    monkeypatch.setattr(graph_ops, "_add_attention_core_with_fp32_scores", explicit)
    monkeypatch.setattr(graph_ops, "add_attention_core", lambda *args, **kwargs: pytest.fail(
        "Explicit score math must not use the IAttention decomposition"))
    output = graph_ops.add_attention_from_rows(
        _Network(), *tensors, num_heads=4, num_kv_heads=1, head_dim=8,
        q_seq=3, kv_seq=7, mask=mask, scale=0.25, logit_softcap=softcap,
        fp32_scores=fp32_scores)
    assert output.dtype == trt.bfloat16
    assert seen == {"num_heads": 4, "num_kv_heads": 1, "head_dim": 8,
                    "mask": mask, "scale": 0.25, "logit_softcap": softcap}
    with pytest.raises(NotImplementedError, match="explicit additive mask"):
        graph_ops.add_attention_from_rows(
            _Network(), *tensors, num_heads=4, num_kv_heads=1, head_dim=8,
            q_seq=3, kv_seq=7, causal=True, fp32_scores=True)
