# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from families.phi4_multimodal import model


def test_vision_loader_keeps_the_indexed_reader_collection(monkeypatch, tmp_path) -> None:
    key = "model.embed_tokens_extend.image_embed.proj.weight"

    class Reader:
        @staticmethod
        def keys():
            return [key]

    class Readers(list):
        tensor_map = {key: Reader()}

    readers = Readers([Reader()])
    seen = []
    monkeypatch.setattr(model, "_open_safetensors", lambda _path: readers)
    monkeypatch.setattr(
        model,
        "_load_tensor",
        lambda collection, name: seen.append((collection, name)) or np.ones((1,), np.float32),
    )

    weights = model._load_vision_weights(str(tmp_path))

    assert list(weights) == ["proj.weight"]
    assert seen == [(readers, key)]

    # Execution admission is independent of the owning image quality result.
    # These are the four already executed packed-image recipes, not new modes.
    from families.phi4_multimodal.edge_llm import builder, dispatch
    from dataclasses import replace
    from tensorrt_model_connect.build import BuildRequest

    monkeypatch.setattr(builder, "package_present", lambda: True)
    monkeypatch.setattr(builder, "builder_python", lambda *_a, **_k: "/python")
    monkeypatch.setattr(builder, "installed_package", lambda *_a, **_k: {
        "python": "/python", "plugin": "/sdk/plugin.so", "all_native_kernels": True,
        "onnx_builder": "/sdk/llm", "onnx_visual_builder": "/sdk/visual",
    })
    for algorithm, group_size, weight_format, sm, packed_head in (
        ("FP8", None, "fp8", 120, False),
        ("NVFP4", 16, "nvfp4", 120, False),
        ("W4A16_AWQ", 128, "int4_awq", 80, False),
        ("NVFP4", 16, "nvfp4", 120, True),
    ):
        source = tmp_path / f"{weight_format}-{packed_head}"
        source.mkdir()
        raw = {
            "model_type": "phi4mm", "architectures": ["Phi4MMForCausalLM"],
            "num_hidden_layers": 32, "hidden_size": 3072, "intermediate_size": 8192,
            "num_attention_heads": 24, "num_key_value_heads": 8, "vocab_size": 200064,
            "partial_rotary_factor": 0.75, "eos_token_id": 199999,
            "max_position_embeddings": 131072, "vision_lora": None,
            "quantization_config": {"quant_method": "modelopt", "quant_algo": algorithm},
        }
        (source / "config.json").write_text(json.dumps(raw))
        (source / "hf_quant_config.json").write_text(json.dumps({
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": algorithm, "group_size": group_size,
                             "kv_cache_quant_algo": None},
        }))
        tokenizer = {"post_processor": {
            "type": "TemplateProcessing",
            "single": [{"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}},
                     {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {}},
                     "added_tokens": [{"content": "<|endoftext|>", "id": 199999}]}
        tokenizer_config = {"eos_token": "<|endoftext|>"}
        builder.validate_tokenizer(tokenizer, tokenizer_config)
        builder.validate_tokenizer(dict(tokenizer, post_processor=None), tokenizer_config)
        bad_post = dict(tokenizer["post_processor"], special_tokens={"injected": {}})
        with pytest.raises(ValueError, match="token-preserving"):
            builder.validate_tokenizer(dict(tokenizer, post_processor=bad_post), tokenizer_config)
        (source / "tokenizer.json").write_text(json.dumps(tokenizer))
        (source / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
        tensors = {"lm_head.weight": np.ones((2, 2), dtype=np.float16)}
        if packed_head:
            tensors["lm_head.weight_scale"] = np.ones((1,), dtype=np.float32)
        save_file(tensors, str(source / "model.safetensors"))
        request = SimpleNamespace(
            model_dir=source, output_path=tmp_path / "bundle.trtmc", backend="trt",
            task="vision_language_generation", precision="fp16", quantization=None,
            max_batch_size=2, tensor_parallel_size=1, context_parallel_size=1,
            max_sequence_length=8192, dynamic_kv_cache=False, fp32_layers=None,
            image_height=None, image_width=None, video_num_frames=None, verbose=False,
        )
        target = {"os": "linux", "arch": "x86_64", "sm": sm}
        monkeypatch.setattr(builder, "local_target", lambda: target)
        commands, published = [], []

        def execute(command, **kwargs):
            assert kwargs["timeout"] == 12 * 60 * 60
            commands.append(command)
            staging = kwargs["cwd"]
            if "tensorrt_edgellm.scripts.export" in command:
                (staging / "onnx").mkdir()
            if command[0] == "/sdk/visual":
                engine = staging / "edge_llm/engine"
                (engine / "visual").mkdir(parents=True)
                for name in ("llm.engine", "visual/visual.engine", "chat_template.jinja"):
                    (engine / name).write_text("test artifact")
                (engine / "config.json").write_text(json.dumps({"eos_token_id": [199999, 200020]}))
                (engine / "visual/config.json").write_text("{}")
                (engine / "tokenizer.json").write_text(json.dumps(tokenizer))
                (engine / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
                save_file({k: np.zeros((3072,), dtype=np.float16) for k in ("glb_GN", "sub_GN")},
                          str(engine / "visual/phi4mm_gn_proj.safetensors"))

        def publish(_request, _writer, files, marker):
            assert "edge_llm/checkpoint/model.safetensors" not in files
            published.append(marker)

        def native(*_args):
            raise AssertionError("A quality-only failure must remain admitted to Edge")

        monkeypatch.setattr(builder.subprocess, "run", execute)
        monkeypatch.setattr(builder, "publish", publish)
        dispatch.build(request, object(), native)
        assert not list(tmp_path.glob(".bundle.trtmc.edge-*.log"))
        assert len(commands) == 3 and len(published) == 1
        assert ("--externalize-weights" in commands[0]) is not packed_head
        assert commands[1][-6:] == ["--maxBatchSize", "2", "--maxInputLen", "7168",
                                    "--maxKVCacheCapacity", "8192"]
        assert commands[2][-6:] == ["--minImageTokens", "256", "--maxImageTokens", "6400",
                                    "--maxImageTokensPerImage", "1280"]
        assert published[0]["weight_format"] == weight_format
        assert published[0]["builder_flow"] == "onnx"
        assert published[0]["max_batch_size"] == 2

    original = {**raw, "vision_lora": {"r": 256, "lora_alpha": 512}}
    original.pop("quantization_config")
    (source / "config.json").write_text(json.dumps(original))
    original_request = BuildRequest(
        source, tmp_path / "original.bundle", "phi4_multimodal",
        "vision_language_generation", "fp16", max_sequence_length=768,
    )
    assert dispatch.candidate(original_request, original)
    native_calls = []
    monkeypatch.setattr(builder, "package_present", lambda: pytest.fail("Edge probed for a non-match"))
    for limit in (None, 256, 767):
        short_request = replace(original_request, max_sequence_length=limit)
        assert not dispatch.candidate(short_request, original)
        dispatch.build(short_request, object(), lambda request, writer: native_calls.append(request))
        assert native_calls[-1] is short_request
    assert not list(tmp_path.glob(".original.bundle.edge-*.log"))

    monkeypatch.setattr(builder, "package_present", lambda: True)
    target = {"os": "linux", "arch": "x86_64", "sm": 80}
    monkeypatch.setattr(builder, "local_target", lambda: target)
    monkeypatch.setattr(builder, "request_weight_format", lambda *_: "fp16")
    timeout = builder.subprocess.TimeoutExpired(["edge-builder"], 12 * 60 * 60)
    def timed_out(*_args):
        raise timeout
    monkeypatch.setattr(dispatch, "EDGE_DISPATCH", {("linux", "x86_64", 80, "fp16"): timed_out})
    dispatch.build(original_request, object(), lambda request, writer: native_calls.append(request))
    assert native_calls[-1] is original_request
    logs = list(tmp_path.glob(".original.bundle.edge-*.log"))
    assert len(logs) == 1 and "TimeoutExpired" in logs[0].read_text()
    logs[0].unlink()

    monkeypatch.setattr(dispatch, "EDGE_DISPATCH", {("linux", "x86_64", 80, "fp16"): lambda *_: ({}, {})})
    def failed_publication(*_args):
        raise RuntimeError("publication failed")
    monkeypatch.setattr(builder, "publish", failed_publication)
    count = len(native_calls)
    with pytest.raises(RuntimeError, match="publication failed"):
        dispatch.build(original_request, object(), lambda request, writer: native_calls.append(request))
    assert len(native_calls) == count
    assert len(list(tmp_path.glob(".original.bundle.edge-*.log"))) == 1


@pytest.mark.parametrize("format", ["fp8", "nvfp4", "int4_awq"])
def test_prepared_fused_projections_keep_source_scales_and_head(tmp_path, format):
    """A real safetensors fixture covers source packing and fused row slicing."""
    import ml_dtypes
    from families.phi4_multimodal.config import ModelConfig
    from families.phi4_multimodal.prepared_quantization import PreparedLinear

    rng = np.random.default_rng(29)
    h, mlp, vocab = 128, 256, 32
    algorithm, group = {"fp8": ("FP8", None), "nvfp4": ("NVFP4", 16),
                        "int4_awq": ("W4A16_AWQ", 128)}[format]
    raw = {"model_type": "phi4mm", "hidden_size": h, "intermediate_size": mlp,
           "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 1,
           "vocab_size": vocab, "vision_lora": None,
           "quantization_config": {"quant_method": "modelopt", "quant_algo": algorithm}}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({
        "producer": {"name": "modelopt"},
        "quantization": {"quant_algo": algorithm, "group_size": group},
    }))
    tensors = {"model.embed_tokens.weight": rng.normal(size=(vocab, h)).astype(np.float16),
               "model.norm.weight": np.ones(h, np.float16)}
    for norm in ("input_layernorm", "post_attention_layernorm"):
        tensors[f"model.layers.0.{norm}.weight"] = np.ones(h, np.float16)
    source_codes = {}
    for prefix, rows, cols in [
        ("model.layers.0.self_attn.qkv_proj", 256, h),
        ("model.layers.0.self_attn.o_proj", h, h),
        ("model.layers.0.mlp.gate_up_proj", 2*mlp, h),
        ("model.layers.0.mlp.down_proj", h, mlp), ("lm_head", vocab, h),
    ]:
        if format == "fp8":
            value = rng.normal(size=(rows, cols)).astype(ml_dtypes.float8_e4m3fn)
            tensors[prefix + ".weight"] = value
            tensors[prefix + ".weight_scale"] = np.array([0.5], np.float32)
            tensors[prefix + ".input_scale"] = np.array([0.25], np.float32)
            source_codes[prefix] = value.view(np.uint8)
        else:
            codes = rng.integers(0, 16, (rows, cols), dtype=np.uint8)
            source_codes[prefix] = codes
            if format == "int4_awq":
                tensors[prefix + ".weight"] = codes[0::2] | (codes[1::2] << 4)
                tensors[prefix + ".weight_scale"] = rng.uniform(0.1, 0.3, (rows, cols//128)).astype(np.float32)
                tensors[prefix + ".pre_quant_scale"] = rng.uniform(0.5, 1.5, cols).astype(np.float16)
            else:
                tensors[prefix + ".weight"] = codes[:, 0::2] | (codes[:, 1::2] << 4)
                tensors[prefix + ".weight_scale"] = rng.uniform(0.1, 0.3, (rows, cols//16)).astype(ml_dtypes.float8_e4m3fn)
                tensors[prefix + ".weight_scale_2"] = np.array([0.5], np.float32)
                tensors[prefix + ".input_scale"] = np.array([0.25], np.float32)
    # NumPy safetensors cannot write FP8 on every supported NumPy release.
    # Write its public container format; the production reader validates it.
    header, payload = {}, bytearray()
    names = {np.dtype(np.float16): "F16", np.dtype(np.float32): "F32",
             np.dtype(np.uint8): "U8", np.dtype(ml_dtypes.float8_e4m3fn): "F8_E4M3"}
    for name, value in tensors.items():
        start = len(payload)
        payload.extend(np.ascontiguousarray(value).tobytes())
        header[name] = {"dtype": names[value.dtype], "shape": list(value.shape),
                        "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (tmp_path / "model.safetensors").write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
    weights = model._Phi4MultimodalModel().load_weights(str(tmp_path), ModelConfig.from_dir(tmp_path))
    for logical, prefix, begin, end in [
        ("w_q", "self_attn.qkv_proj", 0, 128), ("w_k", "self_attn.qkv_proj", 128, 192),
        ("w_v", "self_attn.qkv_proj", 192, 256), ("w_o", "self_attn.o_proj", 0, h),
        ("w_gate", "mlp.gate_up_proj", 0, mlp), ("w_up", "mlp.gate_up_proj", mlp, 2*mlp),
        ("w_down", "mlp.down_proj", 0, h), ("w_out", "lm_head", 0, vocab),
    ]:
        name = prefix if prefix == "lm_head" else "model.layers.0." + prefix
        value = weights[logical if logical == "w_out" else "layer.0." + logical]
        assert isinstance(value, PreparedLinear) and value.transposed
        assert value.shape == (source_codes[name].shape[1], end-begin)
        if format == "fp8":
            np.testing.assert_array_equal(value.packed, source_codes[name][begin:end])
            np.testing.assert_array_equal(value.weight_scale, [0.5])
        else:
            unpacked = np.empty((value.out_features, value.in_features), np.uint8)
            unpacked[:, 0::2], unpacked[:, 1::2] = value.packed & 15, value.packed >> 4
            np.testing.assert_array_equal(unpacked, source_codes[name][begin:end])
            expected = tensors[name + ".weight_scale"][begin:end]
            if format == "nvfp4":
                expected = expected.view(np.uint8)
                np.testing.assert_array_equal(value.global_scale, [0.5])
            else:
                np.testing.assert_array_equal(value.pre_quant_scale, tensors[name + ".pre_quant_scale"])
            np.testing.assert_array_equal(value.weight_scale, expected)
        if format != "int4_awq":
            np.testing.assert_array_equal(value.input_scale, [0.25])


def test_prepared_metadata_and_scales_fail_closed(tmp_path):
    from families.phi4_multimodal.prepared_quantization import checkpoint_format, PreparedLinear
    raw = {"quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({
        "quantization": {"quant_algo": "NVFP4", "group_size": 16}}))
    with pytest.raises(ValueError, match="Conflicting"):
        checkpoint_format(tmp_path)
    values = {"proj.weight_scale": np.ones((4, 1), np.float32)}
    packed = np.zeros((2, 128), np.uint8)
    for invalid in (np.zeros(128), np.ones(127), np.full(128, np.nan)):
        with pytest.raises(ValueError, match="smoothing"):
            PreparedLinear.from_tensor(packed, "proj.weight", values.__getitem__, "int4_awq", invalid)
    values["proj.weight_scale"][0] = np.nan
    with pytest.raises(ValueError, match="invalid AWQ scales"):
        PreparedLinear.from_tensor(packed, "proj.weight", values.__getitem__, "int4_awq")



def test_native_generation_retains_checkpoint_stop_tokens(tmp_path):
    from families.phi4_multimodal.config import ModelConfig
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "phi4mm", "vocab_size": 200064, "eos_token_id": 199999,
    }))
    config = ModelConfig.from_dir(tmp_path)
    assert model._generation_eos_ids(tmp_path, config) == [199999]
    generation = tmp_path / "generation_config.json"
    generation.write_text(json.dumps({"eos_token_id": [200020, 199999]}))
    assert model._generation_eos_ids(tmp_path, config) == [200020, 199999]
    for invalid in ([], [True], [-1], [200064], ["200020"]):
        generation.write_text(json.dumps({"eos_token_id": invalid}))
        with pytest.raises(ValueError, match="invalid eos_token_id"):
            model._generation_eos_ids(tmp_path, config)
