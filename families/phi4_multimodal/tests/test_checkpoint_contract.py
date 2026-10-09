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
        assert len(commands) == 3 and len(published) == 1
        assert ("--externalize-weights" in commands[0]) is not packed_head
        assert commands[1][-6:] == ["--maxBatchSize", "2", "--maxInputLen", "7168",
                                    "--maxKVCacheCapacity", "8192"]
        assert commands[2][-6:] == ["--minImageTokens", "256", "--maxImageTokens", "6400",
                                    "--maxImageTokensPerImage", "1280"]
        assert published[0]["weight_format"] == weight_format
        assert published[0]["builder_flow"] == "onnx"
        assert published[0]["max_batch_size"] == 2
