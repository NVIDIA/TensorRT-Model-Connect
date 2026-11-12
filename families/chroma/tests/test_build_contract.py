# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the Chroma build contract: rejected options, checkpoint layout, tokenizer configuration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ml_dtypes")

from families.chroma.checkpoint import load_checkpoint  # noqa: E402
from families.chroma import model  # noqa: E402
from families.chroma.text_encoder_builder import relative_position_buckets  # noqa: E402
from families.chroma.tokenizer_asset import validate_tokenizer  # noqa: E402
from tensorrt_model_connect import BuildRequest  # noqa: E402


def _request(tmp_path: Path, **overrides) -> BuildRequest:
    values = {"model_dir": tmp_path, "output_path": tmp_path / "x.bundle", "family": "chroma",
              "task": "text_to_image", "precision": "bf16"}
    values.update(overrides)
    return BuildRequest(**values)


class _Stop(Exception):
    pass


def _geometry(monkeypatch, request) -> tuple:
    """The image size and text length build() hands to the component builders."""
    seen = {}

    def capture(writer, checkpoint, precision, verbose, height, width, length):
        seen.update(precision=precision, size=(height, width, length))
        raise _Stop

    monkeypatch.setattr(model, "load_checkpoint", lambda root: SimpleNamespace(root=root))
    monkeypatch.setattr("families.chroma.tokenizer_asset.tokenizer_json", lambda root: b"{}")
    monkeypatch.setattr(model, "_runtime_config", lambda *args: {})
    monkeypatch.setattr(model, "_write_components", capture)

    class Writer:
        def set_header(self, **header):
            seen["header"] = header

        def add_json(self, name, value):
            pass

        def add_bytes(self, name, value):
            pass

    with pytest.raises(_Stop):
        model.build(request, Writer())
    return seen


def test_defaults_are_the_pipeline_defaults(tmp_path: Path, monkeypatch) -> None:
    seen = _geometry(monkeypatch, _request(tmp_path))
    assert seen["size"] == (1024, 1024, 512) and seen["precision"] == "bf16"
    assert seen["header"] == {"family": "chroma", "task": "text_to_image", "backend": "trt"}
    seen = _geometry(monkeypatch, _request(tmp_path, precision="fp32", image_height=512, image_width=768,
                                           max_sequence_length=256))
    assert seen["size"] == (512, 768, 256) and seen["precision"] == "fp32"


@pytest.mark.parametrize("overrides", [
    {"precision": "fp16"},
    {"task": "image_generation"},
    {"backend": "trt_rtx"},
    {"tensor_parallel_size": 2},
    {"context_parallel_size": 2},
    {"max_batch_size": 2},
    {"quantization": "fp8"},
    {"fp32_layers": (1,)},
    {"dynamic_kv_cache": True},
    {"video_num_frames": 9},
    {"mtp_seq_len": 4},
    {"image_height": 1000},
    {"image_width": 520},
    {"image_height": 4112},
    {"max_sequence_length": 513},
])
def test_unsupported_options_are_rejected(tmp_path: Path, overrides: dict) -> None:
    # Rejected before the checkpoint is read or the writer is touched.
    with pytest.raises((ValueError, NotImplementedError)):
        model.build(_request(tmp_path, **overrides), writer=None)


def _pipeline_dir(root: Path, **changes) -> Path:
    files = {
        "model_index.json": {
            "_class_name": "ChromaPipeline", "feature_extractor": [None, None], "image_encoder": [None, None],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "T5EncoderModel"], "tokenizer": ["transformers", "T5Tokenizer"],
            "transformer": ["diffusers", "ChromaTransformer2DModel"], "vae": ["diffusers", "AutoencoderKL"],
        },
        "transformer/config.json": {
            "_class_name": "ChromaTransformer2DModel", "approximator_hidden_dim": 5120, "approximator_layers": 5,
            "approximator_num_channels": 64, "attention_head_dim": 128, "axes_dims_rope": [16, 56, 56],
            "in_channels": 64, "joint_attention_dim": 4096, "num_attention_heads": 24, "num_layers": 19,
            "num_single_layers": 38, "out_channels": None, "patch_size": 1,
        },
        "text_encoder/config.json": {
            "model_type": "t5", "d_model": 4096, "d_kv": 64, "d_ff": 10240, "num_heads": 64, "num_layers": 24,
            "vocab_size": 32128, "relative_attention_num_buckets": 32, "relative_attention_max_distance": 128,
            "layer_norm_epsilon": 1e-6, "feed_forward_proj": "gated-gelu", "dense_act_fn": "gelu_new",
        },
        "vae/config.json": {
            "_class_name": "AutoencoderKL", "act_fn": "silu", "block_out_channels": [128, 256, 512, 512],
            "latent_channels": 16, "layers_per_block": 2, "norm_num_groups": 32, "mid_block_add_attention": True,
            "scaling_factor": 0.3611, "shift_factor": 0.1159, "use_post_quant_conv": False,
            "up_block_types": ["UpDecoderBlock2D"] * 4, "latents_mean": None, "latents_std": None,
        },
        "scheduler/scheduler_config.json": {
            "_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": 3.0,
            "use_dynamic_shifting": False,
        },
    }
    for name, change in changes.items():
        files[name.replace("__", "/")].update(change)
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    return root


def test_the_released_layout_is_accepted(tmp_path: Path) -> None:
    checkpoint = load_checkpoint(_pipeline_dir(tmp_path))
    assert checkpoint.transformer.modulation_rows == 344
    assert checkpoint.vae_scale_factor == 8
    assert checkpoint.scheduler.shift == 3.0


@pytest.mark.parametrize("changes", [
    {"model_index.json": {"_class_name": "ChromaRadiancePipeline"}},
    {"model_index.json": {"image_encoder": ["transformers", "CLIPVisionModelWithProjection"]}},
    {"transformer__config.json": {"patch_size": 2}},
    {"transformer__config.json": {"axes_dims_rope": [16, 56, 64]}},
    {"text_encoder__config.json": {"feed_forward_proj": "relu"}},
    {"vae__config.json": {"use_post_quant_conv": True}},
    {"vae__config.json": {"latents_mean": [0.0] * 16}},
    {"scheduler__scheduler_config.json": {"use_dynamic_shifting": True}},
    {"scheduler__scheduler_config.json": {"stochastic_sampling": True}},
])
def test_other_layouts_are_rejected(tmp_path: Path, changes: dict) -> None:
    with pytest.raises(ValueError):
        load_checkpoint(_pipeline_dir(tmp_path, **changes))


def _tokenizer() -> dict:
    return {
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "WhitespaceSplit"},
            {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "always", "split": True}]},
        "post_processor": {"type": "TemplateProcessing", "single": [
            {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}]},
        "model": {"type": "Unigram", "unk_id": 2, "byte_fallback": False,
                  "vocab": [["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0], ["\u2581a", -1.0]]},
        "added_tokens": [{"id": 1, "content": "</s>", "normalized": False, "lstrip": False, "rstrip": False,
                          "single_word": False, "special": True}],
    }


def test_the_runtime_tokenizer_configuration_is_accepted() -> None:
    validate_tokenizer(_tokenizer())


@pytest.mark.parametrize("change", [
    {"normalizer": {"type": "Precompiled", "precompiled_charsmap": ""}},
    {"pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "first", "split": True}},
    {"post_processor": {"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}]}},
    {"model": {"type": "BPE", "vocab": {}, "merges": []}},
])
def test_other_tokenizer_configurations_are_rejected(change: dict) -> None:
    data = _tokenizer()
    data.update(change)
    with pytest.raises(ValueError):
        validate_tokenizer(data)


def test_t5_relative_position_buckets() -> None:
    buckets = relative_position_buckets(512, 32, 128)
    # Bidirectional: 16 buckets per direction, 8 exact offsets, then logarithmic up to 128.
    assert buckets[0, 0] == 0 and buckets[0, 7] == 16 + 7 and buckets[7, 0] == 7
    assert buckets[0, 8] == 16 + 8 and buckets[0, 511] == 31 and buckets[511, 0] == 15
