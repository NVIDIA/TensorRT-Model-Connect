# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks: configuration resolution, build-option ownership, checkpoint layouts, tokenizer."""

import json
from pathlib import Path

import numpy as np
import pytest

from families.paligemma.config import ModelConfig, eos_token_ids, image_processing
from families.paligemma.model import build
from tensorrt_model_connect import BuildRequest

# google/paligemma-3b-pt-224 config.json at the pinned revision (transformers 4.42 save).
PALIGEMMA_3B_PT_224 = {
    "_name_or_path": "final-pt-224",
    "architectures": ["PaliGemmaForConditionalGeneration"],
    "bos_token_id": 2,
    "eos_token_id": 1,
    "hidden_size": 2048,
    "ignore_index": -100,
    "image_token_index": 257152,
    "model_type": "paligemma",
    "pad_token_id": 0,
    "projection_dim": 2048,
    "text_config": {
        "hidden_size": 2048,
        "intermediate_size": 16384,
        "model_type": "gemma",
        "num_attention_heads": 8,
        "num_hidden_layers": 18,
        "num_image_tokens": 256,
        "num_key_value_heads": 1,
        "torch_dtype": "float32",
        "vocab_size": 257216,
    },
    "torch_dtype": "float32",
    "transformers_version": "4.42.0.dev0",
    "vision_config": {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "num_image_tokens": 256,
        "patch_size": 14,
        "projection_dim": 2048,
        "projector_hidden_act": "gelu_fast",
        "vision_use_head": False,
    },
}


def _with(document, path, value):
    result = json.loads(json.dumps(document))
    target = result
    for key in path[:-1]:
        target = target[key]
    if value is None:
        target.pop(path[-1], None)
    else:
        target[path[-1]] = value
    return result


def test_published_configuration_resolves_with_reference_defaults():
    config = ModelConfig.from_dict(PALIGEMMA_3B_PT_224)
    vision, text = config.vision, config.text
    assert (vision.hidden_size, vision.intermediate_size, vision.num_hidden_layers) == (1152, 4304, 27)
    assert (vision.num_attention_heads, vision.image_size, vision.patch_size) == (16, 224, 14)
    assert vision.layer_norm_eps == 1e-6 and vision.num_patches == 256
    assert (text.hidden_size, text.intermediate_size, text.num_hidden_layers) == (2048, 16384, 18)
    assert (text.num_attention_heads, text.num_key_value_heads, text.head_dim) == (8, 1, 256)
    assert (text.vocab_size, text.max_position_embeddings) == (257216, 8192)
    assert text.rms_norm_eps == 1e-6 and text.rope_theta == 10000.0
    assert config.image_token_id == 257152 and config.num_image_tokens == 256


def test_transformers_5_rope_parameters_resolve():
    document = _with(PALIGEMMA_3B_PT_224, ("text_config", "rope_parameters"),
                     {"rope_type": "default", "rope_theta": 10000.0})
    assert ModelConfig.from_dict(document).text.rope_theta == 10000.0


@pytest.mark.parametrize("path, value", [
    (("text_config", "model_type"), "gemma2"),
    (("text_config", "hidden_act"), "gelu"),
    (("text_config", "attention_bias"), True),
    (("text_config", "rope_scaling"), {"rope_type": "linear", "factor": 2.0}),
    (("text_config", "rope_parameters"), {"rope_type": "yarn", "rope_theta": 10000.0}),
    (("vision_config", "model_type"), "siglip2_vision_model"),
    (("vision_config", "hidden_act"), "gelu"),
    (("projection_dim",), 1152),
    (("tie_word_embeddings",), False),
    (("image_token_index",), 300000),
])
def test_unimplemented_configurations_are_rejected(path, value):
    with pytest.raises(ValueError):
        ModelConfig.from_dict(_with(PALIGEMMA_3B_PT_224, path, value))


class _Writer:
    def __getattr__(self, name):
        raise AssertionError(f"rejected requests must not write ({name})")


@pytest.mark.parametrize("change, error", [
    ({"task": "text_generation"}, ValueError),
    ({"backend": "trt_rtx"}, ValueError),
    ({"dynamic_kv_cache": True}, NotImplementedError),
    ({"max_batch_size": 2}, ValueError),
    ({"tensor_parallel_size": 2}, ValueError),
    ({"context_parallel_size": 2}, ValueError),
    ({"image_height": 448}, ValueError),
    ({"image_width": 448}, ValueError),
    ({"video_num_frames": 4}, ValueError),
    ({"fp32_layers": (0,)}, ValueError),
    ({"mtp_seq_len": 2}, ValueError),
    ({"quantization": "fp8"}, ValueError),
    ({"precision": "int8"}, ValueError),
    ({"precision": "fp16"}, ValueError),
])
def test_build_owns_every_request_option(tmp_path, change, error):
    request = BuildRequest(**{"model_dir": tmp_path, "output_path": tmp_path / "x.bundle",
                              "family": "paligemma", "task": "images_text_to_text",
                              "precision": "bf16", **change})
    with pytest.raises(error):
        build(request, _Writer())


def test_build_rejects_a_capacity_without_room_for_one_image(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(PALIGEMMA_3B_PT_224))
    request = BuildRequest(model_dir=tmp_path, output_path=tmp_path / "x.bundle", family="paligemma",
                           task="images_text_to_text", precision="bf16", max_sequence_length=257)
    with pytest.raises(ValueError, match="max_sequence_length"):
        build(request, _Writer())


def test_generation_config_must_keep_greedy_decoding(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "_from_model_config": True, "bos_token_id": 2, "eos_token_id": [1, 107], "pad_token_id": 0,
        "use_cache": True, "do_sample": False, "transformers_version": "5.2.0"}))
    assert eos_token_ids(tmp_path, PALIGEMMA_3B_PT_224) == (1, 107)
    (tmp_path / "generation_config.json").write_text(json.dumps({"do_sample": True}))
    with pytest.raises(ValueError, match="greedy"):
        eos_token_ids(tmp_path, PALIGEMMA_3B_PT_224)
    (tmp_path / "generation_config.json").unlink()
    assert eos_token_ids(tmp_path, PALIGEMMA_3B_PT_224) == (1,)


SIGLIP_PROCESSOR = {
    "do_normalize": True, "do_rescale": True, "do_resize": True, "image_mean": [0.5, 0.5, 0.5],
    "image_processor_type": "SiglipImageProcessor", "image_seq_length": 256,
    "image_std": [0.5, 0.5, 0.5], "resample": 3, "rescale_factor": 0.00392156862745098,
    "size": {"height": 224, "width": 224},
}


def test_image_processor_from_either_transformers_layout(tmp_path):
    config = ModelConfig.from_dict(PALIGEMMA_3B_PT_224)
    (tmp_path / "processor_config.json").write_text(json.dumps({"image_processor": SIGLIP_PROCESSOR}))
    modern = image_processing(tmp_path, config.vision, config.num_image_tokens)
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(SIGLIP_PROCESSOR))
    legacy = image_processing(tmp_path, config.vision, config.num_image_tokens)
    assert modern == legacy and legacy.height == legacy.width == 224
    for key, value in (("resample", 2), ("size", {"height": 448, "width": 448}),
                       ("do_normalize", False), ("image_seq_length", 1024)):
        (tmp_path / "preprocessor_config.json").write_text(json.dumps({**SIGLIP_PROCESSOR, key: value}))
        with pytest.raises(ValueError):
            image_processing(tmp_path, config.vision, config.num_image_tokens)


def test_checkpoint_layouts_map_to_one_canonical_name(tmp_path):
    pytest.importorskip("torch")
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    from families.paligemma.checkpoint import CheckpointWeights

    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    layouts = {
        "v4": {"vision_tower.vision_model.post_layernorm.weight": value,
               "multi_modal_projector.linear.bias": value,
               "language_model.model.norm.weight": value},
        "v5": {"model.vision_tower.vision_model.post_layernorm.weight": value,
               "model.multi_modal_projector.linear.bias": value,
               "model.language_model.norm.weight": value, "lm_head.weight": value},
    }
    names = {}
    for name, tensors in layouts.items():
        directory = tmp_path / name
        directory.mkdir()
        safetensors_numpy.save_file(tensors, directory / "model.safetensors")
        weights = CheckpointWeights(directory)
        names[name] = weights.names()
        assert weights.shape("text.norm.weight") == (2, 3)
        np.testing.assert_array_equal(weights["projector.linear.bias"], value)
    assert set(names["v5"]) - set(names["v4"]) == {"lm_head.weight"}
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    safetensors_numpy.save_file({"vision_tower.vision_model.a": value, "model.language_model.b": value},
                                mixed / "model.safetensors")
    with pytest.raises(ValueError, match="layout"):
        CheckpointWeights(mixed)


def _sentencepiece_model(path: Path):
    model_pb2 = pytest.importorskip("sentencepiece.sentencepiece_model_pb2")
    model = model_pb2.ModelProto()
    model.trainer_spec.model_type = model_pb2.TrainerSpec.BPE
    model.trainer_spec.byte_fallback = True
    model.trainer_spec.unk_id, model.trainer_spec.bos_id = 3, 2
    model.trainer_spec.eos_id, model.trainer_spec.pad_id = 1, 0
    model.normalizer_spec.name = "identity"
    model.normalizer_spec.add_dummy_prefix = False
    model.normalizer_spec.remove_extra_whitespaces = False
    model.normalizer_spec.escape_whitespaces = True
    pieces = [("<pad>", 3), ("<eos>", 3), ("<bos>", 3), ("<unk>", 2), ("<mask>", 4)]
    pieces += [(f"<0x{value:02X}>", 6) for value in range(256)]
    pieces += [(f"<loc{i:04d}>", 4) for i in range(1024)] + [(f"<seg{i:03d}>", 4) for i in range(128)]
    pieces += [("a", 1), ("\u2581", 1)]
    for text, kind in pieces:
        piece = model.pieces.add()
        piece.piece, piece.type, piece.score = text, kind, 0.0
    path.write_bytes(model.SerializeToString())
    return len(pieces)


def test_tokenizer_special_and_reserved_ids_follow_the_reference_loader(tmp_path):
    from families.paligemma.tokenizer import load_tokenizer
    from families.paligemma.tokenizer_export import parse_trtmcbpe_asset

    count = _sentencepiece_model(tmp_path / "tokenizer.model")
    added = [{"id": 0, "content": "<pad>", "special": True}, {"id": 1, "content": "<eos>", "special": True},
             {"id": count, "content": "<image>", "special": True}]
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": added}))
    asset, facts = load_tokenizer(tmp_path, count, count + 64)
    assert len(parse_trtmcbpe_asset(asset).pieces) == count
    # GemmaTokenizer's named special tokens (BOS, UNK, <mask>) are special even when unlisted;
    # only pieces SentencePiece does not match atomically are reserved (<mask> is user-defined).
    assert facts == {"bos_token_id": 2, "tokenizer_pieces": count,
                     "special_token_ids": [0, 1, 2, 3, 4, count],
                     "reserved_token_ids": [0, 1, 2, 3, count]}
    added[1]["content"] = "<bos>"
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": added}))
    with pytest.raises(ValueError, match="is not that vocabulary piece"):
        load_tokenizer(tmp_path, count, count + 64)


def test_tokenizer_config_added_tokens_take_precedence(tmp_path):
    from families.paligemma.tokenizer import load_tokenizer

    count = _sentencepiece_model(tmp_path / "tokenizer.model")
    loc = 5 + 256  # after the four control/unknown pieces, <mask> and the byte pieces
    # tokenizer.json is ignored when tokenizer_config.json carries added_tokens_decoder.
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": [
        {"id": loc, "content": "<loc0000>", "special": True}]}))
    decoder = {"0": {"content": "<pad>", "special": True},
               str(loc): {"content": "<loc0000>", "special": False},
               str(loc + 1): {"content": "<loc0001>", "special": False, "lstrip": True},
               str(count): {"content": "<image>", "special": True}}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "added_tokens_decoder": decoder, "bos_token": "<bos>", "eos_token": {"content": "<eos>"},
        "unk_token": "<unk>", "pad_token": "<pad>", "mask_token": None}))
    _, facts = load_tokenizer(tmp_path, count, count + 64)
    assert facts["special_token_ids"] == [0, 1, 2, 3, count]
    # A whitespace-stripping added token matches differently from SentencePiece.
    assert facts["reserved_token_ids"] == [0, 1, 2, 3, loc + 1, count]


def test_tokenizer_legacy_files_and_special_token_arguments(tmp_path):
    from families.paligemma.tokenizer import load_tokenizer

    count = _sentencepiece_model(tmp_path / "tokenizer.model")
    loc, seg = 5 + 256, 5 + 256 + 1024
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": [
        {"id": 0, "content": "<pad>", "special": True}]}))
    (tmp_path / "added_tokens.json").write_text(json.dumps({"<image>": count, "<loc0001>": loc + 1}))
    # extra_special_tokens replaces additional_special_tokens; other *_token keys are special.
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "additional_special_tokens": ["<loc0000>"], "extra_special_tokens": ["<seg000>"],
        "custom_token": "<loc0007>", "add_bos_token": False}))
    _, facts = load_tokenizer(tmp_path, count, count + 64)
    assert facts["special_token_ids"] == [0, 1, 2, 3, 4, loc + 7, seg, count]
    assert facts["reserved_token_ids"] == [0, 1, 2, 3, count]
