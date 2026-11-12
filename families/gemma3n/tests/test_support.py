# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checks of Gemma 3n ownership, configuration and processing rules."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from families.gemma3n import processing
from families.gemma3n.config import ModelConfig
from families.gemma3n.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family

DATA = Path(__file__).resolve().parent / "data"
LAYER_TYPES = ["full_attention" if (i + 1) % 5 == 0 else "sliding_attention" for i in range(35)]
CONFIG = {
    "architectures": ["Gemma3nForConditionalGeneration"],
    "model_type": "gemma3n",
    "image_token_id": 262145,
    "boi_token_id": 255999,
    "eoi_token_id": 262144,
    "eos_token_id": 106,
    "vision_soft_tokens_per_image": 256,
    "text_config": {
        "model_type": "gemma3n_text", "hidden_size": 2048, "num_hidden_layers": 35,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256,
        "intermediate_size": [16384] * 35, "layer_types": LAYER_TYPES, "sliding_window": 512,
        "num_kv_shared_layers": 15, "altup_num_inputs": 4, "altup_active_idx": 0,
        "laurel_rank": 64, "hidden_size_per_layer_input": 256, "vocab_size": 262400,
        "vocab_size_per_layer_input": 262144, "rms_norm_eps": 1e-6,
        "final_logit_softcapping": 30.0, "rope_theta": 1000000.0, "rope_local_base_freq": 10000.0,
        "rope_scaling": None, "activation_sparsity_pattern": [0.95] * 10 + [0.0] * 25,
        "hidden_activation": "gelu_pytorch_tanh", "max_position_embeddings": 32768,
    },
    "vision_config": {"model_type": "gemma3n_vision", "architecture": "mobilenetv5_300m_enc",
                      "hidden_size": 2048, "vocab_offset": 262144, "vocab_size": 128,
                      "rms_norm_eps": 1e-6, "model_args": None},
    "audio_config": {"model_type": "gemma3n_audio", "hidden_size": 1536, "vocab_offset": 262272,
                     "vocab_size": 128, "rms_norm_eps": 1e-6},
}


def _config(**text):
    raw = copy.deepcopy(CONFIG)
    raw["text_config"].update(text)
    return raw


def test_gemma3n_has_one_owner_with_one_task():
    name, support = resolve_family(ModelMetadata(config=CONFIG, model_index={}))
    assert name == "gemma3n"
    assert support.tasks == ("images_text_to_text",)
    assert support.default_task == "images_text_to_text"
    assert support.default_precision == "bf16"


def test_earlier_gemma_generations_are_declined():
    for model_type, architecture in (("gemma3", "Gemma3ForConditionalGeneration"),
                                     ("gemma3_text", "Gemma3ForCausalLM"),
                                     ("gemma4", "Gemma4ForConditionalGeneration")):
        config = {"model_type": model_type, "architectures": [architecture]}
        assert describe(ModelMetadata(config=config, model_index={})) is None


def test_kv_sharing_follows_the_last_owner_of_each_attention_type():
    text = ModelConfig.from_raw(CONFIG).text
    owners = text.kv_owners()
    assert owners[:20] == tuple(range(20))
    # Sliding layers reuse layer 18, full-attention layers reuse layer 19.
    assert all(owners[i] == (19 if LAYER_TYPES[i] == "full_attention" else 18) for i in range(20, 35))
    assert text.cache_layers() == tuple(range(20))


def test_rope_bases_resolve_from_either_config_format():
    legacy = ModelConfig.from_raw(CONFIG).text.rope_theta
    assert legacy == {"full_attention": 1e6, "sliding_attention": 1e4}
    raw = _config(rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 5e5},
                                   "sliding_attention": {"rope_type": "default", "rope_theta": 1e4}})
    assert ModelConfig.from_raw(raw).text.rope_theta["full_attention"] == 5e5
    with pytest.raises(NotImplementedError, match="RoPE scaling"):
        ModelConfig.from_raw(_config(rope_scaling={"rope_type": "linear", "factor": 2.0}))
    with pytest.raises(NotImplementedError, match="default rotation"):
        ModelConfig.from_raw(_config(rope_parameters={"full_attention": {"rope_type": "yarn"}}))


@pytest.mark.parametrize("change, error", [
    ({"num_kv_shared_layers": 35}, ValueError),
    ({"altup_active_idx": 1}, NotImplementedError),
    ({"attention_bias": True}, NotImplementedError),
    ({"hidden_activation": "gelu"}, NotImplementedError),
    ({"intermediate_size": [16384] * 34}, ValueError),
    ({"activation_sparsity_pattern": [1.0] * 35}, ValueError),
    ({"layer_types": ["chunked_attention"] * 35}, ValueError),
])
def test_unsupported_text_configurations_are_refused(change, error):
    with pytest.raises(error):
        ModelConfig.from_raw(_config(**change))


def test_untied_language_model_head_is_refused():
    top_level = copy.deepcopy(CONFIG)
    top_level["tie_word_embeddings"] = False
    for raw in (top_level, _config(tie_word_embeddings=False)):
        with pytest.raises(NotImplementedError, match="tied language-model head"):
            ModelConfig.from_raw(raw)


def test_vision_tower_and_vocabularies_are_checked():
    raw = copy.deepcopy(CONFIG)
    raw["vision_config"]["architecture"] = "mobilenetv5_base"
    with pytest.raises(NotImplementedError, match="vision tower"):
        ModelConfig.from_raw(raw)
    raw = copy.deepcopy(CONFIG)
    raw["audio_config"]["vocab_offset"] = 262300
    with pytest.raises(ValueError, match="tile"):
        ModelConfig.from_raw(raw)


def test_native_rendering_matches_the_reference_processor():
    """The runtime's rendering, expanded like the processor, equals the recorded goldens."""
    goldens = json.loads((DATA / "goldens.json").read_text(encoding="utf-8"))["prompts"]
    expanded = "\n\n<start_of_image>" + "<image_soft_token>" * 256 + "<end_of_image>\n\n"
    for golden in goldens:
        text = processing.native_chat_text(golden["messages"])
        assert text.replace("<image_soft_token>", expanded) == golden["rendered"]


def test_chat_template_check_renders_with_jinja():
    pytest.importorskip("jinja2")
    template = (
        "{{ bos_token }}{%- for message in messages -%}"
        "{{ '<start_of_turn>' + ('model' if message['role'] == 'assistant' else message['role']) + '\n' }}"
        "{%- for item in message['content'] -%}"
        "{%- if item['type'] == 'image' -%}{{ '<image_soft_token>' }}"
        "{%- else -%}{{ item['text'] | trim }}{%- endif -%}{%- endfor -%}"
        "{{ '<end_of_turn>\n' }}{%- endfor -%}{{ '<start_of_turn>model\n' }}")
    # Without the system-prefix rule this template renders the probes differently.
    with pytest.raises(NotImplementedError, match="chat template"):
        processing.validate_chat_template(template)


def _image_processor(**change):
    processor = {"image_processor_type": "SiglipImageProcessorFast", "do_resize": True,
                 "size": {"height": 768, "width": 768}, "resample": 2, "do_rescale": True,
                 "rescale_factor": 0.00392156862745098, "do_normalize": False,
                 "do_center_crop": None, "do_convert_rgb": None}
    processor.update(change)
    return processor


def test_image_processing_is_the_implemented_one():
    assert processing.validate_image_processor(_image_processor()) == 0.00392156862745098
    for change in ({"resample": 3}, {"do_normalize": True}, {"size": {"height": 512, "width": 512}},
                   {"image_processor_type": "Gemma3ImageProcessor"}):
        with pytest.raises(NotImplementedError):
            processing.validate_image_processor(_image_processor(**change))


def _fixture_tokenizer() -> dict:
    return json.loads((DATA / "tokenizer_fixture.json").read_text(encoding="utf-8"))


def _fixture_config(**ids) -> ModelConfig:
    """A model configuration whose vocabularies match the 291-token fixture."""
    raw = copy.deepcopy(CONFIG)
    raw.update({"image_token_id": 278, "boi_token_id": 279, "eoi_token_id": 280, **ids})
    raw["text_config"].update(vocab_size=291, vocab_size_per_layer_input=278)
    raw["vision_config"].update(vocab_offset=278, vocab_size=6)
    raw["audio_config"].update(vocab_offset=284, vocab_size=7)
    return ModelConfig.from_raw(raw)


def _rename_token(tokenizer: dict, old: str, new: str) -> None:
    vocab = tokenizer["model"]["vocab"]
    vocab[new] = vocab.pop(old)


def _drop_added_token(tokenizer: dict, content: str) -> None:
    tokenizer["added_tokens"] = [token for token in tokenizer["added_tokens"]
                                 if token["content"] != content]


# One change each to a fixture that passes every check: (change, error, message).
TOKENIZER_CHANGES = {
    "normalizer": (lambda t: t.update(normalizer={"type": "NFC"}), NotImplementedError,
                   "normalizer"),
    "decoder": (lambda t: t["decoder"]["decoders"].pop(), NotImplementedError, "decoder"),
    "dropout": (lambda t: t["model"].update(dropout=0.1), NotImplementedError, "without dropout"),
    "added token flags": (lambda t: t["added_tokens"][0].update(lstrip=True), NotImplementedError,
                          "verbatim"),
    "vocabulary size": (lambda t: t["model"]["vocab"].update({"<extra>": 291}), ValueError,
                        "vocabulary"),
    "merges as strings": (lambda t: t["model"].update(
        merges=[" ".join(pair) for pair in t["model"]["merges"]]), NotImplementedError, "merges"),
    "byte-fallback token": (lambda t: _rename_token(t, "<0x41>", "<0x41>?"), ValueError,
                            "byte-fallback token"),
    "added token id": (lambda t: t["added_tokens"][2].update(id=3), ValueError,
                       "disagree with the vocabulary"),
    "special token": (lambda t: _drop_added_token(t, "<start_of_image>"), ValueError,
                      "special token"),
}


def test_tokenizer_fixture_passes_the_pipeline_checks():
    assert processing.validate_tokenizer(_fixture_tokenizer(), _fixture_config()) == {
        "<bos>": 2, "<start_of_turn>": 276, "<end_of_turn>": 277, "<start_of_image>": 279,
        "<image_soft_token>": 278, "<end_of_image>": 280}


@pytest.mark.parametrize("name", sorted(TOKENIZER_CHANGES))
def test_tokenizer_changes_are_refused_one_at_a_time(name):
    change, error, message = TOKENIZER_CHANGES[name]
    tokenizer = _fixture_tokenizer()
    change(tokenizer)
    with pytest.raises(error, match=message):
        processing.validate_tokenizer(tokenizer, _fixture_config())


def test_special_token_ids_must_match_the_model_config():
    # Token 281 lies in the vision vocabulary but is not the image placeholder.
    with pytest.raises(ValueError, match="<image_soft_token> id"):
        processing.validate_tokenizer(_fixture_tokenizer(), _fixture_config(image_token_id=281))


@pytest.mark.parametrize("files, accepted", [
    ({"tokenizer_config.json": {"bos_token": "<bos>"}}, True),
    ({"tokenizer_config.json": {"bos_token": {"content": "<bos>", "special": True}},
      "special_tokens_map.json": {"bos_token": {"content": "<bos>"}}}, True),
    ({"tokenizer_config.json": {"bos_token": "<s>"}}, False),
    ({"tokenizer_config.json": {}}, False),
    ({"tokenizer_config.json": {"bos_token": "<bos>"},
      "special_tokens_map.json": {"bos_token": "<eos>"}}, False),
])
def test_reference_must_skip_special_tokens_on_rendered_prompts(tmp_path, files, accepted):
    for name, content in files.items():
        (tmp_path / name).write_text(json.dumps(content), encoding="utf-8")
    if accepted:
        processing.validate_bos_token(tmp_path)
    else:
        with pytest.raises(NotImplementedError, match="BOS token"):
            processing.validate_bos_token(tmp_path)
