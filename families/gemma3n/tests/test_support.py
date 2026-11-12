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


def test_earlier_gemma_generations_stay_with_their_owner():
    for model_type in ("gemma3", "gemma3_text", "gemma4"):
        name, _ = resolve_family(ModelMetadata(config={"model_type": model_type}, model_index={}))
        assert name == "gemma"


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


def test_tokenizer_pipeline_is_checked_on_the_fixture():
    tokenizer = json.loads((DATA / "tokenizer_fixture.json").read_text(encoding="utf-8"))
    config = ModelConfig.from_raw(CONFIG)
    altered = copy.deepcopy(tokenizer)
    altered["normalizer"] = {"type": "NFC"}
    with pytest.raises(NotImplementedError, match="normalizer"):
        processing.validate_tokenizer(altered, config)
    altered = copy.deepcopy(tokenizer)
    altered["added_tokens"][0]["lstrip"] = True
    with pytest.raises(NotImplementedError, match="verbatim"):
        processing.validate_tokenizer(altered, config)
    # The fixture's vocabulary is smaller than the model's, which is refused.
    with pytest.raises(ValueError, match="vocabulary"):
        processing.validate_tokenizer(tokenizer, config)
