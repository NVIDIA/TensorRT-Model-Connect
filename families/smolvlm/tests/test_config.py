# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the checkpoint contract the builder enforces before any engine build."""

import copy
import json

import pytest

from families.smolvlm.config import CHAT_TEMPLATE, ModelConfig

# The fields of HuggingFaceTB/SmolVLM2-500M-Video-Instruct that the family reads.
CONFIG = {
    "architectures": ["SmolVLMForConditionalGeneration"], "model_type": "smolvlm",
    "image_token_id": 49190, "scale_factor": 4, "tie_word_embeddings": False, "vocab_size": 49280,
    "text_config": {
        "head_dim": 64, "hidden_size": 960, "intermediate_size": 2560, "max_position_embeddings": 8192,
        "model_type": "llama", "num_attention_heads": 15, "num_hidden_layers": 32, "num_key_value_heads": 5,
        "qk_layer_norms": False, "rms_norm_eps": 1e-05, "rope_interleaved": False, "rope_theta": 100000,
        "use_resampler": False, "vocab_size": 49280, "tie_word_embeddings": True,
    },
    "vision_config": {
        "hidden_size": 768, "image_size": 512, "model_type": "smolvlm_vision", "num_attention_heads": 12,
        "patch_size": 16,
    },
}
PREPROCESSOR = {
    "do_convert_rgb": True, "do_image_splitting": True, "do_normalize": True, "do_pad": True,
    "do_rescale": True, "do_resize": True, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    "image_processor_type": "SmolVLMImageProcessor", "max_image_size": {"longest_edge": 512},
    "resample": 1, "rescale_factor": 0.00392156862745098, "size": {"longest_edge": 2048},
}
PROCESSOR = {"image_seq_len": 64}
GENERATION = {"bos_token_id": 0, "eos_token_id": 49279, "pad_token_id": 2}


def parse(config=CONFIG, preprocessor=PREPROCESSOR, processor=PROCESSOR, generation=GENERATION):
    return ModelConfig.from_dicts(copy.deepcopy(config), copy.deepcopy(preprocessor),
                                  copy.deepcopy(processor), copy.deepcopy(generation))


def test_reference_defaults_fill_absent_vision_and_text_keys():
    config = parse()
    assert (config.vision.num_hidden_layers, config.vision.intermediate_size) == (12, 3072)
    assert config.vision.layer_norm_eps == 1e-6 and config.vision.patches_per_side == 32
    assert config.text.rope_theta == 100000.0 and config.text.head_dim == 64
    assert config.image_tokens_per_tile == 64 and config.connector_width == 12288
    assert config.processing.longest_edge == 2048 and config.processing.tile_size == 512
    assert config.eos_token_id == 49279


def test_rope_parameters_spelling_is_accepted():
    config = copy.deepcopy(CONFIG)
    del config["text_config"]["rope_theta"]
    config["text_config"]["rope_parameters"] = {"rope_type": "default", "rope_theta": 100000.0}
    assert parse(config).text.rope_theta == 100000.0


def _mutated(path, value):
    config = copy.deepcopy(CONFIG)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return config


@pytest.mark.parametrize("path,value", [
    (("model_type",), "idefics3"),
    (("tie_word_embeddings",), True),
    (("text_config", "use_resampler"), True),
    (("text_config", "rope_scaling"), {"rope_type": "linear", "factor": 2.0}),
    (("text_config", "num_key_value_heads"), 4),
    (("text_config", "hidden_act"), "gelu"),
    (("vision_config", "hidden_act"), "gelu"),
    (("vision_config", "patch_size"), 15),
    (("scale_factor", ), 3),
    (("image_token_id",), 10 ** 6),
])
def test_unsupported_model_variants_are_rejected(path, value):
    with pytest.raises(ValueError):
        parse(_mutated(path, value))


@pytest.mark.parametrize("key,value", [
    ("do_image_splitting", False), ("resample", 3), ("size", {"height": 512, "width": 512}),
    ("max_image_size", {"longest_edge": 384}), ("image_std", [0.5, 0.0, 0.5]),
    ("image_processor_type", "Idefics3ImageProcessor"),
])
def test_unsupported_preprocessing_is_rejected(key, value):
    with pytest.raises(ValueError):
        parse(preprocessor=dict(PREPROCESSOR, **{key: value}))


def test_processor_and_generation_contracts():
    with pytest.raises(ValueError):
        parse(processor={"image_seq_len": 169})
    with pytest.raises(ValueError):
        parse(generation=dict(GENERATION, do_sample=True))
    with pytest.raises(ValueError):
        parse(generation={"eos_token_id": [49279, 2]})


# The pinned checkpoint's chat_template.json, copied verbatim (JSON-escaped) and kept apart from
# the family's constant so that a change anywhere in either one fails the test.
CHECKPOINT_CHAT_TEMPLATE_JSON = (
    '{"chat_template": "'
    r"<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}"
    r"{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}"
    r"{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}"
    r"{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}"
    r"<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
    '"}'
)


def test_chat_template_is_the_checkpoint_template():
    # Rendered by the runtime; any other template must be rejected at build time.
    assert CHAT_TEMPLATE == json.loads(CHECKPOINT_CHAT_TEMPLATE_JSON)["chat_template"]


def _tokenizer_document():
    added = [{"id": 49190, "content": "<image>"}, {"id": 49279, "content": "<end_of_utterance>"},
             {"id": 49189, "content": "<fake_token_around_image>"}, {"id": 49152, "content": "<global-img>"},
             {"id": 1, "content": "<|im_start|>"}]
    added += [{"id": 49152 + 4 * (row - 1) + col, "content": f"<row_{row}_col_{col}>"}
              for row in range(1, 5) for col in range(1, 5)]
    for token in added:
        token.update(special=True, lstrip=False, rstrip=False, single_word=False, normalized=False)
    return {
        "added_tokens": added, "normalizer": None, "post_processor": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Digits", "individual_digits": True},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}]},
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "dropout": None, "byte_fallback": False, "ignore_merges": False},
    }


def test_tokenizer_contract():
    pytest.importorskip("tensorrt")
    from families.smolvlm.model import _validate_tokenizer

    config = parse()
    _validate_tokenizer(_tokenizer_document(), config)
    for mutate in (
        lambda d: d.update(normalizer={"type": "NFC"}),
        lambda d: d["pre_tokenizer"]["pretokenizers"][0].update(individual_digits=False),
        lambda d: d["pre_tokenizer"]["pretokenizers"][1].update(add_prefix_space=True),
        lambda d: d["added_tokens"][0].update(id=7),
        lambda d: d["added_tokens"][3].update(lstrip=True),
        lambda d: d.update(added_tokens=[t for t in d["added_tokens"] if t["content"] != "<row_4_col_4>"]),
    ):
        document = _tokenizer_document()
        mutate(document)
        with pytest.raises(ValueError):
            _validate_tokenizer(document, config)


def test_checkpoint_key_inventory_and_tile_profile():
    pytest.importorskip("tensorrt")
    from families.smolvlm.graph import max_tiles_per_image
    from families.smolvlm.model import _expected_checkpoint_keys

    config = parse()
    keys = _expected_checkpoint_keys(config)
    # 3 embedding, 12 x 16 vision, 2 post-norm, 1 connector, 1 + 32 x 9 + 1 text, 1 lm_head.
    assert len(keys) == 489
    assert "lm_head.weight" in keys and "model.text_model.embed_tokens.weight" in keys
    assert max_tiles_per_image(config) == 17
