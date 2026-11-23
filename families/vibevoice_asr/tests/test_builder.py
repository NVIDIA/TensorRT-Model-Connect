# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the builder's configuration, request and prompt contracts."""

from __future__ import annotations

import copy
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from families.vibevoice_asr import checkpoint
from families.vibevoice_asr import model
from families.vibevoice_asr.decoder_builder import rope_tables

_TOKENIZER = {
    "causal": True,
    "channels": 1,
    "conv_bias": True,
    "conv_norm": "none",
    "disable_last_norm": True,
    "encoder_depths": "3-3-3-3-3-3-8",
    "encoder_n_filters": 32,
    "encoder_ratios": [8, 5, 5, 4, 2, 2],
    "layernorm": "RMSNorm",
    "layernorm_elementwise_affine": True,
    "layernorm_eps": 1e-05,
    "mixer_layer": "depthwise_conv",
    "pad_mode": "constant",
}
# The released checkpoint's config.json, reduced to the fields the family reads.
CONFIG = {
    "architectures": ["VibeVoiceForASRTraining"],
    "model_type": "vibevoice",
    "acoustic_vae_dim": 64,
    "semantic_vae_dim": 128,
    "acoustic_tokenizer_config": dict(_TOKENIZER, vae_dim=64, std_dist_type="gaussian"),
    "semantic_tokenizer_config": dict(_TOKENIZER, vae_dim=128, std_dist_type="none"),
    "decoder_config": {
        "model_type": "qwen2",
        "hidden_act": "silu",
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "vocab_size": 152064,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "rope_scaling": None,
        "max_position_embeddings": 131072,
        "use_sliding_window": False,
        "use_mrope": False,
        "layer_types": ["full_attention"] * 28,
    },
}


def _config(**changes) -> dict:
    raw = copy.deepcopy(CONFIG)
    for path, value in changes.items():
        target = raw
        keys = path.split("__")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return raw


def _request(**overrides):
    values = dict(
        task="speech_transcription",
        backend="trt",
        precision="bf16",
        quantization=None,
        fp32_layers=(),
        dynamic_kv_cache=False,
        max_batch_size=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        mtp_seq_len=None,
        graph_transform=None,
        max_sequence_length=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_the_released_configuration_parses():
    cfg = checkpoint.parse_config(CONFIG)
    assert cfg.acoustic.ratios == (
        2,
        2,
        4,
        5,
        5,
        8,
    )  # encoder order: the config lists them reversed
    assert cfg.acoustic.depths == (3, 3, 3, 3, 3, 3, 8)
    assert cfg.hop_length == 3200
    assert (cfg.acoustic.vae_dim, cfg.semantic.vae_dim) == (64, 128)
    assert cfg.decoder.head_dim == 128 and cfg.decoder.num_kv_heads == 4


@pytest.mark.parametrize(
    "change",
    [
        {"architectures": ["VibeVoiceForConditionalGeneration"]},
        {"acoustic_tokenizer_config__causal": False},
        {"semantic_tokenizer_config__layernorm": "LN"},
        {"acoustic_tokenizer_config__mixer_layer": "conv"},
        {"semantic_tokenizer_config__encoder_ratios": [8, 5, 5, 4, 2, 3]},
        {"acoustic_vae_dim": 32},
        {"decoder_config__tie_word_embeddings": True},
        {"decoder_config__use_mrope": True},
        {"decoder_config__rope_scaling": {"type": "yarn", "factor": 4.0}},
    ],
)
def test_unsupported_configurations_are_rejected(change):
    with pytest.raises(ValueError):
        checkpoint.parse_config(_config(**change))


def _earliest_input(cfg: checkpoint.EncoderConfig, frame: int) -> int:
    """First input sample output ``frame`` reads, walking the causal windows back."""
    low = frame
    for kernel, stride in reversed(cfg.convolutions()):
        low = low * stride - (kernel - stride)
    return low


def test_the_receptive_field_bounds_the_chunk_context():
    cfg = checkpoint.parse_config(CONFIG)
    frame = 1000
    history = frame * cfg.hop_length - _earliest_input(cfg.acoustic, frame)
    assert history == cfg.acoustic.history_samples() == cfg.semantic.history_samples() == 185562
    # The runtime prepends whole frames of context; 58 cover the 185,562 samples.
    context = -(-cfg.acoustic.history_samples() // cfg.hop_length)
    assert context == 58 and context * cfg.hop_length >= history


def test_special_tokens_must_be_special_added_tokens():
    tokens = [
        {"id": 151643 + i, "content": content, "special": True}
        for i, content in enumerate(model.SPECIAL_TOKENS.values())
    ]
    ids = model.special_token_ids(json.dumps({"added_tokens": tokens}).encode())
    assert ids["eos"] == 151643 + list(model.SPECIAL_TOKENS).index("eos")
    tokens[0]["special"] = False
    with pytest.raises(ValueError):
        model.special_token_ids(json.dumps({"added_tokens": tokens}).encode())
    with pytest.raises(ValueError):
        model.special_token_ids(json.dumps({"added_tokens": tokens[1:]}).encode())
    with pytest.raises(ValueError, match="lacks the special token"):
        model.special_token_ids(b"{}")


def _qwen_tokenizer(vocab, added):
    """A tokenizer.json with Qwen2.5's layout around the given vocabulary."""
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": model.QWEN_SPLIT_PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
            ],
        },
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": [],
        },
    }


def test_tokenizer_ids_must_be_decoder_rows(monkeypatch):
    # Every id indexes the decoder's embedding table; runtime_contract refuses a
    # tokenizer with any other before the reference tokenizer runs.
    monkeypatch.setattr(model, "tokenizer_checks", lambda tokenizer_json: [])
    cfg = checkpoint.parse_config(CONFIG)
    specials = [
        {"id": 151643 + i, "content": content, "special": True}
        for i, content in enumerate(model.SPECIAL_TOKENS.values())
    ]

    def tokenizer(vocab, extra=()):
        return json.dumps(_qwen_tokenizer(vocab, specials + list(extra)))

    contract = model.runtime_contract(cfg, 4096, tokenizer({"a": 0, "b": 151642}).encode())
    assert contract["vocab_size"] == 152064
    for vocab, extra in (
        ({"a": 0, "b": 152064}, ()),
        ({"a": -1}, ()),
        ({"a": 1.5}, ()),
        ({"a": 0}, ({"id": 152064, "content": "<|stray|>", "special": True},)),
    ):
        with pytest.raises(ValueError, match="outside the decoder vocabulary"):
            model.runtime_contract(cfg, 4096, tokenizer(vocab, extra).encode())


def test_only_the_qwen_tokenizer_layout_builds():
    # The runtime tokenizer reproduces Qwen2.5's layout only; the build refuses any other.
    base = _qwen_tokenizer({"a": 0}, [{"id": 1, "content": "<|endoftext|>", "special": True}])
    model.check_tokenizer_layout(json.dumps(base).encode())

    def changed(path, value):
        raw = copy.deepcopy(base)
        target = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return json.dumps(raw).encode()

    for path, value in (
        (("normalizer",), {"type": "Lowercase"}),
        (("pre_tokenizer",), {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False}),
        (("pre_tokenizer", "pretokenizers", 1, "use_regex"), True),
        (("pre_tokenizer", "pretokenizers", 0, "pattern", "Regex"), r"\s+"),
        (("decoder",), {"type": "Metaspace"}),
        (("model", "byte_fallback"), True),
        (("model", "ignore_merges"), True),
        (("added_tokens", 0, "lstrip"), True),
        (("added_tokens", 0, "normalized"), True),
        (("pre_tokenizer", "pretokenizers", 0, "pattern"), "not an object"),
        (("model",), None),
        # Settings with which the reference would change the ids.
        (("truncation",), {"direction": "Right", "max_length": 4, "strategy": "LongestFirst", "stride": 0}),
        (("padding",), {"strategy": "BatchLongest", "direction": "Right", "pad_id": 1, "pad_token": "x"}),
        (("post_processor",), _TEMPLATE_POST_PROCESSOR),
        (("model", "unk_token"), "<|endoftext|>"),
        # Options are booleans, as the runtime requires: 0 or "" is not false.
        (("pre_tokenizer", "pretokenizers", 0, "invert"), 0),
        (("model", "byte_fallback"), 0),
        (("added_tokens", 0, "lstrip"), ""),
        (("added_tokens",), None),
    ):
        with pytest.raises(ValueError, match="is not supported"):
            model.check_tokenizer_layout(changed(path, value))
    for path, value in ((("added_tokens", 0, "content"), None), (("added_tokens", 0, "id"), 1.0)):
        with pytest.raises(ValueError, match="needs a string content and an integer id"):
            model.check_tokenizer_layout(changed(path, value))
    # A null option is its default, as for the runtime.
    model.check_tokenizer_layout(changed(("model", "byte_fallback"), None))


_TEMPLATE_POST_PROCESSOR = {
    "type": "TemplateProcessing",
    "single": [
        {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
    ],
    "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
    "special_tokens": {"<|endoftext|>": {"id": "<|endoftext|>", "ids": [1], "tokens": ["<|endoftext|>"]}},
}


def test_the_build_applies_the_layout_check(monkeypatch):
    # runtime_contract refuses a layout the runtime would refuse at load.
    monkeypatch.setattr(model, "tokenizer_checks", lambda tokenizer_json: [])
    cfg = checkpoint.parse_config(CONFIG)
    added = [
        {"id": 151643 + i, "content": content, "special": True}
        for i, content in enumerate(model.SPECIAL_TOKENS.values())
    ]
    raw = _qwen_tokenizer({"a": 0}, added)
    raw["post_processor"] = _TEMPLATE_POST_PROCESSOR
    with pytest.raises(ValueError, match="post_processor is not supported"):
        model.runtime_contract(cfg, 4096, json.dumps(raw).encode())


def test_added_token_ids_are_the_ids_the_reference_assigns():
    # The reference re-adds tokens by content after the vocabulary: with one vocabulary
    # entry, a token listed as id 5 gets id 1, which the runtime would not use.
    pytest.importorskip("tokenizers")
    options = {"single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
    token = {"id": 5, "content": "<|endoftext|>", "special": True, **options}
    raw = _qwen_tokenizer({"a": 0}, [token])
    model.tokenizer_checks(json.dumps(_qwen_tokenizer({"a": 0}, [dict(token, id=1)])).encode())
    with pytest.raises(ValueError, match="the reference tokenizer assigns it 1"):
        model.tokenizer_checks(json.dumps(raw).encode())


def test_tokenizer_config_switches_must_be_off(tmp_path):
    # transformers reads these from tokenizer_config.json; the runtime reads only tokenizer.json.
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "tokenizer_config.json"
    qwen = {
        "add_bos_token": False,
        "add_prefix_space": False,
        "clean_up_tokenization_spaces": False,
        "split_special_tokens": False,
        "tokenizer_class": "Qwen2Tokenizer",
    }
    config.write_text(json.dumps(qwen), encoding="utf-8")
    assert set(model._tokenizer_files(tmp_path)) == set(model.TOKENIZER_FILES)
    for switch in model.TOKENIZER_CONFIG_SWITCHES:
        config.write_text(json.dumps(dict(qwen, **{switch: True})), encoding="utf-8")
        with pytest.raises(ValueError, match=f"{switch} is not supported"):
            model._tokenizer_files(tmp_path)


def test_the_user_turn_matches_the_official_processor():
    keys = "Start time, End time, Speaker ID, Content"
    assert model.user_suffix("7.56") == (
        f"This is a 7.56 seconds audio, please transcribe it with these keys: {keys}"
    )
    assert model.user_suffix("1.00", "  Names: Ada \n") == (
        f"This is a 1.00 seconds audio, with extra info: Names: Ada\n\n"
        f"Please transcribe it with these keys: {keys}"
    )
    assert model.user_suffix("1.00", " \n ") == model.user_suffix("1.00")


def test_the_request_accepts_only_what_the_family_builds(tmp_path):
    for override in (
        dict(precision="fp16"),
        dict(quantization="fp8"),
        dict(fp32_layers=(1,)),
        dict(dynamic_kv_cache=True),
        dict(max_batch_size=2),
        dict(tensor_parallel_size=2),
        dict(context_parallel_size=2),
        dict(image_height=224),
        dict(video_num_frames=8),
        dict(mtp_seq_len=2),
        dict(graph_transform=lambda *args: None),
        dict(backend="trt_rtx"),
        dict(task="speech_translation"),
    ):
        with pytest.raises((ValueError, NotImplementedError)):
            model.build(_request(model_dir=tmp_path, **override), writer=None)
    # A supported request gets past the option checks to the checkpoint itself.
    with pytest.raises(FileNotFoundError):
        model.build(_request(model_dir=tmp_path), writer=None)


def test_the_kv_capacity_defaults_to_the_single_pass_budget():
    cfg = checkpoint.parse_config(CONFIG)
    assert model.resolve_capacity(None, cfg) == 65536
    assert model.resolve_capacity(8192, cfg) == 8192
    for invalid in (100, 8000, 131073, 131584):
        with pytest.raises(ValueError):
            model.resolve_capacity(invalid, cfg)


def test_rope_tables_follow_qwen2_rotary_embedding():
    torch = pytest.importorskip("torch")
    cfg = checkpoint.parse_config(CONFIG).decoder
    cos, sin = rope_tables(cfg, 4096)
    inv_freq = 1.0 / (
        cfg.rope_theta
        ** (
            torch.arange(0, cfg.head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / cfg.head_dim
        )
    )
    freqs = torch.arange(4096, dtype=torch.float32)[:, None] * inv_freq[None, :]
    # torch's vectorized FP32 pow (it depends on the CPU) and NumPy's give inverse
    # frequencies up to a few ulps apart, which moves an angle by up to about seven
    # of its ulps and cos and sin by as much; a wrong formula is off by far more.
    tolerance = 2e-6 + 8 * np.spacing(np.abs(freqs.numpy()))
    assert np.all(np.abs(cos - freqs.cos().numpy()) <= tolerance)
    assert np.all(np.abs(sin - freqs.sin().numpy()) <= tolerance)


def test_a_checkpoint_tokenizer_wins_only_when_it_ships_a_tokenizer_config(tmp_path):
    # VibeVoiceASRProcessor keys on tokenizer_config.json, not tokenizer.json.
    assert model.tokenizer_root(tmp_path) is None
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert model.tokenizer_root(tmp_path) is None
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert model.tokenizer_root(tmp_path) == tmp_path


def test_the_pinned_tokenizer_is_read_from_the_cache_only(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls = []

    def not_cached(*args, **kwargs):
        calls.append((args, kwargs))
        raise LocalEntryNotFoundError("not in the cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", not_cached)
    command = (
        "hf download Qwen/Qwen2.5-7B tokenizer.json tokenizer_config.json "
        f"--revision {model.TOKENIZER_REVISION}"
    )
    with pytest.raises(FileNotFoundError, match=re.escape(command)):
        model._tokenizer_files(tmp_path)
    assert calls == [
        (
            (model.TOKENIZER_REPOSITORY,),
            {
                "revision": model.TOKENIZER_REVISION,
                "allow_patterns": model.TOKENIZER_FILES,
                "local_files_only": True,
            },
        )
    ]


def test_a_partly_cached_tokenizer_names_the_fetch_command(tmp_path, monkeypatch):
    import huggingface_hub

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: snapshot)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=re.escape(model.TOKENIZER_FETCH_COMMAND)):
        model._tokenizer_files(checkpoint_dir)
