# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit checks for the rules the WavLM graph depends on.

Each rule that is read from the checkpoint rather than stated in the config has
a control here that fails when the rule is broken. The controls matter because
each of these mistakes loads without error and still produces output.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from families.wavlm import config as config_module, graph as g, model

# The geometry of the released checkpoints, as their config.json states it.
_LARGE = {
    "model_type": "wavlm", "hidden_size": 1024, "num_hidden_layers": 24,
    "num_attention_heads": 16, "intermediate_size": 4096, "layer_norm_eps": 1e-05,
    "do_stable_layer_norm": True, "feat_extract_norm": "layer", "conv_bias": False,
    "hidden_act": "gelu", "feat_extract_activation": "gelu", "add_adapter": False,
    "conv_dim": [512] * 7, "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2], "num_conv_pos_embeddings": 128,
    "num_conv_pos_embedding_groups": 16, "num_buckets": 320, "max_bucket_distance": 800,
}
_BASE_PLUS = {**_LARGE, "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
              "intermediate_size": 3072, "do_stable_layer_norm": False,
              "feat_extract_norm": "group"}
_CTC = {"architectures": ["WavLMForCTC"], "vocab_size": 31, "pad_token_id": 28,
        "tokenizer_class": "Wav2Vec2CTCTokenizer"}
_VOCAB = {"|": 0, **{chr(ord("a") + i): i + 1 for i in range(26)}, "[UNK]": 27, "[PAD]": 28}
_ADDED = {"<s>": 29, "</s>": 30}
_TOKENIZER = {"unk_token": "[UNK]", "bos_token": "<s>", "eos_token": "</s>",
              "pad_token": "[PAD]", "word_delimiter_token": "|",
              "tokenizer_class": "Wav2Vec2CTCTokenizer"}
_PRE = {"feature_size": 1, "padding_value": 0.0, "padding_side": "right",
        "sampling_rate": 16000, "do_normalize": True}


def _write(tmp_path, cfg=None, pre=None, vocab=None, added=None, tokenizer=None):
    (tmp_path / "config.json").write_text(json.dumps(cfg or {**_LARGE, **_CTC}))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(pre or _PRE))
    (tmp_path / "vocab.json").write_text(json.dumps(_VOCAB if vocab is None else vocab))
    (tmp_path / "added_tokens.json").write_text(json.dumps(_ADDED if added is None else added))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(_TOKENIZER if tokenizer is None else tokenizer))
    return tmp_path


def test_resolve_reads_the_large_layout_and_its_tokenizer(tmp_path):
    cfg = config_module.resolve(_write(tmp_path), {"lm_head.weight": (31, 1024)})
    assert cfg["do_stable_layer_norm"] is True
    assert cfg["feat_extract_norm"] == "layer"
    assert len(cfg["tokens"]) == 31
    assert cfg["tokens"][:2] == ["|", "a"]
    assert cfg["tokens"][28:] == ["[PAD]", "<s>", "</s>"]
    # The blank is the tokenizer's [PAD], not id 0: id 0 is the word delimiter.
    assert cfg["blank_token_id"] == 28
    assert cfg["word_delimiter_token"] == "|"
    assert cfg["do_normalize"] is True
    assert (cfg["num_buckets"], cfg["max_distance"]) == (320, 800)


def test_resolve_reads_the_base_plus_layout(tmp_path):
    pre = {**_PRE, "do_normalize": False}
    cfg = config_module.resolve(_write(tmp_path, cfg={**_BASE_PLUS, **_CTC}, pre=pre),
                                {"lm_head.weight": (31, 768)})
    assert cfg["do_stable_layer_norm"] is False
    assert cfg["feat_extract_norm"] == "group"
    # The base checkpoints were trained on unnormalised waveforms.
    assert cfg["do_normalize"] is False


def test_the_pretrained_encoder_resolves_but_is_not_a_ctc_checkpoint(tmp_path):
    pretrained = {**_LARGE, "architectures": ["WavLMModel"], "vocab_size": 32, "pad_token_id": 0}
    geometry = config_module.resolve_encoder(pretrained)
    assert geometry["num_hidden_layers"] == 24 and geometry["do_stable_layer_norm"] is True
    with pytest.raises(NotImplementedError, match="WavLMForCTC, not .*WavLMModel"):
        config_module.resolve(_write(tmp_path, cfg=pretrained), {})


@pytest.mark.parametrize("overrides, fragment", [
    ({"model_type": "wav2vec2"}, "model_type 'wavlm'"),
    ({"do_stable_layer_norm": "true"}, "boolean do_stable_layer_norm"),
    ({"feat_extract_norm": "batch"}, "feat_extract_norm"),
    ({"conv_bias": "true"}, "boolean conv_bias"),
    ({"feat_extract_activation": "relu"}, "feature activation"),
    ({"add_adapter": True}, "adapters"),
    ({"adapter_attn_dim": 16}, "adapters"),
    ({"hidden_act": "gelu_new"}, "exact GELU"),
    ({"num_conv_pos_embeddings": 127}, "even num_conv_pos_embeddings"),
    ({"num_buckets": 2}, "num_buckets"),
    ({"max_bucket_distance": 80}, "max_bucket_distance"),
    ({"architectures": ["WavLMForSequenceClassification"]}, "WavLMForCTC"),
    ({"pad_token_id": 0}, "pad_token_id"),
    ({"tokenizer_class": "Wav2Vec2PhonemeCTCTokenizer"}, "Wav2Vec2CTCTokenizer"),
])
def test_resolve_refuses_the_variants_it_does_not_build(tmp_path, overrides, fragment):
    # Each of these loads without error and produces plausible output, so the
    # refusal is the only thing standing between them and a silent wrong answer.
    tokenizer = {key: value for key, value in _TOKENIZER.items() if key != "tokenizer_class"}
    with pytest.raises(NotImplementedError, match=fragment):
        config_module.resolve(
            _write(tmp_path, cfg={**_LARGE, **_CTC, **overrides}, tokenizer=tokenizer), {})


@pytest.mark.parametrize("vocab, added, fragment", [
    ({"|": 0, "a": 1}, {}, "unnamed"),
    ({**_VOCAB, "b": 2, "c": 2}, _ADDED, "two"),
    ({**_VOCAB, "z": 31}, _ADDED, "outside"),
    ({key: value for key, value in _VOCAB.items() if key != "[PAD]"},
     {**_ADDED, "<blank>": 28}, "pad token"),
])
def test_resolve_refuses_a_vocabulary_that_does_not_match(tmp_path, vocab, added, fragment):
    with pytest.raises(NotImplementedError, match=fragment):
        config_module.resolve(_write(tmp_path, vocab=vocab, added=added), {})


def test_resolve_reads_serialised_special_tokens(tmp_path):
    tokenizer = {**_TOKENIZER, "pad_token": {"content": "[PAD]", "lstrip": False}}
    cfg = config_module.resolve(_write(tmp_path, tokenizer=tokenizer), {})
    assert cfg["blank_token_id"] == 28


@pytest.mark.parametrize("tokenizer, vocab, fragment", [
    ({**_TOKENIZER, "do_lower_case": True}, None, "do_lower_case"),
    ({**_TOKENIZER, "clean_up_tokenization_spaces": True},
     {**{key: value for key, value in _VOCAB.items() if key != "x"}, "'": 24},
     "clean_up_tokenization_spaces"),
])
def test_resolve_refuses_decoding_rules_the_runtime_does_not_apply(tmp_path, tokenizer, vocab,
                                                                    fragment):
    # Transformers lowercases the transcript, or rewrites the space before an
    # apostrophe, only when the tokenizer asks; the runtime never does.
    with pytest.raises(NotImplementedError, match=fragment):
        config_module.resolve(_write(tmp_path, vocab=vocab, tokenizer=tokenizer), {})


def test_clean_up_is_accepted_where_it_cannot_change_the_text(tmp_path):
    # The released vocabulary has letters, '|' and special tokens only, none of
    # the characters the clean-up rewrites a space around.
    tokenizer = {**_TOKENIZER, "clean_up_tokenization_spaces": True}
    assert config_module.resolve(_write(tmp_path, tokenizer=tokenizer), {})["blank_token_id"] == 28


def test_resolve_refuses_a_head_of_the_wrong_width(tmp_path):
    with pytest.raises(NotImplementedError, match="lm_head"):
        config_module.resolve(_write(tmp_path), {"lm_head.weight": (31, 768)})


def test_relative_buckets_follow_the_reference_layout():
    cfg = {"num_buckets": 320, "max_distance": 800}
    buckets = model.relative_buckets(np.array([0, -1, 1, -79, 79, -80, 80, -799, 800, 5000,
                                               -5000]), cfg)
    # Zero and the negative offsets use the lower half, positive offsets the
    # upper half; below 80 each distance is its own bucket, from 800 on every
    # distance shares the last one.
    assert buckets.tolist() == [0, 1, 161, 79, 239, 80, 240, 159, 319, 319, 159]
    offsets = np.arange(0, 3000)
    for sign in (1, -1):
        values = model.relative_buckets(sign * offsets, cfg)
        assert np.all(np.diff(values[1:]) >= 0)


def test_position_bias_table_clamps_at_the_last_bucket():
    cfg = {"num_buckets": 320, "max_distance": 800, "num_attention_heads": 2}
    embedding = np.arange(640, dtype=np.float32).reshape(320, 2)
    table = model.position_bias_table(
        {"wavlm.encoder.layers.0.attention.rel_attn_embed.weight": embedding}, cfg)
    assert table.shape == (1601, 2)
    # Offsets -800 and +800 already sit in the last bucket of their sign, so
    # clamping a longer offset to them in the graph changes nothing.
    np.testing.assert_array_equal(table[0], embedding[159])
    np.testing.assert_array_equal(table[-1], embedding[319])
    np.testing.assert_array_equal(table[800], embedding[0])
    with pytest.raises(ValueError, match="rel_attn_embed"):
        model.position_bias_table(
            {"wavlm.encoder.layers.0.attention.rel_attn_embed.weight": embedding[:, :1]}, cfg)


def test_gate_projection_sums_the_two_groups_of_four():
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(8, 6)).astype(np.float32)
    bias = rng.normal(size=8).astype(np.float32)
    rows = rng.normal(size=(5, 6)).astype(np.float32)
    folded_weight, folded_bias = model.gate_projection(
        {"x.attention.gru_rel_pos_linear.weight": weight,
         "x.attention.gru_rel_pos_linear.bias": bias}, "x.")
    expected = (rows @ weight.T + bias).reshape(5, 2, 4).sum(-1)
    np.testing.assert_allclose(rows @ folded_weight.T + folded_bias, expected, atol=1e-5)
    # Control: pairing the outputs the other way round (even/odd) is a different gate.
    interleaved = (rows @ weight.T + bias).reshape(5, 4, 2).sum(1)
    assert np.abs(interleaved - expected).max() > 1e-2


def test_weight_normed_kernel_reproduces_the_reference_formula():
    rng = np.random.default_rng(0)
    weight_v = rng.normal(size=(8, 4, 6)).astype(np.float32)
    weight_g = rng.normal(size=(1, 1, 6)).astype(np.float32)
    kernel = g.weight_normed_kernel(weight_g, weight_v)
    assert kernel.shape == weight_v.shape

    # Each kernel tap ends up with the length weight_g states for it. That is
    # the whole content of the reparameterisation, so it is checked directly.
    lengths = np.sqrt((kernel.astype(np.float64) ** 2).sum(axis=(0, 1)))
    assert np.abs(lengths - np.abs(weight_g).reshape(-1)).max() < 1e-5

    # Controls. Using weight_v directly, or taking the norm over the output and
    # tap axes instead, both give a kernel with the wrong tap lengths.
    raw = np.sqrt((weight_v.astype(np.float64) ** 2).sum(axis=(0, 1)))
    assert np.abs(raw - np.abs(weight_g).reshape(-1)).max() > 1e-2
    wrong_axis = weight_v * (weight_g.reshape(1, 1, -1) / np.sqrt(
        (weight_v.astype(np.float64) ** 2).sum(axis=(1, 2), keepdims=True)))
    assert np.abs(np.asarray(wrong_axis) - kernel).max() > 1e-2


def test_sample_range_is_a_real_range():
    assert config_module.MIN_SAMPLES < config_module.OPT_SAMPLES < config_module.MAX_SAMPLES
    # The profile has to cover a useful clip: 30 s at 16 kHz.
    assert config_module.MAX_SAMPLES >= 30 * 16000


def test_unsupported_precision_is_refused():
    with pytest.raises(NotImplementedError, match="fp32 or fp16, not bf16"):
        model.build_engine({}, {}, max_samples=config_module.MIN_SAMPLES, precision="bf16")


def _request(**overrides):
    fields = {
        "task": "speech_transcription", "backend": "trt", "dynamic_kv_cache": False,
        "max_batch_size": 1, "tensor_parallel_size": 1, "context_parallel_size": 1,
        "video_num_frames": None, "image_height": None, "image_width": None,
        "quantization": None, "fp32_layers": (), "mtp_seq_len": None, "precision": "fp32",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_e2e_oracle_is_exact_unless_the_case_states_a_bound():
    from families.wavlm.tests import test_e2e

    # Only the fp16 case carries a bound; both fp32 cases keep the exact gate.
    assert test_e2e._thresholds("wavlm-libri-clean-100h-large-fp16") == {
        "max_character_error_rate": 0.01}
    assert test_e2e._thresholds("wavlm-libri-clean-100h-large") == {}
    assert test_e2e._thresholds("wavlm-libri-clean-100h-base-plus") == {}

    reference = "the queen mother gave the french the most affectionate reception"
    one_edit = {"text": reference.replace("french", "frenh"), "token_ids": [1, 2]}
    with pytest.raises(AssertionError, match="transcript mismatch"):
        test_e2e._assert_parity(one_edit, reference, [1, 2], {})
    # The tokenizer strips its transcript; a space the runtime left at an edge
    # is a mismatch, not something the gate trims away.
    with pytest.raises(AssertionError, match="transcript mismatch"):
        test_e2e._assert_parity({"text": reference + " ", "token_ids": [1, 2]}, reference,
                                [1, 2], {})
    with pytest.raises(AssertionError, match="token id mismatch"):
        test_e2e._assert_parity({"text": reference, "token_ids": [1, 3]}, reference, [1, 2], {})
    # One edit in 64 characters is 0.0156: inside a 0.02 bound, outside 0.01.
    test_e2e._assert_parity(one_edit, reference, [1, 3], {"max_character_error_rate": 0.02})
    with pytest.raises(AssertionError, match="character error"):
        test_e2e._assert_parity(one_edit, reference, [1, 2], {"max_character_error_rate": 0.01})
    with pytest.raises(AssertionError, match="empty transcript"):
        test_e2e._assert_parity({"text": "", "token_ids": []}, " ", [], {})


@pytest.mark.parametrize("overrides, error, fragment", [
    ({"task": "audio_to_class_scores"}, ValueError, "speech_transcription"),
    ({"backend": "trt_rtx"}, ValueError, "backend=trt"),
    ({"max_batch_size": 2}, NotImplementedError, "max_batch_size"),
    ({"tensor_parallel_size": 2}, NotImplementedError, "tensor parallelism"),
    ({"quantization": "fp8"}, NotImplementedError, "quantization"),
    ({"image_height": 224}, NotImplementedError, "image size"),
    ({"precision": "bf16"}, NotImplementedError, "fp32 or fp16"),
])
def test_build_refuses_what_it_does_not_build(overrides, error, fragment):
    # Every refusal happens before the checkpoint is read or a writer is touched.
    with pytest.raises(error, match=fragment):
        model.build(_request(**overrides), None)


def test_build_names_a_checkpoint_without_a_ctc_head_before_reading_weights(tmp_path):
    # microsoft/wavlm-large as released: WavLMModel, a pickle and no
    # safetensors. The refusal names the missing head, not the missing file.
    pretrained = {**_LARGE, "architectures": ["WavLMModel"], "vocab_size": 32, "pad_token_id": 0}
    model_dir = _write(tmp_path, cfg=pretrained)
    (model_dir / "pytorch_model.bin").write_bytes(b"")
    request = _request(model_dir=model_dir, max_sequence_length=None, verbose=False)
    with pytest.raises(NotImplementedError, match="builds WavLMForCTC, not .*WavLMModel"):
        model.build(request, None)
