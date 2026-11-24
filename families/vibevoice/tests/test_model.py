# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the vibevoice builder: config admission, request handling, schedules."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ml_dtypes")
pytest.importorskip("safetensors")
pytest.importorskip("tensorrt")

from families.vibevoice import config as config_module  # noqa: E402
from families.vibevoice import model  # noqa: E402
from families.vibevoice.codec_engines import stream_state_layout, stream_state_size  # noqa: E402
from families.vibevoice.tests import reference  # noqa: E402
from families.vibevoice.weights import Checkpoint, expected_tensors  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    return reference.write_tiny_checkpoint(tmp_path_factory.mktemp("vibevoice_tiny"), seed=3)


def _request(model_dir: Path, **overrides):
    values = dict(
        model_dir=model_dir,
        output_path=model_dir / "out.bundle",
        family="vibevoice",
        task="text_to_speech",
        precision="bf16",
        backend="trt",
        max_sequence_length=None,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        quantization=None,
        fp32_layers=(),
        dynamic_kv_cache=False,
        verbose=False,
        graph_transform=None,
        mtp_seq_len=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _NoWriter:
    def __getattr__(self, name):
        raise AssertionError(f"build wrote {name} before refusing the request")


def test_tiny_checkpoint_config_is_read(tiny):
    cfg = config_module.load(tiny)
    assert (cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) == (
        64,
        2,
        4,
        2,
        16,
    )
    assert cfg.acoustic.ratios == (2, 2, 4) and cfg.acoustic.hop_length == 16
    assert cfg.acoustic.depths == (1, 1, 1, 2) and cfg.semantic.vae_dim == 6
    assert cfg.acoustic_vae_std == pytest.approx(0.625)
    assert cfg.head_intermediate_size == 192


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("acoustic_tokenizer_config", "causal", False),
        ("acoustic_tokenizer_config", "mixer_layer", "conv"),
        ("semantic_tokenizer_config", "layernorm", "LN"),
        ("acoustic_tokenizer_config", "decoder_ratios", [2, 2, 4]),
        ("decoder_config", "rope_scaling", {"type": "yarn", "factor": 4.0}),
        ("decoder_config", "tie_word_embeddings", False),
        ("decoder_config", "use_sliding_window", True),
        ("diffusion_head_config", "prediction_type", "epsilon"),
        ("diffusion_head_config", "ddpm_beta_schedule", "linear"),
    ],
)
def test_unimplemented_settings_are_refused_by_name(tiny, tmp_path, section, key, value):
    copy = tmp_path / "ckpt"
    copy.mkdir()
    for name in ("preprocessor_config.json",):
        (copy / name).write_text((tiny / name).read_text())
    config = json.loads((tiny / "config.json").read_text())
    config[section][key] = value
    (copy / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=key):
        config_module.load(copy)


def test_another_tokenizer_source_is_refused(tiny, tmp_path):
    copy = tmp_path / "ckpt"
    copy.mkdir()
    (copy / "config.json").write_text((tiny / "config.json").read_text())
    processor = json.loads((tiny / "preprocessor_config.json").read_text())
    processor["language_model_pretrained_name"] = "Qwen/Qwen2.5-7B"
    (copy / "preprocessor_config.json").write_text(json.dumps(processor))
    with pytest.raises(ValueError, match="tokenizer source"):
        config_module.load(copy)


@pytest.mark.parametrize(
    "overrides",
    [
        {"task": "text_to_audio"},
        {"backend": "trt_rtx"},
        {"max_batch_size": 2},
        {"tensor_parallel_size": 2},
        {"context_parallel_size": 2},
        {"dynamic_kv_cache": True},
        {"image_height": 224},
        {"video_num_frames": 8},
        {"quantization": "fp8"},
        {"fp32_layers": (1,)},
        {"mtp_seq_len": 2},
        {"precision": "fp16"},
        {"max_sequence_length": 8},
    ],
)
def test_build_refuses_options_it_cannot_honour(tiny, overrides):
    with pytest.raises((ValueError, NotImplementedError)):
        model.build(_request(tiny, **overrides), _NoWriter())


def test_sequence_length_must_be_an_integer():
    for value in (64.9, 4096.0, True, "4096"):
        with pytest.raises(ValueError, match="must be an integer"):
            model.sequence_geometry(value, 65536)
    assert model.sequence_geometry(np.int64(4096), 65536) == (4096, model.PREFILL_CHUNK)


def test_short_sequences_shrink_the_prefill_chunk():
    assert model.sequence_geometry(None, 65536) == (model.DEFAULT_CAPACITY, model.PREFILL_CHUNK)
    assert model.sequence_geometry(4096, 65536) == (4096, model.PREFILL_CHUNK)
    assert model.sequence_geometry(model.MIN_CAPACITY, 65536) == (
        model.MIN_CAPACITY,
        model.MIN_CAPACITY,
    )
    assert model.sequence_geometry(300, 65536) == (300, 300)
    for capacity in (model.MIN_CAPACITY - 1, 65537):
        with pytest.raises(ValueError, match="max_sequence_length"):
            model.sequence_geometry(capacity, 65536)


def test_rope_frequencies_follow_the_float32_formula():
    cfg = SimpleNamespace(head_dim=128, rope_theta=1_000_000.0)
    inverse = model.rope_inverse_frequencies(cfg)
    assert inverse.dtype == np.float32 and inverse.shape == (64,)
    assert inverse[0] == 1.0
    assert inverse[-1] == pytest.approx(1_000_000.0 ** (-126 / 128), rel=1e-6)


def test_train_sigmas_match_diffusers():
    diffusers = pytest.importorskip("diffusers")
    scheduler = diffusers.DPMSolverMultistepScheduler(
        beta_schedule="squaredcos_cap_v2", prediction_type="v_prediction"
    )
    np.testing.assert_array_equal(
        np.cumprod((1.0 - scheduler.betas.numpy()).astype(np.float64)).astype(np.float32),
        scheduler.alphas_cumprod.numpy(),
    )
    expected = (((1 - scheduler.alphas_cumprod) / scheduler.alphas_cumprod) ** 0.5).numpy()
    # torch's float32 CPU square root is not always correctly rounded (one ulp here and there).
    np.testing.assert_allclose(model.train_sigmas(1000), expected, rtol=2.5e-7, atol=0)


def test_stream_state_layout_of_the_released_geometry():
    tokenizer = config_module.SpeechTokenizerConfig(
        channels=1,
        vae_dim=64,
        num_filters=32,
        ratios=(2, 2, 4, 5, 5, 8),
        depths=(3, 3, 3, 3, 3, 3, 8),
        eps=1e-5,
    )
    semantic = config_module.SpeechTokenizerConfig(
        channels=1,
        vae_dim=128,
        num_filters=32,
        ratios=(2, 2, 4, 5, 5, 8),
        depths=(3, 3, 3, 3, 3, 3, 8),
        eps=1e-5,
    )
    cfg = SimpleNamespace(acoustic=tokenizer, semantic=semantic)
    layout = stream_state_layout(cfg)
    # decoder: stem conv, 8 stem blocks, 6 x (transposed conv + 3 blocks), head;
    # encoder: stem conv, 3 stem blocks, 6 x (strided conv + stage blocks), head.
    assert len(layout) == (1 + 8 + 6 * 4 + 1) + (1 + 3 + 6 + 5 * 3 + 8 + 1)
    assert layout[0] == (64, 6) and layout[9] == (2048, 1)
    assert stream_state_size(cfg) == sum(c * f for c, f in layout)
    assert math.prod(tokenizer.ratios) == 3200


def _edited(tiny: Path, tmp_path: Path, *, config=None, processor=None, text=None) -> Path:
    """A copy of the tiny checkpoint (weights linked) with edited JSON files."""
    copy_dir = tmp_path / "ckpt"
    copy_dir.mkdir()
    for path in tiny.iterdir():
        if path.name not in ("config.json", "preprocessor_config.json"):
            (copy_dir / path.name).symlink_to(path)
    raw = json.loads((tiny / "config.json").read_text())
    audio = json.loads((tiny / "preprocessor_config.json").read_text())
    if config is not None:
        config(raw)
    if processor is not None:
        processor(audio)
    written = json.dumps(raw)
    (copy_dir / "config.json").write_text(text(written) if text is not None else written)
    (copy_dir / "preprocessor_config.json").write_text(json.dumps(audio))
    return copy_dir


def _set(section, key, value):
    def edit(raw):
        (raw[section] if section else raw)[key] = value

    return edit


def _widen(section, key):
    def edit(raw):
        raw[section][key] += 32

    return edit


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (_set("decoder_config", "hidden_size", 64.0), "hidden_size=64.0 is not an integer"),
        (_set("decoder_config", "num_hidden_layers", True), "num_hidden_layers=True is not"),
        (_set("decoder_config", "tie_word_embeddings", "true"), "tie_word_embeddings"),
        (_set("decoder_config", "rope_theta", 1e39), "rope_theta=1e\\+39 is not a finite float32"),
        (_set("decoder_config", "rope_theta", 0.5), "rope_theta=0.5 must exceed 1"),
        (_set("acoustic_tokenizer_config", "causal", 1), "causal=1 is not implemented"),
        (_set("acoustic_tokenizer_config", "encoder_depths", "01-1-1-2"), "encoder_depths"),
        (_set("acoustic_tokenizer_config", "encoder_ratios", [4, 2.0, 2]), "encoder_ratios"),
        (_set("diffusion_head_config", "head_ffn_ratio", 3.01), "no integer FFN width"),
        (_set("diffusion_head_config", "ddpm_num_steps", 10), "ddpm_num_steps=10 lies outside"),
        (_set("decoder_config", "rms_norm_eps", 1e-100), "rms_norm_eps=1e-100 must be a positive"),
        (_set("diffusion_head_config", "rms_norm_eps", 1e-40), "rms_norm_eps=1e-40 must be a positive"),
        (_set("acoustic_tokenizer_config", "fix_std", 3e38), "noise scale 3.75e\\+38, above float32"),
    ],
)
def test_configuration_values_are_read_without_coercion(tiny, tmp_path, edit, match):
    with pytest.raises(ValueError, match=match):
        config_module.load(_edited(tiny, tmp_path, config=edit))


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (_set("audio_processor", "target_dB_FS", 4000), "no finite nonzero float32 loudness gain"),
        (_set("audio_processor", "eps", "1e-6"), "eps='1e-6' is not a number"),
        (_set("audio_processor", "normalize_audio", 1), "loudness normalization"),
    ],
)
def test_voice_preparation_settings_are_checked(tiny, tmp_path, edit, match):
    with pytest.raises(ValueError, match=match):
        config_module.load(_edited(tiny, tmp_path, processor=edit))


def test_repeated_keys_and_nonnumbers_are_refused(tiny, tmp_path):
    repeated = _edited(tiny, tmp_path, text=lambda text: '{"model_type": "vibevoice", ' + text[1:])
    with pytest.raises(ValueError, match="repeats the key 'model_type'"):
        config_module.load(repeated)
    raw = json.loads((tiny / "config.json").read_text())
    eps = json.dumps(raw["decoder_config"]["rms_norm_eps"])
    nan = tmp_path / "nan"
    nan.mkdir()
    nan_dir = _edited(
        tiny,
        nan,
        text=lambda text: text.replace(f'"rms_norm_eps": {eps}', '"rms_norm_eps": NaN', 1),
    )
    with pytest.raises(ValueError, match="non-number NaN"):
        config_module.load(nan_dir)


def test_usable_gain_mirrors_the_runtime_check():
    assert config_module.usable_gain(-25.0, 1e-6)
    assert not config_module.usable_gain(4000.0, 1e-6)
    assert not config_module.usable_gain(-25.0, 1e-45)
    assert not config_module.usable_gain(-1e6, 1e-6)


def test_expected_tensors_match_the_original_layout_listing(tiny):
    raw = json.loads((tiny / "config.json").read_text())
    assert expected_tensors(config_module.load(tiny)) == reference.original_tensor_shapes(raw)
    Checkpoint(tiny, config_module.load(tiny))


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (_widen("decoder_config", "intermediate_size"), "mlp.gate_proj.weight has shape"),
        (_set("decoder_config", "num_hidden_layers", 1), "its configuration implies"),
        (_set("diffusion_head_config", "head_layers", 3), "its configuration implies"),
        # Counted, not listed: a million blocks per stage would be listed for seconds.
        (_set("acoustic_tokenizer_config", "encoder_depths", "100000-1-1-2"), "implies 2000"),
    ],
)
def test_checkpoint_headers_must_match_the_configuration(tiny, tmp_path, edit, match):
    copy_dir = _edited(tiny, tmp_path, config=edit)
    with pytest.raises(ValueError, match=match):
        Checkpoint(copy_dir, config_module.load(copy_dir))


def test_tensor_count_is_computed_like_the_listing(tiny):
    from families.vibevoice.weights import implied_tensor_count

    cfg = config_module.load(tiny)
    assert implied_tensor_count(cfg) == len(expected_tensors(cfg))


def _rewritten(tiny: Path, tmp_path: Path, edit) -> Path:
    """A copy of the tiny checkpoint whose tensors ``edit`` changes."""
    from safetensors.numpy import load_file, save_file

    tensors = load_file(str(tiny / "model.safetensors"))
    edit(tensors)
    copy_dir = tmp_path / "rewritten"
    copy_dir.mkdir()
    for name in ("config.json", "preprocessor_config.json"):
        (copy_dir / name).write_text((tiny / name).read_text())
    index = {"metadata": {}, "weight_map": {name: "model.safetensors" for name in tensors}}
    (copy_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    save_file(tensors, str(copy_dir / "model.safetensors"))
    return copy_dir


def test_checkpoint_tensor_names_must_match_the_configuration(tiny, tmp_path):
    def rename(tensors):
        tensors["model.language_model.final_norm.weight"] = tensors.pop(
            "model.language_model.norm.weight"
        )

    copy_dir = _rewritten(tiny, tmp_path, rename)
    with pytest.raises(ValueError, match="unexpected \\['model.language_model.final_norm"):
        Checkpoint(copy_dir, config_module.load(copy_dir))


def test_checkpoint_tensors_must_be_bfloat16(tiny, tmp_path):
    from safetensors.numpy import load_file, save_file

    tensors = load_file(str(tiny / "model.safetensors"))
    tensors["model.language_model.norm.weight"] = tensors[
        "model.language_model.norm.weight"
    ].astype(np.float32)
    copy_dir = tmp_path / "f32"
    copy_dir.mkdir()
    for name in ("config.json", "preprocessor_config.json", "model.safetensors.index.json"):
        (copy_dir / name).write_text((tiny / name).read_text())
    save_file(tensors, str(copy_dir / "model.safetensors"))
    with pytest.raises(ValueError, match="norm.weight is F32, not bfloat16"):
        Checkpoint(copy_dir, config_module.load(copy_dir))


_SPECIALS = ("<|endoftext|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>")


def _qwen_tokenizer(merges=()) -> dict:
    """A byte-level BPE in Qwen2.5's layout: one token per byte, the four control tokens."""
    direct = [*range(ord("!"), ord("~") + 1), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
    vocab, extra = {}, 0
    for byte in range(256):
        vocab[chr(byte) if byte in direct else chr(256 + extra)] = byte
        extra += byte not in direct
    added = []
    for content in _SPECIALS:
        vocab[content] = len(vocab)
        added.append(
            {
                "id": vocab[content],
                "content": content,
                "special": True,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            }
        )
    for left, right in merges:
        vocab[left + right] = len(vocab)
    byte_level = {"add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    return {
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
                {"type": "ByteLevel", **byte_level},
            ],
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel", **byte_level},
        "model": {"type": "BPE", "vocab": vocab, "merges": [list(pair) for pair in merges]},
    }


def _bytes(document: dict) -> bytes:
    return json.dumps(document).encode()


def _with_merges(document: dict, merges: list) -> dict:
    changed = copy.deepcopy(document)
    changed["model"]["merges"] = merges
    return changed


def test_the_build_refuses_a_tokenizer_the_runtime_would_refuse():
    good = _qwen_tokenizer([("a", "b")])
    model.check_tokenizer_layout(_bytes(good))
    own_regex = copy.deepcopy(good)
    own_regex["pre_tokenizer"]["pretokenizers"][1]["use_regex"] = True
    lowercase = copy.deepcopy(good)
    lowercase["normalizer"] = {"type": "Lowercase"}
    worded = copy.deepcopy(good)
    worded["added_tokens"][0]["special"] = "false"
    for document, match in (
        (own_regex, "pre_tokenizer is not supported"),
        (lowercase, "normalizer is not supported"),
        (worded, "special is not supported"),
        (_with_merges(good, [["a", "zz"]]), "merge is not supported"),
    ):
        with pytest.raises(ValueError, match=match):
            model.check_tokenizer_layout(_bytes(document))
    with pytest.raises(ValueError, match="repeats the key"):
        model.check_tokenizer_layout(b'{"model": {}, ' + _bytes(good)[1:])
    with pytest.raises(ValueError, match="add_prefix_space is not supported"):
        model.check_tokenizer_config(b'{"add_prefix_space": true}')


def test_tokenizer_ids_index_the_embedding_table():
    good = _qwen_tokenizer()
    rows = len(good["model"]["vocab"])
    model.check_tokenizer_ids(_bytes(good), rows)
    with pytest.raises(ValueError, match="outside the embedding table"):
        model.check_tokenizer_ids(_bytes(good), rows - 1)
    shared = copy.deepcopy(good)
    shared["model"]["vocab"]["UNUSED"] = 97
    with pytest.raises(ValueError, match="used by more than one token"):
        model.check_tokenizer_ids(_bytes(shared), rows)
    # A new added token takes the next id after the vocabulary entries, as the reference
    # assigns it; the runtime refuses any other.
    skipped = copy.deepcopy(good)
    skipped["added_tokens"].append({**good["added_tokens"][0], "content": "<|new|>", "id": rows + 1})
    with pytest.raises(ValueError, match="but the reference assigns it 260"):
        model.check_tokenizer_ids(_bytes(skipped), rows + 2)
    following = copy.deepcopy(good)
    following["added_tokens"].append({**good["added_tokens"][0], "content": "<|new|>", "id": rows})
    model.check_tokenizer_ids(_bytes(following), rows + 1)
    special = model._special_token_ids(good)
    assert sorted(special.values()) == [rows - 4, rows - 3, rows - 2, rows - 1]
    twice = copy.deepcopy(good)
    twice["added_tokens"][1]["id"] = twice["added_tokens"][0]["id"]
    with pytest.raises(ValueError, match="share an id"):
        model._special_token_ids(twice)


def test_runtime_contract_mirrors_the_load_checks(tiny):
    cfg = config_module.load(tiny)
    special = {"eos": 151643, "audio_bos": 151652, "audio_eos": 151653, "audio_token": 151654}
    runtime = model.runtime_contract(
        cfg, precision="bf16", capacity=256, prefill_chunk=64, special=special
    )
    assert runtime["vocab_size"] == cfg.vocab_size
    assert runtime["control_token_ids"] == sorted(special.values())
    assert runtime["max_voice_samples"] % runtime["hop_length"] == 0
    # The voice encoder stores the noise scale in the engine's type: 3.4e38 is a float32,
    # but bfloat16 rounds it to infinity.
    import dataclasses

    wide = dataclasses.replace(cfg, acoustic_vae_std=3.4e38)
    model.runtime_contract(wide, precision="fp32", capacity=256, prefill_chunk=64, special=special)
    with pytest.raises(ValueError, match="not a normal bf16 number"):
        model.runtime_contract(
            wide, precision="bf16", capacity=256, prefill_chunk=64, special=special
        )
    sigmas = model.train_sigmas(1000)
    # linspace(0, 999, 1001) holds 499.5 and 500.499, which both round to timestep 500.
    with pytest.raises(ValueError, match="repeats a noise level"):
        model._check_schedule(sigmas, 1000)
    model._check_schedule(sigmas, 999)
