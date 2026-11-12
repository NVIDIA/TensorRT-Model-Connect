# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the vibevoice builder: config admission, request handling, schedules."""

from __future__ import annotations

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
