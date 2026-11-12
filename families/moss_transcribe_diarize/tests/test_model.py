# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint contract, weight reading and build-request policy, without a GPU or checkpoint."""

from __future__ import annotations

import copy
import json
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt")

from families.moss_transcribe_diarize.checkpoint import Checkpoint  # noqa: E402
from families.moss_transcribe_diarize.config import load_config  # noqa: E402
from families.moss_transcribe_diarize.graph import float32_to_bfloat16_bits  # noqa: E402
from families.moss_transcribe_diarize.model import build  # noqa: E402
from families.moss_transcribe_diarize.tokens import token_bytes  # noqa: E402
from tensorrt_model_connect.build import BuildRequest  # noqa: E402

CONFIG = {
    "architectures": ["MossTranscribeDiarizeForConditionalGeneration"],
    "model_type": "moss_transcribe_diarize",
    "text_config": {
        "model_type": "qwen3", "vocab_size": 151936, "hidden_size": 1024,
        "intermediate_size": 3072, "num_hidden_layers": 2, "num_attention_heads": 16,
        "num_key_value_heads": 8, "head_dim": 128, "hidden_act": "silu",
        "rms_norm_eps": 1e-06, "tie_word_embeddings": True, "rope_theta": 1000000,
        "attention_bias": False, "layer_types": ["full_attention", "full_attention"],
    },
    "audio_config": {
        "model_type": "whisper", "num_mel_bins": 80, "d_model": 1024, "encoder_layers": 2,
        "encoder_attention_heads": 16, "encoder_ffn_dim": 4096, "max_source_positions": 1500,
        "activation_function": "gelu", "scale_embedding": False,
    },
    "audio_token_id": 151671, "audio_merge_size": 4, "adaptor_input_dim": 4096,
    "tie_word_embeddings": True,
}
PREPROCESSOR = {
    "chunk_length": 30, "dither": 0.0, "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 80, "hop_length": 160, "n_fft": 400, "n_samples": 480000,
    "nb_max_frames": 3000, "padding_value": 0.0, "sampling_rate": 16000,
}
PROCESSOR = {
    "processor_class": "MossTranscribeDiarizeProcessor", "audio_tokens_per_second": 12.5,
    "audio_merge_size": 4, "time_marker_every_seconds": 5, "enable_time_marker": True,
}
GENERATION = {"bos_token_id": 151643, "eos_token_id": 151645, "pad_token_id": 151643,
              "max_new_tokens": 5120}


def _write(root: Path, config=CONFIG, preprocessor=PREPROCESSOR, processor=PROCESSOR,
           generation=GENERATION) -> Path:
    for name, value in (("config.json", config), ("preprocessor_config.json", preprocessor),
                        ("processor_config.json", processor),
                        ("generation_config.json", generation)):
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    return root


def test_reads_the_checkpoint_contract(tmp_path):
    config = load_config(_write(tmp_path))
    assert config.tokens_per_chunk == 375
    assert config.samples_per_token == 1280
    assert config.markers.every_seconds == 5
    assert config.eos_token_ids == (151645,)
    assert config.default_max_new_tokens == 5120
    assert config.audio.head_dim == 64


def _variant(path: tuple[str, ...], value):
    config = copy.deepcopy(CONFIG)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return config


@pytest.mark.parametrize("config", [
    _variant(("model_type",), "qwen3"),
    _variant(("architectures",), ["MossTranscribeDiarizeModel"]),
    _variant(("text_config", "rope_scaling"), {"rope_type": "yarn", "factor": 4.0}),
    _variant(("text_config", "use_sliding_window"), True),
    _variant(("text_config", "layer_types"), ["full_attention", "sliding_attention"]),
    _variant(("text_config", "attention_bias"), True),
    _variant(("tie_word_embeddings",), False),
    _variant(("audio_config", "activation_function"), "relu"),
    _variant(("audio_config", "max_source_positions"), 1498),
    _variant(("adaptor_input_dim",), 2048),
])
def test_rejects_unsupported_architecture_variants(tmp_path, config):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, config=config))


@pytest.mark.parametrize("key, value", [
    ("feature_size", 128), ("dither", 1e-5), ("hop_length", 320), ("n_samples", 160000),
])
def test_rejects_other_front_ends(tmp_path, key, value):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, preprocessor={**PREPROCESSOR, key: value}))


@pytest.mark.parametrize("key, value", [
    ("audio_tokens_per_second", 25.0), ("audio_merge_size", 2), ("enable_time_marker", "yes"),
])
def test_rejects_other_placeholder_expansion(tmp_path, key, value):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, processor={**PROCESSOR, key: value}))


@pytest.mark.parametrize("extra", [
    {"do_sample": True}, {"num_beams": 4}, {"repetition_penalty": 1.1}, {"eos_token_id": None},
])
def test_greedy_generation_only(tmp_path, extra):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, generation={**GENERATION, **extra}))


def _safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header, offset, payload = {}, 0, b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        payload += data
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_bf16_tensors_keep_their_stored_bits(tmp_path):
    values = np.array([1.0, -2.5, 3.140625, 1e-30], dtype=np.float32)
    bits = float32_to_bfloat16_bits(values)
    _safetensors(tmp_path / "model.safetensors", {
        "w": ("BF16", (2, 2), bits.tobytes()),
        "f": ("F32", (4,), values.tobytes()),
    })
    checkpoint = Checkpoint(tmp_path)
    assert checkpoint.names() == {"w", "f"}
    assert checkpoint.shape("w") == (2, 2)
    np.testing.assert_array_equal(checkpoint.bfloat16_bits("w").reshape(-1), bits)
    widened = checkpoint.float32("w").reshape(-1)
    np.testing.assert_array_equal(widened.view(np.uint32) >> 16, bits.astype(np.uint32))
    np.testing.assert_array_equal(checkpoint.float32("f"), values)
    with pytest.raises(ValueError):
        checkpoint.bfloat16_bits("f")
    with pytest.raises(ValueError):
        checkpoint.float32("missing")


def test_bfloat16_rounding_is_nearest_even():
    # 1 + 2^-8 lies halfway between two bfloat16 values; ties go to the even one.
    halfway = np.array([1.0 + 2.0 ** -8, 1.0 + 3 * 2.0 ** -8], dtype=np.float32)
    assert float32_to_bfloat16_bits(halfway).tolist() == [0x3F80, 0x3F82]


def test_byte_level_tokens_decode_to_their_bytes():
    assert token_bytes("Ġhello") == b" hello"
    assert token_bytes("Ċ") == b"\n"
    assert token_bytes("<|im_end|>") == b"<|im_end|>"
    # A token with characters outside the byte alphabet keeps its own UTF-8.
    assert token_bytes("你") == "你".encode()


def _request(**overrides) -> BuildRequest:
    fields = {"model_dir": Path("/nonexistent"), "output_path": Path("/nonexistent/out.bundle"),
              "family": "moss_transcribe_diarize", "task": "speech_transcription",
              "precision": "bf16"}
    fields.update(overrides)
    return BuildRequest(**fields)


@pytest.mark.parametrize("overrides, error", [
    ({"task": "transcription"}, ValueError),
    ({"backend": "trt_rtx"}, ValueError),
    ({"precision": "fp16"}, NotImplementedError),
    ({"max_batch_size": 2}, NotImplementedError),
    ({"tensor_parallel_size": 2}, NotImplementedError),
    ({"context_parallel_size": 2}, NotImplementedError),
    ({"quantization": "fp8"}, NotImplementedError),
    ({"fp32_layers": (0,)}, NotImplementedError),
    ({"dynamic_kv_cache": True}, NotImplementedError),
    ({"image_height": 224}, NotImplementedError),
    ({"video_num_frames": 8}, NotImplementedError),
    ({"mtp_seq_len": 2}, NotImplementedError),
])
def test_build_options_are_consumed_or_refused(overrides, error):
    with pytest.raises(error):
        build(_request(**overrides), writer=None)
