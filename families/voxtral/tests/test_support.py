# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU identity, task, configuration and checkpoint-reader checks for Voxtral (NumPy only)."""

import copy
import json
import struct

import numpy as np
import pytest

from families.voxtral.checkpoint import Checkpoint, engine_array, float32_to_bfloat16_bits
from families.voxtral.config import (
    expected_tensors,
    load_json,
    max_audio_chunks,
    parse_config,
    validate_preprocessor,
)
from families.voxtral.support import describe
from tensorrt_model_connect.model_support import ModelMetadata

# config.json of mistralai/Voxtral-Mini-3B-2507 at revision 3060fe34 (training-only keys dropped).
CONFIG = {
    "architectures": ["VoxtralForConditionalGeneration"],
    "model_type": "voxtral",
    "audio_token_id": 24,
    "hidden_size": 3072,
    "projector_hidden_act": "gelu",
    "vocab_size": 131072,
    "text_config": {
        "model_type": "llama", "hidden_size": 3072, "intermediate_size": 8192,
        "num_hidden_layers": 30, "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 128, "vocab_size": 131072, "rms_norm_eps": 1e-05, "rope_theta": 100000000.0,
        "rope_scaling": None, "max_position_embeddings": 131072, "hidden_act": "silu",
        "attention_bias": False, "mlp_bias": False, "sliding_window": None, "pretraining_tp": 1,
    },
    "audio_config": {
        "model_type": "voxtral_encoder", "hidden_size": 1280, "intermediate_size": 5120,
        "num_hidden_layers": 32, "num_attention_heads": 20, "num_key_value_heads": 20,
        "head_dim": 64, "num_mel_bins": 128, "max_source_positions": 1500,
        "activation_function": "gelu", "scale_embedding": False,
    },
}
PREPROCESSOR = {
    "chunk_length": 30, "dither": 0.0, "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128, "hop_length": 160, "n_fft": 400, "n_samples": 480000,
    "nb_max_frames": 3000, "padding_side": "right", "padding_value": 0.0,
    "processor_class": "VoxtralProcessor", "return_attention_mask": False, "sampling_rate": 16000,
}


def _support(config):
    return describe(ModelMetadata(config=config, model_index={}))


def test_speech_transcription_is_offered():
    support = describe(ModelMetadata(config={"model_type": "voxtral"}, model_index={}))
    assert support is not None
    assert support.tasks == ("speech_transcription",)
    assert support.default_task == "speech_transcription"
    assert support.default_precision == "fp16"


def test_architecture_also_resolves():
    support = _support({"architectures": ["VoxtralForConditionalGeneration"]})
    assert support is not None and support.default_task == "speech_transcription"


def test_neighbours_are_not_claimed():
    # The streaming Voxtral Realtime model and the plain Whisper / Llama
    # checkpoints have other owners or none; this family must not claim them.
    for model_type in ("voxtral_realtime", "voxtral_encoder", "whisper", "llama", "mistral"):
        assert _support({"model_type": model_type}) is None
    assert _support({"architectures": ["VoxtralRealtimeForConditionalGeneration"]}) is None


def test_checkpoint_geometry():
    config = parse_config(copy.deepcopy(CONFIG))
    assert config.frames_per_token == 4
    assert config.audio_tokens_per_chunk == 375
    assert config.mel_frames_per_chunk == 3000
    assert config.text.num_attention_heads * config.text.head_dim == 4096
    validate_preprocessor(PREPROCESSOR, config)
    # 4096 positions minus a 3 + 5 token frame leave room for ten 30 s chunks.
    assert max_audio_chunks(4096, config, 8) == 10
    with pytest.raises(ValueError, match="cannot hold one 30 s audio chunk"):
        max_audio_chunks(383, config, 8)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model_type",), "llava", "model_type"),
        (("architectures",), ["VoxtralModel"], "architectures"),
        (("text_config", "model_type"), "mistral", "text_config.model_type"),
        (("text_config", "rope_scaling"), {"rope_type": "yarn", "factor": 4.0}, "rope_scaling"),
        (("text_config", "sliding_window"), 4096, "sliding-window"),
        (("text_config", "attention_bias"), True, "biases"),
        (("text_config", "tie_word_embeddings"), True, "tied output embeddings"),
        (("text_config", "num_key_value_heads"), 7, "multiple of key/value heads"),
        (("audio_config", "activation_function"), "relu", "gelu"),
        (("audio_config", "intermediate_size"), 5000, "whole encoder frames"),
        (("projector_hidden_act", ), "relu", "projector"),
        (("audio_token_id",), 131072, "audio_token_id"),
        # Positive finite numbers that overflow or underflow as the float32 graph constant.
        (("text_config", "rms_norm_eps"), 1e100, "float32"),
        (("text_config", "rms_norm_eps"), 1e-50, "float32"),
        # Present options must have the supported value and its JSON type: null, 0 or 1.0 are not false or 1.
        (("text_config", "attention_bias"), None, "biases"),
        (("text_config", "mlp_bias"), "false", "biases"),
        (("text_config", "tie_word_embeddings"), None, "tied output embeddings"),
        (("text_config", "pretraining_tp"), 1.0, "pretraining_tp"),
        (("text_config", "rope_parameters"), None, "rope_parameters"),
        (("audio_config", "scale_embedding"), 0, "scale_embedding"),
        (("audio_config", "num_key_value_heads"), 20.0, "multi-head attention"),
        (("vocab_size",), 131072.0, "vocab sizes differ"),
        (("hidden_size",), 3072.0, "hidden sizes differ"),
        # Sizes beyond the runtime's load limits are refused before any loop or allocation.
        (("text_config", "head_dim"), 2**40, "head_dim"),
        (("text_config", "hidden_size"), 2**21, "hidden_size"),
        (("text_config", "num_hidden_layers"), 4097, "num_hidden_layers"),
        (("text_config", "vocab_size"), 2**31, "vocab_size"),
        (("text_config", "max_position_embeddings"), 2**31, "max_position_embeddings"),
        (("audio_config", "num_mel_bins"), 2**21, "num_mel_bins"),
        (("text_config", "num_attention_heads"), 8200, "attention width"),
    ],
)
def test_unsupported_variants_are_refused(path, value, message):
    config = copy.deepcopy(CONFIG)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        parse_config(config)


@pytest.mark.parametrize(("key", "value"), [("n_fft", 512), ("sampling_rate", 24000),
                                            ("feature_size", 80), ("dither", 0.1),
                                            ("padding_value", 1.0), ("padding_side", "left"),
                                            # Equal values of another JSON type are refused too.
                                            ("n_fft", 400.0), ("feature_size", 128.0), ("dither", "0"),
                                            ("padding_value", "0"), ("do_normalize", 0)])
def test_other_front_ends_are_refused(key, value):
    config = parse_config(copy.deepcopy(CONFIG))
    preprocessor = dict(PREPROCESSOR, **{key: value})
    with pytest.raises(ValueError):
        validate_preprocessor(preprocessor, config)


@pytest.mark.parametrize("key", ["quantization", "quantization_config"])
def test_quantized_exports_are_refused(key):
    # An MLX or other quantized export declares its scheme in config.json.
    config = copy.deepcopy(CONFIG)
    config[key] = {"group_size": 64, "bits": 4}
    with pytest.raises(ValueError, match="quantized checkpoints are not supported"):
        parse_config(config)


def test_expected_tensors_follow_the_checkpoint_layout():
    shapes = expected_tensors(parse_config(copy.deepcopy(CONFIG)))
    # The two shards of the original checkpoint hold 762 tensors.
    assert len(shapes) == 762
    assert shapes["audio_tower.conv1.weight"] == (1280, 128, 3)
    assert shapes["audio_tower.conv2.weight"] == (1280, 1280, 3)
    assert shapes["audio_tower.embed_positions.weight"] == (1500, 1280)
    assert shapes["multi_modal_projector.linear_1.weight"] == (3072, 5120)
    assert shapes["language_model.model.layers.29.self_attn.k_proj.weight"] == (1024, 3072)
    assert shapes["language_model.model.layers.0.self_attn.o_proj.weight"] == (3072, 4096)
    assert shapes["language_model.lm_head.weight"] == (131072, 3072)
    # Whisper's key projection has no bias; the projector has none at all.
    assert "audio_tower.layers.0.self_attn.k_proj.bias" not in shapes
    assert "audio_tower.layers.31.self_attn.q_proj.bias" in shapes
    assert not any(name.startswith("multi_modal_projector") and name.endswith(".bias") for name in shapes)


# A configuration small enough to write its whole checkpoint in a test.
TINY = {
    "architectures": ["VoxtralForConditionalGeneration"],
    "model_type": "voxtral",
    "audio_token_id": 5,
    "text_config": {
        "model_type": "llama", "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 1,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4, "vocab_size": 32,
        "rms_norm_eps": 1e-05, "rope_theta": 10000.0, "max_position_embeddings": 64,
    },
    "audio_config": {
        "model_type": "voxtral_encoder", "hidden_size": 8, "intermediate_size": 32,
        "num_hidden_layers": 1, "num_attention_heads": 2, "head_dim": 4, "num_mel_bins": 4,
        "max_source_positions": 8,
    },
}


def _write_safetensors(path, tensors, *, dtypes=None, metadata=None):
    """Write float32 arrays as a safetensors file (BF16 unless ``dtypes`` names F32, F16 or U32)."""
    dtypes = dtypes or {}
    header, blobs, offset = {}, [], 0
    for name, values in tensors.items():
        dtype = dtypes.get(name, "BF16")
        if dtype == "BF16":
            blob = float32_to_bfloat16_bits(values).tobytes()
        else:
            blob = np.ascontiguousarray(values, dtype={"F32": "<f4", "F16": "<f2", "U32": "<u4"}[dtype]).tobytes()
        header[name] = {"dtype": dtype, "shape": list(values.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    header["__metadata__"] = metadata if metadata is not None else {"format": "pt"}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def _tiny_tensors():
    generator = np.random.default_rng(7)
    shapes = expected_tensors(parse_config(copy.deepcopy(TINY)))
    return shapes, {name: generator.standard_normal(shape).astype(np.float32) for name, shape in shapes.items()}


def test_the_reader_returns_the_stored_values(tmp_path):
    shapes, tensors = _tiny_tensors()
    names = sorted(tensors)
    half = len(names) // 2
    first = {name: tensors[name] for name in names[:half]}
    second = {name: tensors[name] for name in names[half:]}
    _write_safetensors(tmp_path / "model-00001-of-00002.safetensors", first,
                       dtypes={"audio_tower.embed_positions.weight": "F32"})
    _write_safetensors(tmp_path / "model-00002-of-00002.safetensors", second,
                       dtypes={"audio_tower.embed_positions.weight": "F32"})
    weight_map = {name: "model-00001-of-00002.safetensors" for name in first}
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in second})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    # A Mistral-native file next to the Transformers shards is not read.
    (tmp_path / "consolidated.safetensors").write_bytes(b"not read")

    weights = Checkpoint(tmp_path)
    weights.validate(shapes)
    positions = weights.matrix("audio_tower.embed_positions.weight", (8, 8))
    np.testing.assert_array_equal(positions, tensors["audio_tower.embed_positions.weight"])
    conv = weights.conv1d("audio_tower.conv1", 8, 4, 3)
    bits = float32_to_bfloat16_bits(tensors["audio_tower.conv1.weight"]).astype(np.uint32) << 16
    np.testing.assert_array_equal(conv, bits.view(np.float32))
    weights.linear("language_model.lm_head", 32, 16)
    assert "language_model.lm_head.weight" not in weights.unconsumed()
    assert "language_model.model.embed_tokens.weight" in weights.unconsumed()
    with pytest.raises(ValueError, match="shape"):
        weights.linear("language_model.lm_head", 16, 32)


@pytest.mark.parametrize(
    "change",
    ["missing", "extra", "mlx_conv_layout", "packed", "mlx_format", "index_mismatch"],
)
def test_the_reader_refuses_other_layouts(tmp_path, change):
    shapes, tensors = _tiny_tensors()
    dtypes, metadata = {}, None
    if change == "missing":
        del tensors["language_model.lm_head.weight"]
    elif change == "extra":
        tensors["language_model.lm_head.scales"] = np.ones((32, 1), dtype=np.float32)
    elif change == "mlx_conv_layout":
        tensors["audio_tower.conv1.weight"] = tensors["audio_tower.conv1.weight"].transpose(0, 2, 1).copy()
    elif change == "packed":
        dtypes["language_model.lm_head.weight"] = "U32"
    elif change == "mlx_format":
        metadata = {"format": "mlx"}
    _write_safetensors(tmp_path / "model.safetensors", tensors, dtypes=dtypes, metadata=metadata)
    if change == "index_mismatch":
        # The shard holds a tensor that the index does not list.
        weight_map = {name: "model.safetensors" for name in tensors if name != "language_model.lm_head.weight"}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError):
        Checkpoint(tmp_path).validate(shapes)


def test_non_finite_weights_are_refused(tmp_path):
    shapes, tensors = _tiny_tensors()
    tensors["language_model.model.norm.weight"][3] = np.inf
    _write_safetensors(tmp_path / "model.safetensors", tensors)
    weights = Checkpoint(tmp_path)
    weights.validate(shapes)
    with pytest.raises(ValueError, match="non-finite"):
        weights.vector("language_model.model.norm.weight", 16)


def test_engine_arrays_round_once_and_refuse_fp16_overflow():
    values = np.array([1.0, 1.0 + 2.0**-8, 1.0 + 3 * 2.0**-8, -2.5, 65504.0], dtype=np.float32)
    # bfloat16 keeps 8 significant bits: ties round to even.
    widened = (engine_array(values, "bf16").astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_array_equal(widened, [1.0, 1.0, 1.0 + 4 * 2.0**-8, -2.5, 65536.0])
    np.testing.assert_array_equal(engine_array(values, "fp16").astype(np.float32), values)
    assert engine_array(values, "fp32").dtype == np.float32
    with pytest.raises(ValueError, match="float16 range"):
        engine_array(np.array([70000.0], dtype=np.float32), "fp16")
    # Finite float32 values near its maximum round to infinity in bfloat16.
    with pytest.raises(ValueError, match="bfloat16 range"):
        engine_array(np.array([1.0, 3.4e38], dtype=np.float32), "bf16")
    for precision in ("fp32", "fp16", "bf16"):
        for bad in (np.nan, np.inf):
            with pytest.raises(ValueError, match="not finite"):
                engine_array(np.array([0.5, bad], dtype=np.float32), precision)


def test_json_readers_refuse_repeated_keys_and_non_standard_numbers():
    with pytest.raises(ValueError, match="repeats the key 'rope_theta'"):
        load_json('{"text_config": {"rope_theta": 1e8, "rope_theta": 1e4}}', "config.json")
    for literal in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError, match="non-standard number"):
            load_json('{"rms_norm_eps": %s}' % literal, "config.json")
    assert load_json('{"a": [1, {"b": 2.5}]}', "x") == {"a": [1, {"b": 2.5}]}


def _raw_safetensors(path, header_text: str, data: bytes = b"") -> None:
    encoded = header_text.encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


@pytest.mark.parametrize(
    "header",
    [
        # A repeated tensor name, a float dimension, a boolean dimension, float offsets, offsets past the data,
        # a byte range that does not match the shape, and non-string metadata.
        '{"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},'
        ' "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}',
        '{"w": {"dtype": "F32", "shape": [2.0], "data_offsets": [0, 8]}}',
        '{"w": {"dtype": "F32", "shape": [true, 2], "data_offsets": [0, 8]}}',
        '{"w": {"dtype": "F32", "shape": [2], "data_offsets": [0.0, 8.0]}}',
        '{"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}',
        '{"w": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}',
        '{"__metadata__": {"format": 1}, "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}',
    ],
)
def test_malformed_safetensors_headers_are_refused(tmp_path, header):
    _raw_safetensors(tmp_path / "model.safetensors", header, np.zeros(2, dtype="<f4").tobytes())
    with pytest.raises(ValueError):
        Checkpoint(tmp_path)


@pytest.mark.parametrize(
    ("header", "data_bytes"),
    [
        # Two tensors share bytes; a tensor leaves a gap before it; bytes after the last tensor.
        ('{"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},'
         ' "b": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}', 8),
        ('{"a": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]}}', 16),
        ('{"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}', 12),
    ],
)
def test_safetensors_data_must_be_tiled_exactly(tmp_path, header, data_bytes):
    _raw_safetensors(tmp_path / "model.safetensors", header, bytes(data_bytes))
    with pytest.raises(ValueError, match="overlaps another tensor or leaves a gap|belong to no tensor"):
        Checkpoint(tmp_path)


def test_a_repeated_index_entry_is_refused(tmp_path):
    shapes, tensors = _tiny_tensors()
    _write_safetensors(tmp_path / "model.safetensors", tensors)
    entries = ", ".join(f'"{name}": "model.safetensors"' for name in tensors)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {' + entries + ', "language_model.lm_head.weight": "model.safetensors"}}')
    with pytest.raises(ValueError, match="repeats the key"):
        Checkpoint(tmp_path)


@pytest.mark.parametrize("theta", [5e-324, 1e-310])
def test_rope_bases_with_non_finite_angles_are_refused(theta):
    config = copy.deepcopy(CONFIG)
    config["text_config"]["rope_theta"] = theta
    with pytest.raises(ValueError, match="non-finite rotary angles"):
        parse_config(config)


def test_a_header_length_above_100_mb_is_refused_before_reading(tmp_path):
    # A sparse 200 MB file whose header claims 150 MB: refused from the length alone.
    path = tmp_path / "model.safetensors"
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 150_000_000))
        handle.truncate(200_000_000)
    with pytest.raises(ValueError, match="exceeds the file or 100 MB"):
        Checkpoint(tmp_path)
