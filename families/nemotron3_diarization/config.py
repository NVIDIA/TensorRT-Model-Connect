# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the Nemotron 3 Diarization geometry, cross-checked against the tensors.

``config.json`` states the encoder, the speaker head, the offline chunking and
the speaker-cache policy; ``processor_config.json`` states the front end. Every
value the bundle depends on is read, checked against the tensor shapes where a
tensor pins it, and refused when this family does not implement it, so a
checkpoint that differs in a way the graph would silently ignore is not built.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

# The front end floors the mel energies with this constant before the log.
LOG_GUARD = 2.0**-24
# The checkpoint's prefix of the audio encoder's tensors.
ENCODER_PREFIX = "model.audio_tower"
# The largest finite float32: the engines and the native runtime hold every
# constant in float32.
FLOAT32_MAX = 3.4028234663852886e38
INT32_MAX = 2**31 - 1


def _unique_keys(pairs: list) -> dict:
    """An object hook that refuses a repeated key instead of keeping its last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"nemotron3_diarization config repeats the key {key!r}")
        result[key] = value
    return result


def _read(model_dir: Path, name: str) -> dict:
    path = Path(model_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"Required nemotron3_diarization config is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)


def _int(config: dict, key: str, *, minimum: int = 1, default: int | None = None) -> int:
    # The native runtime reads every count as an int32.
    value = config.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= INT32_MAX:
        raise ValueError(f"nemotron3_diarization config has an unusable {key}: {value!r}")
    return value


def _float(config: dict, key: str, default: float | None = None, *,
           low: float = -FLOAT32_MAX, high: float = FLOAT32_MAX,
           low_open: bool = False, high_open: bool = False) -> float:
    # JSON numbers only: a boolean, a string or an explicit null is refused, never
    # coerced. The range holds for the float32 the engines and the runtime store,
    # so the value is rounded to float32 before it is compared.
    value = config.get(key, default)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not abs(value) <= FLOAT32_MAX):  # NaN fails the comparison too
        raise ValueError(f"nemotron3_diarization config has an unusable {key}: {value!r}")
    stored = struct.unpack("<f", struct.pack("<f", value))[0]
    below = stored <= low if low_open else stored < low
    above = stored >= high if high_open else stored > high
    if below or above:
        raise ValueError(f"nemotron3_diarization config {key} is out of range: {value!r}")
    return float(value)


def _budget(frames: int, rate: float, key: str, limit: int) -> int:
    """A speaker-cache frame budget, ``floor(frames * rate)``, at most ``limit``.

    A boost budget ranks frames with top-k, which needs as many frames as it
    boosts; a compression always ranks more frames than the cache holds.
    """
    budget = math.floor(frames * rate)
    if budget > limit:
        raise ValueError(f"nemotron3_diarization config {key} gives a budget past {limit}: {rate!r}")
    return budget


def _expect(actual, expected, what: str) -> None:
    if actual != expected:
        raise ValueError(f"nemotron3_diarization {what} is {actual!r}; this family needs {expected!r}")


def _encoder(audio: dict, shapes: dict) -> dict:
    hidden = _int(audio, "hidden_size")
    heads = _int(audio, "num_attention_heads")
    kv_heads = _int(audio, "num_key_value_heads", default=heads)
    _expect(kv_heads, heads, "num_key_value_heads")
    if hidden % heads:
        raise ValueError("nemotron3_diarization hidden_size is not divisible by the head count")
    # Rotary embeddings pair dimension i with i + head_dim / 2.
    if (hidden // heads) % 2:
        raise ValueError("nemotron3_diarization needs an even head width for rotary embeddings")
    _expect(audio.get("hidden_act"), "gelu", "hidden_act")
    rope = audio.get("rope_parameters") or {}
    _expect(rope.get("rope_type", "default"), "default", "rope_type")
    # Every dimension of every head rotates: the reference passes nothing through.
    _expect(_float(rope, "partial_rotary_factor", 1.0), 1.0, "partial_rotary_factor")
    encoder = {
        "hidden_size": hidden,
        "num_heads": heads,
        "head_dim": hidden // heads,
        "intermediate_size": _int(audio, "intermediate_size"),
        "num_layers": _int(audio, "num_hidden_layers"),
        "num_mel_bins": _int(audio, "num_mel_bins"),
        "subsampling_factor": _int(audio, "subsampling_factor"),
        "max_position_embeddings": _int(audio, "max_position_embeddings"),
        # A base above one keeps every inverse frequency in (0, 1] in float32.
        "rope_theta": _float(rope, "rope_theta", low=1.0, low_open=True),
        "layer_norm_eps": 1e-5,  # nn.LayerNorm's default; the config does not override it.
    }
    stacked = encoder["subsampling_factor"] * encoder["num_mel_bins"]
    _expect(shapes.get("model.audio_tower.embedder.projection.weight"), (hidden, stacked),
            "feature-stacking projection shape")
    _expect(sum(key.startswith("model.audio_tower.layers.") and key.endswith(".self_attn.q_proj.weight")
                for key in shapes), encoder["num_layers"], "encoder layer count")
    _expect(shapes.get("model.audio_tower.layers.0.mlp.fc1.weight"),
            (encoder["intermediate_size"], hidden), "MLP shape")
    return encoder


def _head(head_config: dict, encoder: dict, shapes: dict) -> dict:
    head = {
        "head_hidden_size": _int(head_config, "hidden_size"),
        "num_speakers": _int(head_config, "num_speakers"),
    }
    _expect(_int(head_config, "audio_hidden_size"), encoder["hidden_size"], "head audio_hidden_size")
    _expect(_int(head_config, "subsampling_factor"), encoder["subsampling_factor"],
            "head subsampling_factor")
    width, speakers, factor = (head["head_hidden_size"], head["num_speakers"],
                               encoder["subsampling_factor"])
    _expect(shapes.get("model.proj.weight"), (width, encoder["hidden_size"]), "projection shape")
    _expect(shapes.get("model.upsampler.conv.weight"), (width * factor, width, 3), "upsampler shape")
    _expect(shapes.get("classifier.dense.weight"), (width, width), "classifier shape")
    _expect(shapes.get("classifier.out_proj.weight"), (speakers, width), "speaker output shape")
    _expect(shapes.get("silence_embeds"), (encoder["hidden_size"],), "silence embedding shape")
    return head


def _speaker_cache(streaming: dict, encoder: dict, head: dict) -> dict:
    """The Arrival-Order Speaker Cache policy, with its derived frame budgets.

    The budgets are computed here, once, with the reference's own arithmetic, so
    the runtime receives integers instead of re-deriving them from rates.
    """
    _expect(_int(streaming, "num_speakers"), head["num_speakers"], "streaming num_speakers")
    _expect(_int(streaming, "subsampling_factor"), encoder["subsampling_factor"],
            "streaming subsampling_factor")
    length = _int(streaming, "speaker_cache_length")
    silence = _int(streaming, "speaker_cache_silence_frames_per_speaker", minimum=0)
    speakers = head["num_speakers"]
    if length < (1 + silence) * speakers:
        raise ValueError("nemotron3_diarization speaker cache cannot hold every speaker")
    budget = length // speakers - silence
    return {
        "speaker_cache_length": length,
        "silence_frames_per_speaker": silence,
        # The probability floor of the cache's log scores: a log of zero would
        # turn saturated probabilities into infinities.
        "prediction_score_threshold": _float(streaming, "prediction_score_threshold", low=0.0,
                                             high=1.0, low_open=True, high_open=True),
        "latest_frames_score_boost": _float(streaming, "latest_frames_score_boost"),
        "min_positive_scores": _budget(budget, _float(streaming, "min_positive_scores_rate",
                                                      low=0.0), "min_positive_scores_rate",
                                       INT32_MAX),
        **{name: _budget(budget, _float(streaming, rate, low=0.0), rate, length)
           for name, rate in (("strong_boosted_frames", "strong_boost_rate"),
                              ("weak_boosted_frames", "weak_boost_rate"))},
    }


def _offline_chunking(model: dict, encoder: dict, cache: dict) -> dict:
    """Whole-recording diarization: the reference's offline chunk loop."""
    chunking = {
        "chunk_length": _int(model, "chunk_length"),
        "chunk_left_context": _int(model, "chunk_left_context", minimum=0, default=0),
        "chunk_right_context": _int(model, "chunk_right_context", minimum=0),
        "fifo_length": _int(model, "fifo_length", minimum=0),
        "speaker_cache_update_period": _int(model, "speaker_cache_update_period"),
    }
    if chunking["chunk_right_context"] >= chunking["chunk_length"]:
        raise ValueError("nemotron3_diarization chunk_right_context must be below chunk_length")
    # One encoder step attends over the whole cache, the FIFO queue and one chunk
    # with both of its contexts; positions restart at zero for every step.
    chunking["max_step_rows"] = (cache["speaker_cache_length"] + chunking["fifo_length"]
                                 + chunking["chunk_left_context"] + chunking["chunk_length"]
                                 + chunking["chunk_right_context"])
    if chunking["max_step_rows"] > encoder["max_position_embeddings"]:
        raise ValueError("nemotron3_diarization step exceeds max_position_embeddings")
    return chunking


def _front_end(processor: dict, encoder: dict) -> dict:
    features = processor.get("feature_extractor") or {}
    _expect(features.get("feature_extractor_type"), "NemotronAsrStreamingFeatureExtractor",
            "feature extractor")
    _expect(features.get("mel_filters_dtype", "float32"), "float32", "mel_filters_dtype")
    _expect(_float(features, "padding_value", 0.0), 0.0, "feature padding_value")
    front = {
        "sample_rate": _int(features, "sampling_rate"),
        "n_fft": _int(features, "n_fft"),
        # The symmetric window divides by win_length - 1.
        "win_length": _int(features, "win_length", minimum=2),
        "hop_length": _int(features, "hop_length"),
        "preemphasis": _float(features, "preemphasis", low=0.0, high=1.0, high_open=True),
        "log_guard": LOG_GUARD,
    }
    _expect(_int(features, "feature_size"), encoder["num_mel_bins"], "feature_size")
    if front["win_length"] > front["n_fft"] or front["n_fft"] & (front["n_fft"] - 1):
        raise ValueError("nemotron3_diarization needs a power-of-two n_fft no shorter than the window")
    _expect(_int(processor, "subsampling_factor"), encoder["subsampling_factor"],
            "processor subsampling_factor")
    return front


def expected_tensors(cfg: dict) -> dict[str, tuple[int, ...]]:
    """Every tensor of the checkpoint, with the shape the configuration implies."""
    hidden, inner = cfg["hidden_size"], cfg["intermediate_size"]
    width, factor, speakers = cfg["head_hidden_size"], cfg["subsampling_factor"], cfg["num_speakers"]
    encoder = ENCODER_PREFIX
    shapes = {
        f"{encoder}.embedder.projection.weight": (hidden, factor * cfg["num_mel_bins"]),
        f"{encoder}.input_layer_norm.weight": (hidden,),
        f"{encoder}.input_layer_norm.bias": (hidden,),
        f"{encoder}.layer_norm.weight": (hidden,),
        f"{encoder}.layer_norm.bias": (hidden,),
        "model.proj.weight": (width, hidden),
        "model.proj.bias": (width,),
        "model.upsampler.conv.weight": (width * factor, width, 3),
        "model.upsampler.conv.bias": (width * factor,),
        "classifier.dense.weight": (width, width),
        "classifier.dense.bias": (width,),
        "classifier.out_proj.weight": (speakers, width),
        "classifier.out_proj.bias": (speakers,),
        "silence_embeds": (hidden,),
    }
    for index in range(cfg["num_layers"]):
        prefix = f"{encoder}.layers.{index}"
        for norm in ("layer_norm1", "layer_norm2"):
            shapes[f"{prefix}.{norm}.weight"] = (hidden,)
            shapes[f"{prefix}.{norm}.bias"] = (hidden,)
        # The query, key and value projections have no bias.
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            shapes[f"{prefix}.self_attn.{name}.weight"] = (hidden, hidden)
        shapes[f"{prefix}.self_attn.o_proj.bias"] = (hidden,)
        shapes[f"{prefix}.mlp.fc1.weight"] = (inner, hidden)
        shapes[f"{prefix}.mlp.fc1.bias"] = (inner,)
        shapes[f"{prefix}.mlp.fc2.weight"] = (hidden, inner)
        shapes[f"{prefix}.mlp.fc2.bias"] = (hidden,)
    return shapes


def check_tensors(cfg: dict, tensors: dict) -> None:
    """The checkpoint holds exactly the expected tensors, each float32 in its configured shape.

    ``tensors`` maps every name in the file to its safetensors dtype and shape,
    read from the header before any tensor is loaded.
    """
    expected = expected_tensors(cfg)
    missing = sorted(set(expected) - set(tensors))
    unexpected = sorted(set(tensors) - set(expected))
    if missing or unexpected:
        raise ValueError(f"nemotron3_diarization checkpoint tensors differ from the configuration: "
                         f"missing {missing[:3]}, unexpected {unexpected[:3]}")
    for name, (dtype, shape) in sorted(tensors.items()):
        if dtype != "F32":
            raise ValueError(f"nemotron3_diarization checkpoint tensor {name} is {dtype}, not F32")
        if tuple(shape) != expected[name]:
            raise ValueError(f"nemotron3_diarization checkpoint tensor {name} has the shape "
                             f"{tuple(shape)}, not {expected[name]}")


RUNTIME_KEYS = (
    "sample_rate", "n_fft", "win_length", "hop_length", "preemphasis", "log_guard",
    "num_mel_bins", "subsampling_factor", "hidden_size", "num_speakers", "chunk_length",
    "chunk_left_context", "chunk_right_context", "fifo_length", "speaker_cache_update_period",
    "speaker_cache_length", "silence_frames_per_speaker", "prediction_score_threshold",
    "latest_frames_score_boost", "min_positive_scores", "strong_boosted_frames",
    "weak_boosted_frames", "max_step_rows",
)


def runtime_config(cfg: dict) -> dict:
    """``runtime.json``: everything the native runtime reads, nothing it re-derives."""
    values = {key: cfg[key] for key in RUNTIME_KEYS}
    # Speakers are anonymous and numbered by first appearance in each input,
    # spelled as the reference toolkit spells them.
    values["labels"] = [f"speaker_{index}" for index in range(cfg["num_speakers"])]
    return values


def resolve(model_dir: Path, shapes: dict) -> dict:
    model = _read(model_dir, "config.json")
    processor = _read(model_dir, "processor_config.json")
    _expect(model.get("model_type"), "nemotron3_diarization", "model_type")
    encoder = _encoder(model.get("audio_config") or {}, shapes)
    head = _head(model.get("head_config") or {}, encoder, shapes)
    cache = _speaker_cache(model.get("streaming_config") or {}, encoder, head)
    cfg = {**encoder, **head, **cache}
    cfg.update(_offline_chunking(model, encoder, cache))
    cfg.update(_front_end(processor, encoder))
    return cfg
