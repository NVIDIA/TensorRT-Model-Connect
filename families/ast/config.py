# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve AST's geometry, cross-checked against the checkpoint's tensor shapes.

``config.json`` states this model completely, unlike the OWLv2 checkpoint, so
the values are read rather than inferred. They are still checked against the
tensors: a stated patch grid that disagrees with the position table would
otherwise build a network that runs and is wrong, and the position table is the
one tensor that pins every stride and both leading tokens at once.
"""

from __future__ import annotations

import json
from pathlib import Path

# Kaldi's framing, in milliseconds, and the floor under the log. The reference
# feature extractor leaves all three at their torchaudio defaults.
FRAME_LENGTH_MS = 25.0
FRAME_SHIFT_MS = 10.0
MEL_FLOOR = 1.1920928955078125e-07
PREEMPHASIS = 0.97
LOW_FREQ = 20.0


def _read(model_dir: Path, name: str) -> dict:
    path = Path(model_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"Required AST config is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _positive(config: dict, key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"ast config.json has an unusable {key}: {value!r}")
    return value


def _labels(config: dict, num_labels: int) -> list[str]:
    """``id2label`` in ordinal order, with every name present and nonempty.

    The Task returns named classes, so a checkpoint that only numbers them is
    refused here rather than silently reported as ordinals.
    """
    mapping = config.get("id2label")
    if not isinstance(mapping, dict) or len(mapping) != num_labels:
        raise ValueError(f"ast config.json needs {num_labels} id2label entries")
    labels = []
    for index in range(num_labels):
        name = mapping.get(str(index), mapping.get(index))
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"ast config.json has no usable label for class {index}")
        labels.append(name)
    return labels


def _n_fft(frame_length: int) -> int:
    """Kaldi rounds the analysis window up to a power of two before the FFT."""
    size = 1
    while size < frame_length:
        size *= 2
    return size


def resolve(model_dir: Path, shapes: dict) -> dict:
    model = _read(model_dir, "config.json")
    pre = _read(model_dir, "preprocessor_config.json")

    hidden = _positive(model, "hidden_size")
    num_layers = _positive(model, "num_hidden_layers")
    num_heads = _positive(model, "num_attention_heads")
    patch = _positive(model, "patch_size")
    frequency_stride = _positive(model, "frequency_stride")
    time_stride = _positive(model, "time_stride")
    num_mel_bins = _positive(model, "num_mel_bins")
    max_length = _positive(model, "max_length")
    if hidden % num_heads:
        raise ValueError("ast hidden_size is not divisible by num_attention_heads")

    frequency_patches = (num_mel_bins - patch) // frequency_stride + 1
    time_patches = (max_length - patch) // time_stride + 1
    if frequency_patches <= 0 or time_patches <= 0:
        raise ValueError("ast patch size does not fit its stated spectrogram")
    num_patches = frequency_patches * time_patches
    # Two leading tokens, not one: AST keeps DeiT's distillation token and the
    # head averages it with the class token rather than dropping it.
    seq_len = num_patches + 2

    positions = shapes.get("audio_spectrogram_transformer.embeddings.position_embeddings")
    if positions != (1, seq_len, hidden):
        raise ValueError(
            f"ast position table {positions} contradicts the stated geometry "
            f"({frequency_patches}x{time_patches} patches plus two tokens, width {hidden})"
        )
    projection = shapes.get(
        "audio_spectrogram_transformer.embeddings.patch_embeddings.projection.weight")
    if projection != (hidden, 1, patch, patch):
        raise ValueError(f"ast patch projection {projection} contradicts patch_size {patch}")

    layers = sorted(
        int(key.split(".")[3])
        for key in shapes
        if key.startswith("audio_spectrogram_transformer.encoder.layer.")
        and key.endswith(".layernorm_before.weight")
    )
    if layers != list(range(num_layers)):
        raise ValueError(f"ast encoder layers {layers} are not 0..{num_layers - 1}")

    classifier = shapes.get("classifier.dense.weight")
    if classifier is None or len(classifier) != 2 or classifier[1] != hidden:
        raise ValueError(f"ast classifier head {classifier} does not read the encoder width")
    num_labels = int(classifier[0])

    sample_rate = _positive(pre, "sampling_rate")
    if pre.get("num_mel_bins") != num_mel_bins or pre.get("max_length") != max_length:
        raise ValueError("ast preprocessor_config.json contradicts config.json")
    if not pre.get("do_normalize", True):
        raise ValueError("ast expects the feature extractor's normalisation to be enabled")
    if pre.get("padding_side", "right") != "right":
        raise ValueError("ast pads the spectrogram on the right, after the real frames")
    if float(pre.get("padding_value", 0.0)) != 0.0:
        raise ValueError("ast pads the raw filter-bank with zeros, before normalising")
    std = float(pre["std"])
    if std <= 0.0:
        raise ValueError(f"ast preprocessor_config.json has an unusable std: {std}")

    frame_length = int(round(FRAME_LENGTH_MS * sample_rate / 1000.0))
    frame_shift = int(round(FRAME_SHIFT_MS * sample_rate / 1000.0))
    return {
        "hidden_size": hidden,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": hidden // num_heads,
        "eps": float(model.get("layer_norm_eps", 1e-12)),
        "patch": patch,
        "frequency_stride": frequency_stride,
        "time_stride": time_stride,
        "frequency_patches": frequency_patches,
        "time_patches": time_patches,
        "num_patches": num_patches,
        "seq_len": seq_len,
        "num_labels": num_labels,
        "labels": _labels(model, num_labels),
        "num_mel_bins": num_mel_bins,
        "max_length": max_length,
        "sample_rate": sample_rate,
        "frame_length": frame_length,
        "frame_shift": frame_shift,
        "n_fft": _n_fft(frame_length),
        # The reference divides by twice the stated deviation, not by the
        # deviation. Folding the two here keeps one spelling of the rule.
        "feature_mean": float(pre["mean"]),
        "feature_scale": 1.0 / (2.0 * std),
    }
