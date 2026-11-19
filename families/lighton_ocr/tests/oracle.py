# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-observation and generation gates for the selected LightOn checkpoint."""

import json
import math
from pathlib import Path

import numpy as np


OBSERVATIONS = ("pixel_values", "image_features", "step_0_logits", "step_1_logits")


def _tensor(directory, record):
    assert record["dtype"] == "float32", "Observation dtype must remain FP32"
    shape = record["shape"]
    assert isinstance(shape, list) and shape and all(type(size) is int and size > 0 for size in shape)
    filename = record["file"]
    assert isinstance(filename, str) and Path(filename).name == filename
    path = directory / filename
    assert path.is_file() and not path.is_symlink(), "Observation file is missing or indirect"
    value = np.fromfile(path, dtype=np.float32)
    assert value.size == math.prod(shape), "Observation byte count does not match its shape"
    assert np.isfinite(value).all(), "Every observation value must be finite"
    return value.reshape(shape)


def compare_observations(native_directory, reference_directory, limits):
    native_directory, reference_directory = Path(native_directory), Path(reference_directory)
    native = json.loads((native_directory / "metadata.json").read_text())
    reference = json.loads((reference_directory / "metadata.json").read_text())
    assert native["precision"] == reference["precision"] == "fp32"
    assert native["prompt_ids"] == reference["prompt_ids"], "Native image-only prompt differs from the processor"
    assert native["cache_prefix_unchanged"] is True and reference["cache_prefix_unchanged"] is True
    assert set(native["tensors"]) == set(reference["tensors"]) == set(OBSERVATIONS)
    config = reference["checkpoint_config"]
    pixel_shape = reference["tensors"]["pixel_values"]["shape"]
    assert len(pixel_shape) == 4 and pixel_shape[:2] == [1, 3]
    patch, merge = config["vision_config"]["patch_size"], config["spatial_merge_size"]
    height, width = pixel_shape[-2:]
    assert height % patch == width % patch == 0
    count = (height // (patch * merge)) * (width // (patch * merge))
    expected_shapes = {
        "pixel_values": pixel_shape,
        "image_features": [count, config["text_config"]["hidden_size"]],
        "step_0_logits": [1, config["text_config"]["vocab_size"]],
        "step_1_logits": [1, config["text_config"]["vocab_size"]],
    }
    assert reference["prompt_ids"].count(config["image_token_id"]) == count
    metrics = {}
    for name in OBSERVATIONS:
        assert native["tensors"][name]["shape"] == reference["tensors"][name]["shape"] == expected_shapes[name], name
        actual = _tensor(native_directory, native["tensors"][name]).astype(np.float64)
        expected = _tensor(reference_directory, reference["tensors"][name]).astype(np.float64)
        difference = actual - expected
        relative = float(np.linalg.norm(difference) / max(float(np.linalg.norm(expected)), 1e-12))
        maximum = float(np.abs(difference).max())
        observed = {"relative_l2": relative, "maximum_absolute_error": maximum}
        if name == "pixel_values":
            assert maximum <= limits["pixel_maximum_absolute_error"], name
        else:
            group = "image_features" if name == "image_features" else "logits"
            assert relative <= limits[group + "_relative_l2"], name
            assert maximum <= limits[group + "_maximum_absolute_error"], name
        if name.startswith("step_"):
            agreement = float(np.mean(actual.argmax(-1) == expected.argmax(-1)))
            observed["argmax_agreement"] = agreement
            assert agreement >= limits["logits_argmax_agreement"], name
        metrics[name] = observed
    return metrics


def validate_generation(observed, limits, *, max_new_tokens, expected_fragments, eos_token_ids, vocab_size):
    assert limits["generation_nonempty"] is True
    assert limits["generation_reaches_eos"] is True
    text, ids = observed["text"], observed["token_ids"]
    assert isinstance(text, str) and text.strip(), "OCR generation must contain text"
    assert isinstance(ids, list) and 0 < len(ids) <= max_new_tokens
    assert all(type(value) is int and 0 <= value < vocab_size for value in ids)
    assert eos_token_ids and ids[-1] in eos_token_ids, "OCR generation must reach EOS within its budget"
    if "reached_eos" in observed:
        assert observed["reached_eos"] is True
    for fragment in expected_fragments:
        assert isinstance(fragment, str) and fragment and fragment in text, fragment
    for name in ("prefill_ms", "decode_ms"):
        if name in observed:
            assert math.isfinite(observed[name]) and observed[name] >= 0


def compare_generation(actual, reference, limits, *, max_new_tokens, expected_fragments, vocab_size):
    assert limits["generation_exact_token_ids"] is True
    assert limits["generation_exact_text"] is True
    options = dict(max_new_tokens=max_new_tokens, expected_fragments=expected_fragments,
                   eos_token_ids=reference["eos_token_ids"], vocab_size=vocab_size)
    assert "prefill_ms" in actual and "decode_ms" in actual, "Native timing fields must be present"
    validate_generation(reference, limits, **options)
    validate_generation(actual, limits, **options)
    assert actual["token_ids"] == reference["token_ids"], "Native/reference generated IDs differ"
    assert actual["text"] == reference["text"], "Native/reference OCR text differs"
