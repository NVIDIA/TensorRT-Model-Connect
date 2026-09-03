# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_model_connect.families.boltz2 import random_samples
from tensorrt_model_connect.families.boltz2.random_samples import (
    MAGIC,
    VERSION,
    serialize_predict_random_samples,
    serialize_random_arrays,
)


def test_random_sample_section_layout_is_stable():
    initial = np.arange(6, dtype=np.float32).reshape(2, 3)
    rotations = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    translations = np.arange(6, dtype=np.float32).reshape(2, 3)
    noise = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    data = serialize_random_arrays(initial, rotations, translations, noise, seed=42)
    assert data[:4] == MAGIC
    assert struct.unpack("<IIII", data[4:20]) == (VERSION, 42, 2, 2)
    expected = b"".join(value.tobytes() for value in (initial, rotations, translations, noise))
    assert data[20:] == expected


def test_random_sample_section_rejects_inconsistent_shapes():
    with pytest.raises(ValueError, match="inconsistent"):
        serialize_random_arrays(
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((2, 2, 3), dtype=np.float32),
            seed=42,
        )


def test_predict_random_samples_capture_the_real_diffusion_boundary(monkeypatch, tmp_path):
    original_sample = object()
    model = SimpleNamespace(
        structure_module=SimpleNamespace(sample=original_sample),
    )
    features = {"token_pad_mask": object()}
    events = []

    def fake_predict(actual_model, actual_features):
        assert actual_model is model
        assert actual_features is features
        actual_model.structure_module.sample()

    def fake_serialize(**kwargs):
        events.append(("serialize", kwargs))
        return b"resolved-stream"

    monkeypatch.setattr(
        "tensorrt_model_connect.families.boltz2.reference_benchmark._load_model",
        lambda checkpoint, *, compiled: model,
    )
    monkeypatch.setattr(
        "tensorrt_model_connect.families.boltz2.reference_benchmark._predict",
        fake_predict,
    )
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: b"boundary-state")
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state: events.append(("set-state", state)),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append(("empty",)))
    monkeypatch.setattr(random_samples, "_serialize_current_cuda_stream", fake_serialize)

    result = serialize_predict_random_samples(
        tmp_path / "boltz2.ckpt",
        features,
        seed=7,
        sampling_steps=3,
        atom_count=11,
    )

    assert result == b"resolved-stream"
    assert model.structure_module.sample is original_sample
    assert events == [
        ("set-state", b"boundary-state"),
        (
            "serialize",
            {"seed": 7, "sampling_steps": 3, "atom_count": 11},
        ),
        ("empty",),
    ]
