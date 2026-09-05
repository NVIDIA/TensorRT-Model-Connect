# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from . import test_e2e as e2e


def _framework(monkeypatch) -> tuple[dict, object]:
    calls: dict = {}
    bf16 = object()
    torch = ModuleType("torch")
    torch.float16 = object()
    torch.float32 = object()
    torch.bfloat16 = bf16
    torch.Tensor = type("Tensor", (), {})

    class Generator:
        def __init__(self, *args, **kwargs):
            calls["generator_init"] = (args, kwargs)

        def manual_seed(self, seed: int):
            calls["seed"] = seed
            return self

    torch.Generator = Generator

    class Processor:
        image_token_ids = [10]
        video_token_ids = [20]
        audio_token_ids = [30]

    class Pipeline:
        def __init__(self):
            self.processor = Processor()

        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):
            calls["model_dir"] = model_dir
            calls["load_pipeline"] = kwargs
            calls["pipeline"] = cls()
            return calls["pipeline"]

        def load_components(self, **kwargs):
            calls["load_components"] = kwargs

        def to(self, device: str):
            calls["device"] = device
            return self

        def __call__(self, **kwargs):
            calls["kwargs"] = kwargs
            return {"videos": [np.zeros((3, 2, 2, 3), dtype=np.float32)]}

    diffusers = ModuleType("diffusers")
    diffusers.ModularPipeline = Pipeline
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    return calls, bf16


def test_reference_uses_cpu_generator_local_components_and_family_processor(
    monkeypatch, tmp_path: Path
) -> None:
    calls, bf16 = _framework(monkeypatch)
    _, manifest, case = e2e.CASES["minimax-h3-768p"]
    manifest = {**manifest, "video_num_frames": 3, "image_height": 2, "image_width": 2}
    model_dir = Path("model")
    result = e2e._official_reference(model_dir, manifest, case, tmp_path)

    assert calls["load_pipeline"] == {"workflow": "t2va", "local_files_only": True}
    assert calls["load_components"] == {
        "dtype": bf16,
        "pretrained_model_name_or_path": model_dir,
        "local_files_only": True,
    }
    assert calls["generator_init"] == ((), {})
    assert calls["seed"] == 0
    processor = calls["pipeline"].processor
    assert not hasattr(processor, "create_mm_token_type_ids")
    frames = np.load(result["frames_path"], mmap_mode="r")
    assert frames.shape == (3, 2, 2, 3)


def test_visual_metrics_stream_one_pair_and_use_block_activity(monkeypatch, tmp_path: Path) -> None:
    actual_paths = [Path(f"actual-{index}") for index in range(3)]
    block_values = (
        np.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
        np.asarray([[0.3, 0.7], [0.5, 0.9]], dtype=np.float32),
        np.asarray([[0.8, 0.4], [0.2, 0.6]], dtype=np.float32),
    )
    checker = np.asarray([[1, -1, 1, -1], [-1, 1, -1, 1]] * 2, dtype=np.float32)
    values = {}
    expected_frames = []
    for index, actual_path in enumerate(actual_paths):
        expected = np.repeat(np.repeat(block_values[index], 2, axis=0), 2, axis=1)
        expected = np.repeat(expected[..., None], 3, axis=2)
        actual = expected + ((-1) ** index) * checker[..., None] * 0.05
        values[actual_path] = actual
        expected_frames.append(expected)
    loaded = []

    def load(path: Path) -> np.ndarray:
        loaded.append(path)
        return values[path]

    monkeypatch.setattr(e2e, "_load_rgb", load)
    expected_path = tmp_path / "expected.npy"
    np.save(expected_path, np.asarray(expected_frames))
    original_load = np.load
    load_options = {}

    def load_frames(path, **kwargs):
        load_options.update(kwargs)
        return original_load(path, **kwargs)

    monkeypatch.setattr(e2e.np, "load", load_frames)
    metrics = e2e._streaming_visual_metrics(actual_paths, expected_path, 2)

    assert loaded == actual_paths
    assert load_options == {"mmap_mode": "r", "allow_pickle": False}
    assert metrics["minimum_frame_low_frequency_correlation"] == pytest.approx(1.0)
    assert metrics["temporal_motion_ratio"] == pytest.approx(1.0)
    assert metrics["temporal_profile_correlation"] == pytest.approx(1.0)
