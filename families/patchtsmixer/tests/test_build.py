# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from families.patchtsmixer import model


def _request(tmp_path, tensor_parallel_size: int):
    return SimpleNamespace(
        model_dir=tmp_path,
        backend="trt",
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="time_series_forecast",
        precision="fp32",
        max_sequence_length=None,
        tensor_parallel_size=tensor_parallel_size,
        quantization=None,
        fp32_layers=(),
        verbose=False,
    )


@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4, 8])
def test_build_replicates_one_plan_into_rank_sections(
    monkeypatch, tmp_path, tensor_parallel_size: int
) -> None:
    config = {
        "num_layers": 1,
        "context_length": 4,
        "num_input_channels": 2,
        "prediction_length": 2,
    }
    builds = []
    sections = {}
    runtime = {}
    monkeypatch.setattr(model, "_read_config", lambda _path: dict(config))
    monkeypatch.setattr(model, "infer_patchtsmixer_task_kind", lambda _config: "forecast")
    monkeypatch.setattr(model, "_require_supported", lambda _config, _task: None)
    monkeypatch.setattr(model, "_load_all_tensors", lambda *args, **kwargs: {})

    def build_plan(*args, **kwargs):
        builds.append((args, kwargs))
        return b"plan"

    monkeypatch.setattr(model, "_build_patchtsmixer_network", build_plan)
    writer = SimpleNamespace(
        set_header=lambda **_header: None,
        add_bytes=lambda name, data: sections.update({name: data}),
        add_json=lambda name, data: runtime.update({name: data}),
    )

    model.build(_request(tmp_path, tensor_parallel_size), writer)

    assert len(builds) == 1
    expected_sections = (
        {"engine.plan": b"plan"}
        if tensor_parallel_size == 1
        else {f"engine.rank{rank}.plan": b"plan" for rank in range(tensor_parallel_size)}
    )
    assert sections == expected_sections
    assert runtime["runtime.json"]["tensor_parallel_size"] == tensor_parallel_size


def test_build_rejects_unsupported_tensor_parallel_size(tmp_path) -> None:
    with pytest.raises(ValueError, match="1, 2, 4, or 8"):
        model.build(_request(tmp_path, 3), SimpleNamespace())
