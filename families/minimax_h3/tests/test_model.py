# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from tensorrt_model_connect import BuildRequest

from families.minimax_h3 import model


class _Writer:
    def __init__(self) -> None:
        self.header = None

    def set_header(self, **header) -> None:
        self.header = header


def test_build_uses_unified_staged_path_and_family_options(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output = tmp_path / "h3.bundle"
    request = BuildRequest(
        model_dir=model_dir,
        output_path=output,
        family="minimax_h3",
        task="image_generation",
        precision="bf16",
        backend="trt_rtx",
        image_height=480,
        image_width=864,
        video_num_frames=345,
        family_options=(
            ("first_block_cache_threshold", 0.12),
            ("transformer_ref", "ref-model"),
        ),
    )
    observed = {}

    def load_weights(path: str, config):
        observed["load"] = (path, config.raw)
        return {"_model_dir": path}

    def staged(path: str, writer, config, weights, **options):
        observed["staged"] = (path, writer, config.raw, weights, options)

    monkeypatch.setattr(model.plugin, "load_weights", load_weights)
    monkeypatch.setattr(model.plugin, "build_staged_bundle", staged)
    writer = _Writer()

    model.build(request, writer)

    assert writer.header == {
        "family": "minimax_h3",
        "task": "image_generation",
        "backend": "trt_rtx",
    }
    raw = observed["load"][1]
    assert raw["_family_build_options"]["minimax_h3"] == dict(request.family_options)
    assert (raw["height"], raw["width"], raw["video_num_frames"]) == (480, 864, 345)
    assert observed["staged"][4]["plans_dir"] == tmp_path / "h3.bundle.plans"
    assert observed["staged"][4]["precision"] == "bf16"
