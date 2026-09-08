# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest

from families.minimax_h3 import model


class _Writer:
    def __init__(self) -> None:
        self.header = None

    def set_header(self, **header) -> None:
        self.header = header


@pytest.mark.parametrize("quantized", [False, True])
def test_build_uses_unified_staged_path_and_family_options(
    tmp_path: Path, monkeypatch, quantized: bool
) -> None:
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
        quantization="int8_tensorwise_convrot" if quantized else None,
        image_height=480,
        image_width=864,
        video_num_frames=345,
        family_options=(
            ("first_block_cache_threshold", 0.12),
            ("transformer_ref", "ref-model"),
            *(("quantized_transformer", "quant.safetensors"),) * quantized,
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


@pytest.mark.parametrize(
    ("quantization", "options"),
    [
        (None, (("quantized_transformer", "quant.safetensors"),)),
        ("none", (("quantized_transformer", "quant.safetensors"),)),
        ("int8_tensorwise_convrot", ()),
    ],
)
def test_build_rejects_inconsistent_quantization_before_loading_weights(
    tmp_path: Path, monkeypatch, quantization, options
) -> None:
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "h3.bundle",
        family="minimax_h3",
        task="image_generation",
        precision="bf16",
        backend="trt_rtx",
        quantization=quantization,
        family_options=options,
    )
    monkeypatch.setattr(
        model.plugin, "load_weights", lambda *_args: pytest.fail("Loaded incompatible weights")
    )
    with pytest.raises(ValueError, match="requires"):
        model.build(request, _Writer())


def test_core_dispatch_validates_h3_options_and_publishes_staged_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    core = importlib.import_module("tensorrt_model_connect.build")
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "h3.bundle",
        family="minimax_h3",
        task="image_generation",
        precision="bf16",
        backend="trt_rtx",
        family_options=(("first_block_cache_threshold", 0.12),),
    )
    validated = []
    validate = model.validate_build_options

    def validate_options(options):
        validated.append(options)
        return validate(options)

    def staged(_path, writer, config, _weights, **_options):
        assert config.raw["_family_build_options"]["minimax_h3"] == dict(request.family_options)
        writer.add_bytes("test_plan", b"plan")

    monkeypatch.setattr(core, "_select_backend", lambda _backend: None)
    monkeypatch.setattr(model, "validate_build_options", validate_options)
    monkeypatch.setattr(model.plugin, "load_weights", lambda path, _config: {"_model_dir": path})
    monkeypatch.setattr(model.plugin, "build_staged_bundle", staged)
    core.build(request)
    assert validated == [dict(request.family_options)]
    assert request.output_path.is_file()
