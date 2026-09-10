# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("quantization", ("none", "bf16", "fp8"))
def test_build_rejects_non_delivery_quantization_before_loading_weights(
    tmp_path: Path, monkeypatch, quantization
) -> None:
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "h3.bundle",
        family="minimax_h3",
        task="image_generation",
        precision="bf16",
        backend="trt_rtx",
        quantization=quantization,
    )
    monkeypatch.setattr(
        model.plugin, "load_weights", lambda *_args: pytest.fail("Loaded incompatible weights")
    )
    with pytest.raises(ValueError, match="requires"):
        model.build(request, _Writer())


@pytest.mark.parametrize("option", ("transformer_ref", "transformer_ref_path"))
def test_public_build_rejects_manual_bf16_ref_source(tmp_path: Path, monkeypatch, option) -> None:
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "h3.bundle",
        family="minimax_h3",
        task="image_generation",
        precision="bf16",
        backend="trt_rtx",
        family_options=((option, "legacy-ref"),),
    )
    monkeypatch.setattr(
        model.plugin, "load_weights", lambda *_args: pytest.fail("Loaded legacy BF16 source")
    )
    with pytest.raises(ValueError, match="unknown MiniMax-H3 option"):
        model.build(request, _Writer())


@pytest.mark.parametrize("sr", (False, True))
def test_plugin_build_resolves_all_three_quantized_sources_and_explicit_sr(
    tmp_path: Path, monkeypatch, sr: bool
) -> None:
    from families.minimax_h3 import delivery, staged_build

    sources = {
        option: tmp_path / filename for option, filename in delivery.QUANTIZED_SOURCES.items()
    }
    primary, weak = tmp_path / "sr.pth", tmp_path / "sr-weak.pth"
    family_options = {"super_resolution": sr}
    observed = {}

    def resolve_sources(path, options):
        assert path == tmp_path
        assert options["super_resolution"] is sr
        return sources

    def resolve_sr(options):
        assert options["super_resolution"] is sr
        return (primary, weak) if sr else (None, None)

    def staged(path, writer, **options):
        observed.update(options)

    monkeypatch.setattr(delivery, "resolve_quantized_sources", resolve_sources)
    monkeypatch.setattr(delivery, "resolve_super_resolution_sources", resolve_sr)
    monkeypatch.setattr(staged_build, "build_staged_bundle", staged)
    model.plugin.build_staged_bundle(
        str(tmp_path),
        _Writer(),
        SimpleNamespace(raw={"_family_build_options": {"minimax_h3": family_options}}),
        {"_model_dir": str(tmp_path)},
        plans_dir=tmp_path / "plans",
        precision="bf16",
    )
    assert {key: observed[key] for key in sources} == sources
    assert "transformer_ref" not in observed
    assert (
        observed.get("super_resolution_model"),
        observed.get("super_resolution_weak_model"),
    ) == ((primary, weak) if sr else (None, None))
    defaults = observed["runtime_defaults"]
    assert (defaults["height"], defaults["width"]) == ((480, 864) if sr else (768, 1344))
    assert defaults["num_frames"] == 124
    assert defaults["ref2va_first_block_cache"] is True


@pytest.mark.parametrize("height,width", ((768, 1344), (864, 480), (544, 960)))
def test_sr_rejects_noncanonical_source_canvas_before_loading_weights(
    tmp_path: Path, monkeypatch, height: int, width: int
) -> None:
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "h3.bundle",
        family="minimax_h3",
        task="image_generation",
        precision="bf16",
        backend="trt_rtx",
        image_height=height,
        image_width=width,
        family_options=(("super_resolution", True),),
    )
    monkeypatch.setattr(
        model.plugin, "load_weights", lambda *_args: pytest.fail("Loaded invalid SR canvas")
    )
    with pytest.raises(ValueError, match="requires height=480 and width=864"):
        model.build(request, _Writer())


def test_one_public_dynamic_profile_includes_native_480p_without_auto_sr() -> None:
    profile = model._public_dynamic_profile({})
    assert profile.min_video_rows == 14_985
    assert profile.min_sequence_length == 15_400
    assert (profile.min_text_rows, profile.text_rows) == (1, 2641)
    assert model._default_canvas_size({"height": 480, "width": 864}) == (480, 864)
    assert model._default_canvas_size({"super_resolution": True}) == (480, 864)


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
    assert validated == [dict(request.family_options)] * 3
    assert request.output_path.is_file()
