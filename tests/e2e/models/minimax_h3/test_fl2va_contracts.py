# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure FL2VA workflow, keyframe, command, and reference-call contracts."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
from tensorrt_model_connect.families.minimax_h3.config import (
    FL2VA_PLAN_FILENAMES,
    FL2VA_PROCESSOR_ASSET_SECTIONS,
)
from tests.e2e.models.minimax_h3 import hf_reference
from tests.e2e.models.minimax_h3 import e2e_plugins
from tests.e2e.models.minimax_h3.e2e_plugins import reference as reference_plugin
from tests.e2e.models.minimax_h3.e2e_plugins import runner as native_runner
from tests.e2e_harness.contracts import RunContext
from tests.e2e_harness.manifest_loader import load_model_manifest


MODEL_DIR = Path(__file__).resolve().parent
FL2VA_MANIFEST = MODEL_DIR / "manifests" / "minimax-h3-fl2va-768p.json"
T2VA_MANIFEST = MODEL_DIR / "manifests" / "minimax-h3-768p.json"
FIRST_IMAGE = MODEL_DIR / "data" / "fl2va-first.ppm"
LAST_IMAGE = MODEL_DIR / "data" / "fl2va-last.ppm"
BENCHMARK_EXCLUSION_REASON = (
    "Required input/audio parity profile; release-performance qualification is tracked "
    "separately and has not been recorded."
)


def _fl2va_cases():
    return load_model_manifest(FL2VA_MANIFEST).testcases


def test_fl2va_manifest_covers_all_four_required_parity_modes() -> None:
    manifest = json.loads(FL2VA_MANIFEST.read_text())
    qualified_thresholds = json.loads(
        (MODEL_DIR / "thresholds" / "minimax-h3-768p.json").read_text()
    )["threshold_overrides"]
    model = load_model_manifest(FL2VA_MANIFEST)

    assert model.bundle == "minimax-h3-fl2va-768p.bundle"
    assert manifest["benchmark_exclusion_reason"] == BENCHMARK_EXCLUSION_REASON
    assert [case.name for case in model.testcases] == [
        "minimax-h3-fl2va-zero-keyframes",
        "minimax-h3-fl2va-first-keyframe",
        "minimax-h3-fl2va-last-keyframe",
        "minimax-h3-fl2va-first-and-last-keyframes",
    ]
    assert [e2e_plugins.keyframe_mode(case) for case in model.testcases] == [
        "zero",
        "first",
        "last",
        "first_and_last",
    ]
    for case in model.testcases:
        assert case.inputs["workflow"] == "fl2va"
        assert len(case.stages) == 1
        assert case.stages[0].name == "end_to_end"
        assert case.stages[0].required is True
        assert "Required release-parity gate" in case.metadata["notes"]
        assert case.threshold_overrides == qualified_thresholds
        assert (MODEL_DIR / "thresholds" / f"{case.name}.json").is_file()
        assert case.metadata["build_cli_args"] == manifest["build_cli_args"]
        preflight_requirements = {
            (requirement.kind, tuple(sorted(requirement.args.items())), requirement.gating)
            for requirement in case.preflight
        }
        assert ("binary_exists", (), True) in preflight_requirements
        assert ("gpu_count_min", (("count", 1),), True) in preflight_requirements
        assert (
            "python_module_available",
            (("module", "diffusers"), ("phase", "reference")),
            True,
        ) in preflight_requirements
        assert (
            "python_module_available",
            (("module", "huggingface_hub"), ("phase", "reference")),
            True,
        ) in preflight_requirements
        preflight_assets = {
            requirement.args["path"]
            for requirement in case.preflight
            if requirement.kind == "asset_exists"
        }
        assert case.inputs["prompt_file"] in preflight_assets
        for input_name in ("first_image", "last_image"):
            if input_name in case.inputs:
                assert case.inputs[input_name] in preflight_assets
    assert manifest["build_cli_args"][0] == {
        "flag": "--set",
        "value": "minimax_h3.workflow=fl2va",
    }
    assert all(
        item.get("value") != "minimax_h3.first_block_cache=true"
        for item in manifest["build_cli_args"]
    )
    assert "threshold_overrides" not in manifest

    t2va = load_model_manifest(T2VA_MANIFEST).testcases[0]
    assert t2va.inputs["workflow"] == "t2va"
    assert t2va.stages[0].required is True


def test_fl2va_keyframe_fixtures_are_small_owned_distinct_rgb_images() -> None:
    images = []
    for path in (FIRST_IMAGE, LAST_IMAGE):
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
            assert source.size == (4, 4)
            assert source.format == "PPM"
    assert images[0].tobytes() != images[1].tobytes()


@pytest.mark.parametrize(
    ("case_index", "expected_flags"),
    [
        (0, []),
        (1, ["--first-image"]),
        (2, ["--last-image"]),
        (3, ["--first-image", "--last-image"]),
    ],
)
def test_native_and_hf_wrappers_forward_keyframes_in_first_then_last_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_index: int,
    expected_flags: list[str],
) -> None:
    case = _fl2va_cases()[case_index]
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "libtrtmc_model_minimax_h3.so").write_bytes(b"plugin")
    ctx = RunContext(
        case=case,
        binary_path=str(tmp_path / "trtmc"),
        model_plugin_dir=str(plugin_dir),
    )
    monkeypatch.setattr(native_runner, "source_revision", lambda *_args: "a" * 40)

    native = native_runner.build_native_command(
        case,
        ctx,
        tmp_path / "native-output",
        resolved_bundle=tmp_path / "fl2va.bundle",
    )
    reference = reference_plugin.build_hf_command(
        case,
        ctx,
        tmp_path / "hf-output",
        resolved_model=tmp_path / "snapshot",
        revision="a" * 40,
    )

    assert reference[reference.index("--workflow") + 1] == "fl2va"
    native_flags = [flag for flag in ("--first-image", "--last-image") if flag in native]
    reference_flags = [flag for flag in ("--first-image", "--last-image") if flag in reference]
    assert native_flags == expected_flags
    assert reference_flags == expected_flags
    if len(expected_flags) == 2:
        assert native.index("--first-image") < native.index("--last-image")
        assert reference.index("--first-image") < reference.index("--last-image")
    for flag, expected in (
        ("--first-image", FIRST_IMAGE),
        ("--last-image", LAST_IMAGE),
    ):
        if flag in expected_flags:
            native_path = Path(native[native.index(flag) + 1])
            reference_path = Path(reference[reference.index(flag) + 1])
            assert native_path.suffix == ".png"
            assert reference_path.suffix == ".png"
            assert native_path.read_bytes() == reference_path.read_bytes()
            with Image.open(expected) as source, Image.open(native_path) as materialized:
                assert materialized.format == "PNG"
                assert materialized.convert("RGB").tobytes() == source.convert("RGB").tobytes()


@pytest.mark.parametrize(
    ("first", "last", "expected_keys", "expected_mode"),
    [
        (None, None, [], "zero"),
        (object(), None, ["image"], "first"),
        (None, object(), ["last_image"], "last"),
        (object(), object(), ["image", "last_image"], "first_and_last"),
    ],
)
def test_hf_pipeline_arguments_use_official_image_and_last_image_names(
    first,
    last,
    expected_keys: list[str],
    expected_mode: str,
) -> None:
    generator = object()
    arguments = hf_reference.pipeline_arguments(
        prompt="prompt",
        generator=generator,
        steps=50,
        output_type="np",
        image=first,
        last_image=last,
    )

    conditioned_keys = [key for key in arguments if key in ("image", "last_image")]
    assert conditioned_keys == expected_keys
    assert "first_image" not in arguments
    assert arguments["generator"] is generator
    assert arguments["output"] == ["videos", "audio", "sampling_rate"]
    assert hf_reference._keyframe_mode(first, last) == expected_mode


def test_native_cli_keyframe_flags_and_mode_cover_zero_first_last_and_both() -> None:
    first = FIRST_IMAGE.resolve()
    last = LAST_IMAGE.resolve()
    from tests.e2e.models.minimax_h3 import native_reference

    assert native_reference.keyframe_cli_args(None, None) == []
    assert native_reference.keyframe_cli_args(first, None) == ["--first-image", str(first)]
    assert native_reference.keyframe_cli_args(None, last) == ["--last-image", str(last)]
    assert native_reference.keyframe_cli_args(first, last) == [
        "--first-image",
        str(first),
        "--last-image",
        str(last),
    ]
    assert [
        native_reference.keyframe_mode(*pair)
        for pair in ((None, None), (first, None), (None, last), (first, last))
    ] == ["zero", "first", "last", "first_and_last"]


def test_fl2va_bundle_source_revision_binds_workflow_partition_plans_and_assets(
    tmp_path: Path,
) -> None:
    case = _fl2va_cases()[3]
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    config = {
        "workflow": "fl2va",
        "checkpoint_partition": "transformer",
        "source_revision": "a" * 40,
        "builder_source_sha256": "b" * 64,
        "checkpoint_inventory_sha256": "c" * 64,
        "context_parallel_size": 1,
        "padded_sequence_length": 38247,
        "vae_tile_batch": 28,
        "first_block_cache": False,
        "denoiser_cache_mode": "monolithic",
        "min_text_rows": 1,
        "max_text_rows": 4096,
        "fl2va_keyframe_counts": [0, 1, 2],
        "fl2va_keyframe_rows": 1008,
        "fl2va_vae_tile_size": 256,
        "fl2va_vae_tile_min_overlap": 64,
        "fl2va_vae_temporal_frames": [1],
        "processor_asset_sections": list(FL2VA_PROCESSOR_ASSET_SECTIONS),
        "plan_sha256": {filename: "d" * 64 for filename in FL2VA_PLAN_FILENAMES},
        "asset_sha256": {
            name: "e" * 64 for name in ("tokenizer.json", *FL2VA_PROCESSOR_ASSET_SECTIONS)
        },
    }
    write_bundle(
        engine_dir / case.bundle,
        BundleInfo(model_id="MiniMaxAI/MiniMax-H3"),
        [BundleSection("config.json", json.dumps(config).encode())],
    )
    ctx = RunContext(case=case, engine_dir=str(engine_dir))

    assert e2e_plugins.source_revision(case, ctx) == "a" * 40

    config["fl2va_vae_temporal_frames"] = [1, 17]
    write_bundle(
        engine_dir / case.bundle,
        BundleInfo(model_id="MiniMaxAI/MiniMax-H3"),
        [BundleSection("config.json", json.dumps(config).encode())],
    )
    with pytest.raises(ValueError, match="FL2VA bundle profile"):
        e2e_plugins.source_revision(case, ctx)

    config["fl2va_vae_temporal_frames"] = [1]
    config["checkpoint_partition"] = "transformer_ref"
    write_bundle(
        engine_dir / case.bundle,
        BundleInfo(model_id="MiniMaxAI/MiniMax-H3"),
        [BundleSection("config.json", json.dumps(config).encode())],
    )
    with pytest.raises(ValueError, match="checkpoint partition"):
        e2e_plugins.source_revision(case, ctx)
