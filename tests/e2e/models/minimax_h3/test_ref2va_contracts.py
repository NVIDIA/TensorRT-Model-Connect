# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Ref2VA manifest, reference-order, fixture, and bundle contracts."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import wave

from PIL import Image
import pytest

from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
from tensorrt_model_connect.families.minimax_h3.config import (
    FL2VA_PROCESSOR_ASSET_SECTIONS,
    REF2VA_MAX_CONDITION_AUDIO_ROWS,
    REF2VA_MAX_CONDITION_VIDEO_ROWS,
    REF2VA_MAX_TEXT_ROWS,
    REF2VA_PLAN_FILENAMES,
)
from tests.e2e.models.minimax_h3 import e2e_plugins, hf_reference, native_reference
from tests.e2e.models.minimax_h3.e2e_plugins import reference as reference_plugin
from tests.e2e.models.minimax_h3.e2e_plugins import runner as native_runner
from tests.e2e_harness.contracts import RunContext
from tests.e2e_harness.manifest_loader import load_model_manifest


MODEL_DIR = Path(__file__).resolve().parent
REF2VA_MANIFEST = MODEL_DIR / "manifests" / "minimax-h3-ref2va-768p.json"
T2VA_MANIFEST = MODEL_DIR / "manifests" / "minimax-h3-768p.json"
FL2VA_MANIFEST = MODEL_DIR / "manifests" / "minimax-h3-fl2va-768p.json"
REFERENCE_FLAGS = {"--reference-image", "--reference-video", "--reference-audio"}
BENCHMARK_EXCLUSION_REASON = (
    "Optional conditioned plumbing profile; weight-backed HF/native parity and "
    "release-performance qualification have not been recorded."
)


def _ref2va_cases():
    return load_model_manifest(REF2VA_MANIFEST).testcases


def _reference_flag_pairs(command: list[str]) -> list[tuple[str, Path]]:
    return [
        (value, Path(command[index + 1]))
        for index, value in enumerate(command[:-1])
        if value in REFERENCE_FLAGS
    ]


def test_ref2va_manifest_is_optional_plumbing_with_visual_and_audio_gates() -> None:
    manifest = json.loads(REF2VA_MANIFEST.read_text(encoding="utf-8"))
    qualified_thresholds = json.loads(
        (MODEL_DIR / "thresholds" / "minimax-h3-768p.json").read_text()
    )["threshold_overrides"]
    model = load_model_manifest(REF2VA_MANIFEST)

    assert model.bundle == "minimax-h3-ref2va-768p.bundle"
    assert manifest["benchmark_exclusion_reason"] == BENCHMARK_EXCLUSION_REASON
    assert [case.name for case in model.testcases] == [
        "minimax-h3-ref2va-image-only",
        "minimax-h3-ref2va-video-with-soundtrack",
        "minimax-h3-ref2va-image-and-audio",
        "minimax-h3-ref2va-mixed-ordered",
    ]
    assert [
        [descriptor.kind for descriptor in e2e_plugins.reference_descriptors(case)]
        for case in model.testcases
    ] == [
        ["image"],
        ["video"],
        ["image", "audio"],
        ["audio", "image", "video", "image"],
    ]
    for case in model.testcases:
        assert case.inputs["workflow"] == "ref2va"
        assert case.stages[0].name == "end_to_end"
        assert case.stages[0].required is False
        assert "Plumbing-only" in case.metadata["notes"]
        assert "authorized" in case.metadata["notes"]
        assert case.threshold_overrides == qualified_thresholds
        assert (MODEL_DIR / "thresholds" / f"{case.name}.json").is_file()
        e2e_plugins.validate_fixed_profile(case)
    assert "threshold_overrides" not in manifest
    assert manifest["build_cli_args"][0] == {
        "flag": "--set",
        "value": "minimax_h3.workflow=ref2va",
    }
    assert all(
        item.get("value") != "minimax_h3.first_block_cache=true"
        for item in manifest["build_cli_args"]
    )
    assert load_model_manifest(T2VA_MANIFEST).testcases[0].stages[0].required is True
    assert all(
        case.stages[0].required is False for case in load_model_manifest(FL2VA_MANIFEST).testcases
    )


def test_reference_fixture_wav_is_generated_deterministically_and_not_checked_in(
    tmp_path: Path,
) -> None:
    case = _ref2va_cases()[2]
    first = e2e_plugins.materialize_reference_inputs(case, tmp_path / "first")
    second = e2e_plugins.materialize_reference_inputs(case, tmp_path / "second")
    first_audio = first[1].path
    second_audio = second[1].path

    assert not (MODEL_DIR / "data" / "ref2va-reference.wav").exists()
    assert first_audio.read_bytes() == second_audio.read_bytes()
    with wave.open(str(first_audio), "rb") as stream:
        assert stream.getnchannels() == 2
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == 32000
        assert stream.getnframes() == 64000
        assert stream.getnframes() / stream.getframerate() >= 2.0


def test_native_and_hf_wrappers_preserve_mixed_reference_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _ref2va_cases()[3]
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
        resolved_bundle=tmp_path / "ref2va.bundle",
    )
    reference = reference_plugin.build_hf_command(
        case,
        ctx,
        tmp_path / "hf-output",
        resolved_model=tmp_path / "snapshot",
        revision="a" * 40,
    )

    expected_flags = [
        "--reference-audio",
        "--reference-image",
        "--reference-video",
        "--reference-image",
    ]
    native_pairs = _reference_flag_pairs(native)
    reference_pairs = _reference_flag_pairs(reference)
    assert [flag for flag, _path in native_pairs] == expected_flags
    assert [flag for flag, _path in reference_pairs] == expected_flags
    assert reference[reference.index("--workflow") + 1] == "ref2va"
    assert all(path.is_file() for _flag, path in native_pairs)
    assert all(path.is_file() for _flag, path in reference_pairs)
    assert native_reference.reference_cli_args(
        [(flag.removeprefix("--reference-"), path) for flag, path in native_pairs]
    ) == [value for pair in native_pairs for value in (pair[0], str(pair[1]))]

    video_path = next(path for flag, path in native_pairs if flag == "--reference-video")
    video_manifest = json.loads(video_path.read_text(encoding="utf-8"))
    assert video_manifest["fps"] == 1
    assert len(video_manifest["frames"]) == 2
    assert len(video_manifest["frames"]) / video_manifest["fps"] == 2.0
    assert all(Path(value).suffix == ".png" for value in video_manifest["frames"])
    for value in video_manifest["frames"]:
        with Image.open(video_path.parent / value) as frame:
            assert frame.format == "PNG"
    assert (video_path.parent / video_manifest["audio"]).is_file()
    for pairs in (native_pairs, reference_pairs):
        for flag, path in pairs:
            if flag == "--reference-image":
                assert path.suffix == ".png"
                with Image.open(path) as image:
                    assert image.format == "PNG"


def test_hf_helper_emits_official_reference_objects_and_references_kwarg(
    tmp_path: Path,
) -> None:
    descriptors = e2e_plugins.materialize_reference_inputs(
        _ref2va_cases()[3],
        tmp_path,
    )

    class FakeMiniMaxH3Reference:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_load_image(path: str) -> str:
        return f"loaded:{Path(path).name}"

    references = hf_reference.build_official_references(
        [(descriptor.kind, descriptor.path) for descriptor in descriptors],
        FakeMiniMaxH3Reference,
        fake_load_image,
    )

    assert [set(reference.kwargs) for reference in references] == [
        {"audio", "sample_rate"},
        {"image"},
        {"video", "fps", "audio", "sample_rate"},
        {"image"},
    ]
    assert references[0].kwargs["sample_rate"] == 32000
    assert references[0].kwargs["audio"].shape == (2, 64000)
    assert references[2].kwargs["fps"] == 1.0
    assert references[2].kwargs["video"] == [
        "loaded:ref2va-subject.png",
        "loaded:ref2va-style.png",
    ]
    assert references[2].kwargs["sample_rate"] == 32000
    assert references[2].kwargs["audio"].shape == (2, 64000)
    arguments = hf_reference.pipeline_arguments(
        prompt="prompt",
        generator=object(),
        steps=50,
        output_type="np",
        references=references,
    )
    assert arguments["references"] is references
    assert "image" not in arguments
    assert "last_image" not in arguments


@pytest.mark.parametrize("module", [hf_reference, native_reference])
def test_reference_flag_parsers_preserve_heterogeneous_encounter_order(module) -> None:
    parser = argparse.ArgumentParser()
    for flag in module._REFERENCE_FLAGS:
        parser.add_argument(flag, dest="reference_specs", action=module._OrderedReferenceAction)

    parsed = parser.parse_args(
        [
            "--reference-audio",
            "voice.wav",
            "--reference-image",
            "subject.ppm",
            "--reference-video",
            "motion.json",
            "--reference-image",
            "style.ppm",
        ]
    )

    assert parsed.reference_specs == [
        ("audio", "voice.wav"),
        ("image", "subject.ppm"),
        ("video", "motion.json"),
        ("image", "style.ppm"),
    ]


def test_native_ref2va_perf_pattern_captures_condition_rows_and_reference_count() -> None:
    stderr = (
        "[minimax-h3.ref2va.perf] prepare_ms=1.0 language_ms=2.0 condition_ms=3.0 "
        "adaln_ms=4.0 denoiser_ms=5.0 vae_decoder_ms=6.0 "
        "audio_vae_decoder_ms=7.0 total_ms=28.0 references=4 text_rows=128 "
        "condition_video_rows=2048 condition_audio_rows=160 full_denoiser_steps=50"
    )

    match = native_reference.REF2VA_PERF_PATTERN.search(stderr)

    assert match is not None
    assert match.groupdict()["references"] == "4"
    assert match.groupdict()["condition_video_rows"] == "2048"
    assert match.groupdict()["condition_audio_rows"] == "160"


def test_ref2va_requires_visual_media_and_rejects_fl2va_keyframes() -> None:
    image_audio = _ref2va_cases()[2]
    audio_only = replace(
        image_audio,
        inputs={
            **image_audio.inputs,
            "references": [image_audio.inputs["references"][1]],
        },
    )
    with pytest.raises(ValueError, match="at least one image or video"):
        e2e_plugins.validate_fixed_profile(audio_only)

    without_references = replace(
        image_audio,
        inputs={key: value for key, value in image_audio.inputs.items() if key != "references"},
    )
    with pytest.raises(ValueError, match="require ordered references"):
        e2e_plugins.validate_fixed_profile(without_references)

    with_keyframe = replace(
        image_audio,
        inputs={
            **image_audio.inputs,
            "first_image": "tests/e2e/models/minimax_h3/data/fl2va-first.ppm",
        },
    )
    with pytest.raises(ValueError, match="cannot provide FL2VA keyframes"):
        e2e_plugins.validate_fixed_profile(with_keyframe)


def test_ref2va_bundle_source_revision_binds_partition_plans_assets_and_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTMC_MINIMAX_H3_SOURCE_REVISION", raising=False)
    case = _ref2va_cases()[0]
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    config = {
        "workflow": "ref2va",
        "checkpoint_partition": "transformer_ref",
        "source_revision": "a" * 40,
        "builder_source_sha256": "b" * 64,
        "checkpoint_inventory_sha256": "c" * 64,
        "context_parallel_size": 1,
        "padded_sequence_length": 38247,
        "vae_tile_batch": 28,
        "first_block_cache": False,
        "denoiser_cache_mode": "monolithic",
        "plan_sha256": {filename: "d" * 64 for filename in REF2VA_PLAN_FILENAMES},
        "asset_sha256": {
            name: "e" * 64 for name in ("tokenizer.json", *FL2VA_PROCESSOR_ASSET_SECTIONS)
        },
        "min_text_rows": 1,
        "opt_text_rows": 8192,
        "max_text_rows": REF2VA_MAX_TEXT_ROWS,
        "ref2va_max_condition_video_rows": REF2VA_MAX_CONDITION_VIDEO_ROWS,
        "ref2va_max_condition_audio_rows": REF2VA_MAX_CONDITION_AUDIO_ROWS,
        "ref2va_max_images": 9,
        "ref2va_max_videos": 3,
        "ref2va_max_audios": 3,
        "ref2va_max_references": 12,
        "ref2va_reference_min_seconds": 2,
        "ref2va_reference_max_seconds": 15,
        "ref2va_vae_tile_size": 256,
        "ref2va_vae_tile_min_overlap": 64,
        "ref2va_vae_temporal_frames": [1, 17],
        "processor_asset_sections": list(FL2VA_PROCESSOR_ASSET_SECTIONS),
    }
    bundle = engine_dir / case.bundle
    write_bundle(
        bundle,
        BundleInfo(model_id="MiniMaxAI/MiniMax-H3"),
        [BundleSection("config.json", json.dumps(config).encode())],
    )
    ctx = RunContext(case=case, engine_dir=str(engine_dir))

    assert e2e_plugins.source_revision(case, ctx) == "a" * 40

    config["checkpoint_partition"] = "transformer"
    write_bundle(
        bundle,
        BundleInfo(model_id="MiniMaxAI/MiniMax-H3"),
        [BundleSection("config.json", json.dumps(config).encode())],
    )
    with pytest.raises(ValueError, match="checkpoint partition"):
        e2e_plugins.source_revision(case, ctx)
