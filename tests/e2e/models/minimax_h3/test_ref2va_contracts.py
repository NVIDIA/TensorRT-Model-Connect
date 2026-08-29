# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Ref2VA manifest, reference-order, fixture, and bundle contracts."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import wave

from PIL import Image
import pytest

from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
from tensorrt_model_connect.families.minimax_h3.config import (
    FL2VA_PROCESSOR_ASSET_SECTIONS,
    REF2VA_MAX_CONDITION_AUDIO_ROWS,
    REF2VA_MAX_CONDITION_VIDEO_ROWS,
    REF2VA_MAX_TEXT_ROWS,
    REF2VA_MIN_CONDITION_VIDEO_ROWS,
    REF2VA_OPT_CONDITION_VIDEO_ROWS,
    REF2VA_PLAN_FILENAMES,
)
from tensorrt_model_connect.families.minimax_h3.provenance import (
    ref2va_input_specification_record,
)
from tests.e2e.models.minimax_h3 import e2e_plugins, hf_reference, native_reference
from tests.e2e.models.minimax_h3.e2e_plugins import reference as reference_plugin
from tests.e2e.models.minimax_h3.e2e_plugins import runner as native_runner
from tests.e2e.models.minimax_h3.receipt_contracts import validate_ref2va_receipt_contract
from tests.e2e_harness.contracts import RunContext
from tests.e2e_harness.manifest_loader import load_model_manifest


MODEL_DIR = Path(__file__).resolve().parent
REF2VA_MANIFEST = MODEL_DIR / "manifests" / "minimax-h3-ref2va-768p.json"
T2VA_MANIFEST = MODEL_DIR / "manifests" / "minimax-h3-768p.json"
FL2VA_MANIFEST = MODEL_DIR / "manifests" / "minimax-h3-fl2va-768p.json"
REFERENCE_FLAGS = {"--reference-image", "--reference-video", "--reference-audio"}
BENCHMARK_EXCLUSION_REASON = (
    "Required input/audio parity profile; release-performance qualification is tracked "
    "separately and has not been recorded."
)


def _ref2va_cases():
    return load_model_manifest(REF2VA_MANIFEST).testcases


def _reference_flag_pairs(command: list[str]) -> list[tuple[str, Path]]:
    return [
        (value, Path(command[index + 1]))
        for index, value in enumerate(command[:-1])
        if value in REFERENCE_FLAGS
    ]


def test_ref2va_manifest_covers_all_five_required_parity_modes() -> None:
    manifest = json.loads(REF2VA_MANIFEST.read_text(encoding="utf-8"))
    qualified_thresholds = json.loads(
        (MODEL_DIR / "thresholds" / "minimax-h3-768p.json").read_text()
    )["threshold_overrides"]
    model = load_model_manifest(REF2VA_MANIFEST)

    assert model.bundle == "minimax-h3-ref2va-768p.bundle"
    assert manifest["benchmark_exclusion_reason"] == BENCHMARK_EXCLUSION_REASON
    assert manifest["official_input_specification"] == ref2va_input_specification_record()
    assert [case.name for case in model.testcases] == [
        "minimax-h3-ref2va-image-only",
        "minimax-h3-ref2va-video-with-soundtrack",
        "minimax-h3-ref2va-image-and-audio",
        "minimax-h3-ref2va-mixed-ordered",
        "minimax-h3-ref2va-audio-only",
    ]
    assert [
        [descriptor.kind for descriptor in e2e_plugins.reference_descriptors(case)]
        for case in model.testcases
    ] == [
        ["image"],
        ["video"],
        ["image", "audio"],
        ["audio", "image", "video", "image"],
        ["audio"],
    ]
    for case in model.testcases:
        assert case.inputs["workflow"] == "ref2va"
        assert case.stages[0].name == "end_to_end"
        assert case.stages[0].required is True
        assert "Required release-parity gate" in case.metadata["notes"]
        assert case.metadata["official_input_specification"] == (
            ref2va_input_specification_record()
        )
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
        case.stages[0].required is True for case in load_model_manifest(FL2VA_MANIFEST).testcases
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


def test_native_and_hf_wrappers_forward_audio_only_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _ref2va_cases()[4]
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

    for command in (native, reference):
        pairs = _reference_flag_pairs(command)
        assert [flag for flag, _path in pairs] == ["--reference-audio"]
        assert pairs[0][1].is_file()
        with wave.open(str(pairs[0][1]), "rb") as stream:
            assert stream.getnchannels() == 2
            assert stream.getframerate() == 32000
            assert stream.getnframes() == 64000


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


def test_hf_audio_only_compatibility_runs_upstream_and_suppresses_only_exact_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSetupStep:
        @staticmethod
        def _check_inputs(_components, block_state) -> None:
            if block_state.invalid:
                raise ValueError("another upstream validation failure")
            if {reference.kind for reference in block_state.references} == {"audio"}:
                raise ValueError(hf_reference._REF2VA_AUDIO_ONLY_GATE_ERROR)

    original_descriptor = inspect.getattr_static(FakeSetupStep, "_check_inputs")
    source = inspect.getsource(FakeSetupStep._check_inputs)
    monkeypatch.setattr(
        hf_reference,
        "_REF2VA_CHECK_INPUTS_SOURCE_SHA256",
        hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )
    state = SimpleNamespace(
        invalid=False,
        references=[SimpleNamespace(kind="audio")],
    )

    with hf_reference.ref2va_audio_only_compatibility(
        [("audio", Path("reference.wav"))], FakeSetupStep
    ) as record:
        assert FakeSetupStep._check_inputs(None, state) is None
        assert record["suppressed_calls"] == 1
        assert record["official_input_specification"] == ref2va_input_specification_record()
        state.invalid = True
        with pytest.raises(ValueError, match="another upstream validation failure"):
            FakeSetupStep._check_inputs(None, state)

    assert inspect.getattr_static(FakeSetupStep, "_check_inputs") is original_descriptor


def test_hf_audio_only_compatibility_rejects_source_drift_and_wrong_runtime_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSetupStep:
        @staticmethod
        def _check_inputs(_components, _block_state) -> None:
            raise ValueError(hf_reference._REF2VA_AUDIO_ONLY_GATE_ERROR)

    monkeypatch.setattr(hf_reference, "_REF2VA_CHECK_INPUTS_SOURCE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="input validator changed"):
        with hf_reference.ref2va_audio_only_compatibility(
            [("audio", Path("reference.wav"))], FakeSetupStep
        ):
            pass

    source = inspect.getsource(FakeSetupStep._check_inputs)
    monkeypatch.setattr(
        hf_reference,
        "_REF2VA_CHECK_INPUTS_SOURCE_SHA256",
        hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )
    wrong_state = SimpleNamespace(references=[SimpleNamespace(kind="image")])
    with hf_reference.ref2va_audio_only_compatibility(
        [("audio", Path("reference.wav"))], FakeSetupStep
    ):
        with pytest.raises(ValueError, match="cannot be used on its own"):
            FakeSetupStep._check_inputs(None, wrong_state)


def _audio_only_receipts(reference_count: int = 1) -> tuple[dict, dict]:
    specification = ref2va_input_specification_record()
    workload = {
        "workflow": "ref2va",
        "reference_kinds": ["audio"] * reference_count,
    }
    trt_receipt = {
        "workload": workload,
        "official_input_specification": specification,
        "runtime": {
            "references": reference_count,
            "condition_video_rows": 0,
            "condition_audio_rows": 160,
        },
        "engine_execute": {
            "language_conditioner_plan_ms": 1.0,
            "audio_vae_encoder_plan_ms": 1.0,
            "ref2va_denoiser_plan_ms": 1.0,
        },
    }
    ref_receipt = {
        "request": {**workload, "warmup": 0, "measure": 1},
        "official_input_specification": specification,
        "ref2va_audio_only_compatibility": {
            "name": "pinned-diffusers-ref2va-audio-only-input-gate",
            "diffusers_revision": hf_reference.DIFFUSERS_REVISION,
            "upstream_method": (
                "diffusers.modular_pipelines.minimax_h3.before_encoder."
                "MiniMaxH3Ref2VASetupStep._check_inputs"
            ),
            "upstream_method_source_sha256": (hf_reference._REF2VA_CHECK_INPUTS_SOURCE_SHA256),
            "suppressed_error": hf_reference._REF2VA_AUDIO_ONLY_GATE_ERROR,
            "suppressed_calls": 1,
            "scope": "audio-only Ref2VA requests",
            "official_input_specification": specification,
        },
    }
    return trt_receipt, ref_receipt


def _visual_routing_receipts(kinds: list[str]) -> tuple[dict, dict]:
    workload = {"workflow": "ref2va", "reference_kinds": kinds}
    engine_execute = {
        "language_conditioner_plan_ms": 1.0,
        "ref2va_denoiser_plan_ms": 1.0,
    }
    if "image" in kinds:
        engine_execute["vision_conditioner_image_plan_ms"] = 1.0
    if "video" in kinds:
        engine_execute["vision_conditioner_video_plan_ms"] = 1.0
    return (
        {
            "workload": workload,
            "runtime": {"references": len(kinds)},
            "engine_execute": engine_execute,
        },
        {"request": workload},
    )


@pytest.mark.parametrize(
    "kinds",
    [
        ["image"],
        ["video"],
        ["image", "audio"],
        ["audio", "image", "video", "image"],
    ],
)
def test_visual_receipts_bind_kind_specialized_vision_routing(kinds: list[str]) -> None:
    validate_ref2va_receipt_contract(*_visual_routing_receipts(kinds))


@pytest.mark.parametrize(
    ("kinds", "mutation"),
    [
        (["image"], "missing_image"),
        (["video"], "missing_video"),
        (["image"], "extra_video"),
        (["video"], "extra_image"),
        (["image"], "legacy"),
    ],
)
def test_visual_receipts_reject_wrong_vision_plan_routing(kinds: list[str], mutation: str) -> None:
    trt_receipt, ref_receipt = _visual_routing_receipts(kinds)
    timings = trt_receipt["engine_execute"]
    if mutation == "missing_image":
        del timings["vision_conditioner_image_plan_ms"]
    elif mutation == "missing_video":
        del timings["vision_conditioner_video_plan_ms"]
    elif mutation == "extra_video":
        timings["vision_conditioner_video_plan_ms"] = 1.0
    elif mutation == "extra_image":
        timings["vision_conditioner_image_plan_ms"] = 1.0
    else:
        timings["vision_conditioner_plan_ms"] = 1.0
    with pytest.raises(ValueError, match="vision engine routing"):
        validate_ref2va_receipt_contract(trt_receipt, ref_receipt)


def test_audio_only_receipts_bind_shim_and_zero_visual_engine_routing() -> None:
    trt_receipt, ref_receipt = _audio_only_receipts()
    validate_ref2va_receipt_contract(trt_receipt, ref_receipt)

    trt_receipt, ref_receipt = _audio_only_receipts(reference_count=2)
    validate_ref2va_receipt_contract(trt_receipt, ref_receipt)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("vision_rows", "engine routing"),
        ("vision_engine", "engine routing"),
        ("visual_vae_t1", "engine routing"),
        ("visual_vae_t17", "engine routing"),
        ("missing_audio_encoder", "engine routing"),
        ("wrong_shim_calls", "compatibility evidence"),
        ("missing_spec", "official input specification"),
        ("mismatched_kinds", "different Ref2VA inputs"),
    ],
)
def test_audio_only_receipts_reject_incomplete_runtime_evidence(
    mutation: str,
    message: str,
) -> None:
    trt_receipt, ref_receipt = _audio_only_receipts()
    if mutation == "vision_rows":
        trt_receipt["runtime"]["condition_video_rows"] = 1
    elif mutation == "vision_engine":
        trt_receipt["engine_execute"]["vision_conditioner_plan_ms"] = 1.0
    elif mutation == "visual_vae_t1":
        trt_receipt["engine_execute"]["vae_encoder_tile_t1_plan_ms"] = 1.0
    elif mutation == "visual_vae_t17":
        trt_receipt["engine_execute"]["vae_encoder_tile_t17_plan_ms"] = 1.0
    elif mutation == "missing_audio_encoder":
        del trt_receipt["engine_execute"]["audio_vae_encoder_plan_ms"]
    elif mutation == "wrong_shim_calls":
        ref_receipt["ref2va_audio_only_compatibility"]["suppressed_calls"] = 0
    elif mutation == "missing_spec":
        del trt_receipt["official_input_specification"]
    else:
        ref_receipt["request"]["reference_kinds"] = ["image", "audio"]

    with pytest.raises(ValueError, match=message):
        validate_ref2va_receipt_contract(trt_receipt, ref_receipt)


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


def test_ref2va_accepts_audio_only_and_still_rejects_missing_references_or_keyframes() -> None:
    image_audio = _ref2va_cases()[2]
    audio_only = replace(
        image_audio,
        inputs={
            **image_audio.inputs,
            "references": [image_audio.inputs["references"][1]],
        },
    )
    e2e_plugins.validate_fixed_profile(audio_only)
    assert [descriptor.kind for descriptor in e2e_plugins.reference_descriptors(audio_only)] == [
        "audio"
    ]

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
        "ref2va_min_condition_video_rows": REF2VA_MIN_CONDITION_VIDEO_ROWS,
        "ref2va_opt_condition_video_rows": REF2VA_OPT_CONDITION_VIDEO_ROWS,
        "ref2va_min_condition_audio_rows": 0,
        "ref2va_opt_condition_audio_rows": 0,
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
