# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the MiniMax-H3 model-owned E2E plugins."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import wave

from tensorrt_model_connect.families.minimax_h3.config import (
    FL2VA_PLAN_FILENAMES,
    FL2VA_PROCESSOR_ASSET_SECTIONS,
    REF2VA_MAX_CONDITION_AUDIO_ROWS,
    REF2VA_MAX_CONDITION_VIDEO_ROWS,
    REF2VA_MAX_TEXT_ROWS,
    REF2VA_PLAN_FILENAMES,
)
from tests.e2e_harness.contracts import E2ECase, RunContext


MODEL_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = MODEL_DIR.parents[3]
_SOURCE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLAN_FILENAMES = {
    "text_encoder.plan",
    "adaln_precompute.plan",
    "denoiser.plan",
    "vae_tile_decoder.plan",
    "audio_vae_decoder.plan",
}
_FIRST_BLOCK_CACHE_PLAN_FILENAMES = {
    "text_encoder.plan",
    "adaln_precompute.plan",
    "denoiser_head.plan",
    "denoiser_tail.plan",
    "denoiser_finish.plan",
    "vae_tile_decoder.plan",
    "audio_vae_decoder.plan",
}
_FL2VA_PLAN_FILENAMES = set(FL2VA_PLAN_FILENAMES)
_REF2VA_PLAN_FILENAMES = set(REF2VA_PLAN_FILENAMES)
_REFERENCE_FLAGS = {
    "image": "--reference-image",
    "video": "--reference-video",
    "audio": "--reference-audio",
}
_REFERENCE_LIMITS = {"image": 9, "video": 3, "audio": 3}
_MAX_REFERENCES = 12


@dataclass(frozen=True)
class ReferenceDescriptor:
    """One resolved Ref2VA medium; list position is part of the request."""

    kind: str
    path: Path


def artifact_dir(ctx: RunContext, case: E2ECase, name: str) -> Path:
    root = Path(ctx.artifacts_dir) if ctx.artifacts_dir else Path("/tmp/e2e_artifacts")
    output = root / case.name / name
    output.mkdir(parents=True, exist_ok=True)
    return output


def resolve_owned_file(value: str) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [PROJECT_DIR / path, MODEL_DIR / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"MiniMax-H3 E2E input file does not exist: {value}")


def _owned_file_candidates(value: str) -> tuple[Path, ...]:
    path = Path(value)
    candidates = (path,) if path.is_absolute() else (PROJECT_DIR / path, MODEL_DIR / path)
    return tuple(candidate.absolute() for candidate in candidates)


def _resolve_reference_path(kind: str, value: str) -> Path:
    try:
        return resolve_owned_file(value)
    except FileNotFoundError:
        if kind != "audio":
            raise
    for candidate in _owned_file_candidates(value):
        recipe = Path(f"{candidate}.json")
        if recipe.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "MiniMax-H3 E2E audio reference is neither a WAV nor backed by a textual "
        f"fixture recipe: {value}"
    )


def reference_descriptors(case: E2ECase) -> tuple[ReferenceDescriptor, ...]:
    """Validate and resolve Ref2VA references without changing encounter order."""

    raw = case.inputs.get("references", [])
    if raw in (None, []):
        return ()
    if not isinstance(raw, list):
        raise ValueError("MiniMax-H3 references must be an ordered list")
    if len(raw) > _MAX_REFERENCES:
        raise ValueError(f"MiniMax-H3 accepts at most {_MAX_REFERENCES} references")

    resolved = []
    counts = {kind: 0 for kind in _REFERENCE_FLAGS}
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict) or set(entry) != {"kind", "path"}:
            raise ValueError(f"MiniMax-H3 references[{index}] must contain exactly kind and path")
        kind = entry.get("kind")
        path = entry.get("path")
        if kind not in _REFERENCE_FLAGS:
            raise ValueError(f"MiniMax-H3 references[{index}] has an invalid kind: {kind!r}")
        if not isinstance(path, str) or not path:
            raise ValueError(f"MiniMax-H3 references[{index}].path must be a non-empty string")
        counts[str(kind)] += 1
        if counts[str(kind)] > _REFERENCE_LIMITS[str(kind)]:
            raise ValueError(
                f"MiniMax-H3 accepts at most {_REFERENCE_LIMITS[str(kind)]} {kind} references"
            )
        resolved.append(ReferenceDescriptor(str(kind), _resolve_reference_path(str(kind), path)))
    if not counts["image"] and not counts["video"]:
        raise ValueError(
            "MiniMax-H3 audio references require at least one image or video reference"
        )
    return tuple(resolved)


def _load_audio_fixture_recipe(path: Path) -> dict:
    recipe_path = Path(f"{path}.json")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    expected_fields = {
        "format",
        "sample_rate_hz",
        "duration_seconds",
        "channels",
        "frequencies_hz",
        "amplitude",
    }
    if not isinstance(recipe, dict) or set(recipe) != expected_fields:
        raise ValueError(f"MiniMax-H3 audio fixture recipe has invalid fields: {recipe_path}")
    if recipe["format"] != "pcm_s16le_wav":
        raise ValueError("MiniMax-H3 audio fixture recipe must request pcm_s16le_wav")
    sample_rate = recipe["sample_rate_hz"]
    duration = recipe["duration_seconds"]
    channels = recipe["channels"]
    frequencies = recipe["frequencies_hz"]
    amplitude = recipe["amplitude"]
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("MiniMax-H3 audio fixture sample_rate_hz must be a positive integer")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(float(duration))
        or not 2.0 <= float(duration) <= 15.0
    ):
        raise ValueError("MiniMax-H3 audio fixture duration must be between 2 and 15 seconds")
    if channels not in (1, 2):
        raise ValueError("MiniMax-H3 audio fixture channels must be 1 or 2")
    if (
        not isinstance(frequencies, list)
        or len(frequencies) != channels
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in frequencies
        )
    ):
        raise ValueError("MiniMax-H3 audio fixture frequencies_hz must cover every channel")
    if (
        not isinstance(amplitude, (int, float))
        or isinstance(amplitude, bool)
        or not math.isfinite(float(amplitude))
        or not 0.0 < float(amplitude) <= 1.0
    ):
        raise ValueError("MiniMax-H3 audio fixture amplitude must be in (0, 1]")
    return recipe


def write_deterministic_reference_wav(recipe_source: Path, output: Path) -> Path:
    """Generate a small deterministic PCM WAV from a checked-in textual recipe."""

    recipe = _load_audio_fixture_recipe(recipe_source)
    sample_rate = int(recipe["sample_rate_hz"])
    frame_count = round(float(recipe["duration_seconds"]) * sample_rate)
    channels = int(recipe["channels"])
    peak = round(float(recipe["amplitude"]) * 32767)
    periods = [max(2, round(sample_rate / float(value))) for value in recipe["frequencies_hz"]]
    payload = bytearray()
    for frame in range(frame_count):
        for period in periods:
            sample = peak if frame % period < period // 2 else -peak
            payload.extend(struct.pack("<h", sample))
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(payload)
    return output.resolve(strict=True)


def _materialize_video_manifest(source: Path, output: Path) -> Path:
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not {"fps", "frames"} <= set(manifest):
        raise ValueError(f"MiniMax-H3 reference video manifest is invalid: {source}")
    if set(manifest) - {"fps", "frames", "audio"}:
        raise ValueError(f"MiniMax-H3 reference video manifest has unknown fields: {source}")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("MiniMax-H3 reference video manifest needs at least one frame")

    output.parent.mkdir(parents=True, exist_ok=True)
    for relative_value in frames:
        if not isinstance(relative_value, str) or not relative_value:
            raise ValueError("MiniMax-H3 reference video frame paths must be non-empty strings")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("MiniMax-H3 reference video frame paths must stay beside the manifest")
        source_frame = (source.parent / relative).resolve(strict=True)
        target_frame = output.parent / relative
        target_frame.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_frame, target_frame)

    audio_value = manifest.get("audio")
    if audio_value is not None:
        if not isinstance(audio_value, str) or not audio_value:
            raise ValueError("MiniMax-H3 reference video audio must be a non-empty path string")
        relative_audio = Path(audio_value)
        if relative_audio.is_absolute() or ".." in relative_audio.parts:
            raise ValueError("MiniMax-H3 reference video audio must stay beside the manifest")
        source_audio = (source.parent / relative_audio).resolve()
        target_audio = output.parent / relative_audio
        target_audio.parent.mkdir(parents=True, exist_ok=True)
        if source_audio.is_file():
            shutil.copyfile(source_audio, target_audio)
        else:
            write_deterministic_reference_wav(source_audio, target_audio)

    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output.resolve(strict=True)


def materialize_reference_inputs(
    case: E2ECase,
    output_dir: Path,
) -> tuple[ReferenceDescriptor, ...]:
    """Make every ordered descriptor a concrete native/HF-readable media path."""

    result = []
    fixture_root = output_dir / "reference_inputs"
    for index, descriptor in enumerate(reference_descriptors(case)):
        if descriptor.kind == "audio" and not descriptor.path.is_file():
            path = write_deterministic_reference_wav(
                descriptor.path,
                fixture_root / f"{index:02d}" / descriptor.path.name,
            )
        elif descriptor.kind == "video":
            path = _materialize_video_manifest(
                descriptor.path,
                fixture_root / f"{index:02d}" / descriptor.path.name,
            )
        else:
            path = descriptor.path.resolve(strict=True)
        result.append(ReferenceDescriptor(descriptor.kind, path))
    return tuple(result)


def reference_cli_args(descriptors: tuple[ReferenceDescriptor, ...]) -> list[str]:
    result = []
    for descriptor in descriptors:
        result.extend((_REFERENCE_FLAGS[descriptor.kind], str(descriptor.path)))
    return result


def bundle_path(case: E2ECase, ctx: RunContext) -> Path:
    path = Path(case.bundle)
    return path if path.is_absolute() else Path(ctx.engine_dir) / path


def validate_fixed_profile(case: E2ECase) -> None:
    expected = {
        "video_num_frames": 124,
        "video_height": 768,
        "video_width": 1344,
        "num_inference_steps": 50,
        "fps": 24,
        "audio_channels": 2,
        "audio_sample_rate_hz": 32000,
        "audio_num_samples_per_channel": 165600,
    }
    mismatches = {
        name: (case.inputs.get(name), value)
        for name, value in expected.items()
        if case.inputs.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported MiniMax-H3 fixed E2E profile: {mismatches}")
    selected_workflow = workflow(case)
    if selected_workflow == "t2va" and any(
        case.inputs.get(name) for name in ("first_image", "last_image")
    ):
        raise ValueError("MiniMax-H3 T2VA E2E cases cannot provide keyframe images")
    references = reference_descriptors(case)
    if selected_workflow == "ref2va":
        if any(case.inputs.get(name) for name in ("first_image", "last_image")):
            raise ValueError("MiniMax-H3 Ref2VA E2E cases cannot provide FL2VA keyframes")
        if not references:
            raise ValueError("MiniMax-H3 Ref2VA E2E cases require ordered references")
    elif references:
        raise ValueError(f"MiniMax-H3 {selected_workflow.upper()} E2E cases cannot use references")


def workflow(case: E2ECase) -> str:
    value = case.inputs.get("workflow", "t2va")
    if value not in ("t2va", "fl2va", "ref2va"):
        raise ValueError(f"Unsupported MiniMax-H3 E2E workflow: {value!r}")
    return str(value)


def keyframe_inputs(case: E2ECase) -> tuple[tuple[str, str, Path], ...]:
    """Resolve FL2VA keyframes in their semantic first-then-last order."""

    resolved = []
    for input_name, flag in (
        ("first_image", "--first-image"),
        ("last_image", "--last-image"),
    ):
        value = case.inputs.get(input_name)
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            raise ValueError(f"MiniMax-H3 {input_name} must be a file path string")
        resolved.append((input_name, flag, resolve_owned_file(value)))
    return tuple(resolved)


def keyframe_mode(case: E2ECase) -> str:
    names = {name for name, _flag, _path in keyframe_inputs(case)}
    if names == {"first_image", "last_image"}:
        return "first_and_last"
    if names == {"first_image"}:
        return "first"
    if names == {"last_image"}:
        return "last"
    return "zero"


def _bundle_config(path: Path) -> dict:
    with path.open("rb") as bundle:
        if bundle.read(8) != b"BUNDLE\x01\x00":
            raise ValueError(f"Not a valid TRTMC bundle: {path}")
        header_length = struct.unpack("<Q", bundle.read(8))[0]
        header = json.loads(bundle.read(header_length).decode("utf-8"))
        section = header.get("sections", {}).get("config.json")
        if not isinstance(section, dict):
            return {}
        bundle.seek(16 + header_length + int(section["offset"]))
        return json.loads(bundle.read(int(section["size"])).decode("utf-8"))


def source_revision(case: E2ECase, ctx: RunContext) -> str:
    """Resolve the exact source revision recorded by both E2E backends."""

    path = bundle_path(case, ctx)
    if not path.is_file():
        raise FileNotFoundError(f"MiniMax-H3 E2E bundle does not exist: {path}")
    config = _bundle_config(path)
    selected_workflow = workflow(case)
    if config.get("workflow", "t2va") != selected_workflow:
        raise ValueError("MiniMax-H3 E2E bundle workflow does not match the testcase")
    expected_partition = "transformer_ref" if selected_workflow == "ref2va" else "transformer"
    if config.get("checkpoint_partition", "transformer") != expected_partition:
        raise ValueError("MiniMax-H3 E2E bundle has the wrong checkpoint partition")
    revision = str(config.get("source_revision", "")).strip().lower()
    if _SOURCE_REVISION.fullmatch(revision) is None:
        raise ValueError("MiniMax-H3 bundle has no valid source_revision")
    if _SHA256.fullmatch(str(config.get("builder_source_sha256", ""))) is None:
        raise ValueError("MiniMax-H3 bundle has no valid builder_source_sha256")
    if _SHA256.fullmatch(str(config.get("checkpoint_inventory_sha256", ""))) is None:
        raise ValueError("MiniMax-H3 bundle has no valid checkpoint_inventory_sha256")
    if config.get("context_parallel_size") != 1:
        raise ValueError("MiniMax-H3 E2E bundle is not single-device")
    if config.get("padded_sequence_length") != 38247:
        raise ValueError("MiniMax-H3 E2E bundle does not use the unpadded sequence")
    if config.get("vae_tile_batch") != 28:
        raise ValueError("MiniMax-H3 E2E bundle does not decode all spatial tiles in one batch")
    cache_mode = config.get("denoiser_cache_mode", "monolithic")
    if cache_mode not in ("monolithic", "first_block"):
        raise ValueError("MiniMax-H3 E2E bundle has an invalid denoiser cache mode")
    first_block_cache = config.get("first_block_cache", False)
    if not isinstance(first_block_cache, bool) or first_block_cache != (
        cache_mode == "first_block"
    ):
        raise ValueError("MiniMax-H3 E2E bundle cache profile is inconsistent")
    if selected_workflow in {"fl2va", "ref2va"}:
        if first_block_cache:
            raise ValueError(
                f"MiniMax-H3 {selected_workflow.upper()} E2E bundle cannot enable first-block cache"
            )
        expected_plans = (
            _REF2VA_PLAN_FILENAMES if selected_workflow == "ref2va" else _FL2VA_PLAN_FILENAMES
        )
    else:
        expected_plans = _FIRST_BLOCK_CACHE_PLAN_FILENAMES if first_block_cache else _PLAN_FILENAMES
    plan_sha = config.get("plan_sha256")
    if not isinstance(plan_sha, dict) or set(plan_sha) != expected_plans:
        raise ValueError("MiniMax-H3 bundle does not identify the selected native plans")
    if any(_SHA256.fullmatch(str(value)) is None for value in plan_sha.values()):
        raise ValueError("MiniMax-H3 bundle contains an invalid native plan SHA256")
    if selected_workflow in {"fl2va", "ref2va"}:
        expected_assets = {"tokenizer.json", *FL2VA_PROCESSOR_ASSET_SECTIONS}
        asset_sha = config.get("asset_sha256")
        if not isinstance(asset_sha, dict) or set(asset_sha) != expected_assets:
            raise ValueError(
                f"MiniMax-H3 {selected_workflow.upper()} bundle does not identify every processor asset"
            )
        if any(_SHA256.fullmatch(str(value)) is None for value in asset_sha.values()):
            raise ValueError(
                f"MiniMax-H3 {selected_workflow.upper()} bundle contains an invalid asset SHA256"
            )
    if selected_workflow == "ref2va":
        expected_ref2va = {
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
        mismatches = {
            name: (config.get(name), value)
            for name, value in expected_ref2va.items()
            if config.get(name) != value
        }
        if mismatches:
            raise ValueError(f"MiniMax-H3 Ref2VA bundle profile is invalid: {mismatches}")

    explicit_revision = os.environ.get("TRTMC_MINIMAX_H3_SOURCE_REVISION", "").strip().lower()
    if explicit_revision:
        if _SOURCE_REVISION.fullmatch(explicit_revision) is None:
            raise ValueError("TRTMC_MINIMAX_H3_SOURCE_REVISION is not an exact Git SHA")
        if explicit_revision != revision:
            raise ValueError(
                "MiniMax-H3 bundle source_revision does not match TRTMC_MINIMAX_H3_SOURCE_REVISION"
            )
    return revision


def model_plugin_dir(ctx: RunContext) -> Path:
    candidates = []
    if ctx.model_plugin_dir:
        root = Path(ctx.model_plugin_dir)
        candidates.extend((root / "minimax_h3", root))
    candidates.append(PROJECT_DIR / "build" / "models" / "minimax_h3")
    for candidate in candidates:
        if (candidate / "libtrtmc_model_minimax_h3.so").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "MiniMax-H3 E2E requires libtrtmc_model_minimax_h3.so via "
        "--model-plugin-dir or build/models/minimax_h3"
    )


def subprocess_env(ctx: RunContext) -> dict[str, str]:
    env = os.environ.copy()
    if env.get("TRTMC_TEST_INSTALLED_WHEEL") != "1":
        python_path = str(PROJECT_DIR / "python")
        if env.get("PYTHONPATH"):
            python_path = f"{python_path}:{env['PYTHONPATH']}"
        env["PYTHONPATH"] = python_path
    if ctx.ld_library_path:
        env["LD_LIBRARY_PATH"] = ctx.ld_library_path
    return env
