# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the single-device native H3 pipeline and preserve frames and timing."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tests.e2e.models.minimax_h3.audio_metrics import (
        EXPECTED_AUDIO_SAMPLE_RATE,
        audio_summary,
        read_float32_wav,
        validate_fixed_audio,
    )
except ModuleNotFoundError:  # Direct execution exposes the sibling script directory.
    from audio_metrics import (  # type: ignore[no-redef]
        EXPECTED_AUDIO_SAMPLE_RATE,
        audio_summary,
        read_float32_wav,
        validate_fixed_audio,
    )
from tensorrt_model_connect.families.minimax_h3.provenance import (
    CHECKPOINT_REVISION,
    atomic_write_json,
    file_identity,
    ref2va_input_specification_record,
    stable_file_record,
    validate_file_identity,
    validate_native_bundle_config,
    validate_source_revision,
)

PERF_PATTERN = re.compile(
    r"\[minimax-h3\.perf\] text_encoder_ms=(?P<text>[0-9.]+) "
    r"adaln_ms=(?P<adaln>[0-9.]+) denoiser_ms=(?P<denoiser>[0-9.]+) "
    r"vae_decoder_ms=(?P<vae>[0-9.]+) "
    r"audio_vae_decoder_ms=(?P<audio_vae>[0-9.]+) total_ms=(?P<total>[0-9.]+)"
)
FL2VA_PERF_PATTERN = re.compile(
    r"\[minimax-h3\.fl2va\.perf\] language_ms=(?P<language>[0-9.]+) "
    r"condition_ms=(?P<condition>[0-9.]+) adaln_ms=(?P<adaln>[0-9.]+) "
    r"denoiser_ms=(?P<denoiser>[0-9.]+) vae_decoder_ms=(?P<vae>[0-9.]+) "
    r"audio_vae_decoder_ms=(?P<audio_vae>[0-9.]+) total_ms=(?P<total>[0-9.]+) "
    r"keyframes=(?P<keyframes>[0-9]+) text_rows=(?P<text_rows>[0-9]+) "
    r"full_denoiser_steps=(?P<full_denoiser_steps>[0-9]+)"
)
REF2VA_PERF_PATTERN = re.compile(
    r"\[minimax-h3\.ref2va\.perf\] prepare_ms=(?P<prepare>[0-9.]+) "
    r"language_ms=(?P<language>[0-9.]+) condition_ms=(?P<condition>[0-9.]+) "
    r"adaln_ms=(?P<adaln>[0-9.]+) denoiser_ms=(?P<denoiser>[0-9.]+) "
    r"vae_decoder_ms=(?P<vae>[0-9.]+) "
    r"audio_vae_decoder_ms=(?P<audio_vae>[0-9.]+) total_ms=(?P<total>[0-9.]+) "
    r"references=(?P<references>[0-9]+) text_rows=(?P<text_rows>[0-9]+) "
    r"condition_video_rows=(?P<condition_video_rows>[0-9]+) "
    r"condition_audio_rows=(?P<condition_audio_rows>[0-9]+) "
    r"full_denoiser_steps=(?P<full_denoiser_steps>[0-9]+)"
)
ENGINE_PATTERN = re.compile(
    r'\[trtmc\.engine_timing\] label="(?P<label>[^"]+)" execute_ms=(?P<execute>[0-9.]+) '
    r"launches=(?P<launches>[0-9]+)"
)
BACKEND_PATTERN = re.compile(
    r"\[trtmc\] Backend loaded: [^\n]* \((?P<dso>libtrtmc_backend_[^)]+\.so)\)"
)
CACHE_THRESHOLD_PATTERN = re.compile(
    r"\[minimax-h3\.perf\][^\n]* cache_threshold=(?P<threshold>[0-9.]+)"
)
CACHE_THRESHOLD_CONFIG_KEY = "minimax_h3.first_block_cache_threshold"
_REFERENCE_FLAGS = {
    "--reference-image": "image",
    "--reference-video": "video",
    "--reference-audio": "audio",
}


class _OrderedReferenceAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        del parser
        references = list(getattr(namespace, self.dest, None) or [])
        references.append((_REFERENCE_FLAGS[str(option_string)], str(values)))
        setattr(namespace, self.dest, references)


def evict_file_pages(path: Path) -> dict[str, bool | str]:
    """Best-effort eviction of clean cache pages for one file only."""

    posix_fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if posix_fadvise is None or dontneed is None:
        return {"supported": False, "attempted": False, "succeeded": False}

    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            posix_fadvise(descriptor, 0, 0, dontneed)
        finally:
            os.close(descriptor)
    except OSError as error:
        return {
            "supported": True,
            "attempted": True,
            "succeeded": False,
            "error": f"{type(error).__name__}: {error}",
        }
    return {"supported": True, "attempted": True, "succeeded": True}


def cache_threshold_cli_args(value: float | None) -> list[str]:
    if value is None:
        return []
    return ["--set", f"{CACHE_THRESHOLD_CONFIG_KEY}={value:.9g}"]


def keyframe_mode(first_image: Path | None, last_image: Path | None) -> str:
    if first_image is not None and last_image is not None:
        return "first_and_last"
    if first_image is not None:
        return "first"
    if last_image is not None:
        return "last"
    return "zero"


def keyframe_cli_args(first_image: Path | None, last_image: Path | None) -> list[str]:
    """Preserve MiniMax-H3 first/last keyframe semantics at the CLI boundary."""

    result = []
    if first_image is not None:
        result.extend(("--first-image", str(first_image)))
    if last_image is not None:
        result.extend(("--last-image", str(last_image)))
    return result


def reference_cli_args(references: list[tuple[str, Path]]) -> list[str]:
    """Preserve heterogeneous Ref2VA reference order at the native CLI boundary."""

    flags = {kind: flag for flag, kind in _REFERENCE_FLAGS.items()}
    result = []
    for kind, path in references:
        result.extend((flags[kind], str(path)))
    return result


def resolve_trt_backend_dso(executable: Path, bundle_config: dict) -> Path:
    """Resolve the exact adjacent backend DSO selected by the runtime loader."""

    if bundle_config.get("engine_backend") != "trt":
        raise ValueError("MiniMax-H3 native evidence requires engine_backend=trt")
    abi = bundle_config.get("trt_abi")
    match = re.fullmatch(r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)", str(abi))
    if match is None:
        raise ValueError("MiniMax-H3 bundle config has an invalid TensorRT ABI")
    names = (
        f"libtrtmc_backend_trt_{match.group('major')}_{match.group('minor')}.so",
        "libtrtmc_backend_trt.so",
    )
    for name in names:
        candidate = executable.parent / name
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise FileNotFoundError(
        "MiniMax-H3 could not bind the adjacent TensorRT backend DSO: "
        + ", ".join(str(executable.parent / name) for name in names)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--trtf", required=True)
    parser.add_argument("--plugin-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--first-image")
    parser.add_argument("--last-image")
    for flag in _REFERENCE_FLAGS:
        parser.add_argument(flag, dest="reference_specs", action=_OrderedReferenceAction)
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="forward CUDA graph enablement to the native TRTMC runtime",
    )
    parser.add_argument(
        "--cache-threshold",
        type=float,
        help=f"override {CACHE_THRESHOLD_CONFIG_KEY} for this visual run",
    )
    args = parser.parse_args()
    if args.cache_threshold is not None and (
        not math.isfinite(args.cache_threshold) or args.cache_threshold <= 0.0
    ):
        raise ValueError("cache threshold must be finite and positive")
    source_revision = validate_source_revision(args.source_revision)
    bundle = Path(args.bundle).resolve(strict=True)
    bundle_identity = file_identity(bundle)
    trtf = Path(args.trtf).resolve(strict=True)
    plugin_dir = Path(args.plugin_dir).resolve(strict=True)
    plugin = (plugin_dir / "libtrtmc_model_minimax_h3.so").resolve(strict=True)
    prompt_path = Path(args.prompt_file).resolve(strict=True)
    prompt_identity = file_identity(prompt_path)
    prompt_spec = json.loads(prompt_path.read_text())
    prompt_record, prompt_hashed_identity = stable_file_record(prompt_path, "prompt file")
    if prompt_hashed_identity != prompt_identity:
        raise ValueError("MiniMax-H3 prompt file changed while it was being read")
    if not isinstance(prompt_spec.get("prompt"), str) or not prompt_spec["prompt"]:
        raise ValueError("MiniMax-H3 prompt file must contain a non-empty prompt")
    if not isinstance(prompt_spec.get("seed"), int) or isinstance(prompt_spec["seed"], bool):
        raise ValueError("MiniMax-H3 prompt file must contain an integer seed")
    bundle_config = validate_native_bundle_config(bundle, source_revision=source_revision)
    first_image = Path(args.first_image).resolve(strict=True) if args.first_image else None
    last_image = Path(args.last_image).resolve(strict=True) if args.last_image else None
    references = [
        (kind, Path(value).resolve(strict=True)) for kind, value in (args.reference_specs or [])
    ]
    workflow = bundle_config.get("workflow", "t2va")
    if workflow == "ref2va":
        if first_image is not None or last_image is not None:
            raise ValueError("MiniMax-H3 Ref2VA bundle does not accept FL2VA keyframes")
        if not references:
            raise ValueError("MiniMax-H3 Ref2VA bundle requires ordered references")
    elif references:
        raise ValueError(f"MiniMax-H3 {workflow.upper()} bundle does not accept omni-references")
    elif workflow == "t2va" and (first_image is not None or last_image is not None):
        raise ValueError("MiniMax-H3 T2VA bundle does not accept keyframe images")
    backend = resolve_trt_backend_dso(trtf, bundle_config)
    script_path = Path(__file__).resolve()
    bound_paths = {
        "bundle": bundle,
        "trtf": trtf,
        "trt_backend": backend,
        "plugin": plugin,
        "prompt_file": prompt_path,
        "native_reference": script_path,
    }
    if first_image is not None:
        bound_paths["first_image"] = first_image
    if last_image is not None:
        bound_paths["last_image"] = last_image
    inputs = {}
    identities = {}
    for label, path in bound_paths.items():
        if label == "prompt_file":
            inputs[label] = prompt_record
            identities[label] = prompt_hashed_identity
        else:
            inputs[label], identities[label] = stable_file_record(path, label)
    if identities["bundle"] != bundle_identity:
        raise ValueError("MiniMax-H3 bundle changed while its config was being read")
    reference_records = []
    reference_identities = []
    for index, (kind, path) in enumerate(references):
        record, identity = stable_file_record(path, f"reference {index} {kind}")
        reference_records.append({"kind": kind, **record})
        reference_identities.append(identity)
    if reference_records:
        inputs["references"] = reference_records
    workload = {
        "prompt": prompt_spec["prompt"],
        "seed": int(prompt_spec["seed"]),
        "workflow": workflow,
        "keyframe_mode": keyframe_mode(first_image, last_image),
        "height": 768,
        "width": 1344,
        "num_frames": 124,
        "num_inference_steps": 50,
        "output_type": "decoded_png_frames_and_stereo_float32_audio",
    }
    if references:
        workload["reference_kinds"] = [kind for kind, _path in references]
    output = Path(args.output_dir)
    frames_dir = output / "frames"
    audio_wav_path = output / "audio.wav"
    output.mkdir(parents=True, exist_ok=True)
    for stale in (
        output / "trt_receipt.json",
        output / "trt_frames.npy",
        output / "trt_audio.npy",
        audio_wav_path,
    ):
        stale.unlink(missing_ok=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    command = [
        str(trtf),
        "generate-video",
        str(bundle),
        "--prompt",
        prompt_spec["prompt"],
        "--output",
        str(frames_dir),
        "--seed",
        str(prompt_spec["seed"]),
        "--num-steps",
        "50",
        "--height",
        "768",
        "--width",
        "1344",
        "--audio-output",
        str(audio_wav_path),
    ]
    command.extend(keyframe_cli_args(first_image, last_image))
    command.extend(reference_cli_args(references))
    if args.cuda_graphs:
        command.append("--cuda-graphs")
    command.extend(cache_threshold_cli_args(args.cache_threshold))
    environment = os.environ.copy()
    environment["TRTMC_MODEL_PLUGIN_DIR"] = str(plugin_dir)
    environment["TRTMC_PNG_WRITE_WORKERS"] = "8"
    environment["WORLD_SIZE"] = "1"
    environment["RANK"] = "0"
    bundle_page_cache_eviction = evict_file_pages(bundle)
    started = time.perf_counter()
    stdout_path = output / "native_stdout.txt"
    stderr_path = output / "native_stderr.txt"
    with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
        returncode = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        ).returncode
    elapsed = time.perf_counter() - started
    if returncode:
        raise RuntimeError(f"Native H3 single-device run failed ({returncode}); see {output}")
    for label, path in bound_paths.items():
        validate_file_identity(path, identities[label], label)
    for index, ((_kind, path), identity) in enumerate(zip(references, reference_identities)):
        validate_file_identity(path, identity, f"reference {index}")
    paths = sorted(frames_dir.glob("frame_*.png"))
    if len(paths) != 124:
        raise RuntimeError(f"Native H3 returned {len(paths)} frames instead of 124")
    frames = np.stack([np.asarray(Image.open(path), dtype=np.float32) / 255.0 for path in paths])
    frames_path = output / "trt_frames.npy"
    np.save(frames_path, frames, allow_pickle=False)
    frames_record, _ = stable_file_record(frames_path, "native decoded frames")
    if not audio_wav_path.is_file():
        raise RuntimeError("Native H3 did not produce the required audio.wav artifact")
    native_wav = read_float32_wav(audio_wav_path)
    audio = validate_fixed_audio(
        native_wav.samples,
        native_wav.sample_rate,
        label="native",
    )
    audio_path = output / "trt_audio.npy"
    np.save(audio_path, audio, allow_pickle=False)
    audio_record, _ = stable_file_record(audio_path, "native decoded audio")
    audio_wav_record, _ = stable_file_record(audio_wav_path, "native decoded audio WAV")
    audio_evidence = audio_summary(audio, EXPECTED_AUDIO_SAMPLE_RATE)
    native_stderr = stderr_path.read_text()
    loaded_backends = [match.group("dso") for match in BACKEND_PATTERN.finditer(native_stderr)]
    if loaded_backends != [backend.name]:
        raise RuntimeError(
            "Native H3 runtime did not load the provenance-bound TensorRT backend DSO"
        )
    if workflow == "ref2va":
        matches = [match.groupdict() for match in REF2VA_PERF_PATTERN.finditer(native_stderr)]
        if matches:
            latest = matches[-1]
            perf = {
                f"{name}_ms": float(latest[name])
                for name in (
                    "prepare",
                    "language",
                    "condition",
                    "adaln",
                    "denoiser",
                    "vae",
                    "audio_vae",
                    "total",
                )
            }
            perf.update(
                {
                    name: int(latest[name])
                    for name in (
                        "references",
                        "text_rows",
                        "condition_video_rows",
                        "condition_audio_rows",
                        "full_denoiser_steps",
                    )
                }
            )
        else:
            raise RuntimeError("Native H3 Ref2VA run did not emit its required performance receipt")
    elif workflow == "fl2va":
        matches = [match.groupdict() for match in FL2VA_PERF_PATTERN.finditer(native_stderr)]
        if not matches:
            raise RuntimeError("Native H3 FL2VA run did not emit its required performance receipt")
        latest = matches[-1]
        perf = {
            f"{name}_ms": float(latest[name])
            for name in ("language", "condition", "adaln", "denoiser", "vae", "audio_vae", "total")
        }
        perf.update(
            {name: int(latest[name]) for name in ("keyframes", "text_rows", "full_denoiser_steps")}
        )
    else:
        matches = [match.groupdict() for match in PERF_PATTERN.finditer(native_stderr)]
        if not matches:
            raise RuntimeError("Native H3 T2VA run did not emit its required performance receipt")
        perf = {name + "_ms": float(value) for name, value in matches[-1].items()}
    threshold_matches = [
        float(match.group("threshold")) for match in CACHE_THRESHOLD_PATTERN.finditer(native_stderr)
    ]
    effective_cache_threshold = threshold_matches[-1] if threshold_matches else None
    if args.cache_threshold is not None and (
        effective_cache_threshold is None
        or not math.isclose(
            effective_cache_threshold, args.cache_threshold, rel_tol=0.0, abs_tol=1e-6
        )
    ):
        raise RuntimeError("Native H3 runtime did not apply the requested cache threshold")
    engine_matches = [match.groupdict() for match in ENGINE_PATTERN.finditer(native_stderr)]
    engine_execute: dict[str, float] = {}
    if engine_matches:
        for match in engine_matches:
            name = f"{match['label']}_ms"
            engine_execute[name] = engine_execute.get(name, 0.0) + float(match["execute"])
        engine_execute["total_ms"] = sum(engine_execute.values())
    receipt = {
        "backend": "tensorrt_native_single_device",
        "status": "passed",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_revision": source_revision,
        "checkpoint_inventory_sha256": bundle_config["checkpoint_inventory_sha256"],
        "builder_source_sha256": bundle_config["builder_source_sha256"],
        "workspace_limit_bytes": bundle_config["workspace_limit_bytes"],
        "plan_sha256": bundle_config["plan_sha256"],
        "inputs": inputs,
        "workload": workload,
        "world_size": 1,
        "cuda_graphs_requested": args.cuda_graphs,
        "cache_threshold_override": args.cache_threshold,
        "effective_cache_threshold": effective_cache_threshold,
        "wall_s": elapsed,
        "runtime": perf,
        "engine_execute": engine_execute,
        "loaded_backend_dso": loaded_backends[0],
        "runtime_includes_plan_deserialization": True,
        "collective_transport": "none",
        "shape": list(frames.shape),
        "frames": frames_record,
        "audio_shape": audio_evidence["shape"],
        "audio_sample_rate_hz": audio_evidence["sample_rate_hz"],
        "audio_num_samples_per_channel": audio_evidence["num_samples_per_channel"],
        "audio_duration_s": audio_evidence["duration_s"],
        "audio_all_finite": audio_evidence["all_finite"],
        "audio_rms": audio_evidence["rms"],
        "audio_peak_absolute": audio_evidence["peak_absolute"],
        "audio_layout": audio_evidence["layout"],
        "audio_encoding": audio_evidence["encoding"],
        "audio_wav_encoding": "ieee_float32le",
        "audio": audio_record,
        "audio_wav": audio_wav_record,
        "bundle_page_cache_eviction": bundle_page_cache_eviction,
        "host": platform.node(),
        "command": command,
    }
    if workflow == "ref2va":
        receipt["official_input_specification"] = ref2va_input_specification_record()
    atomic_write_json(output / "trt_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
