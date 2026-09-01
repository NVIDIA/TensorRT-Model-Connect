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
from tensorrt_model_connect.families.minimax_h3.provenance import (
    CHECKPOINT_REVISION,
    atomic_write_json,
    file_identity,
    stable_file_record,
    validate_file_identity,
    validate_native_bundle_config,
    validate_source_revision,
)

PERF_PATTERN = re.compile(
    r"\[minimax-h3\.perf\] text_encoder_ms=(?P<text>[0-9.]+) "
    r"adaln_ms=(?P<adaln>[0-9.]+) denoiser_ms=(?P<denoiser>[0-9.]+) "
    r"vae_decoder_ms=(?P<vae>[0-9.]+) total_ms=(?P<total>[0-9.]+)"
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
EXPECTED_FRAME_COUNT = 124
EXPECTED_FRAME_SIZE = (1344, 768)


def parse_retained_frame_indices(value: str) -> tuple[int, ...]:
    """Parse a strict, ordered subset of the fixed MiniMax-H3 frame profile."""

    if not value:
        return ()
    try:
        indices = tuple(int(token) for token in value.split(","))
    except ValueError as error:
        raise ValueError("retained frame indices must be comma-separated integers") from error
    if not indices or tuple(sorted(set(indices))) != indices:
        raise ValueError("retained frame indices must be unique and strictly increasing")
    if indices[0] < 0 or indices[-1] >= EXPECTED_FRAME_COUNT:
        raise ValueError(f"retained frame indices must be within [0, {EXPECTED_FRAME_COUNT - 1}]")
    return indices


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
    parser.add_argument(
        "--retain-frame-indices",
        default="",
        help=(
            "retain only this comma-separated frame subset after validating all "
            "decoded frames; omits the full decoded NPY artifact"
        ),
    )
    args = parser.parse_args()
    retained_frame_indices = parse_retained_frame_indices(args.retain_frame_indices)
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
    workload = {
        "prompt": prompt_spec["prompt"],
        "seed": int(prompt_spec["seed"]),
        "height": 768,
        "width": 1344,
        "num_frames": EXPECTED_FRAME_COUNT,
        "num_inference_steps": 50,
        "output_type": "decoded_png_frames",
    }
    output = Path(args.output_dir)
    frames_dir = output / "frames"
    output.mkdir(parents=True, exist_ok=True)
    for stale in (output / "trt_receipt.json", output / "trt_frames.npy"):
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
    ]
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
    paths = sorted(frames_dir.glob("frame_*.png"))
    if len(paths) != EXPECTED_FRAME_COUNT:
        raise RuntimeError(
            f"Native H3 returned {len(paths)} frames instead of {EXPECTED_FRAME_COUNT}"
        )
    decoded_frames = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != EXPECTED_FRAME_SIZE:
                raise RuntimeError(
                    f"Native H3 frame {index} has mode/size {image.mode}/{image.size}; "
                    f"expected RGB/{EXPECTED_FRAME_SIZE}"
                )
            if not retained_frame_indices:
                decoded_frames.append(np.asarray(image, dtype=np.float32) / 255.0)
    frames_record = None
    if retained_frame_indices:
        retained = set(retained_frame_indices)
        for index, path in enumerate(paths):
            if index not in retained:
                path.unlink()
    else:
        frames = np.stack(decoded_frames)
        frames_path = output / "trt_frames.npy"
        np.save(frames_path, frames)
        frames_record, _ = stable_file_record(frames_path, "native decoded frames")
    native_stderr = stderr_path.read_text()
    loaded_backends = [match.group("dso") for match in BACKEND_PATTERN.finditer(native_stderr)]
    if loaded_backends != [backend.name]:
        raise RuntimeError(
            "Native H3 runtime did not load the provenance-bound TensorRT backend DSO"
        )
    matches = [match.groupdict() for match in PERF_PATTERN.finditer(native_stderr)]
    perf = {name + "_ms": float(value) for name, value in matches[-1].items()} if matches else {}
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
        "shape": [EXPECTED_FRAME_COUNT, EXPECTED_FRAME_SIZE[1], EXPECTED_FRAME_SIZE[0], 3],
        "retained_frame_indices": list(retained_frame_indices),
        "bundle_page_cache_eviction": bundle_page_cache_eviction,
        "host": platform.node(),
        "command": command,
    }
    if frames_record is not None:
        receipt["frames"] = frames_record
    atomic_write_json(output / "trt_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
