# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the single-device native H3 pipeline and preserve frames and timing."""

from __future__ import annotations

import argparse
import json
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
    r'\[trtmc\.engine_timing\] label="engine" execute_ms=(?P<execute>[0-9.]+) '
    r"launches=(?P<launches>[0-9]+)"
)
ENGINE_STAGES = ("text_encoder", "adaln", "denoiser", "vae_decoder")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--trtf", required=True)
    parser.add_argument("--plugin-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
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
    script_path = Path(__file__).resolve()
    bound_paths = {
        "bundle": bundle,
        "trtf": trtf,
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
        "num_frames": 124,
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
    environment = os.environ.copy()
    environment["TRTMC_MODEL_PLUGIN_DIR"] = str(plugin_dir)
    environment["TRTMC_PNG_WRITE_WORKERS"] = "8"
    environment["WORLD_SIZE"] = "1"
    environment["RANK"] = "0"
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
    if len(paths) != 124:
        raise RuntimeError(f"Native H3 returned {len(paths)} frames instead of 124")
    frames = np.stack([np.asarray(Image.open(path), dtype=np.float32) / 255.0 for path in paths])
    frames_path = output / "trt_frames.npy"
    np.save(frames_path, frames)
    frames_record, _ = stable_file_record(frames_path, "native decoded frames")
    native_stderr = stderr_path.read_text()
    matches = [match.groupdict() for match in PERF_PATTERN.finditer(native_stderr)]
    perf = {name + "_ms": float(value) for name, value in matches[-1].items()} if matches else {}
    engine_matches = [match.groupdict() for match in ENGINE_PATTERN.finditer(native_stderr)]
    engine_execute = {}
    if len(engine_matches) >= len(ENGINE_STAGES):
        selected = engine_matches[-len(ENGINE_STAGES) :]
        engine_execute = {
            f"{stage}_ms": float(match["execute"])
            for stage, match in zip(ENGINE_STAGES, selected, strict=True)
        }
        engine_execute["total_ms"] = sum(engine_execute.values())
    receipt = {
        "backend": "tensorrt_native_single_device",
        "status": "passed",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_revision": source_revision,
        "checkpoint_inventory_sha256": bundle_config["checkpoint_inventory_sha256"],
        "builder_source_sha256": bundle_config["builder_source_sha256"],
        "plan_sha256": bundle_config["plan_sha256"],
        "inputs": inputs,
        "workload": workload,
        "world_size": 1,
        "wall_s": elapsed,
        "runtime": perf,
        "engine_execute": engine_execute,
        "runtime_includes_plan_deserialization": True,
        "collective_transport": "none",
        "shape": list(frames.shape),
        "frames": frames_record,
        "host": platform.node(),
        "command": command,
    }
    atomic_write_json(output / "trt_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
