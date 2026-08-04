# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the four-rank native H3 pipeline and preserve decoded frames and timing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

import numpy as np
from PIL import Image


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
    args = parser.parse_args()
    output = Path(args.output_dir)
    frames_dir = output / "frames"
    output.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    prompt_spec = json.loads(Path(args.prompt_file).read_text())
    rendezvous = output / "nccl-rendezvous.bin"
    rendezvous.unlink(missing_ok=True)
    command = [
        args.trtf,
        "generate-video",
        args.bundle,
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
    environment["TRTMC_MODEL_PLUGIN_DIR"] = str(Path(args.plugin_dir).resolve())
    environment["TRTMC_NCCL_RENDEZVOUS"] = str(rendezvous.resolve())
    environment["TRTMC_PNG_WRITE_WORKERS"] = "8"
    started = time.perf_counter()
    processes = []
    log_handles = []
    for rank in range(4):
        rank_environment = environment.copy()
        rank_environment["WORLD_SIZE"] = "4"
        rank_environment["RANK"] = str(rank)
        stdout_handle = (output / f"native_rank{rank}_stdout.txt").open("w")
        stderr_handle = (output / f"native_rank{rank}_stderr.txt").open("w")
        log_handles.extend((stdout_handle, stderr_handle))
        processes.append(
            subprocess.Popen(
                command, env=rank_environment, text=True, stdout=stdout_handle, stderr=stderr_handle
            )
        )
    returncodes = [process.wait() for process in processes]
    for handle in log_handles:
        handle.close()
    elapsed = time.perf_counter() - started
    for rank, returncode in enumerate(returncodes):
        if returncode:
            raise RuntimeError(f"Native H3 rank {rank} failed ({returncode}); see {output}")
    paths = sorted(frames_dir.glob("frame_*.png"))
    if len(paths) != 124:
        raise RuntimeError(f"Native H3 returned {len(paths)} frames instead of 124")
    frames = np.stack([np.asarray(Image.open(path), dtype=np.float32) / 255.0 for path in paths])
    np.save(output / "trt_frames.npy", frames)
    rank0_stderr = (output / "native_rank0_stderr.txt").read_text()
    matches = [match.groupdict() for match in PERF_PATTERN.finditer(rank0_stderr)]
    perf = {name + "_ms": float(value) for name, value in matches[-1].items()} if matches else {}
    engine_matches = [match.groupdict() for match in ENGINE_PATTERN.finditer(rank0_stderr)]
    engine_execute = {}
    if len(engine_matches) >= len(ENGINE_STAGES):
        selected = engine_matches[-len(ENGINE_STAGES) :]
        engine_execute = {
            f"{stage}_ms": float(match["execute"])
            for stage, match in zip(ENGINE_STAGES, selected, strict=True)
        }
        engine_execute["total_ms"] = sum(engine_execute.values())
    receipt = {
        "backend": "tensorrt_native_cp4",
        "checkpoint_revision": "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
        "world_size": 4,
        "wall_s": elapsed,
        "runtime": perf,
        "engine_execute": engine_execute,
        "runtime_includes_plan_deserialization": True,
        "collective_transport": (
            "host_staged_diagnostic" if environment.get("NCCL_P2P_DISABLE") == "1" else "native"
        ),
        "shape": list(frames.shape),
        "host": platform.node(),
        "command": command,
    }
    (output / "trt_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
