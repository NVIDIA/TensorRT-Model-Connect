#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the family-owned LeRobot ACT PyTorch reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from families.lerobot_act.tests.official_reference import (
    load_observation,
    load_policy,
    predict_actions,
)


def _checkpoint(model: str, revision: str | None, local_files_only: bool) -> Path:
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            allow_patterns=("config.json", "model.safetensors"),
            local_files_only=local_files_only,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.iterations < 1:
        parser.error("warmup must be non-negative and iterations must be positive")

    checkpoint = _checkpoint(arguments.model, arguments.revision, arguments.local_files_only)
    torch, policy, config, device = load_policy(arguments.source_root, checkpoint)
    pixels, state = load_observation(arguments.image, arguments.state)

    actions = np.empty((0, 0), dtype=np.float32)
    for _ in range(arguments.warmup):
        actions = predict_actions(torch, policy, config, device, pixels, state)
    samples = []
    for _ in range(arguments.iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        actions = predict_actions(torch, policy, config, device, pixels, state)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)

    payload = {
        "samples_ms": samples,
        "metrics": {"latency_ms": {"p50": statistics.median(samples)}},
        "output_summary": {
            "action_steps": int(actions.shape[0]),
            "action_dim": int(actions.shape[1]),
            "action_values": int(actions.size),
            "finite": bool(np.isfinite(actions).all()),
        },
        "framework": f"lerobot-torch-{torch.__version__}",
        "resolved_revision": arguments.revision or "unreported",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
