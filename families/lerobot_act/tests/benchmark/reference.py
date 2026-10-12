#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned official Accuracy and Performance reference for LeRobot ACT."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping

import numpy as np

from families.lerobot_act.tests.official_reference import (
    load_observation,
    load_policy,
    predict_actions,
)


def _source_root() -> Path:
    root = Path(sys.prefix) / "trtmc-reference/lerobot"
    if not (root / "lerobot/common/policies/act/modeling_act.py").is_file():
        raise RuntimeError(f"LeRobot reference is incomplete: {root}")
    return root


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


def _summary(actions: np.ndarray) -> dict[str, Any]:
    return {
        "action_steps": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "action_values": int(actions.size),
        "actions": actions.reshape(-1).tolist(),
        "finite": bool(np.isfinite(actions).all()),
    }


def _run_accuracy(arguments: argparse.Namespace) -> int:
    payload = json.loads(arguments.request.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Accuracy reference requires samples")
    checkpoint = _checkpoint(
        str(payload["model"]),
        payload.get("revision"),
        os.environ.get("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY") == "1",
    )
    torch, policy, config, device = load_policy(_source_root(), checkpoint)
    output_samples = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"Accuracy sample {index} must be an object")
        pixels, state = load_observation(
            Path(str(sample["image_path"])), Path(str(sample["state_path"]))
        )
        actions = predict_actions(torch, policy, config, device, pixels, state)
        output_samples.append(
            {"sample_id": str(sample["sample_id"]), **_summary(actions)}
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps({"samples": output_samples}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def _run_performance(arguments: argparse.Namespace) -> int:
    import torch

    request = json.loads(arguments.request_json)
    timing = json.loads(arguments.timing_contract_json)
    checkpoint = _checkpoint(arguments.model, arguments.revision, arguments.local_files_only)
    _, policy, config, device = load_policy(_source_root(), checkpoint)
    pixels, state = load_observation(
        Path(str(request["image_path"])), Path(str(request["state_path"]))
    )

    def invoke():
        return predict_actions(torch, policy, config, device, pixels, state)

    actions = None
    for _ in range(arguments.warmup):
        actions = invoke()
    samples_ms = []
    for _ in range(arguments.iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        actions = invoke()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    if actions is None:
        raise RuntimeError("LeRobot reference produced no actions")
    value = {
        "schema_version": "trtmc.perf-baseline/v1",
        "status": "completed",
        "model": arguments.model,
        "family": arguments.family,
        "operation": arguments.operation,
        "case_name": arguments.case_name,
        "selected_task": arguments.selected_task,
        "precision": arguments.precision,
        "mode": arguments.mode,
        "framework": f"lerobot-torch-{torch.__version__}",
        "measurement": {
            "warmup": arguments.warmup,
            "iterations": arguments.iterations,
        },
        "measurement_policy": timing,
        **timing,
        "samples_ms": samples_ms,
        "metrics": {"latency_ms": {"p50": statistics.median(samples_ms)}},
        "output_summary": _summary(actions),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return 0


def _accuracy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _performance_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for name in (
        "model",
        "family",
        "manifest",
        "operation",
        "selected-task",
        "request-json",
        "adapter-options-json",
        "timing-contract-json",
        "precision",
        "mode",
        "padding",
        "case-name",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> int:
    if "--request" in sys.argv[1:]:
        return _run_accuracy(_accuracy_parser().parse_args())
    return _run_performance(_performance_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
