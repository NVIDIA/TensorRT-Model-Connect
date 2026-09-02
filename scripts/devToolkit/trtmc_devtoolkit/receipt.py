# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist reproducible, secret-free preparation and failure receipts."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import DevToolkitError, EnvironmentHandle, PreparationPlan, ProbeResult


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def write_plan(plan: PreparationPlan) -> Path:
    return write_json(plan.state_dir / "plan.json", plan.as_dict())


def write_doctor(plan: PreparationPlan, probes: tuple[ProbeResult, ...], sm: str) -> Path:
    return write_json(
        plan.state_dir / "environment.json",
        {
            "schema_version": 1,
            "architecture": plan.architecture,
            "selected_sm": sm,
            "probes": [asdict(probe) for probe in probes],
        },
    )


def write_success(
    plan: PreparationPlan,
    environment: EnvironmentHandle,
    *,
    wheel: Path | None,
    bundle: Path | None,
) -> Path:
    observation = environment.observation
    if observation is None:
        raise DevToolkitError("A success receipt requires observed toolchain attestation")
    observed_tensorrt = {
        observation.tensorrt_python_version,
        observation.tensorrt_native_version,
        observation.tensorrt_header_version,
    }
    if observed_tensorrt != {plan.cohort.tensorrt_version}:
        raise DevToolkitError(
            "Observed TensorRT Python/native/header versions do not match the plan"
        )
    if observation.cuda_version != plan.cohort.cuda_version:
        raise DevToolkitError("Observed CUDA version does not match the plan")
    (plan.state_dir / "failure-summary.json").unlink(missing_ok=True)
    return write_json(
        plan.state_dir / "receipt.json",
        {
            "schema_version": 1,
            "status": "ready",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": plan.run_id,
            "environment_id": plan.environment_id,
            "source_revision": plan.source_revision,
            "cohort": plan.cohort.id,
            "tensorrt": observation.tensorrt_native_version,
            "cuda": observation.cuda_version,
            "architecture": plan.architecture,
            "mode": plan.request.mode,
            "environment": asdict(environment),
            "attestation": asdict(observation),
            "artifacts": {
                "wheel": str(wheel) if wheel else None,
                "bundle": str(bundle) if bundle else None,
            },
        },
    )


def write_failure(plan: PreparationPlan, error: BaseException) -> Path:
    (plan.state_dir / "receipt.json").unlink(missing_ok=True)
    return write_json(
        plan.state_dir / "failure-summary.json",
        {
            "schema_version": 1,
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": plan.run_id,
            "environment_id": plan.environment_id,
            "source_revision": plan.source_revision,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
