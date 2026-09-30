# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Candidate bundles, built on demand with trtmc-bench into the environment's bundle root.

Builds run in the model's family environment (the reference environment, which holds the family's
declared requirements, as CI installs them before a build) and hold the host GPU lock. The catalog
manifest is built as is; ``candidate.build`` overrides and family-prepared model directories go
through a descriptor derived from it. An existing bundle is reused as is.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from .config import Environment
from .services import build_env, gpu_exclusive

BUILD_TIMEOUT_S = 3 * 3600


class BuildError(RuntimeError):
    pass


def bundle_path(environment: Environment, model: Mapping[str, Any]) -> Path:
    return environment.path("bundle_root") / model["candidate"]["bundle"]


def _absolute_assets(value: Any, root: Path) -> Any:
    """Catalog testcase assets are relative to the manifest's tests/ directory; a descriptor elsewhere
    needs them absolute."""
    if isinstance(value, dict):
        return {key: _absolute_assets(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_absolute_assets(item, root) for item in value]
    if isinstance(value, str) and value and not value.startswith("/") and len(value) < 256 and "\n" not in value:
        try:
            if (root / value).exists():
                return str((root / value).resolve())
        except OSError:  # not a path (for example a long prompt)
            pass
    return value


def _descriptor(model: Mapping[str, Any], python: str, work: Path) -> Path | None:
    candidate = model["candidate"]
    if not candidate.get("build") and not candidate.get("model_directory"):
        return None
    manifest_path = Path(candidate["manifest"])
    manifest = {**json.loads(manifest_path.read_text()), **candidate.get("build", {})}
    manifest["testcases"] = _absolute_assets(manifest.get("testcases", []), manifest_path.parent.parent)
    if candidate.get("model_directory"):
        # The family environment's hook prepares the upstream checkout the family builds from.
        environment_root = Path(python).parent.parent.resolve()
        directory = (environment_root / candidate["model_directory"]).resolve()
        if not directory.is_relative_to(environment_root) or not directory.is_dir():
            raise BuildError(f"prepared model directory is missing: {directory}")
        manifest.update(hf_id=str(directory), hf_revision="")
    work.mkdir(parents=True, exist_ok=True)
    path = work / "candidate-model.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def build_command(environment: Environment, model: Mapping[str, Any], python: str, work: Path) -> list[str]:
    descriptor = _descriptor(model, python, work)
    target = (["--model", str(descriptor)] if descriptor else
              ["--manifest-root", str(environment.path("repo") / "families"), "--model", model["catalog_profile"]])
    return [python, "-m", "trtmc_benchmark", "run", *target, "--bundle-cache", str(environment["bundle_root"]),
            "--prepare-only", "--runtime-root", str(environment["runtime_root"]),
            "--worker", str(environment["worker"]), "-o", str(work / "prepare")]


def _failure_reason(log: Path) -> str:
    lines = log.read_text(errors="replace").splitlines() if log.is_file() else []
    errors = [line for line in lines if "rror" in line]
    return (errors or lines[-3:] or [""])[-1][:300]


def ensure_bundle(environment: Environment, model: Mapping[str, Any], python: str, out: Path) -> dict[str, Any]:
    """Build the candidate bundle unless it exists: {"status": reused | built | failed, ...}."""
    path = bundle_path(environment, model)
    if path.is_file():
        return {"status": "reused", "bundle": str(path)}
    work = out / "build"
    work.mkdir(parents=True, exist_ok=True)
    log = work / "build.log"
    started = time.time()
    try:
        command = build_command(environment, model, python, work)
    except BuildError as error:
        return {"status": "failed", "exit": None, "reason": str(error), "log": str(log)}
    (work / "command.json").write_text(json.dumps(command))
    with gpu_exclusive(environment), open(log, "w") as handle:
        try:
            code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=build_env(environment),
                                  cwd=environment.path("repo"), timeout=BUILD_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            code = None
    seconds = round(time.time() - started)
    if code == 0 and path.is_file():
        return {"status": "built", "bundle": str(path), "seconds": seconds}
    reason = _failure_reason(log) if code is not None else f"build timed out after {BUILD_TIMEOUT_S} s"
    return {"status": "failed", "exit": code, "reason": reason, "log": str(log), "seconds": seconds}


def prefetch(environment: Environment, model: Mapping[str, Any]) -> None:
    """Download the model's checkpoints outside the GPU lock, the way the build resolves them."""
    for repo in _downloads(model):
        subprocess.run([str(environment["serve_python"]), "-c",
                        "import sys; from tensorrt_model_connect.model_support import resolve_model; "
                        "resolve_model(sys.argv[1], sys.argv[2] or None)", repo, str(_revision(model, repo) or "")],
                       env=build_env(environment), cwd=environment.path("repo"), capture_output=True,
                       timeout=BUILD_TIMEOUT_S, check=False)


def _downloads(model: Mapping[str, Any]) -> list[str]:
    from .models import checkpoints

    return [] if model["candidate"].get("model_directory") else sorted(checkpoints(model))


def _revision(model: Mapping[str, Any], repo: str) -> str | None:
    return model["candidate"].get("revision") if repo == model["candidate"].get("checkpoint") else None
