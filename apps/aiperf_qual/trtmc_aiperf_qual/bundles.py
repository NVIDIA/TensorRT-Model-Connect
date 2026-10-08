# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Candidate bundles, built on demand with trtmc-bench into the environment's bundle root.

Builds run in the model's reference environment (the serving interpreter plus the family's declared
requirements, as CI installs them before a build: some builds unpickle checkpoints with the family's
packages) and hold the host GPU lock. The catalog manifest is built as is; ``candidate.build``
overrides go through a descriptor derived from it. trtmc-bench reuses an existing bundle only when its
build receipt matches (catalog manifest or descriptor sha256, checkpoint snapshot revision, build
command, TRTMC core and family source digest, package versions) and rebuilds it otherwise.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from .config import Environment
from .services import build_env, gpu_exclusive

BUILD_TIMEOUT_S = 3 * 3600


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


def _descriptor(model: Mapping[str, Any], work: Path, python: str | None) -> Path | None:
    """A model descriptor with config/models' build overrides (None: the catalog manifest as is). A
    ``model_directory`` (relative to the reference environment, prepared by the family's reference/prepare.py)
    replaces the checkpoint for families whose build reads an upstream checkout."""
    candidate = model["candidate"]
    if not candidate.get("build") and not candidate.get("model_directory"):
        return None
    manifest_path = Path(candidate["manifest"])
    manifest = {**json.loads(manifest_path.read_text()), **(candidate.get("build") or {})}
    manifest["testcases"] = _absolute_assets(manifest.get("testcases", []), manifest_path.parent.parent)
    if candidate.get("model_directory"):
        if not python:
            raise RuntimeError("a model_directory build needs the model's reference environment")
        directory = (Path(python).parent.parent / candidate["model_directory"]).resolve()
        if not directory.is_dir():
            raise RuntimeError(f"prepared model directory is missing: {directory}")
        manifest.update(hf_id=str(directory), hf_revision="")
    work.mkdir(parents=True, exist_ok=True)
    path = work / "candidate-model.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def build_command(environment: Environment, model: Mapping[str, Any], work: Path, python: str | None = None) -> list[str]:
    descriptor = _descriptor(model, work, python)
    target = (["--model", str(descriptor)] if descriptor else
              ["--manifest-root", str(environment.path("repo") / "families"), "--model", model["catalog_profile"]])
    return [python or str(environment["serve_python"]), "-m", "trtmc_benchmark", "run", *target, "--bundle-cache", str(environment["bundle_root"]),
            "--prepare-only", "--runtime-root", str(environment["runtime_root"]),
            "--worker", str(environment["worker"]), "-o", str(work / "prepare")]


def _failure_reason(log: Path) -> str:
    lines = log.read_text(errors="replace").splitlines() if log.is_file() else []
    errors = [line for line in lines if "rror" in line]
    return (errors or lines[-3:] or [""])[-1][:300]


def _state(path: Path) -> tuple[str | None, int | None]:
    """The bundle's trtmc-bench build receipt and modification time (None when absent)."""
    receipt = path.with_suffix(path.suffix + ".benchmark.json")
    return (receipt.read_text() if receipt.is_file() else None, path.stat().st_mtime_ns if path.is_file() else None)


def ensure_bundle(environment: Environment, model: Mapping[str, Any], out: Path, python: str | None = None) -> dict[str, Any]:
    """Prepare the candidate bundle (trtmc-bench reuses it when its build receipt matches, else rebuilds):
    {"status": reused | built | failed, ...}."""
    path = bundle_path(environment, model)
    before = _state(path)
    work = out / "build"
    work.mkdir(parents=True, exist_ok=True)
    log = work / "build.log"
    started = time.time()
    try:
        command = build_command(environment, model, work, python)
    except RuntimeError as error:
        return {"status": "failed", "exit": None, "reason": str(error), "log": str(log)}
    (work / "command.json").write_text(json.dumps(command))
    env = build_env(environment)
    if cached(environment, model):  # the build reads the cache; gated repositories refuse anonymous checks
        env["HF_HUB_OFFLINE"] = "1"
    with gpu_exclusive(environment), open(log, "w") as handle:
        try:
            code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                  cwd=environment.path("repo"), timeout=BUILD_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            code = None
    seconds = round(time.time() - started)
    if code == 0 and path.is_file():
        reused = before[1] is not None and _state(path) == before
        return {"status": "reused" if reused else "built", "bundle": str(path), "seconds": seconds, **identity(path)}
    reason = _failure_reason(log) if code is not None else f"build timed out after {BUILD_TIMEOUT_S} s"
    return {"status": "failed", "exit": code, "reason": reason, "log": str(log), "seconds": seconds}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    """The bundle that received the verdict: TensorRT builds are not bit-reproducible, so the build inputs alone
    do not identify it. sha256 of the bundle file and of its trtmc-bench receipt."""
    receipt = path.with_suffix(path.suffix + ".benchmark.json")
    return {"bundle_bytes": path.stat().st_size, "bundle_sha256": file_sha256(path),
            "receipt_sha256": file_sha256(receipt) if receipt.is_file() else None}


def cached(environment: Environment, model: Mapping[str, Any]) -> bool:
    """Every checkpoint the model's run reads is in the hub cache (at its pinned revision)."""
    if not environment.values.get("serve_python") or not _downloads(model):
        return False
    command = [str(environment["serve_python"]), "-c",
               "import sys; from huggingface_hub import snapshot_download; "
               "[snapshot_download(r, revision=v or None, local_files_only=True) for r, v in zip(sys.argv[1::2], sys.argv[2::2])]"]
    for repo in _downloads(model):
        command += [repo, str(_revision(model, repo) or "")]
    env = {**build_env(environment), "HF_HUB_OFFLINE": "1"}
    return subprocess.run(command, env=env, capture_output=True, timeout=600).returncode == 0


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
    if repo == model["candidate"].get("checkpoint"):
        return model["candidate"].get("revision")
    reference = model.get("reference") or {}
    return reference.get("revision") if repo == reference.get("model") else None
