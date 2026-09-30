# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start and stop the candidate/reference HTTP servers (trtmc-perf-serve, later trtmc-server)."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from .config import Environment


class ServiceError(RuntimeError):
    pass


def _serve_env(environment: Environment) -> dict[str, str]:
    repo = environment.path("repo")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": f"{repo}/apps/perf_serving:{repo}/apps/benchmark:{repo}/core/builder:{repo}"}
    if environment.values.get("hf_hub_cache"):  # the cache that checkpoint retention manages
        env["HF_HUB_CACHE"] = str(environment["hf_hub_cache"])
    return env


def build_env(environment: Environment) -> dict[str, str]:
    """Environment of bundle builds and checkpoint downloads (online, plus the environment's build_env)."""
    env = {**_serve_env(environment), **{key: str(value) for key, value in
                                         (environment.values.get("build_env") or {}).items()}}
    env.pop("HF_HUB_OFFLINE", None)
    return env


@contextmanager
def gpu_exclusive(environment: Environment) -> Iterator[None]:
    """Hold the host's GPU lock (``gpu_lock`` in the environment): bundle builds and timing phases take it,
    so no measurement overlaps a build on the same GPU. Acc phases run without it."""
    path = environment.values.get("gpu_lock")
    if not path:
        yield
        return
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def reference_python(environment: Environment, model: dict[str, Any]) -> str:
    """The family's reference interpreter (created on first use and cached by requirements digest)."""
    completed = subprocess.run(
        [str(environment["serve_python"]), "-m", "trtmc_perf_serving", "reference-env", "--profile",
         model["catalog_profile"], "--manifest-root", str(environment.path("repo") / "families"),
         "--root", str(environment.path("reference_env_root"))],
        capture_output=True, text=True, env=_serve_env(environment), cwd=environment.path("repo"), timeout=10800)
    if completed.returncode != 0:
        raise ServiceError(f"reference environment for {model['catalog_profile']} failed: {completed.stderr[-800:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])["python"]


def platform_fingerprint(environment: Environment, python: str | None = None) -> dict[str, Any]:
    """Fingerprint of a serving environment (GPU architecture, framework and CUDA versions)."""
    completed = subprocess.run([python or str(environment["serve_python"]), "-m", "trtmc_perf_serving", "platform"],
                               capture_output=True, text=True, env=_serve_env(environment),
                               cwd=environment.path("repo"), check=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


@contextmanager
def serving(environment: Environment, model: dict[str, Any], backend: str, out: Path, *,
            mode: str = "eager", precision: str | None = None, deterministic: bool = False,
            isolate_requests: bool = False, python: str | None = None,
            script_measurement: Mapping[str, int] | None = None) -> Iterator[dict[str, Any]]:
    """Run one server for the model; yields its URL and /v1/serving/info.

    backend: ``trtmc`` (candidate), ``reference`` (generic HF adapters), or ``script`` (the family's
    qualification reference). References run in ``python``, the family's reference environment.
    """
    repo = environment.path("repo")
    port = int(environment["ports"]["candidate" if backend == "trtmc" else "reference"])
    out.mkdir(parents=True, exist_ok=True)
    command = [python or str(environment["serve_python"]), "-m", "trtmc_perf_serving", "serve",
               "--manifest-root", str(repo / "families"), "--profile", model["catalog_profile"],
               "--backend", backend, "--port", str(port), "--full-observations",
               "--records", str(out / "records.jsonl"), "--scratch", str(out / "scratch")]
    reference = model["reference"]
    if backend == "trtmc":
        command += ["--bundle", str(environment.path("bundle_root") / model["candidate"]["bundle"]),
                    "--runtime-root", str(environment["runtime_root"]), "--worker", str(environment["worker"])]
        if isolate_requests:
            command.append("--isolate-requests")
    else:
        command += ["--mode", mode, "--precision", precision or reference.get("precision", "fp32")]
        if reference.get("trust_remote_code"):
            command.append("--trust-remote-code")
        if backend == "script":
            command += ["--runtime-root", str(environment["runtime_root"]),
                        "--reference-options", json.dumps(dict(script_measurement or {"warmup": 0, "iterations": 1}))]
        else:
            if deterministic:
                command.append("--deterministic")
            if reference.get("model"):
                command += ["--reference-model", reference["model"]]
            if reference.get("options"):
                command += ["--reference-options", json.dumps(reference["options"])]
    env = _serve_env(environment)
    if backend != "trtmc":
        # References may fetch what their checkpoint does not carry (pipeline parts, remote code).
        env.pop("HF_HUB_OFFLINE", None)
    (out / "command.json").write_text(json.dumps(command))
    process = subprocess.Popen(command, stdout=open(out / "server.log", "w"), stderr=subprocess.STDOUT,
                               env=env, cwd=repo, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    try:
        info = _wait_ready(process, url, out)
        yield {"url": url, "info": info, "records": out / "records.jsonl"}
    finally:
        _stop(process)


def _wait_ready(process: subprocess.Popen, url: str, out: Path, timeout_s: float = 1800) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise ServiceError(f"server exited {process.returncode}: {(out / 'server.log').read_text()[-600:]}")
        try:
            urllib.request.urlopen(f"{url}/health/ready", timeout=2)
            return json.loads(urllib.request.urlopen(f"{url}/v1/serving/info", timeout=5).read())
        except OSError:
            time.sleep(2)
    raise ServiceError("server did not become ready")


def _stop(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(120)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
