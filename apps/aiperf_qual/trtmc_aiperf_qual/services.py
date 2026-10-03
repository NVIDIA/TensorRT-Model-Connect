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
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from .config import Environment


class ServiceError(RuntimeError):
    pass


REPLICA_PORT_OFFSET = 10  # replica i > 0 listens on <port> + 10 + i
REPLICA_HEADROOM_MIB = 24 * 1024  # GPU memory left free next to the replicas
REPLICA_GROWTH = 1.5  # a replica's peak over its memory once loaded (activations, KV cache)


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


def reference_python(environment: Environment, model: Mapping[str, Any]) -> str:
    """The native reference's interpreter: the serving one, or an environment that layers the model's
    ``reference.requirements`` (a file relative to the repository) on it, created once and cached."""
    reference = model.get("reference") or {}
    if not reference.get("requirements"):
        return str(environment["serve_python"])
    command = [str(environment["serve_python"]), "-m", "trtmc_perf_serving", "reference-env",
               "--requirements", str(environment.path("repo") / reference["requirements"]),
               "--root", str(environment.path("reference_env_root"))]
    if reference.get("build_isolation") is False:
        command.append("--no-build-isolation")
    if reference.get("prepare"):
        command += ["--prepare", str(environment.path("repo") / reference["prepare"])]
    completed = subprocess.run(command, capture_output=True, text=True, env=build_env(environment),
                               cwd=environment.path("repo"), timeout=10800)
    if completed.returncode != 0:
        raise ServiceError(f"reference environment for {model['catalog_profile']} failed: {completed.stderr[-800:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])["python"]


def platform_fingerprint(environment: Environment, python: str | None = None) -> dict[str, Any]:
    """Fingerprint of a serving environment (GPU architecture, framework and CUDA versions)."""
    completed = subprocess.run([python or str(environment["serve_python"]), "-m", "trtmc_perf_serving", "platform"],
                               capture_output=True, text=True, env=_serve_env(environment),
                               cwd=environment.path("repo"), check=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def platform_id(fingerprint: Mapping[str, Any]) -> str:
    """Readable, stable platform name: GPU architecture plus a hash of the whole fingerprint."""
    from .suites import canonical, sha256_text

    return f"{fingerprint.get('gpu_arch', 'unknown')}-{sha256_text(canonical(dict(fingerprint)))[:10]}"


@contextmanager
def serving(environment: Environment, model: dict[str, Any], backend: str, out: Path, *,
            mode: str = "eager", precision: str | None = None, deterministic: bool = False,
            isolate_requests: bool = False, python: str | None = None, keep_artifacts: bool = False, memory_probe: bool = False,
            port: int | None = None, extra_env: Mapping[str, str] | None = None) -> Iterator[dict[str, Any]]:
    """Run one server for the model; yields its URL and /v1/serving/info.

    backend: ``trtmc`` (candidate) or ``reference`` (generic HF adapters, run in ``python``: the model's
    reference environment).
    """
    repo = environment.path("repo")
    port = port or int(environment["ports"]["candidate" if backend == "trtmc" else "reference"])
    out.mkdir(parents=True, exist_ok=True)
    command = [python or str(environment["serve_python"]), "-m", "trtmc_perf_serving", "serve",
               "--manifest-root", str(repo / "families"), "--profile", model["catalog_profile"],
               "--backend", backend, "--port", str(port), "--full-observations",
               "--records", str(out / "records.jsonl"), "--scratch", str(out / "scratch")]
    reference = model["reference"]
    if model["candidate"].get("revision"):  # the pinned checkpoint (tokenizer, latent replay, reference)
        command += ["--revision", str(model["candidate"]["revision"])]
    if backend == "trtmc":
        command += ["--bundle", str(environment.path("bundle_root") / model["candidate"]["bundle"]),
                    "--runtime-root", str(environment["runtime_root"]), "--worker", str(environment["worker"])]
        if isolate_requests:
            command.append("--isolate-requests")
    else:
        command += ["--mode", mode, "--precision", precision or reference.get("precision", "fp32")]
        if reference.get("trust_remote_code"):
            command.append("--trust-remote-code")
        if deterministic:
            command.append("--deterministic")
        if reference.get("model"):
            command += ["--reference-model", reference["model"]]
        if reference.get("revision"):
            command += ["--reference-revision", str(reference["revision"])]
        if reference.get("options"):
            command += ["--reference-options", json.dumps(reference["options"])]
        if reference.get("adapter"):  # the family's own native pipeline
            command += ["--reference-adapter", str(repo / reference["adapter"])]
    if keep_artifacts:  # checks that read output artifacts (audio, images) after the run
        command.append("--keep-artifacts")
    if memory_probe:  # peak GPU memory per call; only meaningful while this server is alone on the GPU
        command.append("--memory-probe")
    env = _serve_env(environment)
    if backend != "trtmc" and not reference.get("offline"):
        # References may fetch what their checkpoint does not carry (pipeline parts, remote code); one whose
        # gated repository refuses even the checks for optional files declares ``reference.offline``.
        env.pop("HF_HUB_OFFLINE", None)
    env.update(extra_env or {})
    (out / "command.json").write_text(json.dumps(command))
    process = subprocess.Popen(command, stdout=open(out / "server.log", "w"), stderr=subprocess.STDOUT,
                               env=env, cwd=repo, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    try:
        info = _wait_ready(process, url, out)
        yield {"url": url, "info": info, "records": out / "records.jsonl"}
    finally:
        _stop(process)


def gpu_memory_mib() -> tuple[int, int] | None:
    """(used, total) MiB of the first visible GPU, or None without nvidia-smi."""
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=60, check=True).stdout.splitlines()[0]
        used, total = (int(value) for value in line.split(","))
        return used, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def replicas_that_fit(before: tuple[int, int] | None, after: tuple[int, int] | None, wanted: int) -> int:
    """How many copies of a server fit: ``after`` (one copy loaded) - ``before`` is one copy's memory;
    each needs REPLICA_GROWTH times that at its peak, with REPLICA_HEADROOM_MIB left free."""
    if wanted <= 1 or before is None or after is None:
        return 1
    one = max(after[0] - before[0], 1)
    return max(1, min(wanted, int((after[1] - before[0] - REPLICA_HEADROOM_MIB) // (one * REPLICA_GROWTH))))


@contextmanager
def mps(environment: Environment, directory: Path) -> Iterator[dict[str, str]]:
    """With ``acc_mps`` (environment), a CUDA MPS daemon of its own (private pipe directory): the processes given
    the yielded variables share the GPU's SMs concurrently instead of time-slicing it; others are unaffected.
    Yields {} without it or when the daemon does not start (the copies then time-slice, still correct)."""
    if not environment.values.get("acc_mps"):
        yield {}
        return
    variables = {"CUDA_MPS_PIPE_DIRECTORY": str(directory / "pipe"), "CUDA_MPS_LOG_DIRECTORY": str(directory / "log")}
    for path in variables.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **variables}
    try:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        (directory / "mps-unavailable.txt").write_text(f"{type(error).__name__}: {error}\n")
        yield {}
        return
    try:
        yield variables
    finally:  # after the clients stopped (the caller's servers exit first)
        try:
            subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env=env, capture_output=True,
                           timeout=120)
        except (OSError, subprocess.SubprocessError) as error:
            (directory / "mps-quit-failed.txt").write_text(f"{type(error).__name__}: {error}\n")


@contextmanager
def serving_replicas(environment: Environment, model: dict[str, Any], backend: str, out: Path, *, count: int,
                     **options: Any) -> Iterator[dict[str, Any]]:
    """Up to ``count`` copies of one server on the GPU, as many as its free memory holds (the first copy
    measures what one needs); yields the first copy's service with ``urls`` of all and ``replicas``. Each
    copy still serves one request at a time; clients spread their requests over ``urls``."""
    base = int(environment["ports"]["candidate" if backend == "trtmc" else "reference"])
    before = gpu_memory_mib() if count > 1 else None
    with ExitStack() as stack:
        shared = stack.enter_context(mps(environment, out.parent / f"{out.name}-mps")) if count > 1 else {}
        first = stack.enter_context(serving(environment, model, backend, out, extra_env=shared, **options))
        urls = [first["url"]]
        for index in range(1, replicas_that_fit(before, gpu_memory_mib() if before else None, count)):
            try:
                extra = stack.enter_context(serving(environment, model, backend, out.parent / f"{out.name}-replica{index}",
                                                    port=base + REPLICA_PORT_OFFSET + index, extra_env=shared, **options))
            except ServiceError:  # the copies started so far serve
                break
            urls.append(extra["url"])
        yield {**first, "urls": urls, "replicas": len(urls), "mps": bool(shared)}


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
