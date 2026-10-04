# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start and stop the candidate/reference HTTP servers (trtmc-perf-serve, later trtmc-server)."""

from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from .config import Environment


class ServiceError(RuntimeError):
    pass


class GpuStateError(BaseException):
    """The GPU is left in a state that no measurement may follow (an MPS daemon that would not stop). A
    BaseException, so no phase, precision fallback, or per-model handler swallows it: the run stops."""


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
        if backend == "trtmc":  # which TensorRT and CUDA libraries the loaded bundle actually runs on
            (out / LOADED_LIBRARIES).write_text(json.dumps(loaded_libraries(process.pid), indent=2) + "\n")
        yield {"url": url, "info": info, "records": out / "records.jsonl"}
    finally:
        _stop(process)


LOADED_LIBRARIES = "loaded-libraries.json"
LIBRARY_NAMES = ("libnvinfer", "libnvonnxparser", "libcudart", "libcublas", "libcudnn")


def loaded_libraries(group: int, names: tuple[str, ...] = LIBRARY_NAMES) -> list[str]:
    """The TensorRT and CUDA shared libraries mapped by the processes of a server's process group (its worker
    loads TensorRT at run time, from wherever the library path resolves it)."""
    found: set[str] = set()
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            if int(stat.read_text().rsplit(") ", 1)[1].split()[2]) != group:
                continue
            maps = (stat.parent / "maps").read_text()
        except (OSError, IndexError, ValueError):
            continue
        found.update(line.split()[-1] for line in maps.splitlines()
                     if "/" in line and any(name in line.rsplit("/", 1)[-1] for name in names))
    return sorted(found)


GPU_FIELDS = ("uuid", "name", "driver_version", "clocks.max.sm", "clocks.max.mem", "power.limit", "persistence_mode",
              "compute_mode")


def gpu_identity(environment: Environment) -> dict[str, Any]:
    """Where a result was measured: the host and the GPU (UUID, driver, clocks, power limit, persistence and
    compute mode); fields that cannot be read are absent. The libraries TRTMC ran on are ``loaded_libraries``."""
    identity: dict[str, Any] = {"hostname": socket.gethostname()}
    selector = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    try:
        line = subprocess.run(["nvidia-smi", *(["-i", selector] if selector else []), f"--query-gpu={','.join(GPU_FIELDS)}",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60,
                              check=True).stdout.splitlines()[0]
        identity["gpu"] = dict(zip(GPU_FIELDS, (value.strip() for value in line.split(","))))
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return identity


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


MPS_CONTROL = re.compile(r"Control (\d+)\] Starting control daemon")
MPS_SERVER = re.compile(r"Starting new server (\d+)")
MPS_START_S, MPS_QUIT_S, MPS_EXIT_S = 30, 120, 60  # the daemon's log line; its quit; its processes' exit


def _alive(pid: int) -> bool:
    """A process that still runs (a zombie has exited: only its parent has not collected it)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


MPS_PROGRAMS = ("nvidia-cuda-mps-control", "nvidia-cuda-mps-server")


def _ours(pid: int, pipe: str) -> bool:
    """A process of this run's daemon: an MPS program whose environment names this run's private pipe directory
    (a logged pid that the system reused, or a log naming another process, is never signalled)."""
    try:
        program = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0", 1)[0].decode(errors="replace")
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return False
    return os.path.basename(program) in MPS_PROGRAMS and f"CUDA_MPS_PIPE_DIRECTORY={pipe}".encode() in environ


def _mps_pids(directory: Path) -> list[int]:
    """The daemon's control and server processes, as its log names them ([] before it logged)."""
    try:
        text = (directory / "log" / "control.log").read_text(errors="replace")
    except OSError:
        return []
    return [int(pid) for pattern in (MPS_CONTROL, MPS_SERVER) for pid in pattern.findall(text)]


def _mps_devices(env: Mapping[str, str]) -> str | None:
    """CUDA_VISIBLE_DEVICES as GPU UUIDs (MPS renumbers its clients' devices), None when unset. Raises
    ServiceError for ordinals that do not name one GPU unambiguously (several GPUs not in PCI bus order)."""
    entries = [entry.strip() for entry in (env.get("CUDA_VISIBLE_DEVICES") or "").split(",") if entry.strip()]
    if not entries or all(entry.startswith(("GPU-", "MIG-")) for entry in entries):
        return ",".join(entries) or None
    try:
        lines = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True,
                               text=True, timeout=60, check=True).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        raise ServiceError(f"GPU UUIDs unavailable: {error}") from error
    uuids = dict(tuple(part.strip() for part in line.split(",", 1)) for line in lines if "," in line)
    if len(uuids) > 1 and env.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ServiceError("CUDA ordinals in CUDA_VISIBLE_DEVICES are ambiguous without CUDA_DEVICE_ORDER=PCI_BUS_ID")
    try:
        return ",".join(entry if entry.startswith(("GPU-", "MIG-")) else uuids[entry] for entry in entries)
    except KeyError as error:
        raise ServiceError(f"no GPU {error} for CUDA_VISIBLE_DEVICES") from error


def _mps_stop(env: Mapping[str, str], directory: Path) -> None:
    """Quit the daemon and make sure its processes exited (signalled if they outlive the quit; only processes that
    ``_ours`` confirms); raises GpuStateError when one still runs, so nothing follows on this GPU. Nothing to do
    when no daemon logged."""
    pids = _mps_pids(directory)
    if not pids:
        return
    pipe = str(env["CUDA_MPS_PIPE_DIRECTORY"])
    running = lambda candidates: [pid for pid in candidates if _alive(pid) and _ours(pid, pipe)]  # noqa: E731
    try:
        completed = subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env=dict(env),
                                   capture_output=True, timeout=MPS_QUIT_S)
        note = f"quit exited {completed.returncode}"
    except (OSError, subprocess.SubprocessError) as error:
        note = f"quit failed: {type(error).__name__}"
    for sig, wait_s in ((None, MPS_EXIT_S), (signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        living = running(pids)
        for pid in living if sig else []:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.time() + wait_s
        while living and time.time() < deadline:
            time.sleep(1)
            living = running(living)
        if not living:
            return
    raise GpuStateError(f"MPS processes {living} still run after the daemon's quit ({note})")


@contextmanager
def mps(environment: Environment, directory: Path) -> Iterator[dict[str, str]]:
    """With ``acc_mps`` (environment), a CUDA MPS daemon of its own (private pipe directory): the processes given
    the yielded variables share the GPU's SMs concurrently instead of time-slicing it; others are unaffected.
    The daemon and its clients name the GPU by UUID. Yields {} without it or when the daemon does not start (the
    copies then time-slice, still correct); a daemon that started is stopped and its exit verified either way."""
    if not environment.values.get("acc_mps"):
        yield {}
        return
    # A directory of this attempt's own (a retry gets the next one): its log names only this daemon's processes.
    directory = next(path for path in (directory, *(directory.with_name(f"{directory.name}-{n}") for n in range(2, 100)))
                     if not path.exists())
    variables = {"CUDA_MPS_PIPE_DIRECTORY": str(directory / "pipe"), "CUDA_MPS_LOG_DIRECTORY": str(directory / "log")}
    for path in variables.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    env = {**_serve_env(environment), **variables}
    try:
        devices = _mps_devices(env)
        if devices:
            env["CUDA_VISIBLE_DEVICES"] = variables["CUDA_VISIBLE_DEVICES"] = devices
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True, capture_output=True, timeout=60)
        deadline = time.time() + MPS_START_S
        while not _mps_pids(directory) and time.time() < deadline:
            time.sleep(0.5)
        if not _mps_pids(directory):
            raise ServiceError("the MPS daemon did not log its start")
    except (OSError, subprocess.SubprocessError, ServiceError) as error:
        (directory / "mps-unavailable.txt").write_text(f"{type(error).__name__}: {error}\n")
        _mps_stop(env, directory)
        yield {}
        return
    try:
        yield variables
    finally:  # after the clients stopped (the caller's servers exit first)
        _mps_stop(env, directory)


@contextmanager
def serving_replicas(environment: Environment, model: dict[str, Any], backend: str, out: Path, *, count: int,
                     **options: Any) -> Iterator[dict[str, Any]]:
    """Up to ``count`` copies of one server on the GPU, as many as its free memory holds (the first copy
    measures what one needs; the others then start together); yields the first copy's service with ``urls`` of
    all and ``replicas``. Each copy still serves one request at a time; clients spread their requests over
    ``urls``."""
    base = int(environment["ports"]["candidate" if backend == "trtmc" else "reference"])
    before = gpu_memory_mib() if count > 1 else None
    with ExitStack() as stack:
        shared = stack.enter_context(mps(environment, out.parent / f"{out.name}-mps")) if count > 1 else {}
        first = stack.enter_context(serving(environment, model, backend, out, extra_env=shared, **options))
        urls = [first["url"]]
        # The other copies start together (sized above for their peak); each one that starts serves.
        copies = [serving(environment, model, backend, out.parent / f"{out.name}-replica{index}",
                          port=base + REPLICA_PORT_OFFSET + index, extra_env=shared, **options)
                  for index in range(1, replicas_that_fit(before, gpu_memory_mib() if before else None, count))]
        if copies:
            failure: BaseException | None = None
            with ThreadPoolExecutor(max_workers=len(copies)) as pool:
                started = [pool.submit(copy.__enter__) for copy in copies]
                for copy, future in zip(copies, started):  # every copy that started is stopped with the rest
                    try:
                        service = future.result()
                    except ServiceError:  # this copy did not start; the others serve
                        continue
                    except BaseException as error:  # noqa: BLE001 - raised once the started copies are registered
                        failure = failure or error
                        continue
                    stack.push(copy)
                    urls.append(service["url"])
            if failure is not None:
                raise failure
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
