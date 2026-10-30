# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run trtmc-server with the bundles a step needs and call its OpenAI-style routes."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import httpx

log = logging.getLogger(__name__)

CALL_TIMEOUT = httpx.Timeout(3600.0, connect=10.0)
VIDEO_POLL_SECONDS = 5.0
STORAGE_RETRIES, STORAGE_RETRY_SECONDS = 40, 30.0


def drop_page_cache(paths) -> None:
    """Ask the kernel to drop the cached pages of bundle files once the server has read them.

    trtmc-server reads each bundle once into GPU memory. Where GPU memory is a NUMA node of the
    host (GB300, GH200), the cache of a 100 GB bundle can land there as memory no process owns,
    and the next stage's model then fails to load.
    """

    if not hasattr(os, "posix_fadvise"):
        return
    for path in paths:
        for file in [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()):
            try:
                descriptor = os.open(file, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(descriptor)


@dataclass(frozen=True)
class Served:
    """A bundle as trtmc-server registers it."""

    name: str
    bundle: Path
    byok: tuple[str, str, str] | None = None


class Engine:
    """One trtmc-server process, restarted only when a step needs bundles it has not loaded.

    Loading stage by stage lets models that do not fit together share one GPU.
    """

    def __init__(self, server: Path, runtime_root: Path, port: int, log_path: Path,
                 cuda_device: str | None = None, python_path: list[Path] = (), scratch: Path | None = None):
        self.server = server
        self.python_path = list(python_path)
        self.scratch = scratch
        self.runtime_root = runtime_root
        self.port = port
        self.log_path = log_path
        self.cuda_device = cuda_device
        self.url = f"http://127.0.0.1:{port}"
        self._process: subprocess.Popen | None = None
        self._loaded: frozenset[Served] = frozenset()
        self._lock = threading.Lock()

    @property
    def loaded(self) -> list[str]:
        return sorted(served.name for served in self._loaded)

    def ensure(self, models: list[Served]) -> None:
        wanted = frozenset(models)
        with self._lock:
            if self._process is not None and self._process.poll() is None and wanted <= self._loaded:
                return
            self._stop()
            if wanted:
                self._start(wanted)

    def stop(self) -> None:
        with self._lock:
            self._stop()

    def command(self, models: frozenset[Served]) -> list[str]:
        kernels = {served.byok for served in models if served.byok}
        if len(kernels) > 1:
            raise ValueError("one trtmc-server process registers one bring-your-own kernel")
        command = [str(self.server), "--host", "127.0.0.1", "--port", str(self.port),
                   "--runtime-root", str(self.runtime_root),
                   "--startup-timeout", "3600", "--request-timeout", "3600",
                   "--max-body-bytes", str(64 << 20), "--max-prompt-bytes", str(1 << 20),
                   "--max-new-tokens", "16384", "--max-audio-seconds", "60",
                   # The app downloads each clip at once; finished jobs only wait out their TTL.
                   "--max-video-jobs", "16", "--max-video-bytes", str(32 << 30),
                   "--video-ttl-seconds", "900"]
        for served in sorted(models, key=lambda served: served.name):
            command += ["--model", f"{served.name}={served.bundle}"]
        if kernels:
            library, function, name = kernels.pop()
            command += ["--byok-library", str(self.runtime_root / library),
                        "--byok-function", function, "--byok-name", name]
        return command

    def _start(self, models: frozenset[Served]) -> None:
        environment = dict(os.environ)
        if self.cuda_device is not None:  # nvidia-smi numbering
            environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=self.cuda_device)
        if self.python_path:  # a source checkout's video packaging module
            environment["PYTHONPATH"] = os.pathsep.join(
                [*map(str, self.python_path), *filter(None, [environment.get("PYTHONPATH")])])
        if self.scratch is not None:  # the server keeps video jobs here
            self.scratch.mkdir(parents=True, exist_ok=True)
            environment["TMPDIR"] = str(self.scratch)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        drop_page_cache(served.bundle for served in models)  # left over from earlier runs
        log.info("starting trtmc-server with %s", ", ".join(sorted(m.name for m in models)))
        with self.log_path.open("ab") as output:
            self._process = subprocess.Popen(self.command(models), stdout=output,
                                             stderr=subprocess.STDOUT, env=environment,
                                             start_new_session=True)
        self._loaded = models
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                self._loaded = frozenset()
                raise RuntimeError(f"trtmc-server exited while loading; see {self.log_path}")
            try:
                if httpx.get(f"{self.url}/health/ready", timeout=5).status_code == 200:
                    drop_page_cache(served.bundle for served in models)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
        self._stop()
        raise TimeoutError("trtmc-server did not become ready within an hour")

    def _stop(self) -> None:
        process, self._process, self._loaded = self._process, None, frozenset()
        if process is None or process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    # The calls below target the running server.

    def image(self, model: str, prompt: str, *, size: str, seed: int) -> bytes:
        response = httpx.post(f"{self.url}/v1/images/generations", timeout=CALL_TIMEOUT,
                              json={"model": model, "prompt": prompt, "size": size, "seed": seed})
        _check(response)
        return base64.b64decode(response.json()["data"][0]["b64_json"])

    def video(self, model: str, prompt: str, references: list[Path], output: Path, *,
              seed: int, continuation: str | None = None) -> None:
        """Reference-to-video: submit, wait, and download the clip with its audio."""

        fields = {"model": model, "prompt": prompt, "seed": str(seed)}
        if continuation:
            fields["continuation_prompt"] = continuation
        files = [("input_reference", (path.name, path.read_bytes(), "image/png")) for path in references]
        for _ in range(STORAGE_RETRIES):  # finished clips free their storage when their TTL expires
            response = httpx.post(f"{self.url}/v1/videos", data=fields, files=files, timeout=CALL_TIMEOUT)
            if response.status_code != 429:
                break
            log.info("trtmc-server video storage is busy; retrying in %s s", STORAGE_RETRY_SECONDS)
            time.sleep(STORAGE_RETRY_SECONDS)
        _check(response)
        job = response.json()
        while job["status"] not in {"completed", "failed"}:
            time.sleep(VIDEO_POLL_SECONDS)
            response = httpx.get(f"{self.url}/v1/videos/{job['id']}", timeout=30)
            _check(response)
            job = response.json()
        if job["status"] == "failed":
            raise RuntimeError((job.get("error") or {}).get("message", "video generation failed"))
        partial = output.with_suffix(".part")
        with httpx.stream("GET", f"{self.url}/v1/videos/{job['id']}/content",
                          timeout=CALL_TIMEOUT) as stream:
            _check(stream)
            with partial.open("wb") as file:
                for chunk in stream.iter_bytes(1 << 20):
                    file.write(chunk)
        partial.replace(output)

    def transcribe(self, model: str, wav: bytes) -> str:
        response = httpx.post(f"{self.url}/v1/audio/transcriptions", timeout=CALL_TIMEOUT,
                              data={"model": model}, files={"file": ("speech.wav", wav, "audio/wav")})
        _check(response)
        return response.json()["text"]


def _check(response: httpx.Response) -> None:
    if response.is_success:
        return
    if isinstance(response, httpx.Response) and not response.is_stream_consumed:
        response.read()
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = response.text[:300]
    raise RuntimeError(f"trtmc-server {response.request.url.path}: {response.status_code} {message}")
