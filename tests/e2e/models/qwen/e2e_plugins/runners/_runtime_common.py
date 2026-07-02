# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-local E2E subprocess and distributed runtime helpers."""

from __future__ import annotations

import logging
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .. import _case_artifact_dir
from ..contracts import E2ECase, RunContext

logger = logging.getLogger(__name__)

_SUPPORTED_STAGES = {"full_generation", "prefill", "decode"}
_TRTMC_TIMING_RE = re.compile(
    r"^\[trtmc\.timing\]\s+"
    r"prefill_ms=(?P<prefill_ms>[-+0-9.eE]+)\s+"
    r"decode_ms=(?P<decode_ms>[-+0-9.eE]+)\s+"
    r"total_ms=(?P<total_ms>[-+0-9.eE]+)\s*$",
    re.MULTILINE,
)
_TRTMC_LOAD_TIMING_RE = re.compile(
    r"^\[trtmc\.load_timing\]\s+.*?"
    r"load_deserialize_ms=(?P<load_deserialize_ms>[-+0-9.eE]+)",
    re.MULTILINE,
)
_TRT_RUNTIME_ERROR_RE = re.compile(
    r"(?im)^.*("
    r"\[trt\]\s+ERROR:"
    r"|IExecutionContext::enqueueV3:\s+Error Code"
    r"|Internal Error:"
    r"|Cuda Runtime"
    r"|illegal memory access"
    r").*$"
)
_MPI_TAGGED_STDOUT_RE = re.compile(
    r"^\[[^\]]+,(?P<rank>\d+)\]<stdout>:(?P<text>.*)$")
_MPI_STREAM_TAG_RE = re.compile(r"\[[^\]]+,\d+\]<(?:stdout|stderr)>:")


def _distributed_runtime_config(case: E2ECase | None) -> dict:
    if case is None:
        return {}
    config = case.metadata.get("distributed_runtime", {})
    return config if isinstance(config, dict) and config.get("enabled") else {}


def _extract_rank_zero_stdout(stdout: str) -> str:
    """Return rank-0 stdout from OpenMPI --tag-output, falling back to raw text."""
    rank0_lines: list[str] = []
    saw_tagged = False
    for line in (stdout or "").splitlines():
        match = _MPI_TAGGED_STDOUT_RE.match(line)
        if match is None:
            continue
        saw_tagged = True
        if int(match.group("rank")) == 0:
            rank0_lines.append(match.group("text"))
    if saw_tagged:
        return "\n".join(rank0_lines).strip()
    return (stdout or "").strip()


def _strip_mpi_stream_tags(text: str) -> str:
    return _MPI_STREAM_TAG_RE.sub("", text or "")


def _safe_artifact_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "case")


def _read_text_generation_sample(path: Path) -> dict:
    """Read the first JSONL text-generation sample written by the C++ CLI."""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict):
                return {}
            token_ids = sample.get("token_ids")
            if isinstance(token_ids, list):
                sample["token_ids"] = [int(token) for token in token_ids]
            return sample
    return {}


def _expected_answers(case: E2ECase) -> list[str]:
    raw = case.metadata.get("expected_answers", case.metadata.get("expected_answer", []))
    if isinstance(raw, str):
        raw_values = [raw]
    elif isinstance(raw, (list, tuple)):
        raw_values = list(raw)
    else:
        raw_values = []
    return [str(value) for value in raw_values if str(value).strip()]


def _ensure_distributed_runtime_env(
    case: E2ECase,
    ctx: RunContext,
    env: dict[str, str],
    rendezvous_suffix: str = "",
) -> None:
    """Populate shared env values needed by all distributed ranks."""
    if not _distributed_runtime_config(case):
        return
    if env.get("TRTMC_NCCL_RENDEZVOUS"):
        return

    safe_name = _safe_artifact_name(case.name)
    root = Path(_case_artifact_dir(ctx.artifacts_dir, case.name)) if ctx.artifacts_dir else \
        Path(tempfile.gettempdir())
    path = root / f"{safe_name}{rendezvous_suffix}.nccl_rendezvous.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    env["TRTMC_NCCL_RENDEZVOUS"] = str(path)


def _wrap_distributed_command(
    cmd: list[str], case: E2ECase | None, env: dict[str, str]
) -> list[str]:
    config = _distributed_runtime_config(case)
    if not config:
        return cmd

    launcher = str(config.get("launcher", "mpirun") or "mpirun")
    world_size = int(config.get("world_size", config.get("tp_size", 2)) or 2)
    launcher_args = config.get("launcher_args")
    if isinstance(launcher_args, list):
        prefix = [launcher] + [str(arg) for arg in launcher_args]
    else:
        prefix = [launcher, "--tag-output", "-np", str(world_size)]

    export_env = config.get("export_env", ["LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"])
    if isinstance(export_env, list) and Path(launcher).name == "mpirun":
        export_names = [str(name) for name in export_env]
        for name in ("TRTMC_NCCL_RENDEZVOUS", "TRTMC_EMBEDDING_STDOUT"):
            if name in env and name not in export_names:
                export_names.append(name)
        for name in export_names:
            if name in env:
                prefix.extend(["-x", name])

    return prefix + cmd


def _visible_gpu_indices(env: dict[str, str]) -> list[str]:
    raw = env.get("CUDA_VISIBLE_DEVICES", "")
    if not raw or raw.lower() in {"all", "none", "void"}:
        return []
    indices: list[str] = []
    for part in raw.split(","):
        token = part.strip()
        if token.isdigit():
            indices.append(token)
    return indices


class _GpuMemorySampler:
    def __init__(self, artifacts_dir: str | None, case_name: str, env: dict[str, str],
                 interval_ms: int) -> None:
        root = Path(_case_artifact_dir(artifacts_dir, case_name)) if artifacts_dir else \
            Path(tempfile.gettempdir())
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "gpu_memory_samples.csv"
        self.env = env
        self.interval_ms = max(50, interval_ms)
        self.visible_indices = _visible_gpu_indices(env)
        self.proc: subprocess.Popen | None = None
        self.handle = None
        self.error = ""

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            self.error = "nvidia-smi not found"
            return
        self.handle = self.path.open("w", encoding="utf-8")
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=self.handle,
                stderr=subprocess.DEVNULL,
                text=True,
                env=self.env,
            )
        except Exception as exc:
            self.error = str(exc)
            self.handle.close()
            self.handle = None

    def stop(self) -> dict:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        return self._summary()

    def _summary(self) -> dict:
        meta = {
            "sample_file": str(self.path),
            "sample_interval_ms": self.interval_ms,
            "visible_device_indices": self.visible_indices,
        }
        if self.error:
            meta["error"] = self.error
            return meta
        peaks: dict[str, int] = {}
        sample_count = 0
        if not self.path.is_file():
            meta["error"] = "sample file was not created"
            return meta
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2 or not parts[0].isdigit():
                    continue
                if self.visible_indices and parts[0] not in self.visible_indices:
                    continue
                try:
                    used_mb = int(float(parts[1]))
                except ValueError:
                    continue
                peaks[parts[0]] = max(peaks.get(parts[0], 0), used_mb)
                sample_count += 1
        meta["sample_count"] = sample_count
        meta["peak_memory_mb_by_gpu"] = peaks
        if peaks:
            meta["peak_memory_mb"] = max(peaks.values())
            meta["peak_memory_mb_visible_sum"] = sum(peaks.values())
        return meta


def _maybe_start_gpu_memory_sampler(
    distributed_runtime: dict, ctx: RunContext, case: E2ECase | None, env: dict[str, str]
) -> _GpuMemorySampler | None:
    if case is None or not distributed_runtime.get("capture_gpu_memory"):
        return None
    interval_ms = int(distributed_runtime.get("gpu_memory_sample_interval_ms", 200) or 200)
    sampler = _GpuMemorySampler(ctx.artifacts_dir, case.name, env, interval_ms)
    sampler.start()
    return sampler


def _extract_trtmc_timing(stderr: str) -> dict[str, float]:
    match = _TRTMC_TIMING_RE.search(stderr or "")
    if match is None:
        return {}
    try:
        prefill_ms = float(match.group("prefill_ms"))
        decode_ms = float(match.group("decode_ms"))
        total_ms = float(match.group("total_ms"))
    except ValueError:
        return {}
    return {
        "trt_engine_prefill_s": prefill_ms / 1000.0,
        "trt_engine_decode_s": decode_ms / 1000.0,
        "trt_engine_s": total_ms / 1000.0,
    }


def _extract_trtmc_load_timing(stderr: str) -> dict[str, float]:
    total_ms = 0.0
    found = False
    for match in _TRTMC_LOAD_TIMING_RE.finditer(stderr or ""):
        try:
            total_ms += float(match.group("load_deserialize_ms"))
            found = True
        except ValueError:
            continue
    return {"trt_load_deserialize_s": total_ms / 1000.0} if found else {}


def _detect_trt_runtime_error(stderr: str) -> str:
    match = _TRT_RUNTIME_ERROR_RE.search(stderr or "")
    return match.group(0).strip() if match else ""


def _distributed_debug_logits_required(case: E2ECase) -> bool:
    distributed_runtime = _distributed_runtime_config(case)
    return bool(distributed_runtime and distributed_runtime.get("debug_logits", True))


def _format_debug_runner_error(case: E2ECase, phase: str, meta: dict) -> str:
    detail = meta.get("error")
    if not detail and meta.get("returncode") not in (None, 0):
        detail = f"returncode={meta['returncode']}"
    if not detail:
        detail = "logits were not produced"
    log_path = meta.get("stderr_log")
    if log_path:
        detail = f"{detail}; stderr_log={log_path}"
    return (
        f"Distributed debug logits requested for {case.name} phase={phase}, "
        f"but {detail}"
    )


