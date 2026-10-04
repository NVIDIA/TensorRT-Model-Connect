# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invoke AIPerf and read its exports once they are finalized."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import cancel
from .config import Environment

READY_MARKER = ".aiperf_results_ready.json"
RAW_EXPORT = "profile_export_raw.jsonl"


@dataclass(frozen=True)
class AiperfRun:
    directory: Path
    exit_code: int
    command: list[str]

    def json(self, name: str) -> dict[str, Any]:
        path = self.directory / name
        return json.loads(path.read_text()) if path.is_file() else {}

    @property
    def summary(self) -> dict[str, Any]:
        return self.json("profile_export_aiperf.json")

    def raw_records(self, phase: str = "profiling") -> list[dict[str, Any]]:
        """The per-request records: the merged export, else the per-processor files AIPerf leaves
        unmerged (for example when every request failed)."""
        records = []
        paths = sorted(self.directory.glob(f"**/{RAW_EXPORT}")) or sorted(self.directory.glob("**/raw_records/*.jsonl"))
        for path in paths:
            # JSON Lines split on "\n" only: str.splitlines also breaks at U+2028, U+0085 and the like, which
            # JSON strings may carry unescaped (benchmark text does).
            for line in path.read_text().split("\n"):
                if not line.strip():
                    continue
                item = json.loads(line)
                if item["metadata"].get("benchmark_phase") == phase:
                    records.append(item)
        return records

    def accuracy_records(self) -> list[dict[str, Any]]:
        path = self.directory / "accuracy_export.jsonl"
        if not path.is_file():
            return []
        return [item for item in map(json.loads, filter(str.strip, path.read_text().split("\n")))
                if item.get("benchmark_phase") == "profiling"]


def _run(command: Sequence[str], log: Any, env: Mapping[str, str], timeout_s: float) -> int:
    """AIPerf in a process group of its own, stopped (the whole group) at its deadline (TimeoutExpired), when the
    run is cancelled (``cancel.Cancelled``), or on any interrupt of this thread."""
    process = subprocess.Popen(list(command), stdout=log, stderr=subprocess.STDOUT, env=dict(env),
                               start_new_session=True)
    deadline = time.time() + timeout_s
    try:
        while True:
            try:
                return process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                cancel.check()
                if time.time() > deadline:
                    raise subprocess.TimeoutExpired(list(command), timeout_s) from None
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def run_aiperf(environment: Environment, out: Path, arguments: Sequence[str], *,
               env: Mapping[str, str] | None = None, timeout_s: float = 7200) -> AiperfRun:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    command = [str(environment["aiperf"]), "profile", "--model", "trtmc", "--ui-type", "none",
               "--export-level", "raw", "--artifact-dir", str(out), "--random-seed", "0",
               "--request-timeout-seconds", "1800", *arguments]
    # Tokenizers and public datasets load online (AIPerf's offline path rejects local tokenizer
    # directories); keep the datasets cache private to this environment.
    client_env = {key: value for key, value in os.environ.items()
                  if key not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    client_env["HF_DATASETS_CACHE"] = str(environment["hf_datasets_cache"])
    client_env.update(env or {})
    (out.parent / f"{out.name}.command.json").write_text(json.dumps(command))
    with open(out.parent / f"{out.name}.log", "w") as log:
        code = _run(command, log, client_env, timeout_s)
    _wait_ready(out)
    return AiperfRun(out, code, command)


def _wait_ready(out: Path, timeout_s: float = 180) -> None:
    """AIPerf finalizes exports asynchronously after the CLI returns; the raw export can still be
    aggregating when the ready marker appears, so also wait for its size to settle."""
    deadline = time.time() + timeout_s
    while time.time() < deadline and not any(out.glob(f"**/{READY_MARKER}")):
        time.sleep(0.5)
    previous = None
    while time.time() < deadline:
        sizes = tuple(path.stat().st_size for path in sorted(out.glob(f"**/{RAW_EXPORT}")))
        if sizes and sizes == previous:
            return
        previous = sizes
        time.sleep(1.0)
