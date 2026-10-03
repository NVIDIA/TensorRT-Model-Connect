# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start N local ranks of a TRTMC command without MPI (Linux and Windows).

The distributed family runtimes discover their rank from the OpenMPI
environment contract (``OMPI_COMM_WORLD_SIZE``/``_RANK``/``_LOCAL_RANK``) and
exchange the NCCL unique id through the file named by
``TRTMC_NCCL_RENDEZVOUS``. On Linux, ``mpirun`` provides the rank variables.
This launcher provides the same contract on one machine where OpenMPI is not
available, such as native Windows:

    python tools/launch_ranks.py -n 2 --gpus 0,1 -- trtmc generate-video BUNDLE ...

Every rank sees the same ``CUDA_VISIBLE_DEVICES`` list and selects its device
by local rank, exactly as under ``mpirun``. Each launch uses a fresh
rendezvous file, so a stale unique id from an earlier run cannot be read.
Rank output is prefixed like ``mpirun --tag-output`` (``[1,<rank>]<stdout>:``)
so existing rank-0 output parsers work unchanged. When one rank fails, the
remaining ranks are terminated and the launcher returns the first non-zero
exit status.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

RANK_ENV_NAMES = (
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
)
RENDEZVOUS_ENV = "TRTMC_NCCL_RENDEZVOUS"
NCCL_LIBRARY_ENV = "TRTMC_NCCL_LIBRARY"


def tag(rank: int, stream: str) -> str:
    """Return the ``mpirun --tag-output`` prefix for one rank and stream."""
    return f"[1,{rank}]<{stream}>:"


def library_path_variable(platform: str = sys.platform) -> str:
    """Environment variable the platform loader searches for shared libraries."""
    return "PATH" if platform.startswith("win") else "LD_LIBRARY_PATH"


def parse_gpus(value: str | None, world_size: int) -> list[str] | None:
    """Validate a comma-separated GPU list; None keeps the caller's visibility."""
    if value is None or value == "":
        return None
    gpus = [item.strip() for item in value.split(",")]
    if any(not item for item in gpus):
        raise ValueError(f"invalid --gpus list: {value!r}")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"--gpus lists a device twice: {value!r}")
    if len(gpus) < world_size:
        raise ValueError(f"--gpus lists {len(gpus)} device(s) for {world_size} ranks")
    return gpus


def new_rendezvous_path(directory: str | os.PathLike[str] | None = None) -> Path:
    """Unique, not-yet-existing rendezvous file for one launch."""
    root = Path(directory) if directory is not None else Path(tempfile.gettempdir())
    return root / f"trtmc_nccl_{os.getpid()}_{uuid.uuid4().hex}.bin"


def rank_environments(
    base: Mapping[str, str],
    world_size: int,
    rendezvous: Path,
    gpus: Sequence[str] | None = None,
    nccl_library: str | None = None,
    library_dirs: Sequence[str] = (),
    platform: str = sys.platform,
) -> list[dict[str, str]]:
    """Build the environment of every rank from ``base`` (not modified)."""
    if world_size < 1:
        raise ValueError("world size must be at least 1")
    shared = dict(base)
    if gpus is not None:
        shared["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    shared[RENDEZVOUS_ENV] = str(rendezvous)
    if nccl_library:
        shared[NCCL_LIBRARY_ENV] = nccl_library
    if library_dirs:
        variable = library_path_variable(platform)
        # Windows environment names are case-insensitive; reuse the existing key.
        key = next((name for name in shared if name.upper() == variable.upper()), variable)
        current = shared.get(key, "")
        separator = ";" if platform.startswith("win") else ":"
        shared[key] = separator.join([*library_dirs, *([current] if current else [])])
    environments = []
    for rank in range(world_size):
        env = dict(shared)
        env["OMPI_COMM_WORLD_SIZE"] = str(world_size)
        env["OMPI_COMM_WORLD_RANK"] = str(rank)
        env["OMPI_COMM_WORLD_LOCAL_RANK"] = str(rank)
        env["OMPI_COMM_WORLD_LOCAL_SIZE"] = str(world_size)
        environments.append(env)
    return environments


@dataclass
class _Rank:
    rank: int
    process: subprocess.Popen
    threads: list[threading.Thread] = field(default_factory=list)


def _pump(source, rank: int, stream: str, sink: TextIO, lock: threading.Lock, tagged: bool):
    prefix = tag(rank, stream) if tagged else ""
    for raw in iter(source.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        with lock:
            sink.write(f"{prefix}{line}\n")
            sink.flush()
    source.close()


def _terminate(ranks: Sequence[_Rank], grace_s: float) -> None:
    for item in ranks:
        if item.process.poll() is None:
            item.process.terminate()
    deadline = time.monotonic() + grace_s
    for item in ranks:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            item.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            item.process.kill()
            item.process.wait()


def launch(
    command: Sequence[str],
    world_size: int,
    *,
    gpus: Sequence[str] | None = None,
    rendezvous_dir: str | os.PathLike[str] | None = None,
    nccl_library: str | None = None,
    library_dirs: Sequence[str] = (),
    tagged: bool = True,
    timeout_s: float | None = None,
    grace_s: float = 10.0,
    base_env: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run ``command`` as ``world_size`` ranks; return the launch exit status."""
    if not command:
        raise ValueError("no command to launch")
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    rendezvous = new_rendezvous_path(rendezvous_dir)
    rendezvous.parent.mkdir(parents=True, exist_ok=True)
    environments = rank_environments(
        os.environ if base_env is None else base_env,
        world_size,
        rendezvous,
        gpus=gpus,
        nccl_library=nccl_library,
        library_dirs=library_dirs,
    )
    lock = threading.Lock()
    ranks: list[_Rank] = []
    status = 0
    try:
        for rank, env in enumerate(environments):
            process = subprocess.Popen(
                list(command),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            item = _Rank(rank, process)
            for source, name, sink in (
                (process.stdout, "stdout", stdout),
                (process.stderr, "stderr", stderr),
            ):
                thread = threading.Thread(
                    target=_pump, args=(source, rank, name, sink, lock, tagged), daemon=True
                )
                thread.start()
                item.threads.append(thread)
            ranks.append(item)

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        pending = list(ranks)
        while pending:
            for item in list(pending):
                code = item.process.poll()
                if code is None:
                    continue
                pending.remove(item)
                if code != 0 and status == 0:
                    status = code
                    with lock:
                        stderr.write(
                            f"[launch_ranks] rank {item.rank} exited with status {code}; "
                            "terminating the remaining ranks\n"
                        )
                        stderr.flush()
                    _terminate(pending, grace_s)
            if deadline is not None and pending and time.monotonic() > deadline:
                with lock:
                    stderr.write(f"[launch_ranks] timed out after {timeout_s:g} s\n")
                    stderr.flush()
                _terminate(pending, grace_s)
                status = status or 124
                break
            if pending:
                time.sleep(0.05)
    except KeyboardInterrupt:
        _terminate(ranks, grace_s)
        status = status or 130
    finally:
        _terminate(ranks, grace_s)
        for item in ranks:
            for thread in item.threads:
                thread.join()
        for leftover in (rendezvous, rendezvous.with_name(rendezvous.name + ".tmp")):
            try:
                leftover.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
    return status


def _interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        epilog="Everything after '--' is the command each rank runs.",
    )
    parser.add_argument("-n", "--np", dest="world_size", type=int, required=True)
    parser.add_argument(
        "--gpus", help="comma-separated CUDA devices; sets CUDA_VISIBLE_DEVICES for all ranks"
    )
    parser.add_argument(
        "--rendezvous-dir", help="directory for the per-launch NCCL rendezvous file (default: temp)"
    )
    parser.add_argument(
        "--nccl-library", help=f"NCCL shared library for the ranks (sets {NCCL_LIBRARY_ENV})"
    )
    parser.add_argument(
        "--library-dir",
        action="append",
        default=[],
        help="prepend a directory to the shared-library search path (PATH on Windows, "
        "LD_LIBRARY_PATH elsewhere); repeatable",
    )
    parser.add_argument("--no-tag-output", action="store_true", help="do not prefix rank output")
    parser.add_argument("--timeout", type=float, help="terminate all ranks after this many seconds")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        _parser().error("missing command after '--'")
    if args.world_size < 1:
        _parser().error("-n must be at least 1")
    try:
        gpus = parse_gpus(args.gpus, args.world_size)
    except ValueError as error:
        _parser().error(str(error))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _interrupt)
    return launch(
        command,
        args.world_size,
        gpus=gpus,
        rendezvous_dir=args.rendezvous_dir,
        nccl_library=args.nccl_library,
        library_dirs=args.library_dir,
        tagged=not args.no_tag_output,
        timeout_s=args.timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
