#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence


DEFAULT_CONTAINER_IMAGE = "trtmc-dev-gb300:latest"
DEFAULT_CONTAINER_PREFIX = "trtmc-dev-gb300"
DEFAULT_CONTAINER_WORKDIR = "/workspace/tensorrt-model-connect"
AGENT_PATTERN = re.compile(r"^agent-(\d+)$")
SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class Mount:
    source: str
    destination: str


@dataclass(frozen=True)
class ContainerRecord:
    name: str
    image: str
    running: bool
    mounts: tuple[Mount, ...]


@dataclass(frozen=True)
class ContainerContext:
    repo_root: str
    workspace_id: str | None
    expected_container_name: str
    container_name: str
    container_workdir: str
    container_image: str
    state: str
    created: bool = False


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _real_path(value: str | Path) -> str:
    return os.path.realpath(os.fspath(value))


def find_repo_root(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"no Git repository found above {resolved}")


def verify_repo_identity(repo_root: Path) -> None:
    required = (
        repo_root / ".agents" / "plugins" / "marketplace.json",
        repo_root / "python" / "tensorrt_model_connect",
        repo_root / "scripts" / "docker_build_gb300.sh",
    )
    missing = [str(path.relative_to(repo_root)) for path in required if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"{repo_root} is not a TensorRT-Model-Connect checkout; missing: {joined}"
        )


def _safe_workspace_id(value: str) -> str:
    safe = SAFE_NAME_PATTERN.sub("-", value.strip()).strip("-.")
    if not safe:
        raise RuntimeError("workspace identifier does not contain a safe container-name character")
    return safe


def detect_workspace_id(repo_root: Path) -> str | None:
    marker = repo_root / ".workspace_id"
    if marker.is_file():
        value = marker.read_text(encoding="utf-8").strip()
        if value:
            return _safe_workspace_id(value)

    for part in repo_root.parts:
        match = AGENT_PATTERN.fullmatch(part)
        if match:
            return f"agent-{match.group(1)}"
    return None


def expected_container_name(workspace_id: str | None) -> str:
    override = os.environ.get("TRTMC_CONTAINER_NAME", "").strip()
    if override:
        return _safe_workspace_id(override)
    if workspace_id:
        return f"{DEFAULT_CONTAINER_PREFIX}-{workspace_id}"
    return DEFAULT_CONTAINER_PREFIX


def _completed_error(command: Sequence[str], result: subprocess.CompletedProcess[str]) -> RuntimeError:
    detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
    return RuntimeError(f"{' '.join(command)} failed: {detail}")


class ContainerManager:
    def __init__(self, repo_root: Path, runner: CommandRunner = _default_runner):
        self.repo_root = repo_root.resolve()
        verify_repo_identity(self.repo_root)
        self.runner = runner
        self.workspace_id = detect_workspace_id(self.repo_root)
        self.name_overridden = bool(os.environ.get("TRTMC_CONTAINER_NAME", "").strip())
        self.expected_name = expected_container_name(self.workspace_id)
        self.image = os.environ.get("TRTMC_CONTAINER_IMAGE", DEFAULT_CONTAINER_IMAGE)
        self.requested_workdir = os.environ.get(
            "TRTMC_CONTAINER_WORKDIR", DEFAULT_CONTAINER_WORKDIR
        )

    def _run(self, command: Sequence[str], *, require_success: bool = True) -> subprocess.CompletedProcess[str]:
        result = self.runner(tuple(command))
        if require_success and result.returncode != 0:
            raise _completed_error(command, result)
        return result

    def list_containers(self) -> list[ContainerRecord]:
        ids_result = self._run(("docker", "ps", "-aq"))
        ids = [line.strip() for line in ids_result.stdout.splitlines() if line.strip()]
        if not ids:
            return []

        inspect_result = self._run(("docker", "inspect", *ids))
        payload = json.loads(inspect_result.stdout)
        records: list[ContainerRecord] = []
        for item in payload:
            mounts = tuple(
                Mount(
                    source=str(mount.get("Source", "")),
                    destination=str(mount.get("Destination", "")),
                )
                for mount in item.get("Mounts", [])
                if mount.get("Source") and mount.get("Destination")
            )
            records.append(
                ContainerRecord(
                    name=str(item.get("Name", "")).lstrip("/"),
                    image=str(item.get("Config", {}).get("Image", "")),
                    running=bool(item.get("State", {}).get("Running", False)),
                    mounts=mounts,
                )
            )
        return records

    def _repo_mount(self, container: ContainerRecord) -> Mount | None:
        repo_path = _real_path(self.repo_root)
        matches = [mount for mount in container.mounts if _real_path(mount.source) == repo_path]
        if not matches:
            return None
        if len(matches) > 1:
            raise RuntimeError(
                f"container {container.name} mounts {self.repo_root} more than once"
            )
        mount = matches[0]
        explicit_workdir = os.environ.get("TRTMC_CONTAINER_WORKDIR", "").strip()
        if explicit_workdir and mount.destination != explicit_workdir:
            raise RuntimeError(
                f"container {container.name} mounts the checkout at {mount.destination}, "
                f"not requested {explicit_workdir}"
            )
        return mount

    def _select(self, containers: Sequence[ContainerRecord]) -> tuple[ContainerRecord | None, str]:
        by_name = {container.name: container for container in containers}
        named = by_name.get(self.expected_name)
        if named is not None:
            mount = self._repo_mount(named)
            if mount is None:
                raise RuntimeError(
                    f"container name {self.expected_name} is already used by another checkout"
                )
            return named, mount.destination

        matches: list[tuple[ContainerRecord, Mount]] = []
        for container in containers:
            mount = self._repo_mount(container)
            if mount is not None:
                matches.append((container, mount))

        if self.name_overridden and matches:
            names = ", ".join(sorted(container.name for container, _mount in matches))
            raise RuntimeError(
                f"requested container {self.expected_name} does not exist, but this "
                f"checkout is already mounted by: {names}"
            )
        if len(matches) == 1:
            container, mount = matches[0]
            return container, mount.destination
        if len(matches) > 1:
            names = ", ".join(sorted(container.name for container, _mount in matches))
            raise RuntimeError(
                f"multiple containers mount {self.repo_root}: {names}; "
                "set TRTMC_CONTAINER_NAME explicitly"
            )
        return None, self.requested_workdir

    def resolve(self) -> ContainerContext:
        container, workdir = self._select(self.list_containers())
        if container is None:
            return ContainerContext(
                repo_root=str(self.repo_root),
                workspace_id=self.workspace_id,
                expected_container_name=self.expected_name,
                container_name=self.expected_name,
                container_workdir=workdir,
                container_image=self.image,
                state="missing",
            )
        return ContainerContext(
            repo_root=str(self.repo_root),
            workspace_id=self.workspace_id,
            expected_container_name=self.expected_name,
            container_name=container.name,
            container_workdir=workdir,
            container_image=container.image,
            state="running" if container.running else "stopped",
        )

    def _build_image(self) -> None:
        if self.image != DEFAULT_CONTAINER_IMAGE:
            self._run(("docker", "image", "inspect", self.image))
            return
        build_script = self.repo_root / "scripts" / "docker_build_gb300.sh"
        self._run(("bash", str(build_script)))
        self._run(("docker", "image", "inspect", self.image))

    def _create_container(self, workdir: str) -> None:
        storage_root = Path(
            os.environ.get("TRTMC_STORAGE_ROOT", str(Path.home() / ".cache" / "trtmc"))
        ).expanduser().resolve()
        hf_cache = Path(
            os.environ.get(
                "TRTMC_HF_CACHE",
                os.environ.get(
                    "HF_HOME", str(Path.home() / ".cache" / "huggingface")
                )
                + "/hub",
            )
        ).expanduser().resolve()
        engine_dir = storage_root / "engines"
        engine_dir.mkdir(parents=True, exist_ok=True)
        hf_cache.mkdir(parents=True, exist_ok=True)

        command = (
            "docker",
            "run",
            "-d",
            "--gpus",
            "all",
            "-v",
            f"{self.repo_root}:{workdir}",
            "-v",
            f"{storage_root}:{storage_root}",
            "-v",
            f"{hf_cache}:/root/.cache/huggingface/hub",
            "-e",
            f"ENGINE_DIR={engine_dir}",
            "-w",
            workdir,
            "--name",
            self.expected_name,
            self.image,
            "sleep",
            "infinity",
        )
        self._run(command)

    def ensure(self) -> ContainerContext:
        context = self.resolve()
        if context.state == "running":
            return context
        if context.state == "stopped":
            self._run(("docker", "start", context.container_name))
            return ContainerContext(**{**asdict(context), "state": "running"})

        self._build_image()
        self._create_container(context.container_workdir)
        return ContainerContext(
            **{
                **asdict(context),
                "state": "running",
                "container_image": self.image,
                "created": True,
            }
        )


def build_docker_exec(context: ContainerContext, command: Sequence[str]) -> list[str]:
    escaped = " ".join(shlex.quote(part) for part in command)
    shell_command = f"cd {shlex.quote(context.container_workdir)} && exec {escaped}"
    return ["docker", "exec", context.container_name, "bash", "-lc", shell_command]


def _print_context(context: ContainerContext, pretty: bool) -> None:
    print(json.dumps(asdict(context), indent=2 if pretty else None, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find or provision the dev container for the current TensorRT-Model-Connect checkout."
    )
    parser.add_argument("--cwd", help="override the path used to find the repository")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    resolve = subparsers.add_parser("resolve", help="inspect container mapping without mutation")
    resolve.add_argument("--pretty", action="store_true")

    ensure = subparsers.add_parser("ensure", help="start or provision the matching container")
    ensure.add_argument("--pretty", action="store_true")

    run = subparsers.add_parser("run", help="ensure a container and run a command in it")
    run.add_argument("--print-only", action="store_true")
    run.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        repo_root = find_repo_root(Path(args.cwd or os.getcwd()))
        manager = ContainerManager(repo_root)
        if args.subcommand == "resolve":
            _print_context(manager.resolve(), args.pretty)
            return 0
        if args.subcommand == "ensure":
            _print_context(manager.ensure(), args.pretty)
            return 0

        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            raise RuntimeError("no command provided after 'run --'")
        context = manager.resolve() if args.print_only else manager.ensure()
        docker_command = build_docker_exec(context, command)
        if args.print_only:
            print(" ".join(shlex.quote(part) for part in docker_command))
            return 0
        return subprocess.run(docker_command, check=False).returncode
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
