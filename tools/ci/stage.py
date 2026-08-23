# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attach one named pipeline stage to its run-owned container.

Boundary: host-to-container execution and cancellation; stage definitions live in ``pipeline``.
"""

from __future__ import annotations

import os
import signal
import subprocess

from .container import ContainerConfig
from .environment import COMMON_ENVIRONMENT, TRUSTED_ENVIRONMENT, forwarded_environment
from .process import CiError, CommandRunner


class ContainerStageRunner:
    """Run one stage and remove its exact container immediately on cancellation."""

    def __init__(self, stage: str, env: dict[str, str] | None = None):
        self.stage = stage
        self.env = dict(env or os.environ)
        self.config = ContainerConfig.from_environment(self.env)
        self.commands = CommandRunner(cwd=self.config.workspace, env=self.env)
        self.process: subprocess.Popen[str] | None = None

    def run(self) -> int:
        if not self._container_is_running():
            raise CiError(
                f"CI container '{self.config.name}' is not running. Start it with "
                "'python3 -m tools.ci container start' before running stages."
            )
        previous = {
            signal_number: signal.signal(signal_number, self._cancel)
            for signal_number in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            self.process = subprocess.Popen(
                self._docker_command(),
                cwd=self.config.workspace,
                env=self.env,
                text=True,
                start_new_session=True,
            )
            return self.process.wait()
        finally:
            self.process = None
            for signal_number, handler in previous.items():
                signal.signal(signal_number, handler)

    def _container_is_running(self) -> bool:
        result = self.commands.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.config.name],
            check=False,
            capture_output=True,
        )
        return result.stdout.strip() == "true"

    def _docker_command(self) -> list[str]:
        names = forwarded_environment(
            COMMON_ENVIRONMENT if self.config.hardened else TRUSTED_ENVIRONMENT,
            self.env,
        )
        forwarded = [item for name in names for item in ("-e", name)]
        command = [
            "docker",
            "exec",
            "-w",
            str(self.config.workspace),
            *forwarded,
            self.config.name,
            "python3",
        ]
        if self.config.hardened:
            command.extend(
                [
                    "-I",
                    "-c",
                    "import importlib.util, runpy, sys; root=sys.argv.pop(1); "
                    "spec=importlib.util.spec_from_file_location('tools', "
                    "root+'/tools/__init__.py', submodule_search_locations=[root+'/tools']); "
                    "module=importlib.util.module_from_spec(spec); sys.modules['tools']=module; "
                    "spec.loader.exec_module(module); "
                    "runpy.run_module('tools.ci', run_name='__main__')",
                    str(self.config.workspace),
                ]
            )
        else:
            command.extend(["-m", "tools.ci"])
        command.extend(
            [
                "pipeline",
                self.stage,
            ]
        )
        return command

    def _cancel(self, signal_number: int, _frame: object) -> None:
        subprocess.run(
            ["docker", "rm", "-f", self.config.name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if self.process is not None and self.process.poll() is None:
            # The handler may interrupt ``Popen.wait`` itself, so waiting for the
            # same child again here can deadlock.  The docker client has its own
            # process group and the container was already removed above.
            os.killpg(self.process.pid, signal.SIGKILL)
        raise SystemExit(130 if signal_number == signal.SIGINT else 143)
