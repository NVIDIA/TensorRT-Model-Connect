# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build or reuse the single current TRTMC development image."""

from __future__ import annotations

from pathlib import Path

from .process import CiError, CommandRunner, GitHubFiles


class DockerImageManager:
    def __init__(self, repository: Path, env: dict[str, str]):
        self.repository = repository.resolve()
        self.env = env
        self.commands = CommandRunner(cwd=self.repository, env=env)
        self.github = GitHubFiles(env)

    def ensure(self) -> str:
        image = self.env.get("TRTMC_CI_IMAGE", "trtmc-ci:trt11.1")
        self.commands.run(
            [
                "docker",
                "build",
                "--file",
                "Dockerfile",
                "--tag",
                image,
                ".",
            ]
        )
        if self._inspect(image).returncode:
            raise CiError(f"Docker image was not created: {image}")
        self.github.environment("TRTMC_CI_IMAGE", image)
        print(f"TRTMC_CI_IMAGE={image}")
        return image

    def _inspect(self, image: str):
        return self.commands.run(
            ["docker", "image", "inspect", image],
            check=False,
            capture_output=True,
        )
