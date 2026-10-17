#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the contributor-visible CPU gate used by local hooks and public CI.

The host owns diff selection and Docker lifecycle. Source-only C++ and Python
units run in the same hardened, GPU-free container boundary used by premerge.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import re
import urllib.parse
import urllib.request
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

# The image staging command needs only the standard library and declaration data.
if __package__:
    from tools.ci.container import CiContainer
    from tools.ci.context import CiContext
    from tools.ci.process import CiError, CommandRunner, GitHubFiles
    from tools.ci.quality import SourceQualityChecks
    from tools.ci.stage import ContainerStageRunner
    from tools import test_impact


class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        if urllib.parse.urlsplit(new_url).scheme != "https":
            raise ValueError("Native dependency redirects must retain HTTPS")
        return super().redirect_request(request, response, code, message, headers, new_url)


def _archive(family: str, value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "cmake_variable",
        "filename",
        "url",
        "sha256",
        "systems",
        "machines",
    }:
        raise ValueError(f"Invalid native dependency declaration for {family}")
    variable, filename = value["cmake_variable"], value["filename"]
    if (
        family.upper() in {"CMAKE", "TRTMC", "BUILD"}
        or not isinstance(variable, str)
        or not re.fullmatch(re.escape(family.upper()) + r"_[A-Z0-9_]*ARCHIVE", variable)
    ):
        raise ValueError(f"Native dependency archive cache variable must belong to {family}")
    if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", filename):
        raise ValueError("Native dependency filename must be a plain basename")
    if not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
        raise ValueError("Native dependency requires a SHA256 digest")
    if not isinstance(value["url"], str):
        raise ValueError("Native dependency requires an HTTPS URL")
    url = urllib.parse.urlsplit(value["url"])
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.fragment:
        raise ValueError("Native dependency requires an HTTPS URL without credentials")
    for key in ("systems", "machines"):
        if not isinstance(value[key], list) or not all(
            isinstance(item, str) and item for item in value[key]
        ):
            raise ValueError(f"Native dependency {key} must be a list of names")
    if not value["systems"]:
        raise ValueError("Native dependency requires a target system")
    return value


def _download(archive: dict, destination: Path) -> dict:
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Native dependency destination already exists: {destination}")
    temporary = destination.with_name(destination.name + ".partial")
    digest = hashlib.sha256()
    size = 0
    opener = urllib.request.build_opener(HTTPSRedirectHandler())
    try:
        with temporary.open("xb") as output, opener.open(archive["url"], timeout=120) as source:
            resolved_url = source.geturl()
            if urllib.parse.urlsplit(resolved_url).scheme != "https":
                raise ValueError("Native dependency download must retain HTTPS")
            while block := source.read(1024 * 1024):
                digest.update(block)
                size += len(block)
                output.write(block)
        if digest.hexdigest() != archive["sha256"]:
            raise ValueError(f"Native dependency SHA256 mismatch: {archive['filename']}")
        temporary.rename(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "url": archive["url"],
        "resolved_url": resolved_url,
        "sha256": digest.hexdigest(),
        "bytes": size,
        "path": str(destination),
    }


def stage(manifests: Path, output: Path, *, system: str, machine: str) -> None:
    if output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise ValueError("Native dependency output must not traverse a symlink")
    output.mkdir(parents=True, exist_ok=False)
    selected = []
    required = set()
    variables, destinations = set(), set()
    for manifest in sorted(manifests.glob("*.json")):
        family = manifest.stem
        if manifest.is_symlink() or not re.fullmatch(r"[a-z][a-z0-9_]*", family):
            raise ValueError("Native dependency declaration must be a regular family JSON file")
        declaration = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            not isinstance(declaration, dict)
            or set(declaration) != {"schema_version", "artifacts"}
            or type(declaration["schema_version"]) is not int
            or declaration["schema_version"] != 1
        ):
            raise ValueError(f"Unsupported native dependency declaration: {manifest}")
        if not isinstance(declaration["artifacts"], dict):
            raise ValueError("Native dependency artifacts must be an object")
        for value in declaration["artifacts"].values():
            archive = _archive(family, value)
            if system in archive["systems"]:
                required.add(archive["cmake_variable"])
            if system not in archive["systems"] or (
                archive["machines"] and machine not in archive["machines"]
            ):
                continue
            destination = output / family / archive["filename"]
            variable = archive["cmake_variable"]
            if variable in variables or destination in destinations:
                raise ValueError(
                    f"Duplicate native dependency destination or cache variable: {variable}"
                )
            variables.add(variable)
            destinations.add(destination)
            selected.append((variable, archive, destination))
    if required != variables:
        raise ValueError(
            f"No native dependency archive for {system}/{machine}: {sorted(required - variables)}"
        )
    receipt, cache = [], []
    for variable, archive, destination in selected:
        destination.parent.mkdir(exist_ok=True)
        receipt.append({"cmake_variable": variable, **_download(archive, destination)})
        # Bracket arguments preserve paths literally, including CMake metacharacters.
        delimiter = "="
        while f"]{delimiter}]" in str(destination):
            delimiter += "="
        cache.append(
            f'set({variable} [{delimiter}[{destination}]{delimiter}] CACHE FILEPATH "" FORCE)\n'
        )
    (output / "cache.cmake").write_text("".join(cache), encoding="utf-8")
    (output / "downloads.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


class CommunityCI:
    """Coordinate the public, source-only validation contract."""

    def __init__(
        self,
        repository: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.repository = (repository or Path.cwd()).resolve()
        self.env = dict(env or os.environ)
        self.commands = CommandRunner(cwd=self.repository, env=self.env)
        self.github = GitHubFiles(self.env)

    def source_quality(self, base: str | None) -> None:
        resolved_base = self.resolve_base(base)
        context = CiContext(
            repository=self.repository,
            env={**self.env, "CI_BASE_REF": resolved_base},
        )
        quality = SourceQualityChecks(context)
        name = "New-architecture source quality"
        print(f"::group::{name}")
        try:
            quality.run()
        except CiError as error:
            print(f"::error title={name}::{error}")
            raise
        finally:
            print("::endgroup::")

    def impact(self, base: str | None) -> dict[str, object]:
        resolved_base = self.resolve_base(base)
        try:
            test_impact.validate(self.repository)
            paths = test_impact.changed_files(self.repository, resolved_base, "HEAD")
            result = test_impact.classify(self.repository, paths)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            raise CiError(f"Impact analysis failed: {error}") from error
        summary = {
            "scope": result.scope,
            "families": list(result.families),
            "changed_paths": list(result.changed_files),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        self.github.output(
            "families",
            json.dumps(summary["families"], separators=(",", ":")),
        )
        self.github.summary("### Community CPU ownership and impact")
        self.github.summary(f"- Families: `{len(result.families)}`")
        self.github.summary(
            "- Changed paths: "
            + (", ".join(f"`{path}`" for path in result.changed_files) or "none")
        )
        return summary

    def unit(self) -> None:
        image = self._ensure_cpu_image()
        runner_temp = Path(self.env.get("RUNNER_TEMP", tempfile.gettempdir())).resolve()
        runner_temp.mkdir(parents=True, exist_ok=True)
        container_name = f"trtmc-community-{os.getpid()}"
        with tempfile.TemporaryDirectory(
            prefix="trtmc-community-unit-",
            dir=runner_temp,
        ) as scratch:
            container_env = {
                **self.env,
                "TRTMC_CI_WORKSPACE": str(self.repository),
                "TRTMC_CI_IMAGE": image,
                "TRTMC_CI_CONTAINER_NAME": container_name,
                "TRTMC_CI_HARDENED": "true",
                "TRTMC_CI_SCRATCH_HOST": scratch,
                "GITHUB_RUN_ID": self.env.get("GITHUB_RUN_ID", f"local-{os.getpid()}"),
                "GITHUB_RUN_ATTEMPT": self.env.get("GITHUB_RUN_ATTEMPT", "1"),
            }
            try:
                CiContainer(container_env).start()
                return_code = ContainerStageRunner("premerge-unit", container_env).run()
                if return_code:
                    raise CiError(f"Community CPU unit stage failed with exit code {return_code}")
            finally:
                self.commands.run(
                    ["docker", "rm", "-f", container_name],
                    check=False,
                    capture_output=True,
                )

    def resolve_base(self, explicit: str | None) -> str:
        configured = explicit or self.env.get("TRTMC_COMMUNITY_BASE_REF", "")
        if not configured:
            raise CiError("--base or TRTMC_COMMUNITY_BASE_REF is required")
        result = self.commands.run(
            ["git", "rev-parse", "--verify", f"{configured}^{{commit}}"],
            check=False,
            capture_output=True,
        )
        if result.returncode:
            raise CiError(f"Community CPU base does not resolve: {configured}")
        revision = result.stdout.strip()
        print(f"Community CPU base: {configured} ({revision})")
        return revision

    def _ensure_cpu_image(self) -> str:
        image = "trtmc-community-cpu:local"
        # Do not expose the checkout to a connected dependency-build stage.
        with tempfile.TemporaryDirectory(prefix="trtmc-cpu-image-") as directory:
            context = Path(directory)

            def copy_input(source: Path, destination: Path) -> None:
                if source.is_symlink() or any(parent.is_symlink() for parent in source.parents):
                    raise CiError(f"CPU image input must not traverse a symlink: {source}")
                shutil.copyfile(source, destination)

            copy_input(
                self.repository / "requirements/community-ci.txt", context / "community-ci.txt"
            )
            copy_input(
                self.repository / "tools/community_ci.py",
                context / "native-dependencies.py",
            )
            declarations = context / "native-declarations"
            declarations.mkdir()
            for manifest in sorted(self.repository.glob("families/*/ci/native-dependencies.json")):
                copy_input(manifest, declarations / f"{manifest.parents[1].name}.json")
            self.commands.run(
                [
                    "docker",
                    "build",
                    "--file",
                    str(self.repository / "Dockerfile.community-cpu"),
                    "--tag",
                    image,
                    str(context),
                ]
            )
        return image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ("source-quality", "impact"):
        command = commands.add_parser(name)
        command.add_argument("--base")

    commands.add_parser("unit", help="Run the complete source-only unit suite")
    staging = commands.add_parser(
        "stage-native-dependencies", help="Stage declared archives in the CPU image"
    )
    staging.add_argument("manifests", type=Path)
    staging.add_argument("output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "stage-native-dependencies":
        stage(
            arguments.manifests,
            arguments.output,
            system=platform.system(),
            machine=platform.machine(),
        )
        return 0
    if not __package__:
        raise SystemExit("Run CI stages with python3 -m tools.community_ci")
    runner = CommunityCI()
    try:
        if arguments.command == "source-quality":
            runner.source_quality(arguments.base)
        elif arguments.command == "impact":
            runner.impact(arguments.base)
        elif arguments.command == "unit":
            runner.unit()
    except CiError as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
