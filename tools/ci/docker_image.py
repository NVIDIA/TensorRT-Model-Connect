# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build or reuse the single current TRTMC development image."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import platform
from pathlib import Path


class DockerImageManager:
    def __init__(self, repository: Path, env: dict[str, str]):
        from .process import CommandRunner, GitHubFiles

        self.repository = repository.resolve()
        self.env = env
        self.commands = CommandRunner(cwd=self.repository, env=env)
        self.github = GitHubFiles(env)

    def ensure(self) -> str:
        image = self.env.get("TRTMC_CI_IMAGE", "trtmc-ci:trt11.4")
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
        self.github.environment("TRTMC_CI_IMAGE", image)
        print(f"TRTMC_CI_IMAGE={image}")
        return image


_PACKAGE = "nvidia-cusparselt-cu13"
_VERSION = "0.8.1"
_LIBRARY = "nvidia/cusparselt/lib/libcusparseLt.so.0"
_LIBRARY_SIZE = 274143825
_LIBRARY_SHA256 = "fbb03bbf8d2238670f24256c5452281c2a660162899b838bb4d0cc06dad027df"
_BAD_TAG = "Tag: py3-none-manylinux2014_sbsa\n"
_GOOD_TAG = "Tag: py3-none-manylinux2014_aarch64\n"


def _record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")


def _verify_library(distribution: metadata.Distribution) -> None:
    """Match the public ARM wheel payload, not just a filename or a tag."""
    library = Path(distribution.locate_file(_LIBRARY))
    if library.stat().st_size != _LIBRARY_SIZE:
        raise ValueError("Unexpected cuSPARSELt library size; refusing metadata repair")
    with library.open("rb") as source:
        header = source.read(64)
        source.seek(0)
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 183:
        raise ValueError("cuSPARSELt is not an AArch64 ELF library; refusing metadata repair")
    if digest != _LIBRARY_SHA256:
        raise ValueError("Unexpected cuSPARSELt library digest; refusing metadata repair")


def repair_cusparselt_wheel() -> bool:
    """Repair only the known wheel; all other package compatibility checks remain active."""
    if platform.machine() != "aarch64":
        return False
    try:
        distribution = metadata.distribution(_PACKAGE)
    except metadata.PackageNotFoundError:
        return False
    if distribution.version != _VERSION:
        return False
    info = Path(distribution.locate_file(f"nvidia_cusparselt_cu13-{_VERSION}.dist-info"))
    wheel = info / "WHEEL"
    original = wheel.read_bytes()
    tags = [line for line in original.decode().splitlines(keepends=True) if line.startswith("Tag:")]
    if tags == [_GOOD_TAG]:
        return False
    if tags != [_BAD_TAG] or any((info / name).exists() for name in ("RECORD.jws", "RECORD.p7s")):
        raise ValueError("Unexpected or signed cuSPARSELt wheel metadata; refusing repair")
    _verify_library(distribution)
    record = info / "RECORD"
    rows = list(csv.reader(io.StringIO(record.read_text())))
    wheel_name = f"{info.name}/WHEEL"
    entries = [index for index, row in enumerate(rows) if row[0] == wheel_name]
    if len(entries) != 1 or rows[entries[0]][1:] != [_record_hash(original), str(len(original))]:
        raise ValueError("cuSPARSELt WHEEL disagrees with RECORD; refusing metadata repair")
    corrected = original.replace(_BAD_TAG.encode(), _GOOD_TAG.encode())
    rows[entries[0]] = [wheel_name, _record_hash(corrected), str(len(corrected))]
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    wheel.write_bytes(corrected)
    record.write_text(output.getvalue())
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate known CI image package metadata")
    parser.add_argument("action", choices=("repair-cusparselt-wheel",))
    parser.parse_args()
    print(
        "Corrected verified cuSPARSELt ARM wheel metadata"
        if repair_cusparselt_wheel()
        else "No cuSPARSELt metadata repair needed"
    )
