# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download exact public PyPI artifacts without executing package source code.

Boundary: online artifact retrieval only; profile installation and verification are offline.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request


MAX_PACKAGE_COUNT = 256
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _response_host(response, expected: str, label: str) -> None:
    final_url = urllib.parse.urlparse(response.geturl())
    if final_url.scheme != "https" or final_url.hostname != expected:
        raise RuntimeError(f"{label} redirected to an untrusted URL")


def _download_sdist(requirement: str, destination: Path) -> None:
    distribution, separator, version = requirement.partition("==")
    if not separator:
        raise RuntimeError(f"invalid exact requirement: {requirement}")
    distribution = distribution.partition("[")[0]
    endpoint = (
        "https://pypi.org/pypi/"
        + urllib.parse.quote(distribution, safe="")
        + "/"
        + urllib.parse.quote(version, safe="")
        + "/json"
    )
    request = urllib.request.Request(
        endpoint,
        headers={"User-Agent": "trtmc-profile-preparer/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        _response_host(response, "pypi.org", f"PyPI metadata for {requirement}")
        payload_bytes = response.read(MAX_METADATA_BYTES + 1)
    if len(payload_bytes) > MAX_METADATA_BYTES:
        raise RuntimeError(f"PyPI metadata is unexpectedly large for {requirement}")
    payload = json.loads(payload_bytes)
    candidates = [
        item for item in payload.get("urls", []) if item.get("packagetype") == "sdist"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"no unique source distribution is available for {requirement}")
    record = candidates[0]
    filename = record.get("filename")
    url = record.get("url")
    expected = record.get("digests", {}).get("sha256")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not isinstance(url, str)
        or not isinstance(expected, str)
        or len(expected) != 64
    ):
        raise RuntimeError(f"PyPI returned an invalid source record for {requirement}")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
        raise RuntimeError(f"PyPI returned an untrusted source URL for {requirement}")
    target = destination / filename
    if target.is_file() and _sha256(target) == expected:
        return
    temporary = destination / f".{filename}.{os.getpid()}.tmp"
    artifact_request = urllib.request.Request(
        url,
        headers={"User-Agent": "trtmc-profile-preparer/1"},
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(artifact_request, timeout=300) as response, temporary.open(
            "wb"
        ) as output:
            _response_host(response, "files.pythonhosted.org", requirement)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_ARTIFACT_BYTES:
                    raise RuntimeError(f"source distribution is too large for {requirement}")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"source distribution digest mismatch for {requirement}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def download_packages(destination: Path, requirements: list[str]) -> None:
    if not requirements or len(requirements) > MAX_PACKAGE_COUNT:
        raise ValueError(f"profile package count must be between 1 and {MAX_PACKAGE_COUNT}")
    destination.mkdir(parents=True, exist_ok=True)
    for requirement in requirements:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--no-deps",
                "--only-binary=:all:",
                "--index-url",
                "https://pypi.org/simple",
                "--dest",
                str(destination),
                requirement,
            ],
            check=False,
        )
        if result.returncode:
            _download_sdist(requirement, destination)
        total_bytes = sum(
            path.stat().st_size
            for path in destination.iterdir()
            if path.is_file() and not path.is_symlink()
        )
        if total_bytes > MAX_TOTAL_BYTES:
            raise RuntimeError("prepared profile packages exceed the 16 GiB run limit")


def main(arguments: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if arguments is None else arguments)
    if len(values) < 2:
        raise SystemExit("usage: profile_downloader.py DESTINATION NAME==VERSION [...]")
    download_packages(Path(values[0]), values[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
