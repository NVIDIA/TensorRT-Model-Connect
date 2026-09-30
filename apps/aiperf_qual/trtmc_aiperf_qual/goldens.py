# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference goldens: the reference backend's observations, cached per suite and reference identity.

Reference numerics depend on the platform (GPU architecture, framework and CUDA
versions), so goldens are shared only within one platform fingerprint and are
regenerated automatically on a new platform. Layout (identical locally and in a shared store)::

    <root>/<suite>/<platform_id>/<key>/golden.jsonl    {"request_sha", "sample_id", "observation"} per line
    <root>/<suite>/<platform_id>/<key>/manifest.json   suite, reference, platform, host, numerics
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from .suites import Suite, canonical, sha256_text


def platform_id(fingerprint: Mapping[str, Any]) -> str:
    """Readable, stable directory name: GPU architecture plus a hash of the whole fingerprint."""
    return f"{fingerprint.get('gpu_arch', 'unknown')}-{sha256_text(canonical(dict(fingerprint)))[:10]}"


def golden_key(suite: Suite, reference: Mapping[str, Any], fingerprint: Mapping[str, Any]) -> str:
    """Any change to the suite samples, the reference identity, or the platform yields a new key."""
    return sha256_text(canonical({"suite_key": suite.key, "reference": dict(reference),
                                  "platform": dict(fingerprint)}))[:24]


class GoldenStore:
    def __init__(self, root: Path, read_url: str | None = None) -> None:
        self.root = root
        self.read_url = read_url.rstrip("/") if read_url else None

    def _relative(self, suite: str, platform: str, key: str) -> str:
        return f"{suite}/{platform}/{key}"

    def load(self, suite: Suite, platform: str, key: str) -> dict[str, Any] | None:
        relative = self._relative(suite.name, platform, key)
        directory = self.root / relative
        if not (directory / "golden.jsonl").is_file() and self.read_url:
            self._fetch(relative, directory)
        path = directory / "golden.jsonl"
        if not path.is_file():
            return None
        goldens = {}
        for line in path.read_text().splitlines():
            item = json.loads(line)
            goldens[item["request_sha"]] = item["observation"]
        missing = [s["sample_id"] for s in suite.samples if s["request_sha"] not in goldens]
        return None if missing else goldens

    def save(self, suite: Suite, platform: str, key: str, goldens: Mapping[str, Any],
             manifest: Mapping[str, Any]) -> Path:
        directory = self.root / self._relative(suite.name, platform, key)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "golden.jsonl", "w") as handle:
            for sample in suite.samples:
                handle.write(json.dumps({"request_sha": sample["request_sha"], "sample_id": sample["sample_id"],
                                         "observation": goldens[sample["request_sha"]]}) + "\n")
        (directory / "manifest.json").write_text(json.dumps(dict(manifest), indent=2))
        return directory

    def _fetch(self, relative: str, directory: Path) -> None:
        try:
            files = {name: urllib.request.urlopen(f"{self.read_url}/{relative}/{name}", timeout=30).read()
                     for name in ("manifest.json", "golden.jsonl")}
        except (urllib.error.URLError, TimeoutError):
            return
        directory.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (directory / name).write_bytes(data)
