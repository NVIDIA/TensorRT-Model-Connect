# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Digest-verified public model artifacts for network-disabled model proofs.

Boundary: artifact retrieval, digest validation, and isolated cache projection only;
model-specific artifact selection and semantics remain family-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import urllib.parse
import urllib.request

from .context import CiContext
from .process import CiError

_MAX_FILE_BYTES = 3 << 30
_MAX_CONTRACT_BYTES = 3 << 30
_ALLOWED_ORIGINS = {
    "api.ngc.nvidia.com",
    "raw.githubusercontent.com",
}
_S3_ORIGIN = re.compile(r"[a-z0-9][a-z0-9-]{2,62}\.s3\.amazonaws\.com")


def _allowed_origin(hostname: str | None) -> bool:
    return hostname in _ALLOWED_ORIGINS or bool(
        isinstance(hostname, str) and _S3_ORIGIN.fullmatch(hostname)
    )


@dataclass(frozen=True)
class ModelArtifactFile:
    path: str
    url: str
    sha256: str
    size: int

    def as_payload(self) -> dict[str, object]:
        return {"path": self.path, "url": self.url, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class ModelArtifactContract:
    family: str
    relative_path: str
    environment_variable: str
    files: tuple[ModelArtifactFile, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "environment_variable": self.environment_variable,
            "files": [item.as_payload() for item in self.files],
        }


def _relative(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CiError(f"{field} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CiError(f"{field} must be a canonical relative path")
    return value


def parse_model_artifact_contract(
    owner: dict[str, object], family: str, manifest: Path, suite: str | None
) -> ModelArtifactContract | None:
    raw = owner.get("model_artifact_cache")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise CiError(f"model_artifact_cache must be a table in {manifest}")
    suites = raw.get("suites")
    if suites is not None and (
        not isinstance(suites, list)
        or not suites
        or any(item not in {"premerge", "nightly"} for item in suites)
        or len(suites) != len(set(suites))
    ):
        raise CiError("model_artifact_cache.suites must select premerge and/or nightly")
    if suites is not None and suite is not None and suite not in suites:
        return None
    relative_path = _relative(raw.get("relative_path"), "model_artifact_cache.relative_path")
    if PurePosixPath(relative_path).parts[0] != family:
        raise CiError("model_artifact_cache.relative_path must be owned by its E2E family")
    environment = raw.get("environment_variable")
    if not isinstance(environment, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", environment):
        raise CiError("model_artifact_cache.environment_variable is invalid")
    file_values = raw.get("files")
    if not isinstance(file_values, list) or not file_values or len(file_values) > 8:
        raise CiError("model_artifact_cache.files must contain between one and eight files")
    files: list[ModelArtifactFile] = []
    seen: set[str] = set()
    for index, value in enumerate(file_values):
        if not isinstance(value, dict):
            raise CiError(f"model_artifact_cache.files[{index}] must be a table")
        path = _relative(value.get("path"), f"model_artifact_cache.files[{index}].path")
        if len(PurePosixPath(path).parts) != 1:
            raise CiError(f"model_artifact_cache.files[{index}].path must be a filename")
        url = value.get("url")
        parsed = urllib.parse.urlsplit(url) if isinstance(url, str) else None
        if (
            parsed is None
            or parsed.scheme != "https"
            or not _allowed_origin(parsed.hostname)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise CiError(f"model_artifact_cache.files[{index}].url is not an allowed HTTPS URL")
        digest = value.get("sha256")
        size = value.get("size")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise CiError(f"model_artifact_cache.files[{index}].sha256 is invalid")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or size > _MAX_FILE_BYTES
        ):
            raise CiError(f"model_artifact_cache.files[{index}].size is invalid")
        if path in seen:
            raise CiError(f"duplicate model artifact path: {path}")
        seen.add(path)
        files.append(ModelArtifactFile(path, url, digest, size))
    if sum(item.size for item in files) > _MAX_CONTRACT_BYTES:
        raise CiError("model_artifact_cache exceeds the 3 GiB total limit")
    return ModelArtifactContract(family, relative_path, environment, tuple(files))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_artifact(path: Path, contract: ModelArtifactFile) -> None:
    if path.is_symlink() or not path.is_file():
        raise CiError(f"model artifact is not a regular file: {path}")
    if path.stat().st_size != contract.size or _sha256(path) != contract.sha256:
        raise CiError(f"model artifact does not match its pinned size/digest: {path}")


class ModelArtifactCacheWarmer:
    def __init__(self, context: CiContext):
        self.context = context

    def warm_contract(self, contract: ModelArtifactContract) -> Path:
        configured = self.context.env.get(
            "TRTMC_MODEL_ARTIFACT_CACHE_ROOT"
        ) or self.context.env.get("TRTMC_MODEL_REFERENCE_CACHE_ROOT", "")
        if not configured:
            raise CiError("TRTMC_MODEL_ARTIFACT_CACHE_ROOT is required")
        repository = self.context.repository.resolve(strict=True)
        requested = Path(configured)
        unresolved_root = requested.resolve(strict=False)
        if (
            unresolved_root == Path("/")
            or unresolved_root == repository
            or unresolved_root.is_relative_to(repository)
        ):
            raise CiError("model artifact cache root is invalid")
        requested.mkdir(parents=True, exist_ok=True)
        root = requested.resolve(strict=True)
        if root == Path("/") or root == repository or root.is_relative_to(repository):
            raise CiError("model artifact cache root is invalid")
        destination = root / "model-artifacts" / contract.relative_path
        lock_dir = root / ".artifact-locks"
        if lock_dir.is_symlink():
            raise CiError("model artifact cache lock directory must not be a symlink")
        lock_dir.mkdir(exist_ok=True)
        lock_path = lock_dir / (
            hashlib.sha256(contract.relative_path.encode()).hexdigest() + ".lock"
        )
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            destination.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink() or not destination.resolve().is_relative_to(root):
                raise CiError("model artifact cache destination is unsafe")
            for item in contract.files:
                target = destination.joinpath(*PurePosixPath(item.path).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    validate_artifact(target, item)
                    continue
                handle, temporary_name = tempfile.mkstemp(
                    prefix=f".{target.name}.", dir=target.parent
                )
                os.close(handle)
                temporary = Path(temporary_name)
                try:
                    digest = hashlib.sha256()
                    size = 0
                    request = urllib.request.Request(
                        item.url, headers={"User-Agent": "trtmc-model-proof/1"}
                    )
                    with (
                        urllib.request.urlopen(request, timeout=60) as response,
                        temporary.open("wb") as output,
                    ):
                        while block := response.read(1 << 20):
                            size += len(block)
                            if size > item.size:
                                raise CiError(
                                    f"model artifact exceeds its declared size: {item.path}"
                                )
                            digest.update(block)
                            output.write(block)
                    if size != item.size or digest.hexdigest() != item.sha256:
                        raise CiError(f"downloaded model artifact failed verification: {item.path}")
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
                validate_artifact(target, item)
        return destination


class ModelArtifactCache:
    def __init__(self, context: CiContext, model: str):
        self.context = context
        self.model = model

    def prepare(self, payload: dict[str, object] | None, work: Path, artifacts: Path) -> None:
        if payload is None:
            return
        contract = parse_model_artifact_contract(
            {"model_artifact_cache": payload},
            self.model,
            Path("model proof selection"),
            suite=None,
        )
        if contract is None:  # Defensive: a non-null payload always selects a contract.
            raise CiError("model artifact cache payload did not select a contract")
        files = contract.files
        source = ModelArtifactCacheWarmer(self.context).warm_contract(contract)
        destination = work / "model-artifacts" / contract.relative_path
        if destination.exists():
            raise CiError("proof-private model artifact destination already exists")
        destination.mkdir(parents=True)
        for item in files:
            source_file = source.joinpath(*PurePosixPath(item.path).parts)
            validate_artifact(source_file, item)
            target = destination.joinpath(*PurePosixPath(item.path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
            validate_artifact(target, item)
        evidence = {
            "schema_version": 1,
            "model": self.model,
            "isolation": "selected-digest-private",
            "relative_path": contract.relative_path,
            "container_storage_root": "/work/model-artifacts",
            "files": [item.as_payload() for item in files],
        }
        (artifacts / "model-artifact-cache.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
