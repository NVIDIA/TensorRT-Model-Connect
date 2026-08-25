# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warm pinned model-reference source checkouts on one trusted CI host.

Boundary: this module may fetch public source into the host-local shared cache.
The model proof remains network-disabled and copies only a verified commit into
its private view.
"""

from __future__ import annotations

from dataclasses import dataclass
import tensorrt_model_connect.utils.fcntl_shim as fcntl
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import tomllib

from .context import CiContext
from .process import CiError


_GIT_TIMEOUT = "10m"


@dataclass(frozen=True)
class ModelReferenceContract:
    """One suite-selected pinned source checkout declared by an E2E owner."""

    family: str
    repository: str
    revision: str
    relative_path: str
    entrypoint: str
    environment_variable: str = ""

    def as_payload(self) -> dict[str, str]:
        """Return the exact contract shape embedded in proof selection."""
        payload = {
            "repository": self.repository,
            "revision": self.revision,
            "relative_path": self.relative_path,
            "entrypoint": self.entrypoint,
        }
        if self.environment_variable:
            payload["environment_variable"] = self.environment_variable
        return payload


def parse_model_reference_contract(
    owner: dict[str, object],
    family: str,
    manifest: Path,
    suite: str | None,
) -> ModelReferenceContract | None:
    """Validate and select one owner-declared reference-cache contract."""
    raw = owner.get("model_reference_cache")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise CiError(f"model_reference_cache must be a table in {manifest}")

    suites = raw.get("suites")
    if suites is not None and (
        not isinstance(suites, list)
        or not suites
        or any(not isinstance(item, str) or item not in {"premerge", "nightly"} for item in suites)
        or len(suites) != len(set(suites))
    ):
        raise CiError(
            "model_reference_cache.suites must be a unique non-empty list of premerge or nightly"
        )

    def relative(field: str) -> str:
        value = raw.get(field)
        if (
            not isinstance(value, str)
            or not value
            or "\\" in value
            or any(character in value for character in "\r\n\t")
        ):
            raise CiError(f"model_reference_cache.{field} must be a non-empty POSIX path")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise CiError(f"model_reference_cache.{field} must be a canonical relative path")
        return value

    revision = raw.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise CiError("model_reference_cache.revision must be a full lowercase Git commit")
    repository = raw.get("repository")
    if (
        not isinstance(repository, str)
        or not repository
        or any(character in repository for character in "\r\n\t")
    ):
        raise CiError("model_reference_cache.repository must be a non-empty single-line string")
    relative_path = relative("relative_path")
    if PurePosixPath(relative_path).parts[0] != family:
        raise CiError(
            "model_reference_cache.relative_path must be owned by the selected E2E family"
        )
    entrypoint = relative("entrypoint")
    environment_variable = raw.get("environment_variable", "")
    if not isinstance(environment_variable, str) or (
        environment_variable and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", environment_variable)
    ):
        raise CiError(
            "model_reference_cache.environment_variable must be a valid environment variable name"
        )
    if suites is not None and suite is not None and suite not in suites:
        return None
    return ModelReferenceContract(
        family=family,
        repository=repository,
        revision=revision,
        relative_path=relative_path,
        entrypoint=entrypoint,
        environment_variable=environment_variable,
    )


class ModelReferenceCacheWarmer:
    """Discover and atomically provision every declared reference for one suite."""

    def __init__(self, context: CiContext):
        self.context = context

    def warm(self, suite: str) -> list[Path]:
        if suite not in {"premerge", "nightly"}:
            raise CiError("model reference cache suite must be premerge or nightly")
        root = self._cache_root()
        contracts = self._contracts(suite)
        destinations = [self._warm_one(root, contract) for contract in contracts]
        print(f"Model reference cache ready: {len(destinations)} {suite} contract(s) under {root}")
        return destinations

    def warm_contract(self, contract: ModelReferenceContract) -> Path:
        """Provision one already-validated proof contract on first use."""
        return self._warm_one(self._cache_root(), contract)

    def _cache_root(self) -> Path:
        configured = self.context.env.get("TRTMC_MODEL_REFERENCE_CACHE_ROOT", "")
        if not configured:
            raise CiError("TRTMC_MODEL_REFERENCE_CACHE_ROOT is required")
        requested = Path(configured)
        try:
            requested.mkdir(parents=True, exist_ok=True)
            root = requested.resolve(strict=True)
        except OSError as error:
            raise CiError(f"model reference cache root is unavailable: {requested}") from error
        if not root.is_dir() or root == Path("/") or root == self.context.repository:
            raise CiError("model reference cache root is invalid")
        return root

    def _contracts(self, suite: str) -> list[ModelReferenceContract]:
        models_root = self.context.repository / "tests/e2e/models"
        if not models_root.is_dir():
            raise CiError(f"E2E model ownership root is unavailable: {models_root}")
        contracts: list[ModelReferenceContract] = []
        seen_destinations: set[str] = set()
        for manifest in sorted(models_root.glob("*/MODEL.toml")):
            try:
                owner = tomllib.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
                raise CiError(f"could not read model reference contract: {manifest}") from error
            family = manifest.parent.name
            if owner.get("id") != family:
                raise CiError(f"invalid projected E2E manifest: {manifest}")
            contract = parse_model_reference_contract(owner, family, manifest, suite)
            if contract is None:
                continue
            if contract.relative_path in seen_destinations:
                raise CiError(
                    f"duplicate model reference cache destination: {contract.relative_path}"
                )
            seen_destinations.add(contract.relative_path)
            contracts.append(contract)
        return contracts

    def _warm_one(self, root: Path, contract: ModelReferenceContract) -> Path:
        destination = root.joinpath(*PurePosixPath(contract.relative_path).parts)
        parent = self._safe_parent(root, destination.parent)
        lock_dir = root / ".locks"
        if lock_dir.is_symlink():
            raise CiError("model reference cache lock directory must not be a symlink")
        lock_dir.mkdir(exist_ok=True)
        lock_name = hashlib.sha256(contract.relative_path.encode("utf-8")).hexdigest()
        lock_path = lock_dir / f"{lock_name}.lock"
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if os.path.lexists(destination):
                self._validate_checkout(destination, contract)
                print(f"  CACHED {contract.relative_path} @ {contract.revision[:12]}")
                return destination
            temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
            try:
                self._fetch_checkout(temporary, contract)
                if os.path.lexists(destination):
                    raise CiError(
                        "model reference cache destination appeared during publish: "
                        f"{contract.relative_path}"
                    )
                os.replace(temporary, destination)
            finally:
                if os.path.lexists(temporary):
                    if temporary.is_dir() and not temporary.is_symlink():
                        shutil.rmtree(temporary)
                    else:
                        temporary.unlink()
            print(f"  FETCHED {contract.relative_path} @ {contract.revision[:12]}")
            return destination

    @staticmethod
    def _safe_parent(root: Path, parent: Path) -> Path:
        relative = parent.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise CiError(f"model reference cache path must not contain symlinks: {current}")
            try:
                current.mkdir(exist_ok=True)
            except OSError as error:
                raise CiError(f"model reference cache parent is unavailable: {current}") from error
            if not current.is_dir() or not current.resolve().is_relative_to(root):
                raise CiError(f"model reference cache parent is invalid: {current}")
        return current

    def _fetch_checkout(
        self,
        temporary: Path,
        contract: ModelReferenceContract,
    ) -> None:
        updates = {
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
        self.context.run(
            ["git", "init", "--quiet", temporary],
            limit=_GIT_TIMEOUT,
            capture_output=True,
            updates=updates,
        )
        self.context.run(
            ["git", "-C", temporary, "remote", "add", "origin", contract.repository],
            limit=_GIT_TIMEOUT,
            capture_output=True,
            updates=updates,
        )
        self.context.run(
            [
                "git",
                "-C",
                temporary,
                "fetch",
                "--quiet",
                "--depth=1",
                "--no-tags",
                "origin",
                contract.revision,
            ],
            limit=_GIT_TIMEOUT,
            capture_output=True,
            updates=updates,
        )
        self.context.run(
            [
                "git",
                "-C",
                temporary,
                "-c",
                "advice.detachedHead=false",
                "checkout",
                "--quiet",
                "--detach",
                "FETCH_HEAD",
            ],
            limit=_GIT_TIMEOUT,
            capture_output=True,
            updates=updates,
        )
        self._validate_checkout(temporary, contract)

    def _validate_checkout(
        self,
        source: Path,
        contract: ModelReferenceContract,
    ) -> None:
        if source.is_symlink() or not source.is_dir():
            raise CiError(f"model reference cache destination is invalid: {contract.relative_path}")
        revision = self.context.output(["git", "-C", source, "rev-parse", "HEAD^{commit}"])
        if revision != contract.revision:
            raise CiError(
                f"model reference cache revision mismatch for {contract.relative_path}: "
                f"expected {contract.revision}, found {revision}"
            )
        repository = self.context.output(
            ["git", "-C", source, "config", "--get", "remote.origin.url"]
        )
        if repository != contract.repository:
            raise CiError(
                f"model reference cache repository mismatch for {contract.relative_path}: "
                f"expected {contract.repository}, found {repository}"
            )
        entrypoint = self.context.run(
            [
                "git",
                "-C",
                source,
                "cat-file",
                "-e",
                f"{contract.revision}:{contract.entrypoint}",
            ],
            check=False,
            capture_output=True,
        )
        if entrypoint.returncode:
            raise CiError(
                f"model reference cache entrypoint is absent from {contract.revision}: "
                f"{contract.entrypoint}"
            )
