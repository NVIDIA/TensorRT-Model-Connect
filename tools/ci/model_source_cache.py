# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and privately materialize pinned local model-source packages.

Boundary: the trusted host may read one digest-pinned archive from a configured
cache.  The proof container receives only the validated, symlink-free payload.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import tempfile

from .context import CiContext
from .process import CiError


MODEL_SOURCE_CACHE_ROOT_ENV = "TRTMC_MODEL_SOURCE_CACHE_ROOT"
_MAX_ARCHIVE_MEMBERS = 4096
_MAX_COMPRESSED_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024
_MAX_ARCHIVE_FILE_BYTES = 16 * 1024 * 1024 * 1024
_MAX_MATERIALIZED_BYTES = 32 * 1024 * 1024 * 1024
_MAX_TAR_STREAM_BYTES = _MAX_MATERIALIZED_BYTES + 64 * 1024 * 1024
_MAX_MEMBER_PATH_BYTES = 4096
_MAX_PAX_HEADER_BYTES = 64 * 1024
_TAR_BLOCK_BYTES = 512
_TAR_METADATA_TYPES = {b"g", b"x", b"X", b"L", b"K"}
_MAX_CONSECUTIVE_TAR_METADATA_HEADERS = 16


@dataclass(frozen=True)
class ModelSourcePackageContract:
    """One suite-selected, digest-pinned local source package."""

    family: str
    cache_file: str
    sha256: str
    project_path: str
    entrypoint: str

    def as_payload(self) -> dict[str, str]:
        """Return the exact contract embedded in proof selection."""
        return {
            "cache_file": self.cache_file,
            "sha256": self.sha256,
            "project_path": self.project_path,
            "entrypoint": self.entrypoint,
        }


def _canonical_relative_path(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
        or any(character in value for character in "\r\n\t")
    ):
        raise CiError(f"model_source_package.{field} must be a non-empty POSIX path")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CiError(f"model_source_package.{field} must be valid UTF-8") from error
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CiError(f"model_source_package.{field} must be a canonical relative path")
    return value


def parse_model_source_package_contract(
    owner: dict[str, object],
    family: str,
    manifest: Path,
    suite: str | None,
) -> ModelSourcePackageContract | None:
    """Validate and select one owner-declared local source package."""
    raw = owner.get("model_source_package")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise CiError(f"model_source_package must be a table in {manifest}")

    suites = raw.get("suites")
    if (
        not isinstance(suites, list)
        or not suites
        or any(not isinstance(item, str) or item not in {"premerge", "nightly"} for item in suites)
        or len(suites) != len(set(suites))
    ):
        raise CiError(
            "model_source_package.suites must be a unique non-empty list of premerge or nightly"
        )

    cache_file = _canonical_relative_path(raw.get("cache_file"), "cache_file")
    if PurePosixPath(cache_file).parts[0] != family or not cache_file.endswith(".tar.gz"):
        raise CiError("model_source_package.cache_file must be a family-owned .tar.gz path")
    project_path = _canonical_relative_path(raw.get("project_path"), "project_path")
    if "," in project_path:
        raise CiError("model_source_package.project_path must not contain a comma")
    project_parts = PurePosixPath(project_path).parts
    if len(project_parts) < 3 or project_parts[:2] != ("artifacts", family):
        raise CiError("model_source_package.project_path must be owned by artifacts/<family>")
    entrypoint = _canonical_relative_path(raw.get("entrypoint"), "entrypoint")
    digest = raw.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise CiError("model_source_package.sha256 must be a full lowercase SHA-256 digest")
    if suite is not None and suite not in suites:
        return None
    return ModelSourcePackageContract(
        family=family,
        cache_file=cache_file,
        sha256=digest,
        project_path=project_path,
        entrypoint=entrypoint,
    )


@dataclass(frozen=True)
class _ArchiveMember:
    info: tarfile.TarInfo
    path: PurePosixPath
    kind: str
    link_target: PurePosixPath | None = None


def _normalized_member_path(info: tarfile.TarInfo) -> PurePosixPath:
    name = info.name
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or any(character in name for character in "\r\n\t")
    ):
        raise CiError(f"model source archive has an unsafe member path: {name!r}")
    parts = name.split("/")
    while parts and parts[0] == ".":
        parts.pop(0)
    while info.isdir() and parts and parts[-1] == "":
        parts.pop()
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise CiError(f"model source archive has an unsafe member path: {name!r}")
    try:
        normalized = PurePosixPath(*parts)
        encoded = normalized.as_posix().encode("utf-8")
    except UnicodeEncodeError as error:
        raise CiError("model source archive member path is not valid UTF-8") from error
    if len(encoded) > _MAX_MEMBER_PATH_BYTES:
        raise CiError("model source archive member path is too long")
    if normalized.is_absolute():
        raise CiError(f"model source archive has an absolute member path: {name!r}")
    return normalized


def _normalized_link_target(member_path: PurePosixPath, raw_target: str) -> PurePosixPath:
    if (
        not raw_target
        or "\x00" in raw_target
        or "\\" in raw_target
        or raw_target.startswith("/")
        or any(character in raw_target for character in "\r\n\t")
    ):
        raise CiError(f"model source archive symlink {member_path} has an unsafe target")
    target = PurePosixPath(raw_target)
    if (
        target.is_absolute()
        or target.as_posix() != raw_target
        or any(part in {"", ".", ".."} for part in target.parts)
    ):
        raise CiError(f"model source archive symlink {member_path} has a noncanonical target")
    try:
        encoded = target.as_posix().encode("utf-8")
    except UnicodeEncodeError as error:
        raise CiError("model source archive symlink target is not valid UTF-8") from error
    if len(encoded) > _MAX_MEMBER_PATH_BYTES:
        raise CiError(f"model source archive symlink {member_path} has a target that is too long")
    return member_path.parent / target


def _validate_pax_headers(headers: dict[str, str]) -> None:
    size = 0
    for key, value in headers.items():
        # Tarfile decodes binary SCHILY xattr values with surrogateescape.
        # These attributes are ignored during materialization; preserve their
        # original byte length solely to enforce the metadata bound.
        encoded_key = key.encode("utf-8", "surrogateescape")
        encoded_value = value.encode("utf-8", "surrogateescape")
        size += len(encoded_key) + len(encoded_value)
        if size > _MAX_PAX_HEADER_BYTES:
            raise CiError("model source archive PAX metadata exceeds the size limit")


def _tar_header_number(field: bytes) -> int:
    if field and field[0] & 0x80:
        if field[0] & 0x40:
            raise CiError("model source archive has a negative binary header size")
        return int.from_bytes(bytes([field[0] & 0x7F]) + field[1:], "big")
    stripped = field.strip(b" \0")
    if not stripped:
        return 0
    if any(byte < ord("0") or byte > ord("7") for byte in stripped):
        raise CiError("model source archive has an invalid octal header size")
    return int(stripped, 8)


def _pax_record_keys(payload: bytes) -> set[bytes]:
    keys: set[bytes] = set()
    offset = 0
    while offset < len(payload):
        separator = payload.find(b" ", offset)
        if separator <= offset:
            raise CiError("model source archive has malformed PAX metadata")
        length_field = payload[offset:separator]
        if not length_field.isdigit():
            raise CiError("model source archive has malformed PAX record length")
        record_length = int(length_field)
        record_end = offset + record_length
        if record_length <= separator - offset + 2 or record_end > len(payload):
            raise CiError("model source archive has an invalid PAX record length")
        record = payload[separator + 1 : record_end]
        if not record.endswith(b"\n") or b"=" not in record[:-1]:
            raise CiError("model source archive has a malformed PAX record")
        key, _value = record[:-1].split(b"=", 1)
        if not key:
            raise CiError("model source archive has an empty PAX metadata key")
        keys.add(key)
        offset = record_end
    return keys


def _preflight_tar_stream(path: Path) -> None:
    """Bound tar extension records before ``tarfile`` allocates their payloads."""
    stream_size = path.stat().st_size
    physical_headers = 0
    consecutive_metadata_headers = 0
    with path.open("rb") as stream:
        while stream.tell() < stream_size:
            header = stream.read(_TAR_BLOCK_BYTES)
            if len(header) != _TAR_BLOCK_BYTES:
                raise CiError("model source archive has a truncated tar header")
            if header == b"\0" * _TAR_BLOCK_BYTES:
                second_eof_block = stream.read(_TAR_BLOCK_BYTES)
                if second_eof_block != b"\0" * _TAR_BLOCK_BYTES:
                    raise CiError("model source archive has a noncanonical tar terminator")
                while trailing := stream.read(1024 * 1024):
                    if trailing.strip(b"\0"):
                        raise CiError("model source archive has nonzero data after its terminator")
                return
            physical_headers += 1
            if physical_headers > _MAX_ARCHIVE_MEMBERS:
                raise CiError("model source archive contains too many physical tar headers")
            payload_size = _tar_header_number(header[124:136])
            member_type = header[156:157]
            if member_type in _TAR_METADATA_TYPES:
                consecutive_metadata_headers += 1
                if consecutive_metadata_headers > _MAX_CONSECUTIVE_TAR_METADATA_HEADERS:
                    raise CiError("model source archive has too many consecutive metadata headers")
                if payload_size > _MAX_PAX_HEADER_BYTES:
                    raise CiError("model source archive metadata exceeds the size limit")
            else:
                consecutive_metadata_headers = 0
            if member_type == b"S":
                raise CiError("model source archive contains a GNU sparse member")
            padded_size = (
                (payload_size + _TAR_BLOCK_BYTES - 1) // _TAR_BLOCK_BYTES
            ) * _TAR_BLOCK_BYTES
            if stream.tell() + padded_size > stream_size:
                raise CiError("model source archive has a truncated tar payload")
            if member_type in {b"g", b"x", b"X"}:
                payload = stream.read(padded_size)
                keys = _pax_record_keys(payload[:payload_size])
                if b"size" in keys:
                    raise CiError(
                        "model source archive must not override file size in PAX metadata"
                    )
                if b"SCHILY.realsize" in keys or any(
                    key.startswith(b"GNU.sparse.") for key in keys
                ):
                    raise CiError("model source archive must not contain PAX sparse metadata")
            else:
                stream.seek(padded_size, os.SEEK_CUR)
    raise CiError("model source archive is missing its canonical tar terminator")


def _validated_members(archive: tarfile.TarFile) -> list[_ArchiveMember]:
    members: list[_ArchiveMember] = []
    indexed: dict[PurePosixPath, _ArchiveMember] = {}
    total_regular_bytes = 0
    raw_member_count = 0
    for info in archive:
        raw_member_count += 1
        if raw_member_count > _MAX_ARCHIVE_MEMBERS:
            raise CiError("model source archive contains too many members")
        _validate_pax_headers(info.pax_headers)
        if info.isdir() and info.name in {".", "./"}:
            # A tar root-directory marker carries no payload and has no path
            # inside the materialized project tree.  Ignore only this exact
            # directory form; other empty paths and root non-directories fail.
            continue
        path = _normalized_member_path(info)
        if path in indexed:
            raise CiError(f"model source archive has duplicate normalized member path: {path}")
        if info.isdir():
            member = _ArchiveMember(info, path, "directory")
        elif info.isreg():
            if getattr(info, "sparse", None):
                raise CiError(f"model source archive contains a sparse file: {path}")
            if info.size < 0 or info.size > _MAX_ARCHIVE_FILE_BYTES:
                raise CiError(f"model source archive file is too large: {path}")
            total_regular_bytes += info.size
            if total_regular_bytes > _MAX_MATERIALIZED_BYTES:
                raise CiError("model source archive expands beyond the size limit")
            member = _ArchiveMember(info, path, "file")
        elif info.issym():
            member = _ArchiveMember(
                info,
                path,
                "symlink",
                _normalized_link_target(path, info.linkname),
            )
        elif info.islnk():
            raise CiError(f"model source archive contains a hardlink: {path}")
        else:
            raise CiError(f"model source archive contains a non-file member: {path}")
        indexed[path] = member
        members.append(member)

    _validate_pax_headers(archive.pax_headers)
    for path, member in indexed.items():
        for parent in path.parents:
            if parent == PurePosixPath("."):
                break
            ancestor = indexed.get(parent)
            if ancestor is not None and ancestor.kind != "directory":
                raise CiError(f"model source archive member has a non-directory ancestor: {path}")
        if member.kind != "symlink":
            continue
        target = indexed.get(member.link_target)
        if target is None:
            raise CiError(f"model source archive symlink is dangling: {path}")
        if target.kind == "directory":
            raise CiError(f"model source archive symlink targets a directory: {path}")
        if target.kind == "symlink":
            raise CiError(f"model source archive contains a symlink chain: {path}")
        if target.kind != "file":
            raise CiError(f"model source archive symlink has an invalid target: {path}")
        total_regular_bytes += target.info.size
        if total_regular_bytes > _MAX_MATERIALIZED_BYTES:
            raise CiError("model source archive materialization exceeds the size limit")
    return members


def materialized_tree_sha256(root: Path) -> str:
    """Digest a symlink-free materialized tree by canonical path and contents."""
    root = root.resolve(strict=True)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise CiError(f"materialized model source contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            digest.update(b"D\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise CiError(f"materialized model source contains a special file: {relative}")
        digest.update(b"F\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(metadata.st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


class ModelSourcePackagePreparer:
    """Verify and atomically extract one selected archive before GPU leasing."""

    def __init__(self, context: CiContext, model: str):
        self.context = context
        self.model = model

    def prepare(
        self,
        contract: dict[str, str] | None,
        private_root: Path,
        artifacts_dir: Path,
    ) -> Path | None:
        if contract is None:
            return None
        configured = self.context.env.get(MODEL_SOURCE_CACHE_ROOT_ENV, "")
        if not configured:
            raise CiError(f"{MODEL_SOURCE_CACHE_ROOT_ENV} is required for {self.model}")
        requested_root = Path(configured)
        try:
            root = requested_root.resolve(strict=True)
        except OSError as error:
            raise CiError("model source package cache root is unavailable") from error
        if not root.is_dir() or root in {Path("/"), self.context.repository}:
            raise CiError("model source package cache root is invalid")

        relative = PurePosixPath(contract["cache_file"])
        archive_path = root.joinpath(*relative.parts)
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise CiError("selected model source package path contains a symlink")
        try:
            resolved_archive = archive_path.resolve(strict=True)
            resolved_archive.relative_to(root)
        except (OSError, ValueError) as error:
            raise CiError("selected model source package is unavailable") from error

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(resolved_archive, flags)
        except OSError as error:
            raise CiError("selected model source package is not a readable regular file") from error
        snapshot_descriptor = -1
        snapshot_path: Path | None = None
        tar_descriptor = -1
        tar_path: Path | None = None
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise CiError("selected model source package is not a regular file")
            private_root.mkdir(parents=True, mode=0o700, exist_ok=True)
            if private_root.is_symlink() or not private_root.is_dir():
                raise CiError("proof-private model source root is invalid")
            private_root.chmod(0o700)
            destination = private_root / "payload"
            if os.path.lexists(destination):
                raise CiError("proof-private model source destination already exists")

            snapshot_descriptor, raw_snapshot_path = tempfile.mkstemp(
                prefix=".archive-", suffix=".tar.gz", dir=private_root
            )
            snapshot_path = Path(raw_snapshot_path)
            digest = hashlib.sha256()
            with os.fdopen(snapshot_descriptor, "w+b", closefd=False) as snapshot:
                compressed_bytes = 0
                while chunk := os.read(descriptor, 1024 * 1024):
                    compressed_bytes += len(chunk)
                    if compressed_bytes > _MAX_COMPRESSED_ARCHIVE_BYTES:
                        raise CiError(
                            "selected model source package exceeds the compressed size limit"
                        )
                    digest.update(chunk)
                    snapshot.write(chunk)
                snapshot.flush()
                os.fsync(snapshot_descriptor)
                actual_digest = digest.hexdigest()
                if actual_digest != contract["sha256"]:
                    raise CiError(
                        "selected model source package SHA-256 mismatch: "
                        f"expected {contract['sha256']}, found {actual_digest}"
                    )
                os.fchmod(snapshot_descriptor, 0o400)
                snapshot.seek(0)

                tar_descriptor, raw_tar_path = tempfile.mkstemp(
                    prefix=".archive-unpacked-", suffix=".tar", dir=private_root
                )
                tar_path = Path(raw_tar_path)
                try:
                    with os.fdopen(tar_descriptor, "wb", closefd=False) as unpacked:
                        uncompressed_bytes = 0
                        with gzip.GzipFile(fileobj=snapshot, mode="rb") as compressed:
                            while chunk := compressed.read(1024 * 1024):
                                uncompressed_bytes += len(chunk)
                                if uncompressed_bytes > _MAX_TAR_STREAM_BYTES:
                                    raise CiError(
                                        "selected model source package exceeds the tar stream size limit"
                                    )
                                unpacked.write(chunk)
                        unpacked.flush()
                        os.fsync(tar_descriptor)
                    os.fchmod(tar_descriptor, 0o400)
                except (gzip.BadGzipFile, EOFError) as error:
                    raise CiError(
                        "selected model source package is not a valid gzip stream"
                    ) from error
                _preflight_tar_stream(tar_path)

                temporary = Path(tempfile.mkdtemp(prefix=".extract-", dir=private_root))
                try:
                    try:
                        with tarfile.open(tar_path, mode="r:") as archive:
                            members = _validated_members(archive)
                            self._extract(archive, members, temporary)
                    except (tarfile.TarError, EOFError) as error:
                        raise CiError(
                            "selected model source package is not a valid tar.gz"
                        ) from error

                    entrypoint_path = PurePosixPath(contract["entrypoint"])
                    indexed = {member.path: member for member in members}
                    entrypoint_member = indexed.get(entrypoint_path)
                    entrypoint = temporary.joinpath(*entrypoint_path.parts)
                    if (
                        entrypoint_member is None
                        or entrypoint_member.kind != "file"
                        or entrypoint.is_symlink()
                        or not entrypoint.is_file()
                    ):
                        raise CiError("model source package is missing its regular entrypoint")
                    tree_digest = materialized_tree_sha256(temporary)
                    os.replace(temporary, destination)
                except BaseException:
                    if temporary.exists():
                        shutil.rmtree(temporary)
                    raise
        finally:
            os.close(descriptor)
            if snapshot_descriptor >= 0:
                os.close(snapshot_descriptor)
            if snapshot_path is not None:
                snapshot_path.unlink(missing_ok=True)
            if tar_descriptor >= 0:
                os.close(tar_descriptor)
            if tar_path is not None:
                tar_path.unlink(missing_ok=True)

        counts = {
            kind: sum(member.kind == kind for member in members)
            for kind in ("file", "directory", "symlink")
        }
        evidence = {
            "schema_version": 1,
            "model": self.model,
            "isolation": "selected-digest-private",
            "cache_file": contract["cache_file"],
            "sha256": contract["sha256"],
            "project_path": contract["project_path"],
            "entrypoint": contract["entrypoint"],
            "container_path": f"/src/{contract['project_path']}",
            "copy_method": "verified-tar-materialization",
            "member_count": len(members),
            "regular_file_count": counts["file"],
            "directory_count": counts["directory"],
            "materialized_symlink_count": counts["symlink"],
            "materialized_tree_sha256": tree_digest,
        }
        evidence_path = artifacts_dir / "model-source-package.json"
        temporary_evidence = evidence_path.with_name(f".{evidence_path.name}.tmp-{os.getpid()}")
        temporary_evidence.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_evidence, evidence_path)
        return destination

    @staticmethod
    def _extract(
        archive: tarfile.TarFile,
        members: list[_ArchiveMember],
        destination: Path,
    ) -> None:
        directories = {destination}
        for member in members:
            path = destination.joinpath(*member.path.parts)
            directories.update(path.parents)
            if member.kind == "directory":
                directories.add(path)
        for directory in sorted(
            (path for path in directories if path.is_relative_to(destination)),
            key=lambda path: len(path.parts),
        ):
            directory.mkdir(mode=0o700, exist_ok=True)

        modes: dict[PurePosixPath, int] = {}
        for member in members:
            if member.kind != "file":
                continue
            target = destination.joinpath(*member.path.parts)
            source = archive.extractfile(member.info)
            if source is None:
                raise CiError(f"model source archive file could not be read: {member.path}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if target.stat().st_size != member.info.size:
                raise CiError(f"model source archive file size mismatch: {member.path}")
            mode = 0o700 if member.info.mode & 0o111 else 0o600
            target.chmod(mode)
            modes[member.path] = mode

        for member in members:
            if member.kind != "symlink":
                continue
            assert member.link_target is not None
            source = destination.joinpath(*member.link_target.parts)
            target = destination.joinpath(*member.path.parts)
            with source.open("rb") as input_file, target.open("xb") as output:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
            target.chmod(modes[member.link_target])
