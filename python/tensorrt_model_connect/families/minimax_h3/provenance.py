# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source and artifact provenance helpers for native MiniMax-H3 plans."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
import subprocess
import tempfile
from pathlib import Path

from .config import (
    CANVAS_MAX_ASPECT_RATIO,
    CANVAS_MAX_PIXELS,
    CANVAS_MIN_ASPECT_RATIO,
    CANVAS_MULTIPLE,
    CANVAS_SHORT_EDGE,
    NATIVE_EXPLICIT_CANVAS_SIZES,
    SOL_ENGINE_1344X768_124_TO_345F,
    TRT_DEFAULT_WORKSPACE_POLICY,
    native_plan_filenames,
)

CHECKPOINT_REVISION = "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
CHECKPOINT_REPOSITORY = "MiniMaxAI/MiniMax-H3"
HF_CACHE_REPOSITORY = "models--MiniMaxAI--MiniMax-H3"
DIFFUSERS_REFERENCE_REPOSITORY = "https://github.com/huggingface/diffusers.git"
DIFFUSERS_REFERENCE_REVISION = "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
DIFFUSERS_REFERENCE_TREE = "a9aeec5268dd9661565a3e0af9b298744eb416b2"
DIFFUSERS_REFERENCE_RELATIVE_PATH = "minimax_h3/reference/diffusers-abc5e9bf71fd"
DIFFUSERS_REFERENCE_ENTRYPOINT = "src/diffusers/__init__.py"
DIFFUSERS_REFERENCE_ENTRYPOINT_BYTES = 65_047
DIFFUSERS_REFERENCE_ENTRYPOINT_SHA256 = (
    "78bac2aa899c34b6d504e8dfb128d9475ad7baee179b3ad97d09ccef25999916"
)
DIFFUSERS_REFERENCE_ARCHIVE_ENTRIES = 2_772
DIFFUSERS_REFERENCE_ARCHIVE_BYTES = 50_469_043
DIFFUSERS_REFERENCE_ARCHIVE_SHA256 = (
    "372c820aece801258bd4cea2458a2b85ad536e9262d7b0bbcdd450eda2d664a9"
)
DIFFUSERS_REFERENCE_CONTAINER_ROOT = "/work/reference-private"
PLAN_FILENAMES = native_plan_filenames()
FIRST_BLOCK_CACHE_PLAN_FILENAMES = PLAN_FILENAMES
_FL2VA_CONDITIONING_PLAN_FILENAMES = (
    "vision_encoder.plan",
    "fl2va_keyframe_vae_encoder.plan",
)
_REQUIRED_SNAPSHOT_FILES = (
    "audio_vae/config.json",
    "audio_vae/diffusion_pytorch_model.safetensors",
    "modular_model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/tokenizer.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/diffusion_pytorch_model.safetensors.index.json",
)
_BUNDLE_MAGIC = b"BUNDLE\x01\x00"
_MAX_BUNDLE_HEADER_BYTES = 100 << 20
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_HF_LOCAL_DOWNLOAD_METADATA = Path(".cache/huggingface/download")
_CHECKPOINT_INDEX_FILES = (
    "text_encoder/model.safetensors.index.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/diffusion_pytorch_model.safetensors.index.json",
)


def sha256_file(path: Path, *, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def file_identity(path: Path) -> dict[str, int]:
    try:
        metadata = path.stat()
    except OSError as error:
        raise ValueError(f"MiniMax-H3 artifact is unavailable: {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"MiniMax-H3 artifact is not a regular file: {path}")
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
    }


def stable_file_record(path: Path, label: str) -> tuple[dict[str, int | str], dict[str, int]]:
    before = file_identity(path)
    record = file_record(path)
    after = file_identity(path)
    if after != before:
        raise ValueError(f"MiniMax-H3 artifact changed while hashing: {label}")
    return record, after


def validate_file_identity(path: Path, expected: dict[str, int], label: str) -> None:
    if file_identity(path) != expected:
        raise ValueError(f"MiniMax-H3 artifact changed while it was in use: {label}")


def _validate_record_object(record: object, label: str) -> tuple[int, str]:
    if not isinstance(record, dict):
        raise ValueError(f"MiniMax-H3 receipt is missing {label}")
    expected_size = record.get("bytes")
    expected_sha = record.get("sha256")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0:
        raise ValueError(f"MiniMax-H3 receipt has an invalid byte count for {label}")
    if not isinstance(expected_sha, str) or _SHA256.fullmatch(expected_sha) is None:
        raise ValueError(f"MiniMax-H3 receipt has an invalid SHA256 for {label}")
    return expected_size, expected_sha


def plan_filenames_for_profile(profile) -> tuple[str, ...]:
    if profile.first_block_cache is not True:
        raise ValueError("MiniMax-H3 build receipts require the dense FirstBlockCache profile")
    return native_plan_filenames()


def validate_workspace_limit_bytes(
    record: object,
    *,
    profile=None,
    additional_plan_filenames: tuple[str, ...] = (),
    excluded_plan_filenames: tuple[str, ...] = (),
) -> dict[str, int | str]:
    """Validate the exact per-plan TensorRT tactic-workspace provenance."""

    if profile is not None and profile.first_block_cache is not True:
        raise ValueError("MiniMax-H3 workspace validation requires FirstBlockCache")
    if not isinstance(additional_plan_filenames, tuple) or any(
        not isinstance(filename, str) or not filename for filename in additional_plan_filenames
    ):
        raise ValueError("MiniMax-H3 additional plan filenames must be a tuple of names")
    if not isinstance(excluded_plan_filenames, tuple) or any(
        not isinstance(filename, str) or not filename for filename in excluded_plan_filenames
    ):
        raise ValueError("MiniMax-H3 excluded plan filenames must be a tuple of names")
    base = native_plan_filenames()
    if not set(excluded_plan_filenames).issubset(base):
        raise ValueError("MiniMax-H3 excluded plan filenames are not selected native plans")
    expected = (
        *(filename for filename in base if filename not in excluded_plan_filenames),
        *additional_plan_filenames,
    )
    if len(set(expected)) != len(expected):
        raise ValueError("MiniMax-H3 workspace plan filenames must be unique")
    if not isinstance(record, dict) or set(record) != set(expected):
        raise ValueError(
            "MiniMax-H3 workspace_limit_bytes must cover exactly the selected native plans"
        )
    for filename, value in record.items():
        valid_numeric_limit = isinstance(value, int) and not isinstance(value, bool) and value > 0
        if not valid_numeric_limit and value != TRT_DEFAULT_WORKSPACE_POLICY:
            raise ValueError(
                f"MiniMax-H3 workspace_limit_bytes has an invalid value for {filename}"
            )
    return dict(record)


def validate_record(path: Path, record: object, label: str, *, hash_file: bool) -> None:
    expected_size, expected_sha = _validate_record_object(record, label)
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(f"MiniMax-H3 artifact size does not match its receipt: {label}")
    if hash_file and sha256_file(path) != expected_sha:
        raise ValueError(f"MiniMax-H3 artifact SHA256 does not match its receipt: {label}")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably replace ``path`` without exposing a partial artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_bytes(path, json.dumps(payload, indent=2).encode())


def _canonical_json_sha256(payload: object) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _checkpoint_snapshot_payload(files: dict[str, dict[str, int | str]]) -> dict:
    files = dict(sorted(files.items()))
    payload = {
        "repository": CHECKPOINT_REPOSITORY,
        "revision": CHECKPOINT_REVISION,
        "files": files,
    }
    inventory_sha256 = _canonical_json_sha256(payload)
    return {
        **payload,
        "file_count": len(files),
        "inventory_sha256": inventory_sha256,
    }


def _validate_checkpoint_indexes(snapshot: Path, files: dict[str, dict[str, int | str]]) -> None:
    missing = sorted(set(_REQUIRED_SNAPSHOT_FILES) - set(files))
    if missing:
        raise ValueError(f"MiniMax-H3 snapshot is incomplete; missing: {missing}")
    for index_name in _CHECKPOINT_INDEX_FILES:
        try:
            index_payload = (snapshot / index_name).read_bytes()
        except OSError as error:
            raise ValueError(f"MiniMax-H3 checkpoint index is unavailable: {index_name}") from error
        if hashlib.sha256(index_payload).hexdigest() != files[index_name]["sha256"]:
            raise ValueError(
                f"MiniMax-H3 checkpoint index changed while being validated: {index_name}"
            )
        try:
            index = json.loads(index_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"MiniMax-H3 checkpoint index is invalid: {index_name}") from error
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"MiniMax-H3 checkpoint index has no weight_map: {index_name}")
        filenames = tuple(weight_map.values())
        if any(
            not isinstance(filename, str)
            or not filename
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
            for filename in filenames
        ):
            raise ValueError(f"MiniMax-H3 checkpoint index has an invalid shard path: {index_name}")
        referenced = {(Path(index_name).parent / filename).as_posix() for filename in filenames}
        missing_shards = sorted(referenced - set(files))
        if missing_shards:
            raise ValueError(
                f"MiniMax-H3 checkpoint index references missing shards: {missing_shards}"
            )


def _checkpoint_file_record(path: Path, relative: str, blob_id: str) -> dict[str, int | str]:
    """Record HF identity without rereading LFS payloads.

    A canonical Hugging Face LFS blob name (or a local-dir ETag) is already the
    payload SHA-256. Git-backed metadata files are small, so validate their Git
    blob ID and retain a content SHA-256 in the inventory.
    """

    try:
        metadata = path.stat()
    except OSError as error:
        raise ValueError(f"MiniMax-H3 checkpoint file is unavailable: {relative}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise ValueError(f"MiniMax-H3 checkpoint file is invalid: {relative}")
    if _SHA256.fullmatch(blob_id) is not None:
        return {"blob_id": blob_id, "bytes": metadata.st_size, "sha256": blob_id}
    if _GIT_SHA.fullmatch(blob_id) is None:
        raise ValueError(f"MiniMax-H3 checkpoint file has an invalid HF blob ID: {relative}")

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"MiniMax-H3 checkpoint metadata is unreadable: {relative}") from error
    if len(payload) != metadata.st_size or path.stat().st_size != metadata.st_size:
        raise ValueError(f"MiniMax-H3 checkpoint metadata changed while reading: {relative}")
    try:
        git_blob = hashlib.sha1(usedforsecurity=False)
    except TypeError:  # pragma: no cover - for older Python implementations
        git_blob = hashlib.sha1()
    git_blob.update(f"blob {len(payload)}\0".encode())
    git_blob.update(payload)
    if git_blob.hexdigest() != blob_id:
        raise ValueError(f"MiniMax-H3 checkpoint Git blob ID mismatch: {relative}")
    return {
        "blob_id": blob_id,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _canonical_snapshot_entries(snapshot: Path) -> tuple[Path, ...]:
    entries: list[Path] = []

    def fail_walk(error: OSError) -> None:
        raise ValueError("MiniMax-H3 canonical snapshot could not be inventoried") from error

    for directory, directory_names, filenames in os.walk(
        snapshot, topdown=True, onerror=fail_walk, followlinks=False
    ):
        root = Path(directory)
        directory_names.sort()
        filenames.sort()
        if root == snapshot and "transformer_ref" in directory_names:
            directory_names.remove("transformer_ref")
        for name in directory_names:
            path = root / name
            metadata = path.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or _is_reparse_point(metadata)
                or not stat.S_ISDIR(metadata.st_mode)
            ):
                relative = path.relative_to(snapshot).as_posix()
                raise ValueError(
                    f"MiniMax-H3 canonical snapshot contains a linked directory: {relative}"
                )
        entries.extend(root / name for name in filenames)
    return tuple(entries)


def _canonical_checkpoint_snapshot_record(snapshot: Path) -> dict:
    """Describe an exact-revision canonical HF cache snapshot."""

    if snapshot.name != CHECKPOINT_REVISION or snapshot.parent.name != "snapshots":
        raise ValueError(
            f"MiniMax-H3 model path must resolve to pinned snapshot {CHECKPOINT_REVISION}"
        )
    repository_root = snapshot.parent.parent
    if repository_root.name != HF_CACHE_REPOSITORY:
        raise ValueError(f"MiniMax-H3 snapshot must belong to {CHECKPOINT_REPOSITORY}")
    blob_root = (repository_root / "blobs").resolve(strict=True)

    files: dict[str, dict[str, int | str]] = {}
    for path in _canonical_snapshot_entries(snapshot):
        relative = path.relative_to(snapshot).as_posix()
        if not path.is_symlink():
            raise ValueError(
                f"MiniMax-H3 canonical snapshot entry is not a cache symlink: {relative}"
            )
        try:
            target = (path.parent / os.readlink(path)).resolve(strict=True)
        except OSError as error:
            raise ValueError(f"MiniMax-H3 snapshot has a broken entry: {relative}") from error
        if target.parent != blob_root or not target.is_file():
            raise ValueError(
                f"MiniMax-H3 snapshot entry leaves its canonical blob cache: {relative}"
            )
        blob_id = target.name
        is_lfs_sha256 = _SHA256.fullmatch(blob_id) is not None
        if not is_lfs_sha256 and _GIT_SHA.fullmatch(blob_id) is None:
            raise ValueError(f"MiniMax-H3 snapshot entry has an invalid HF blob ID: {relative}")
        if path.suffix == ".safetensors" and not is_lfs_sha256:
            raise ValueError(
                f"MiniMax-H3 weight shard is not backed by an LFS SHA256 blob: {relative}"
            )
        files[relative] = _checkpoint_file_record(target, relative, blob_id)

    _validate_checkpoint_indexes(snapshot, files)
    return _checkpoint_snapshot_payload(files)


def _is_reparse_point(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _plain_snapshot_content_paths(snapshot: Path) -> tuple[Path, ...]:
    paths: list[Path] = []

    def fail_walk(error: OSError) -> None:
        raise ValueError("MiniMax-H3 local-dir checkpoint could not be inventoried") from error

    for directory, directory_names, filenames in os.walk(
        snapshot, topdown=True, onerror=fail_walk, followlinks=False
    ):
        root = Path(directory)
        directory_names.sort()
        filenames.sort()
        if root == snapshot:
            for excluded_directory in (".cache", "transformer_ref"):
                if excluded_directory in directory_names:
                    directory_names.remove(excluded_directory)
        for name in directory_names:
            path = root / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or _is_reparse_point(metadata):
                relative = path.relative_to(snapshot).as_posix()
                raise ValueError(
                    f"MiniMax-H3 local-dir checkpoint contains a linked directory: {relative}"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                relative = path.relative_to(snapshot).as_posix()
                raise ValueError(
                    f"MiniMax-H3 local-dir checkpoint contains a special entry: {relative}"
                )
        for name in filenames:
            path = root / name
            metadata = path.lstat()
            relative = path.relative_to(snapshot).as_posix()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or _is_reparse_point(metadata)
                or not stat.S_ISREG(metadata.st_mode)
            ):
                raise ValueError(
                    f"MiniMax-H3 local-dir checkpoint entry is not a regular file: {relative}"
                )
            paths.append(path)
    return tuple(paths)


def _plain_snapshot_metadata_paths(
    snapshot: Path, content_relative_paths: set[str]
) -> dict[str, Path]:
    metadata_root = snapshot / _HF_LOCAL_DOWNLOAD_METADATA
    if not metadata_root.is_dir() or metadata_root.is_symlink():
        raise ValueError(
            "MiniMax-H3 local-dir checkpoint is missing Hugging Face download metadata"
        )
    result = {
        relative: metadata_root / f"{relative}.metadata" for relative in content_relative_paths
    }
    missing = sorted(
        relative for relative, path in result.items() if not path.is_file() or path.is_symlink()
    )
    if missing:
        raise ValueError(f"MiniMax-H3 local-dir checkpoint is missing download metadata: {missing}")
    return result


def _read_plain_snapshot_metadata(path: Path, relative: str) -> str:
    try:
        fields = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(
            f"MiniMax-H3 local-dir download metadata is unreadable: {relative}"
        ) from error
    if len(fields) < 2 or fields[0] != CHECKPOINT_REVISION:
        raise ValueError(
            f"MiniMax-H3 local-dir metadata does not identify pinned revision: {relative}"
        )
    blob_id = fields[1]
    if _GIT_SHA.fullmatch(blob_id) is None and _SHA256.fullmatch(blob_id) is None:
        raise ValueError(f"MiniMax-H3 local-dir metadata has an invalid ETag: {relative}")
    return blob_id


def _plain_checkpoint_snapshot_record(snapshot: Path) -> dict:
    """Describe a pinned ``hf download --local-dir`` checkpoint."""

    content_paths = _plain_snapshot_content_paths(snapshot)
    content_by_relative = {path.relative_to(snapshot).as_posix(): path for path in content_paths}
    metadata_paths = _plain_snapshot_metadata_paths(snapshot, set(content_by_relative))
    files: dict[str, dict[str, int | str]] = {}
    for relative, path in sorted(content_by_relative.items()):
        blob_id = _read_plain_snapshot_metadata(metadata_paths[relative], relative)
        if path.suffix == ".safetensors" and len(blob_id) != 64:
            raise ValueError(f"MiniMax-H3 local-dir weight shard has a non-LFS ETag: {relative}")
        files[relative] = _checkpoint_file_record(path, relative, blob_id)

    _validate_checkpoint_indexes(snapshot, files)
    return _checkpoint_snapshot_payload(files)


def checkpoint_snapshot_record(snapshot: Path) -> dict:
    """Describe a pinned HF cache or ``--local-dir`` snapshot.

    LFS payload identity comes from the canonical blob name or local-dir ETag,
    while Git-backed metadata is read and hashed. This keeps the receipt stable
    without rereading the 144-GB checkpoint before and after every staged build.
    """

    snapshot = snapshot.absolute()
    try:
        metadata = snapshot.lstat()
    except OSError as error:
        raise ValueError("MiniMax-H3 model path is unavailable") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse_point(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise ValueError("MiniMax-H3 model path must be a regular snapshot directory")
    if snapshot.parent.name == "snapshots" and snapshot.parent.parent.name == HF_CACHE_REPOSITORY:
        return _canonical_checkpoint_snapshot_record(snapshot)
    return _plain_checkpoint_snapshot_record(snapshot)


def validate_checkpoint_snapshot_record(record: object) -> dict:
    if not isinstance(record, dict):
        raise ValueError("MiniMax-H3 receipt is missing checkpoint_snapshot")
    if set(record) != {
        "repository",
        "revision",
        "files",
        "file_count",
        "inventory_sha256",
    }:
        raise ValueError("MiniMax-H3 checkpoint snapshot has unexpected fields")
    if record.get("repository") != CHECKPOINT_REPOSITORY:
        raise ValueError("MiniMax-H3 checkpoint snapshot has the wrong repository")
    if record.get("revision") != CHECKPOINT_REVISION:
        raise ValueError("MiniMax-H3 checkpoint snapshot has the wrong revision")
    files = record.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("MiniMax-H3 checkpoint snapshot has no file inventory")
    if record.get("file_count") != len(files):
        raise ValueError("MiniMax-H3 checkpoint snapshot has the wrong file count")
    missing = sorted(set(_REQUIRED_SNAPSHOT_FILES) - set(files))
    if missing:
        raise ValueError(f"MiniMax-H3 checkpoint snapshot is incomplete; missing: {missing}")
    for relative, entry in files.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("MiniMax-H3 checkpoint snapshot has an invalid relative path")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.parts[0] in {".cache", "transformer_ref"}
        ):
            raise ValueError("MiniMax-H3 checkpoint snapshot has an invalid relative path")
        if not isinstance(entry, dict) or set(entry) != {"blob_id", "bytes", "sha256"}:
            raise ValueError(f"MiniMax-H3 checkpoint file has unexpected fields: {relative}")
        _, digest = _validate_record_object(entry, f"checkpoint file {relative}")
        blob_id = entry.get("blob_id") if isinstance(entry, dict) else None
        if not isinstance(blob_id, str) or (
            _GIT_SHA.fullmatch(blob_id) is None and _SHA256.fullmatch(blob_id) is None
        ):
            raise ValueError(f"MiniMax-H3 checkpoint file has an invalid blob ID: {relative}")
        if len(blob_id) == 64 and digest != blob_id:
            raise ValueError(f"MiniMax-H3 LFS digest does not match its blob ID: {relative}")
    payload = {
        "repository": record["repository"],
        "revision": record["revision"],
        "files": files,
    }
    expected_digest = record.get("inventory_sha256")
    if not isinstance(expected_digest, str) or _SHA256.fullmatch(expected_digest) is None:
        raise ValueError("MiniMax-H3 checkpoint snapshot has an invalid inventory SHA256")
    if _canonical_json_sha256(payload) != expected_digest:
        raise ValueError("MiniMax-H3 checkpoint snapshot inventory SHA256 does not match")
    return record


def builder_source_sha256() -> str:
    """Hash the semantic native builder surface shared by all entrypoints."""

    family_root = Path(__file__).resolve().parent
    package_root = family_root.parents[1]
    repo_root = package_root.parents[1]
    sources = [*family_root.glob("*.py"), package_root / "trt_compat.py"]
    digest = hashlib.sha256()
    for path in sorted(set(sources)):
        relative = path.relative_to(repo_root)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def validate_source_revision(revision: str) -> str:
    if _GIT_SHA.fullmatch(revision) is None:
        raise ValueError("MiniMax-H3 source revision must be a lowercase 40-character Git SHA")
    return revision


def validated_git_source_record(entrypoint: Path, *, expected_revision: str, label: str) -> dict:
    """Require a clean imported source checkout at an exact upstream commit."""

    expected_revision = validate_source_revision(expected_revision)
    entrypoint = entrypoint.resolve(strict=True)
    try:
        root_result = subprocess.run(
            ["git", "-C", str(entrypoint.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"MiniMax-H3 {label} must be imported from a Git checkout") from error
    root = Path(root_result.stdout.strip()).resolve(strict=True)
    try:
        relative_entrypoint = entrypoint.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"MiniMax-H3 {label} entrypoint is outside its Git checkout") from error
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_revision:
        raise ValueError(
            f"MiniMax-H3 {label} revision mismatch: expected {expected_revision}, got {head}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(f"MiniMax-H3 {label} Git checkout has tracked modifications")
    entrypoint_record, _ = stable_file_record(entrypoint, f"{label} entrypoint")
    return {
        "revision": head,
        "entrypoint": relative_entrypoint,
        "entrypoint_record": entrypoint_record,
        "tracked_worktree_clean": True,
    }


def _lstat_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    metadata = path.lstat()
    return (
        metadata.st_mode,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _resolve_archive_symlink(path: Path) -> Path:
    """Resolve a Git symlink payload without asking Windows to follow it directly.

    Git symlink payloads always use POSIX separators. Windows can create such a
    link, but ``Path.resolve`` on the link itself rejects some relative POSIX
    payloads with ``ERROR_INVALID_NAME``. Resolve the payload relative to its
    parent instead; the caller still validates the canonical target boundary.
    """

    payload = Path(os.readlink(path))
    target = payload if payload.is_absolute() else path.parent / payload
    return target.resolve(strict=True)


def _archive_layout(root: Path, label: str) -> list[tuple[str, str]]:
    try:
        paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    except OSError as error:
        raise ValueError(f"MiniMax-H3 {label} archive could not be inventoried") from error

    layout: list[tuple[str, str]] = []
    populated_directories: set[str] = set()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if ".git" in Path(relative).parts:
            raise ValueError(f"MiniMax-H3 {label} archive must not contain Git metadata")
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise ValueError(
                f"MiniMax-H3 {label} archive entry is unavailable: {relative}"
            ) from error
        if stat.S_ISDIR(mode):
            kind = "D"
        elif stat.S_ISREG(mode):
            kind = "F"
        elif stat.S_ISLNK(mode):
            kind = "L"
            try:
                target = _resolve_archive_symlink(path)
            except (OSError, RuntimeError) as error:
                raise ValueError(
                    f"MiniMax-H3 {label} archive has a broken symlink: {relative}"
                ) from error
            if not target.is_relative_to(root):
                raise ValueError(f"MiniMax-H3 {label} archive has an escaping symlink: {relative}")
        else:
            raise ValueError(f"MiniMax-H3 {label} archive contains a special file: {relative}")
        layout.append((relative, kind))
        if kind != "D":
            parent = Path(relative).parent
            while parent != Path("."):
                populated_directories.add(parent.as_posix())
                parent = parent.parent

    empty_directories = sorted(
        relative
        for relative, kind in layout
        if kind == "D" and relative not in populated_directories
    )
    if empty_directories:
        raise ValueError(
            f"MiniMax-H3 {label} archive contains untracked empty directories: {empty_directories}"
        )
    return layout


def _archive_inventory_record(root: Path, label: str) -> dict[str, int | str]:
    layout_before = _archive_layout(root, label)
    digest = hashlib.sha256()
    entry_count = 0
    total_bytes = 0
    for relative, kind in layout_before:
        if kind == "D":
            continue
        path = root / relative
        try:
            identity_before = _lstat_identity(path)
            if kind == "L":
                payload = os.fsencode(os.readlink(path))
                size = len(payload)
                content_sha256 = hashlib.sha256(payload).hexdigest()
            else:
                size = identity_before[3]
                content_sha256 = sha256_file(path)
            identity_after = _lstat_identity(path)
        except OSError as error:
            raise ValueError(
                f"MiniMax-H3 {label} archive entry changed while hashing: {relative}"
            ) from error
        # Windows reports reparse-point metadata size rather than the Git link
        # payload length for a symlink. The lstat tuple still detects mutation;
        # only regular files can additionally bind st_size to bytes hashed.
        if identity_after != identity_before or (kind != "L" and identity_before[3] != size):
            raise ValueError(f"MiniMax-H3 {label} archive entry changed while hashing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(kind.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_sha256.encode("ascii"))
        digest.update(b"\n")
        entry_count += 1
        total_bytes += size

    if _archive_layout(root, label) != layout_before:
        raise ValueError(f"MiniMax-H3 {label} archive changed while hashing")
    return {
        "entry_count": entry_count,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _stable_json_object(path: Path, label: str) -> tuple[dict, dict[str, int | str]]:
    if path.is_symlink():
        raise ValueError(f"MiniMax-H3 {label} evidence must not be a symlink")
    try:
        identity_before = file_identity(path)
        payload = path.read_bytes()
        identity_after = file_identity(path)
    except OSError as error:
        raise ValueError(f"MiniMax-H3 {label} evidence is unavailable: {path}") from error
    if identity_after != identity_before:
        raise ValueError(f"MiniMax-H3 {label} evidence changed while it was being read")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"MiniMax-H3 {label} evidence contains duplicate keys")
            result[key] = value
        return result

    try:
        decoded = json.loads(payload, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"MiniMax-H3 {label} evidence is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"MiniMax-H3 {label} evidence must be a JSON object")
    return decoded, {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def validated_git_archive_source_record(
    entrypoint: Path,
    *,
    evidence_path: Path,
    label: str,
) -> dict:
    """Bind the imported Diffusers source to the exact proof-private Git archive."""

    evidence_path = Path(evidence_path).absolute()
    evidence, evidence_record = _stable_json_object(evidence_path, label)
    expected_evidence = {
        "schema_version": 1,
        "model": "minimax_h3",
        "isolation": "selected-pinned-private",
        "repository": DIFFUSERS_REFERENCE_REPOSITORY,
        "reference_revision": DIFFUSERS_REFERENCE_REVISION,
        "reference_tree": DIFFUSERS_REFERENCE_TREE,
        "relative_path": DIFFUSERS_REFERENCE_RELATIVE_PATH,
        "entrypoint": DIFFUSERS_REFERENCE_ENTRYPOINT,
        "container_storage_root": DIFFUSERS_REFERENCE_CONTAINER_ROOT,
        "copy_method": "git-archive",
    }
    if set(evidence) != set(expected_evidence):
        raise ValueError(f"MiniMax-H3 {label} evidence has unsupported fields")
    for key, expected in expected_evidence.items():
        actual = evidence.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"MiniMax-H3 {label} evidence mismatch for {key}")

    storage_root = Path(DIFFUSERS_REFERENCE_CONTAINER_ROOT)
    if not storage_root.is_absolute():
        raise ValueError(f"MiniMax-H3 {label} container storage root is not absolute")
    archive_root = storage_root / DIFFUSERS_REFERENCE_RELATIVE_PATH
    try:
        resolved_root = archive_root.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"MiniMax-H3 {label} archive root is unavailable") from error
    if archive_root.is_symlink() or not archive_root.is_dir() or resolved_root != archive_root:
        raise ValueError(f"MiniMax-H3 {label} archive root is not canonical")

    expected_entrypoint = archive_root / DIFFUSERS_REFERENCE_ENTRYPOINT
    imported_entrypoint = Path(entrypoint).absolute()
    if imported_entrypoint != expected_entrypoint or imported_entrypoint.is_symlink():
        raise ValueError(f"MiniMax-H3 {label} was not imported from the selected Git archive")
    try:
        if imported_entrypoint.resolve(strict=True) != expected_entrypoint:
            raise ValueError(f"MiniMax-H3 {label} imported entrypoint is not canonical")
    except OSError as error:
        raise ValueError(f"MiniMax-H3 {label} imported entrypoint is unavailable") from error

    entrypoint_record, _ = stable_file_record(imported_entrypoint, f"{label} entrypoint")
    expected_entrypoint_record = {
        "bytes": DIFFUSERS_REFERENCE_ENTRYPOINT_BYTES,
        "sha256": DIFFUSERS_REFERENCE_ENTRYPOINT_SHA256,
    }
    if entrypoint_record != expected_entrypoint_record:
        raise ValueError(f"MiniMax-H3 {label} entrypoint does not match the pinned source")

    archive_inventory = _archive_inventory_record(archive_root, label)
    expected_inventory = {
        "entry_count": DIFFUSERS_REFERENCE_ARCHIVE_ENTRIES,
        "bytes": DIFFUSERS_REFERENCE_ARCHIVE_BYTES,
        "sha256": DIFFUSERS_REFERENCE_ARCHIVE_SHA256,
    }
    if archive_inventory != expected_inventory:
        raise ValueError(f"MiniMax-H3 {label} archive inventory does not match the pinned source")
    return {
        "qualification": "selected-pinned-git-archive",
        "repository": DIFFUSERS_REFERENCE_REPOSITORY,
        "revision": DIFFUSERS_REFERENCE_REVISION,
        "tree": DIFFUSERS_REFERENCE_TREE,
        "entrypoint": DIFFUSERS_REFERENCE_ENTRYPOINT,
        "entrypoint_record": entrypoint_record,
        "archive_inventory": archive_inventory,
        "evidence_record": evidence_record,
        "copy_method": "git-archive",
        "container_storage_root": DIFFUSERS_REFERENCE_CONTAINER_ROOT,
    }


def validate_git_archive_source_unchanged(
    entrypoint: Path,
    *,
    evidence_path: Path,
    expected_record: dict,
    label: str,
) -> None:
    """Revalidate an imported Git archive after the reference run."""

    if not isinstance(expected_record, dict):
        raise ValueError(f"MiniMax-H3 {label} expected archive record is invalid")
    current = validated_git_archive_source_record(
        entrypoint,
        evidence_path=evidence_path,
        label=label,
    )
    if current != expected_record:
        raise ValueError(f"MiniMax-H3 {label} archive changed while it was in use")


def serialized_profile(profile) -> dict:
    return json.loads(json.dumps(profile.__dict__))


def _validate_build_receipt_metadata(
    receipt: object,
    *,
    build_helper: Path,
    source_revision: str,
    profile,
    additional_plan_filenames: tuple[str, ...] = (),
) -> tuple[str, dict, dict]:
    if not isinstance(receipt, dict):
        raise ValueError("MiniMax-H3 build receipt must be a JSON object")
    source_revision = validate_source_revision(source_revision)
    source_sha = builder_source_sha256()
    expected = {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_revision": source_revision,
        "builder_source_sha256": source_sha,
        "build_helper_sha256": sha256_file(build_helper.resolve()),
        "profile": serialized_profile(profile),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"MiniMax-H3 build receipt does not match current {key}")
    selected_plans = (*plan_filenames_for_profile(profile), *additional_plan_filenames)
    if len(set(selected_plans)) != len(selected_plans):
        raise ValueError("MiniMax-H3 build receipt plan filenames must be unique")
    validate_workspace_limit_bytes(
        receipt.get("workspace_limit_bytes"),
        profile=profile,
        additional_plan_filenames=additional_plan_filenames,
    )
    snapshot_record = validate_checkpoint_snapshot_record(receipt.get("checkpoint_snapshot"))
    components = receipt.get("components")
    if not isinstance(components, dict) or set(components) != set(selected_plans):
        raise ValueError("MiniMax-H3 build receipt must cover exactly the selected native plans")
    for filename in selected_plans:
        _validate_record_object(components.get(filename), filename)
    assets = receipt.get("assets")
    tokenizer_record = assets.get("tokenizer.json") if isinstance(assets, dict) else None
    _validate_record_object(tokenizer_record, "tokenizer.json")
    return source_sha, components, snapshot_record


def validate_build_receipt(
    receipt: object,
    *,
    plans_dir: Path,
    snapshot: Path,
    tokenizer: Path,
    build_helper: Path,
    source_revision: str,
    profile,
    hash_files: bool,
    additional_plan_filenames: tuple[str, ...] = (),
) -> tuple[str, dict, dict, dict]:
    source_sha, components, recorded_snapshot = _validate_build_receipt_metadata(
        receipt,
        build_helper=build_helper,
        source_revision=source_revision,
        profile=profile,
        additional_plan_filenames=additional_plan_filenames,
    )
    current_snapshot = checkpoint_snapshot_record(snapshot)
    if recorded_snapshot != current_snapshot:
        raise ValueError("MiniMax-H3 build receipt does not match current checkpoint_snapshot")
    for filename in (*plan_filenames_for_profile(profile), *additional_plan_filenames):
        validate_record(
            plans_dir / filename,
            components.get(filename),
            filename,
            hash_file=hash_files,
        )
    tokenizer_record = receipt["assets"]["tokenizer.json"]
    validate_record(tokenizer, tokenizer_record, "tokenizer.json", hash_file=hash_files)
    return source_sha, components, tokenizer_record, recorded_snapshot


def validate_component_build_receipt(
    receipt: object,
    *,
    component: str,
    artifact: Path,
    build_helper: Path,
    source_revision: str,
    profile,
    hash_file: bool,
) -> tuple[str, dict, dict]:
    if component not in plan_filenames_for_profile(profile):
        raise ValueError(f"Unknown MiniMax-H3 native component: {component}")
    source_sha, components, snapshot_record = _validate_build_receipt_metadata(
        receipt,
        build_helper=build_helper,
        source_revision=source_revision,
        profile=profile,
    )
    component_record = components[component]
    validate_record(artifact, component_record, component, hash_file=hash_file)
    return source_sha, component_record, snapshot_record


def load_bundle_config(bundle: Path) -> dict:
    with bundle.open("rb") as stream:
        if stream.read(len(_BUNDLE_MAGIC)) != _BUNDLE_MAGIC:
            raise ValueError("MiniMax-H3 bundle has invalid magic")
        raw_header_size = stream.read(8)
        if len(raw_header_size) != 8:
            raise ValueError("MiniMax-H3 bundle has a truncated header size")
        header_size = struct.unpack("<Q", raw_header_size)[0]
        if header_size > _MAX_BUNDLE_HEADER_BYTES:
            raise ValueError("MiniMax-H3 bundle header exceeds the runtime limit")
        raw_header = stream.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError("MiniMax-H3 bundle has a truncated header")
        header = json.loads(raw_header)
        sections = header.get("sections") if isinstance(header, dict) else None
        config_section = sections.get("config.json") if isinstance(sections, dict) else None
        if not isinstance(config_section, dict):
            raise ValueError("MiniMax-H3 bundle is missing config.json")
        offset = config_section.get("offset")
        size = config_section.get("size")
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
        ):
            raise ValueError("MiniMax-H3 bundle config.json section has invalid bounds")
        data_start = len(_BUNDLE_MAGIC) + 8 + header_size
        if offset + size > bundle.stat().st_size - data_start:
            raise ValueError("MiniMax-H3 bundle config.json section is out of bounds")
        stream.seek(data_start + offset)
        raw_config = stream.read(size)
        if len(raw_config) != size:
            raise ValueError("MiniMax-H3 bundle config.json section is truncated")
    config = json.loads(raw_config)
    if not isinstance(config, dict):
        raise ValueError("MiniMax-H3 bundle config.json must be a JSON object")
    return config


def validate_native_bundle_config(bundle: Path, *, source_revision: str) -> dict:
    source_revision = validate_source_revision(source_revision)
    config = load_bundle_config(bundle)
    expected = {
        "model_type": "minimax_h3",
        "runtime_strategy": "diffusion_minimax_h3",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_revision": source_revision,
        "builder_source_sha256": builder_source_sha256(),
        "context_parallel_size": 1,
        "padded_sequence_length": SOL_ENGINE_1344X768_124_TO_345F.padded_sequence_length,
        "packed_sequence_length_min": (SOL_ENGINE_1344X768_124_TO_345F.min_sequence_length),
        "packed_sequence_length_opt": (SOL_ENGINE_1344X768_124_TO_345F.opt_sequence_length),
        "packed_sequence_length_max": SOL_ENGINE_1344X768_124_TO_345F.sequence_length,
        "canvas_multiple": CANVAS_MULTIPLE,
        "canvas_short_edge": CANVAS_SHORT_EDGE,
        "canvas_max_pixels": CANVAS_MAX_PIXELS,
        "explicit_canvas_sizes": [list(size) for size in NATIVE_EXPLICIT_CANVAS_SIZES],
        "min_aspect_ratio": CANVAS_MIN_ASPECT_RATIO,
        "max_aspect_ratio": CANVAS_MAX_ASPECT_RATIO,
        "vae_tile_batch": 28,
        "vae_tile_batch_min": 15,
        "vae_tile_batch_opt": 28,
        "vae_tile_batch_max": 33,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"MiniMax-H3 bundle config does not match current {key}")
    inventory_sha = config.get("checkpoint_inventory_sha256")
    if not isinstance(inventory_sha, str) or _SHA256.fullmatch(inventory_sha) is None:
        raise ValueError("MiniMax-H3 bundle config has an invalid checkpoint inventory SHA256")
    if config.get("denoiser_cache_mode") != "first_block":
        raise ValueError("MiniMax-H3 bundle config must use dense FirstBlockCache")
    plan_sha = config.get("plan_sha256")
    if not isinstance(plan_sha, dict):
        raise ValueError("MiniMax-H3 bundle config must identify the selected native plans")
    public_workflows = config.get("public_workflows")
    supported_workflows = (
        ["t2va"],
        ["t2va", "fl2va"],
        ["t2va", "fl2va", "ref2va"],
    )
    if public_workflows is not None and public_workflows not in supported_workflows:
        raise ValueError(
            "MiniMax-H3 bundle public_workflows must be an ordered prefix of [t2va, fl2va, ref2va]"
        )
    declares_fl2va = public_workflows is not None and "fl2va" in public_workflows
    declares_ref2va = public_workflows is not None and "ref2va" in public_workflows
    ref2va_supported = config.get("ref2va_supported", False)
    if not isinstance(ref2va_supported, bool) or ref2va_supported != declares_ref2va:
        raise ValueError("MiniMax-H3 bundle ref2va_supported must match public_workflows")
    present_conditioning = set(plan_sha).intersection(_FL2VA_CONDITIONING_PLAN_FILENAMES)
    if present_conditioning and present_conditioning != set(_FL2VA_CONDITIONING_PLAN_FILENAMES):
        raise ValueError("MiniMax-H3 FL2VA conditioning plans must be all-or-none")
    include_conditioning = (
        bool(present_conditioning) if public_workflows is None else declares_fl2va
    )
    selected_plans = native_plan_filenames()
    if not include_conditioning:
        selected_plans = tuple(
            filename
            for filename in selected_plans
            if filename not in _FL2VA_CONDITIONING_PLAN_FILENAMES
        )
    additional_plan_filenames: tuple[str, ...] = ()
    if declares_ref2va:
        from .ref2va_bundle_contract import REF2VA_PLAN_SECTIONS

        additional_plan_filenames = tuple(
            filename for _component, filename, _section in REF2VA_PLAN_SECTIONS
        )
        selected_plans = (*selected_plans, *additional_plan_filenames)
    if set(plan_sha) != set(selected_plans):
        raise ValueError("MiniMax-H3 bundle config must identify exactly the selected native plans")
    if any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in plan_sha.values()
    ):
        raise ValueError("MiniMax-H3 bundle config has an invalid native plan SHA256")
    if config.get("first_block_cache") is not True:
        raise ValueError("MiniMax-H3 bundle config must enable FirstBlockCache")
    if config.get("attention_mode") != "dense":
        raise ValueError("MiniMax-H3 bundle config must use dense attention")
    validate_workspace_limit_bytes(
        config.get("workspace_limit_bytes"),
        additional_plan_filenames=additional_plan_filenames,
        excluded_plan_filenames=(
            () if include_conditioning else _FL2VA_CONDITIONING_PLAN_FILENAMES
        ),
    )
    return config
