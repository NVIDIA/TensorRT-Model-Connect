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
PLAN_FILENAMES = (
    "text_encoder.plan",
    "adaln_precompute.plan",
    "denoiser.plan",
    "vae_tile_decoder.plan",
)
_REQUIRED_SNAPSHOT_FILES = (
    "modular_model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/tokenizer.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/diffusion_pytorch_model.safetensors.index.json",
)
_BUNDLE_MAGIC = b"TRTFB\x00\x01\x00"
_MAX_BUNDLE_HEADER_BYTES = 100 << 20
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


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


def validate_workspace_limit_bytes(record: object) -> dict[str, int]:
    """Validate the exact per-plan TensorRT tactic-workspace provenance."""

    if not isinstance(record, dict) or set(record) != set(PLAN_FILENAMES):
        raise ValueError(
            "MiniMax-H3 workspace_limit_bytes must cover exactly all four native plans"
        )
    for filename, value in record.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
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


def checkpoint_snapshot_record(snapshot: Path) -> dict:
    """Describe the canonical HF snapshot without rereading LFS weight blobs.

    Hugging Face names LFS cache blobs by their SHA256. We bind large weight
    shards to those content-addressed names and hash the smaller Git blobs
    directly. Noncanonical copied snapshots fail closed instead of silently
    triggering another 135 GB read.
    """

    snapshot = snapshot.absolute()
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError("MiniMax-H3 model path must be a canonical HF snapshot directory")
    if snapshot.name != CHECKPOINT_REVISION or snapshot.parent.name != "snapshots":
        raise ValueError(
            f"MiniMax-H3 model path must resolve to pinned snapshot {CHECKPOINT_REVISION}"
        )
    repository_root = snapshot.parent.parent
    if repository_root.name != HF_CACHE_REPOSITORY:
        raise ValueError(f"MiniMax-H3 snapshot must belong to {CHECKPOINT_REPOSITORY}")
    blob_root = (repository_root / "blobs").resolve(strict=True)

    files: dict[str, dict[str, int | str]] = {}
    small_blob_digests: dict[Path, str] = {}
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            continue
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
        digest = blob_id if is_lfs_sha256 else small_blob_digests.get(target)
        if digest is None:
            digest = sha256_file(target)
            small_blob_digests[target] = digest
        files[relative] = {
            "blob_id": blob_id,
            "bytes": target.stat().st_size,
            "sha256": digest,
        }

    missing = sorted(set(_REQUIRED_SNAPSHOT_FILES) - set(files))
    if missing:
        raise ValueError(f"MiniMax-H3 snapshot is incomplete; missing: {missing}")
    for index_name in (
        "text_encoder/model.safetensors.index.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/diffusion_pytorch_model.safetensors.index.json",
    ):
        index = json.loads((snapshot / index_name).read_text())
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"MiniMax-H3 checkpoint index has no weight_map: {index_name}")
        referenced = {
            (Path(index_name).parent / filename).as_posix() for filename in weight_map.values()
        }
        missing_shards = sorted(referenced - set(files))
        if missing_shards:
            raise ValueError(
                f"MiniMax-H3 checkpoint index references missing shards: {missing_shards}"
            )

    payload = {
        "repository": CHECKPOINT_REPOSITORY,
        "revision": CHECKPOINT_REVISION,
        "files": files,
    }
    return {
        **payload,
        "file_count": len(files),
        "inventory_sha256": _canonical_json_sha256(payload),
    }


def validate_checkpoint_snapshot_record(record: object) -> dict:
    if not isinstance(record, dict):
        raise ValueError("MiniMax-H3 receipt is missing checkpoint_snapshot")
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
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("MiniMax-H3 checkpoint snapshot has an invalid relative path")
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
                target = path.resolve(strict=True)
            except OSError as error:
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
        if identity_after != identity_before or identity_before[3] != size:
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
    validate_workspace_limit_bytes(receipt.get("workspace_limit_bytes"))
    snapshot_record = validate_checkpoint_snapshot_record(receipt.get("checkpoint_snapshot"))
    components = receipt.get("components")
    if not isinstance(components, dict) or set(components) != set(PLAN_FILENAMES):
        raise ValueError("MiniMax-H3 build receipt must cover exactly all four native plans")
    for filename in PLAN_FILENAMES:
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
) -> tuple[str, dict, dict, dict]:
    source_sha, components, recorded_snapshot = _validate_build_receipt_metadata(
        receipt,
        build_helper=build_helper,
        source_revision=source_revision,
        profile=profile,
    )
    current_snapshot = checkpoint_snapshot_record(snapshot)
    if recorded_snapshot != current_snapshot:
        raise ValueError("MiniMax-H3 build receipt does not match current checkpoint_snapshot")
    for filename in PLAN_FILENAMES:
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
    if component not in PLAN_FILENAMES:
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
        "padded_sequence_length": 38247,
        "vae_tile_batch": 28,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"MiniMax-H3 bundle config does not match current {key}")
    inventory_sha = config.get("checkpoint_inventory_sha256")
    if not isinstance(inventory_sha, str) or _SHA256.fullmatch(inventory_sha) is None:
        raise ValueError("MiniMax-H3 bundle config has an invalid checkpoint inventory SHA256")
    plan_sha = config.get("plan_sha256")
    if not isinstance(plan_sha, dict) or set(plan_sha) != set(PLAN_FILENAMES):
        raise ValueError("MiniMax-H3 bundle config must identify exactly all four native plans")
    if any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in plan_sha.values()
    ):
        raise ValueError("MiniMax-H3 bundle config has an invalid native plan SHA256")
    validate_workspace_limit_bytes(config.get("workspace_limit_bytes"))
    return config
