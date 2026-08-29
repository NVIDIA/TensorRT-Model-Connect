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
    FL2VA_PROCESSOR_ASSET_SECTIONS,
    MINIMAX_H3_NATIVE_PLUGIN_ABI,
    MINIMAX_H3_NATIVE_PLUGIN_FILENAME,
    MINIMAX_H3_NATIVE_PLUGIN_IDENTITY,
    MINIMAX_H3_NATIVE_PLUGIN_SECTION,
    MINIMAX_H3_WORKFLOWS,
    REF2VA_MAX_CONDITION_AUDIO_ROWS,
    REF2VA_MAX_CONDITION_VIDEO_ROWS,
    REF2VA_IMAGE_VISION_ATTENTION_IMPLEMENTATION,
    REF2VA_IMAGE_VISION_ATTENTION_PRECISION,
    REF2VA_IMAGE_VISION_ATTENTION_SCALE,
    REF2VA_IMAGE_VISION_LINEAR_COUNT,
    REF2VA_IMAGE_VISION_LINEAR_IMPLEMENTATION,
    REF2VA_IMAGE_VISION_LAYER_NORM_COUNT,
    REF2VA_IMAGE_VISION_LAYER_NORM_IMPLEMENTATION,
    REF2VA_IMAGE_VISION_PATCH_BIAS_SHAPE,
    REF2VA_IMAGE_VISION_PATCH_IMPLEMENTATION,
    REF2VA_IMAGE_VISION_PATCH_INPUT_SHAPE,
    REF2VA_IMAGE_VISION_PATCH_KERNEL,
    REF2VA_IMAGE_VISION_PATCH_OUTPUT_SHAPE,
    REF2VA_IMAGE_VISION_PATCH_PRECISION,
    REF2VA_IMAGE_VISION_PATCH_PROFILE,
    REF2VA_IMAGE_VISION_PATCH_STRIDE,
    REF2VA_IMAGE_VISION_PATCH_WEIGHT_SHAPE,
    REF2VA_LANGUAGE_ATTENTION_IMPLEMENTATION,
    REF2VA_LANGUAGE_ATTENTION_PRECISION,
    REF2VA_LANGUAGE_Q_PRE_SCALE_PRECISION,
    REF2VA_MAX_TEXT_ROWS,
    REF2VA_MIN_CONDITION_VIDEO_ROWS,
    REF2VA_OPT_CONDITION_VIDEO_ROWS,
    REF2VA_VIDEO_VISION_ATTENTION_IMPLEMENTATION,
    REF2VA_VIDEO_VISION_ATTENTION_PRECISION,
    REF2VA_VIDEO_VISION_PATCH_PROFILE,
    REF2VA_VIDEO_VISION_Q_PRE_SCALE_PRECISION,
    REF2VA_VISION_PLAN_LAYOUT,
    native_plan_filenames,
)

CHECKPOINT_REVISION = "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
CHECKPOINT_REPOSITORY = "MiniMaxAI/MiniMax-H3"
HF_CACHE_REPOSITORY = "models--MiniMaxAI--MiniMax-H3"
MODEL_CARD_PATH = "README.md"
MODEL_CARD_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
MODEL_CARD_SHA256 = "f0116a90332496bdfcc827320c603a26b849c73bf804f2674d03682fbbd2334a"
REF2VA_AUDIO_ONLY_SEMANTICS_REVISION = "939557dc319dd91227e30195a763f272ba7f8765"
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
PLAN_FILENAMES = native_plan_filenames(first_block_cache=False)
FIRST_BLOCK_CACHE_PLAN_FILENAMES = native_plan_filenames(first_block_cache=True)
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
_FL2VA_REQUIRED_SNAPSHOT_FILES = (
    *_REQUIRED_SNAPSHOT_FILES,
    "processor/preprocessor_config.json",
    "processor/video_preprocessor_config.json",
    "text_encoder/config.json",
    "transformer/config.json",
    "vae/config.json",
)
_REF2VA_REQUIRED_SNAPSHOT_FILES = (
    *_FL2VA_REQUIRED_SNAPSHOT_FILES,
    "transformer_ref/config.json",
    "transformer_ref/diffusion_pytorch_model.safetensors.index.json",
)
_BUNDLE_MAGIC = b"BUNDLE\x01\x00"
_MAX_BUNDLE_HEADER_BYTES = 100 << 20
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


def ref2va_input_specification_record() -> dict[str, str]:
    """Return the official model-card provenance for Ref2VA audio-only input."""

    return {
        "repository": CHECKPOINT_REPOSITORY,
        "path": MODEL_CARD_PATH,
        "current_revision": MODEL_CARD_REVISION,
        "sha256": MODEL_CARD_SHA256,
        "audio_only_semantics_revision": REF2VA_AUDIO_ONLY_SEMANTICS_REVISION,
        "semantics": "Ref2VA accepts one or more audio references without image or video",
    }


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


def _validated_workflow(workflow: object) -> str:
    if not isinstance(workflow, str) or workflow not in MINIMAX_H3_WORKFLOWS:
        raise ValueError("MiniMax-H3 provenance has an invalid workflow")
    return workflow


def plan_filenames_for_profile(profile, *, workflow: str = "t2va") -> tuple[str, ...]:
    return native_plan_filenames(
        first_block_cache=profile.first_block_cache,
        workflow=_validated_workflow(workflow),
    )


def validate_workspace_limit_bytes(
    record: object,
    *,
    profile=None,
    first_block_cache: bool | None = None,
    workflow: str = "t2va",
) -> dict[str, int]:
    """Validate the exact per-plan TensorRT tactic-workspace provenance."""

    if profile is not None and first_block_cache is not None:
        raise ValueError("MiniMax-H3 workspace validation received two profile selectors")
    if profile is not None:
        first_block_cache = profile.first_block_cache
    selected = False if first_block_cache is None else first_block_cache
    if not isinstance(selected, bool):
        raise ValueError("MiniMax-H3 first_block_cache selector must be a boolean")
    expected = native_plan_filenames(
        first_block_cache=selected,
        workflow=_validated_workflow(workflow),
    )
    if not isinstance(record, dict) or set(record) != set(expected):
        raise ValueError(
            "MiniMax-H3 workspace_limit_bytes must cover exactly the selected native plans"
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


def _required_snapshot_files(workflow: str) -> tuple[str, ...]:
    workflow = _validated_workflow(workflow)
    if workflow == "fl2va":
        return _FL2VA_REQUIRED_SNAPSHOT_FILES
    if workflow == "ref2va":
        return _REF2VA_REQUIRED_SNAPSHOT_FILES
    return _REQUIRED_SNAPSHOT_FILES


def checkpoint_snapshot_record(snapshot: Path, *, workflow: str = "t2va") -> dict:
    """Describe the canonical HF snapshot without rereading LFS weight blobs.

    Hugging Face names LFS cache blobs by their SHA256. We bind large weight
    shards to those content-addressed names and hash the smaller Git blobs
    directly. Noncanonical copied snapshots fail closed instead of silently
    triggering another 135 GB read.
    """

    required_files = _required_snapshot_files(workflow)
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

    missing = sorted(set(required_files) - set(files))
    if missing:
        raise ValueError(f"MiniMax-H3 snapshot is incomplete; missing: {missing}")
    index_names = [
        "text_encoder/model.safetensors.index.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/diffusion_pytorch_model.safetensors.index.json",
    ]
    if workflow == "ref2va":
        index_names.append("transformer_ref/diffusion_pytorch_model.safetensors.index.json")
    for index_name in index_names:
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


def validate_checkpoint_snapshot_record(record: object, *, workflow: str = "t2va") -> dict:
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
    missing = sorted(set(_required_snapshot_files(workflow)) - set(files))
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

    from .native_plugin_builder import native_plugin_source_files

    family_root = Path(__file__).resolve().parent
    package_root = family_root.parents[1]
    repo_root = package_root.parents[1]
    sources = [*family_root.glob("*.py"), package_root / "trt_compat.py"]
    logical_sources = [(str(path.relative_to(repo_root)), path) for path in sorted(set(sources))]
    logical_sources.extend(
        (
            f"src/runtime/models/minimax_h3/native_plugins/{path.name}",
            path,
        )
        for path in native_plugin_source_files()
    )
    digest = hashlib.sha256()
    for logical_path, path in sorted(logical_sources):
        digest.update(logical_path.encode())
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
    workflow: str = "t2va",
) -> tuple[str, dict, dict]:
    if not isinstance(receipt, dict):
        raise ValueError("MiniMax-H3 build receipt must be a JSON object")
    workflow = _validated_workflow(workflow)
    recorded_workflow = _validated_workflow(receipt.get("workflow", "t2va"))
    if recorded_workflow != workflow:
        raise ValueError("MiniMax-H3 build receipt does not match current workflow")
    expected_partition = "transformer_ref" if workflow == "ref2va" else "transformer"
    if receipt.get("checkpoint_partition", "transformer") != expected_partition:
        raise ValueError("MiniMax-H3 build receipt does not match current checkpoint_partition")
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
    selected_plans = plan_filenames_for_profile(profile, workflow=workflow)
    validate_workspace_limit_bytes(
        receipt.get("workspace_limit_bytes"),
        profile=profile,
        workflow=workflow,
    )
    snapshot_record = validate_checkpoint_snapshot_record(
        receipt.get("checkpoint_snapshot"),
        workflow=workflow,
    )
    components = receipt.get("components")
    if not isinstance(components, dict) or set(components) != set(selected_plans):
        raise ValueError("MiniMax-H3 build receipt must cover exactly the selected native plans")
    for filename in selected_plans:
        _validate_record_object(components.get(filename), filename)
    assets = receipt.get("assets")
    expected_assets = {"tokenizer.json"}
    if workflow in {"fl2va", "ref2va"}:
        expected_assets.update(FL2VA_PROCESSOR_ASSET_SECTIONS)
    if workflow == "ref2va":
        expected_assets.add(MINIMAX_H3_NATIVE_PLUGIN_SECTION)
    if not isinstance(assets, dict) or set(assets) != expected_assets:
        raise ValueError("MiniMax-H3 build receipt must cover exactly the selected assets")
    for name in expected_assets:
        _validate_record_object(assets.get(name), name)
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
    workflow: str = "t2va",
) -> tuple[str, dict, dict, dict]:
    source_sha, components, recorded_snapshot = _validate_build_receipt_metadata(
        receipt,
        build_helper=build_helper,
        source_revision=source_revision,
        profile=profile,
        workflow=workflow,
    )
    current_snapshot = checkpoint_snapshot_record(snapshot, workflow=workflow)
    if recorded_snapshot != current_snapshot:
        raise ValueError("MiniMax-H3 build receipt does not match current checkpoint_snapshot")
    for filename in plan_filenames_for_profile(profile, workflow=workflow):
        validate_record(
            plans_dir / filename,
            components.get(filename),
            filename,
            hash_file=hash_files,
        )
    tokenizer_record = receipt["assets"]["tokenizer.json"]
    validate_record(tokenizer, tokenizer_record, "tokenizer.json", hash_file=hash_files)
    if workflow in {"fl2va", "ref2va"}:
        for relative in FL2VA_PROCESSOR_ASSET_SECTIONS:
            validate_record(
                snapshot / relative,
                receipt["assets"][relative],
                relative,
                hash_file=hash_files,
            )
    if workflow == "ref2va":
        validate_record(
            plans_dir / MINIMAX_H3_NATIVE_PLUGIN_FILENAME,
            receipt["assets"][MINIMAX_H3_NATIVE_PLUGIN_SECTION],
            MINIMAX_H3_NATIVE_PLUGIN_SECTION,
            hash_file=hash_files,
        )
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
    workflow: str = "t2va",
) -> tuple[str, dict, dict]:
    if component not in plan_filenames_for_profile(profile, workflow=workflow):
        raise ValueError(f"Unknown MiniMax-H3 native component: {component}")
    source_sha, components, snapshot_record = _validate_build_receipt_metadata(
        receipt,
        build_helper=build_helper,
        source_revision=source_revision,
        profile=profile,
        workflow=workflow,
    )
    component_record = components[component]
    validate_record(artifact, component_record, component, hash_file=hash_file)
    return source_sha, component_record, snapshot_record


def _load_bundle_sections(bundle: Path) -> tuple[dict, int]:
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
    if not isinstance(sections, dict):
        raise ValueError("MiniMax-H3 bundle has an invalid section index")
    return sections, len(_BUNDLE_MAGIC) + 8 + header_size


def _read_bundle_section(bundle: Path, section_name: str, sections: dict, data_start: int) -> bytes:
    section = sections.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"MiniMax-H3 bundle is missing {section_name}")
    offset = section.get("offset")
    size = section.get("size")
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 0
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
    ):
        raise ValueError(f"MiniMax-H3 bundle {section_name} section has invalid bounds")
    if offset + size > bundle.stat().st_size - data_start:
        raise ValueError(f"MiniMax-H3 bundle {section_name} section is out of bounds")
    with bundle.open("rb") as stream:
        stream.seek(data_start + offset)
        payload = stream.read(size)
    if len(payload) != size:
        raise ValueError(f"MiniMax-H3 bundle {section_name} section is truncated")
    return payload


def load_bundle_config(bundle: Path) -> dict:
    sections, data_start = _load_bundle_sections(bundle)
    raw_config = _read_bundle_section(bundle, "config.json", sections, data_start)
    config = json.loads(raw_config)
    if not isinstance(config, dict):
        raise ValueError("MiniMax-H3 bundle config.json must be a JSON object")
    return config


def validate_native_bundle_config(bundle: Path, *, source_revision: str) -> dict:
    source_revision = validate_source_revision(source_revision)
    config = load_bundle_config(bundle)
    workflow = _validated_workflow(config.get("workflow", "t2va"))
    expected = {
        "model_type": "minimax_h3",
        "runtime_strategy": "diffusion_minimax_h3",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_revision": source_revision,
        "builder_source_sha256": builder_source_sha256(),
        "context_parallel_size": 1,
        "padded_sequence_length": 38247,
        "vae_tile_batch": 28,
        "audio_sample_rate": 32000,
        "audio_latent_frames": 207,
        "audio_output_samples": 165600,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"MiniMax-H3 bundle config does not match current {key}")
    expected_partition = "transformer_ref" if workflow == "ref2va" else "transformer"
    if config.get("checkpoint_partition", "transformer") != expected_partition:
        raise ValueError("MiniMax-H3 bundle config has the wrong checkpoint partition")
    if workflow == "fl2va":
        fl2va_expected = {
            "min_text_rows": 1,
            "max_text_rows": 4096,
            "fl2va_keyframe_counts": [0, 1, 2],
            "fl2va_keyframe_rows": 1008,
            "fl2va_vae_tile_size": 256,
            "fl2va_vae_tile_min_overlap": 64,
            "fl2va_vae_temporal_frames": [1],
            "processor_asset_sections": list(FL2VA_PROCESSOR_ASSET_SECTIONS),
        }
        for key, value in fl2va_expected.items():
            if config.get(key) != value:
                raise ValueError(f"MiniMax-H3 FL2VA bundle config has an invalid {key}")
    elif workflow == "ref2va":
        ref2va_expected = {
            "min_text_rows": 1,
            "opt_text_rows": 8192,
            "max_text_rows": REF2VA_MAX_TEXT_ROWS,
            "ref2va_min_condition_video_rows": REF2VA_MIN_CONDITION_VIDEO_ROWS,
            "ref2va_opt_condition_video_rows": REF2VA_OPT_CONDITION_VIDEO_ROWS,
            "ref2va_min_condition_audio_rows": 0,
            "ref2va_opt_condition_audio_rows": 0,
            "ref2va_max_condition_video_rows": REF2VA_MAX_CONDITION_VIDEO_ROWS,
            "ref2va_max_condition_audio_rows": REF2VA_MAX_CONDITION_AUDIO_ROWS,
            "ref2va_max_images": 9,
            "ref2va_max_videos": 3,
            "ref2va_max_audios": 3,
            "ref2va_max_references": 12,
            "ref2va_vision_plan_layout": REF2VA_VISION_PLAN_LAYOUT,
            "minimax_h3_native_plugin_section": MINIMAX_H3_NATIVE_PLUGIN_SECTION,
            "minimax_h3_native_plugin_artifact": MINIMAX_H3_NATIVE_PLUGIN_FILENAME,
            "minimax_h3_native_plugin_abi": MINIMAX_H3_NATIVE_PLUGIN_ABI,
            "minimax_h3_native_plugin_identity": MINIMAX_H3_NATIVE_PLUGIN_IDENTITY,
            "ref2va_language_attention_implementation": (REF2VA_LANGUAGE_ATTENTION_IMPLEMENTATION),
            "ref2va_language_attention_precision": REF2VA_LANGUAGE_ATTENTION_PRECISION,
            "ref2va_language_q_pre_scale_precision": REF2VA_LANGUAGE_Q_PRE_SCALE_PRECISION,
            "ref2va_image_vision_attention_implementation": (
                REF2VA_IMAGE_VISION_ATTENTION_IMPLEMENTATION
            ),
            "ref2va_image_vision_attention_precision": (REF2VA_IMAGE_VISION_ATTENTION_PRECISION),
            "ref2va_image_vision_attention_scale": REF2VA_IMAGE_VISION_ATTENTION_SCALE,
            "ref2va_image_vision_linear_implementation": (
                REF2VA_IMAGE_VISION_LINEAR_IMPLEMENTATION
            ),
            "ref2va_image_vision_linear_count": REF2VA_IMAGE_VISION_LINEAR_COUNT,
            "ref2va_image_vision_layer_norm_implementation": (
                REF2VA_IMAGE_VISION_LAYER_NORM_IMPLEMENTATION
            ),
            "ref2va_image_vision_layer_norm_count": REF2VA_IMAGE_VISION_LAYER_NORM_COUNT,
            "ref2va_image_vision_patch_implementation": (REF2VA_IMAGE_VISION_PATCH_IMPLEMENTATION),
            "ref2va_image_vision_patch_precision": REF2VA_IMAGE_VISION_PATCH_PRECISION,
            "ref2va_image_vision_patch_input_shape": list(REF2VA_IMAGE_VISION_PATCH_INPUT_SHAPE),
            "ref2va_image_vision_patch_weight_shape": list(REF2VA_IMAGE_VISION_PATCH_WEIGHT_SHAPE),
            "ref2va_image_vision_patch_bias_shape": list(REF2VA_IMAGE_VISION_PATCH_BIAS_SHAPE),
            "ref2va_image_vision_patch_kernel": list(REF2VA_IMAGE_VISION_PATCH_KERNEL),
            "ref2va_image_vision_patch_stride": list(REF2VA_IMAGE_VISION_PATCH_STRIDE),
            "ref2va_image_vision_patch_output_shape": list(REF2VA_IMAGE_VISION_PATCH_OUTPUT_SHAPE),
            "ref2va_video_vision_attention_implementation": (
                REF2VA_VIDEO_VISION_ATTENTION_IMPLEMENTATION
            ),
            "ref2va_video_vision_attention_precision": (REF2VA_VIDEO_VISION_ATTENTION_PRECISION),
            "ref2va_video_vision_q_pre_scale_precision": (
                REF2VA_VIDEO_VISION_Q_PRE_SCALE_PRECISION
            ),
            "ref2va_image_vision_patch_profile": list(REF2VA_IMAGE_VISION_PATCH_PROFILE),
            "ref2va_video_vision_patch_profile": list(REF2VA_VIDEO_VISION_PATCH_PROFILE),
            "ref2va_reference_min_seconds": 2,
            "ref2va_reference_max_seconds": 15,
            "ref2va_vae_tile_size": 256,
            "ref2va_vae_tile_min_overlap": 64,
            "ref2va_vae_temporal_frames": [1, 17],
            "processor_asset_sections": list(FL2VA_PROCESSOR_ASSET_SECTIONS),
        }
        for key, value in ref2va_expected.items():
            if config.get(key) != value:
                raise ValueError(f"MiniMax-H3 Ref2VA bundle config has an invalid {key}")
    inventory_sha = config.get("checkpoint_inventory_sha256")
    if not isinstance(inventory_sha, str) or _SHA256.fullmatch(inventory_sha) is None:
        raise ValueError("MiniMax-H3 bundle config has an invalid checkpoint inventory SHA256")
    cache_mode = config.get("denoiser_cache_mode", "monolithic")
    if cache_mode not in ("monolithic", "first_block"):
        raise ValueError("MiniMax-H3 bundle config has an invalid denoiser cache mode")
    first_block_cache = cache_mode == "first_block"
    selected_plans = native_plan_filenames(
        first_block_cache=first_block_cache,
        workflow=workflow,
    )
    expected_eager = ["tokenizer.json", "config.json"]
    if workflow == "fl2va":
        expected_eager = ["tokenizer.json", *FL2VA_PROCESSOR_ASSET_SECTIONS, "config.json"]
    elif workflow == "ref2va":
        expected_eager = [
            "tokenizer.json",
            *FL2VA_PROCESSOR_ASSET_SECTIONS,
            MINIMAX_H3_NATIVE_PLUGIN_SECTION,
            "config.json",
        ]
    expected_loading = {
        "mode": "staged",
        "eager_sections": expected_eager,
        "lazy_sections": [f"{filename.removesuffix('.plan')}_plan" for filename in selected_plans],
    }
    if config.get("bundle_loading") != expected_loading:
        raise ValueError("MiniMax-H3 bundle config has an invalid staged-loading section set")
    sections, data_start = _load_bundle_sections(bundle)
    expected_sections = set(expected_loading["eager_sections"]) | set(
        expected_loading["lazy_sections"]
    )
    if workflow == "ref2va" and set(sections) != expected_sections:
        raise ValueError("MiniMax-H3 bundle sections do not match the staged-loading contract")
    if workflow != "ref2va" and MINIMAX_H3_NATIVE_PLUGIN_SECTION in sections:
        raise ValueError("MiniMax-H3 non-Ref2VA bundle contains a native plugin section")
    plan_sha = config.get("plan_sha256")
    if not isinstance(plan_sha, dict) or set(plan_sha) != set(selected_plans):
        raise ValueError("MiniMax-H3 bundle config must identify exactly the selected native plans")
    if any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in plan_sha.values()
    ):
        raise ValueError("MiniMax-H3 bundle config has an invalid native plan SHA256")
    if not isinstance(config.get("first_block_cache", False), bool):
        raise ValueError("MiniMax-H3 bundle config has an invalid first_block_cache flag")
    if config.get("first_block_cache", False) != first_block_cache:
        raise ValueError("MiniMax-H3 bundle cache mode and profile flag disagree")
    if workflow in {"fl2va", "ref2va"}:
        expected_assets = {"tokenizer.json", *FL2VA_PROCESSOR_ASSET_SECTIONS}
        if workflow == "ref2va":
            expected_assets.add(MINIMAX_H3_NATIVE_PLUGIN_SECTION)
        asset_sha = config.get("asset_sha256")
        if not isinstance(asset_sha, dict) or set(asset_sha) != expected_assets:
            raise ValueError("MiniMax-H3 conditioned bundle must hash every selected asset")
        if any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in asset_sha.values()
        ):
            raise ValueError("MiniMax-H3 conditioned bundle has an invalid asset SHA256")
        if workflow == "ref2va":
            plugin_payload = _read_bundle_section(
                bundle,
                MINIMAX_H3_NATIVE_PLUGIN_SECTION,
                sections,
                data_start,
            )
            if (
                hashlib.sha256(plugin_payload).hexdigest()
                != asset_sha[MINIMAX_H3_NATIVE_PLUGIN_SECTION]
            ):
                raise ValueError("MiniMax-H3 native plugin section SHA256 does not match")
    validate_workspace_limit_bytes(
        config.get("workspace_limit_bytes"),
        first_block_cache=first_block_cache,
        workflow=workflow,
    )
    return config
