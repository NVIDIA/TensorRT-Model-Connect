# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe recognition and provenance checks for the supplied SAM2 package.

Discovery reads YAML with ``safe_load`` and inspects only the checkpoint ZIP
directory and its small serialization-version member. It never unpickles the
checkpoint. Build/export adapters request SHA-256 verification before loading
the exact checkpoint with PyTorch's ``weights_only`` mode and a strict state
dict load.
"""

from __future__ import annotations

import hashlib
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


PACKAGE_DIRNAME = "sam2_nvidia_repro"
CONFIG_RELATIVE_PATH = Path("config/sam2.1_hiera_s_with_bbox_head.yaml")
CHECKPOINT_RELATIVE_PATH = Path("checkpoint/sam2.1_hiera_small_with_bbox_head.pt")
SHA256SUMS_RELATIVE_PATH = Path("SHA256SUMS")

# Immutable provenance for the package supplied for the L4 bring-up. These
# values identify that delivery; the safe recognizer itself does not hash the
# 194 MB checkpoint unless verification is explicitly requested.
REFERENCE_SOURCE_ARCHIVE_SHA256 = "9521c0483106c5f914f38e061e03ca1fde2af00f6d2c2c7fb8dd1679b010b66a"
REFERENCE_SHA256SUMS_SHA256 = "130a7d860e70aea795c95139dd0ba3b02f16681978c664d77a3f0924d1b4a819"
REFERENCE_CONFIG_SHA256 = "59488bb78c7cc48aaaebd966ea9d054014f683459d062b7a959a4aa501342656"
REFERENCE_CHECKPOINT_SHA256 = "89fd676560809c8504411b574cea305c86db1f65bda790ec7fe16cedc6c6ff73"

_REFERENCE_CHECKPOINT_LAYOUT: Mapping[str, int | str] = {
    "serialization_version": "3",
    "archive_members": 597,
    "pickle_bytes": 80_191,
    "zip_storage_members": 595,
    "stored_nbytes": 193_747_376,
}

_EXPECTED_CONFIG_VALUES: tuple[tuple[tuple[str, ...], object], ...] = (
    (("model", "_target_"), "sam2.modeling.sam2_base.SAM2Base"),
    (
        ("model", "image_encoder", "_target_"),
        "sam2.modeling.backbones.image_encoder.ImageEncoderWithBBoxHead",
    ),
    (
        ("model", "image_encoder", "trunk", "_target_"),
        "sam2.modeling.backbones.hieradet.Hiera",
    ),
    (("model", "image_encoder", "trunk", "embed_dim"), 96),
    (("model", "image_encoder", "trunk", "num_heads"), 1),
    (("model", "image_encoder", "trunk", "stages"), [1, 2, 11, 2]),
    (("model", "image_encoder", "trunk", "global_att_blocks"), [7, 10, 13]),
    (("model", "image_encoder", "neck", "d_model"), 256),
    (
        ("model", "image_encoder", "neck", "backbone_channel_list"),
        [768, 384, 192, 96],
    ),
    (
        ("model", "image_encoder", "bbox_head", "_target_"),
        "sam2.modeling.backbones.bbox_head.RTMDetSepBNHeadModule",
    ),
    (("model", "image_encoder", "bbox_head", "num_classes"), 2),
    (("model", "image_encoder", "bbox_head", "in_channels"), 256),
    (("model", "image_encoder", "bbox_head", "feat_channels"), 256),
    (("model", "image_encoder", "bbox_head", "stacked_convs"), 2),
    (("model", "image_encoder", "bbox_head", "featmap_strides"), [8, 16, 32]),
    (
        ("model", "image_encoder", "bbox_head", "featmap_sizes"),
        [[128, 128], [64, 64], [32, 32]],
    ),
    (("model", "image_encoder", "bbox_head", "share_conv"), True),
    (("model", "memory_attention", "d_model"), 256),
    (("model", "memory_attention", "num_layers"), 4),
    (("model", "memory_encoder", "out_dim"), 64),
    (("model", "num_maskmem"), 7),
    (("model", "image_size"), 1024),
    (("model", "use_high_res_features_in_sam"), True),
    (("model", "use_obj_ptrs_in_encoder"), True),
    (("model", "pred_obj_scores"), True),
)

# The compatible source snapshot also contains a same-named experimental YAML
# with an additional CSPNeXt FPN. That is a different graph and its parameters
# are absent from the delivered 603-tensor checkpoint, so discovery must reject
# it even before build-time exact-hash verification.
_FORBIDDEN_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("model", "image_encoder", "learnable_fpn_module"),
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class Sam2ArchiveContractError(ValueError):
    """The directory resembles the SAM2 package but violates its contract."""


@dataclass(frozen=True)
class Sam2ArchiveDescription:
    """Validated, JSON-serializable package description."""

    root: Path
    config: dict[str, Any]
    checkpoint_inventory: dict[str, Any]
    provenance: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "layout": PACKAGE_DIRNAME,
            "root": str(self.root),
            "config_path": CONFIG_RELATIVE_PATH.as_posix(),
            "checkpoint_path": CHECKPOINT_RELATIVE_PATH.as_posix(),
            "checkpoint_inventory": self.checkpoint_inventory,
            "provenance": self.provenance,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_package_root(model_dir: str | Path) -> Path | None:
    """Return the package root for either it or its one-directory wrapper."""
    path = Path(model_dir)
    required = (
        CONFIG_RELATIVE_PATH,
        CHECKPOINT_RELATIVE_PATH,
        SHA256SUMS_RELATIVE_PATH,
    )
    for candidate in (path, path / PACKAGE_DIRNAME):
        if (
            candidate.is_dir()
            and not candidate.is_symlink()
            and all(_is_regular_descendant(candidate, item) for item in required)
        ):
            return candidate
    return None


def _is_regular_descendant(root: Path, relative: Path) -> bool:
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return False
    return current.is_file()


def _read_yaml_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - declared package dependency
        raise RuntimeError("PyYAML is required to inspect the SAM2 config") from exc

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise Sam2ArchiveContractError(f"unable to read SAM2 config {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise Sam2ArchiveContractError(f"SAM2 config must contain a mapping: {config_path}")
    return loaded


def _nested_value(raw: Mapping[str, Any], path: tuple[str, ...]) -> object:
    value: object = raw
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            raise Sam2ArchiveContractError(
                f"SAM2 config is missing required field {'.'.join(path)}"
            )
        value = value[part]
    return value


def _has_nested_path(raw: Mapping[str, Any], path: tuple[str, ...]) -> bool:
    value: object = raw
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return False
        value = value[part]
    return True


def validate_model_config(config_path: Path) -> dict[str, Any]:
    """Validate the supported Hiera-small, RTMDet, and tracker graph contract."""
    raw = _read_yaml_config(config_path)
    for field_path, expected in _EXPECTED_CONFIG_VALUES:
        actual = _nested_value(raw, field_path)
        if actual != expected:
            raise Sam2ArchiveContractError(
                f"SAM2 config field {'.'.join(field_path)} must be {expected!r}, got {actual!r}"
            )
    for field_path in _FORBIDDEN_CONFIG_PATHS:
        if _has_nested_path(raw, field_path):
            raise Sam2ArchiveContractError(
                f"SAM2 config contains an unsupported graph field {'.'.join(field_path)}"
            )
    return raw


def _safe_zip_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(
        name
        and "\\" not in name
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def validate_checkpoint_inventory(checkpoint_path: Path) -> dict[str, Any]:
    """Inspect the checkpoint ZIP layout without reading ``data.pkl``.

    This deliberately validates only a non-executable discovery envelope.
    Exact SHA-256 plus the adapters' ``weights_only`` strict state-dict load is
    the deeper build-time identity and tensor-schema check.
    """
    try:
        archive = zipfile.ZipFile(checkpoint_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise Sam2ArchiveContractError(
            f"SAM2 checkpoint is not a readable PyTorch ZIP archive: {checkpoint_path}"
        ) from exc

    with archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)) or any(not _safe_zip_name(name) for name in names):
            raise Sam2ArchiveContractError("SAM2 checkpoint has duplicate or unsafe ZIP members")
        if any(member.is_dir() for member in members):
            raise Sam2ArchiveContractError("SAM2 checkpoint must not contain ZIP directories")
        if any(stat.S_ISLNK((member.external_attr >> 16) & 0xFFFF) for member in members):
            raise Sam2ArchiveContractError("SAM2 checkpoint must not contain ZIP symlinks")
        if any(member.compress_type != zipfile.ZIP_STORED for member in members):
            raise Sam2ArchiveContractError("SAM2 checkpoint members must be stored uncompressed")

        pickle_members = [name for name in names if name.endswith("/data.pkl")]
        if len(pickle_members) != 1:
            raise Sam2ArchiveContractError(
                "SAM2 checkpoint must contain exactly one data.pkl member"
            )
        pickle_member = pickle_members[0]
        archive_root = pickle_member.removesuffix("/data.pkl")
        if not archive_root or "/" in archive_root:
            raise Sam2ArchiveContractError("SAM2 checkpoint must have one archive root")

        version_member = f"{archive_root}/version"
        storage_prefix = f"{archive_root}/data/"
        expected_names = {pickle_member, version_member}
        storage_members = []
        storage_keys: set[str] = set()
        for member in members:
            if member.filename.startswith(storage_prefix):
                key = member.filename.removeprefix(storage_prefix)
                if not key.isdigit():
                    raise Sam2ArchiveContractError(
                        f"SAM2 checkpoint has an invalid storage member: {member.filename}"
                    )
                storage_members.append(member)
                storage_keys.add(key)
                expected_names.add(member.filename)
        if set(names) != expected_names:
            raise Sam2ArchiveContractError("SAM2 checkpoint has unsupported ZIP members")
        if storage_keys != {str(index) for index in range(len(storage_members))}:
            raise Sam2ArchiveContractError(
                "SAM2 checkpoint storage members must be consecutively numbered"
            )

        try:
            version_info = archive.getinfo(version_member)
            if version_info.file_size > 16:
                raise Sam2ArchiveContractError("SAM2 checkpoint version member is too large")
            version = archive.read(version_info).decode("ascii").strip()
        except (KeyError, UnicodeDecodeError) as exc:
            raise Sam2ArchiveContractError(
                "SAM2 checkpoint has no readable serialization version"
            ) from exc

        inventory = {
            "format": "pytorch_zip",
            "serialization_version": version,
            "archive_members": len(members),
            "pickle_bytes": archive.getinfo(pickle_member).file_size,
            "zip_storage_members": len(storage_members),
            "stored_nbytes": sum(member.file_size for member in storage_members),
        }
        expected = dict(_REFERENCE_CHECKPOINT_LAYOUT)
        actual = {key: inventory[key] for key in expected}
        if actual != expected:
            raise Sam2ArchiveContractError(
                f"SAM2 checkpoint ZIP layout mismatch: expected {expected!r}, got {actual!r}"
            )
        return inventory


def read_declared_sha256s(root: Path) -> dict[str, str]:
    """Parse the package manifest without following or hashing its entries."""
    manifest = root / SHA256SUMS_RELATIVE_PATH
    if not manifest.is_file() or manifest.is_symlink():
        return {}
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Sam2ArchiveContractError(f"unable to read {manifest}: {exc}") from exc

    records: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not _SHA256_RE.fullmatch(parts[0]):
            raise Sam2ArchiveContractError(f"invalid SHA256SUMS record at line {line_number}")
        relative_text = parts[1].lstrip("*")
        relative = PurePosixPath(relative_text)
        if not _safe_zip_name(relative_text):
            raise Sam2ArchiveContractError(
                f"unsafe SHA256SUMS path at line {line_number}: {relative_text!r}"
            )
        normalized = relative.as_posix()
        if normalized in records:
            raise Sam2ArchiveContractError(f"duplicate SHA256SUMS path: {normalized!r}")
        records[normalized] = parts[0]
    return records


def _manifest_file(root: Path, relative: str) -> Path:
    """Resolve a regular manifest entry without accepting any symlink hop."""
    parts = PurePosixPath(relative).parts
    candidate = root.joinpath(*parts)
    current = root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise Sam2ArchiveContractError(f"declared package file is a symlink: {relative}")
    if not candidate.is_file():
        raise Sam2ArchiveContractError(
            f"declared package file is missing or not regular: {relative}"
        )
    return candidate


def verify_declared_provenance(root: str | Path) -> dict[str, Any]:
    """Hash every declared package file and return a provenance report."""
    package_root = Path(root)
    if package_root.is_symlink():
        raise Sam2ArchiveContractError(f"package root must not be a symlink: {package_root}")
    records = read_declared_sha256s(package_root)
    if not records:
        raise Sam2ArchiveContractError(
            f"package provenance manifest is missing: {package_root / SHA256SUMS_RELATIVE_PATH}"
        )
    verified: dict[str, str] = {}
    for relative, expected in records.items():
        actual = sha256_file(_manifest_file(package_root, relative))
        if actual != expected:
            raise Sam2ArchiveContractError(
                f"package file SHA256 mismatch for {relative}: expected {expected}, got {actual}"
            )
        verified[relative] = actual
    return {
        "manifest_sha256": sha256_file(package_root / SHA256SUMS_RELATIVE_PATH),
        "files": verified,
    }


def describe_archive(
    model_dir: str | Path,
    *,
    verify_provenance: bool = False,
) -> Sam2ArchiveDescription:
    """Validate the safe discovery contract and optionally hash package files."""
    root = resolve_package_root(model_dir)
    if root is None:
        raise Sam2ArchiveContractError(
            f"expected {CONFIG_RELATIVE_PATH.as_posix()}, "
            f"{CHECKPOINT_RELATIVE_PATH.as_posix()}, and "
            f"{SHA256SUMS_RELATIVE_PATH.as_posix()} under {model_dir}"
        )
    config = validate_model_config(root / CONFIG_RELATIVE_PATH)
    inventory = validate_checkpoint_inventory(root / CHECKPOINT_RELATIVE_PATH)
    declared = read_declared_sha256s(root)
    required_records = {
        CONFIG_RELATIVE_PATH.as_posix(),
        CHECKPOINT_RELATIVE_PATH.as_posix(),
    }
    if not required_records.issubset(declared):
        missing = sorted(required_records - declared.keys())
        raise Sam2ArchiveContractError(
            f"SAM2 SHA256SUMS is missing required entries: {', '.join(missing)}"
        )

    declared_config = declared[CONFIG_RELATIVE_PATH.as_posix()]
    declared_checkpoint = declared[CHECKPOINT_RELATIVE_PATH.as_posix()]
    provenance: dict[str, Any] = {
        "source_archive_reference_sha256": REFERENCE_SOURCE_ARCHIVE_SHA256,
        "reference_config_sha256": REFERENCE_CONFIG_SHA256,
        "reference_checkpoint_sha256": REFERENCE_CHECKPOINT_SHA256,
        "declared_config_sha256": declared_config,
        "declared_checkpoint_sha256": declared_checkpoint,
        "declared_matches_reference_config": declared_config == REFERENCE_CONFIG_SHA256,
        "declared_matches_reference_checkpoint": (
            declared_checkpoint == REFERENCE_CHECKPOINT_SHA256
        ),
        "verification_status": "not_requested",
    }
    if verify_provenance:
        report = verify_declared_provenance(root)
        checkpoint_sha256 = report["files"][CHECKPOINT_RELATIVE_PATH.as_posix()]
        config_sha256 = report["files"][CONFIG_RELATIVE_PATH.as_posix()]
        provenance.update(
            {
                "verification_status": "verified",
                "manifest_sha256": report["manifest_sha256"],
                "matches_reference_manifest": (
                    report["manifest_sha256"] == REFERENCE_SHA256SUMS_SHA256
                ),
                "checkpoint_sha256": checkpoint_sha256,
                "config_sha256": config_sha256,
                "matches_reference_checkpoint": (checkpoint_sha256 == REFERENCE_CHECKPOINT_SHA256),
                "matches_reference_config": config_sha256 == REFERENCE_CONFIG_SHA256,
            }
        )
    return Sam2ArchiveDescription(
        root=root,
        config=config,
        checkpoint_inventory=inventory,
        provenance=provenance,
    )


__all__ = [
    "CHECKPOINT_RELATIVE_PATH",
    "CONFIG_RELATIVE_PATH",
    "PACKAGE_DIRNAME",
    "REFERENCE_CHECKPOINT_SHA256",
    "REFERENCE_CONFIG_SHA256",
    "REFERENCE_SOURCE_ARCHIVE_SHA256",
    "Sam2ArchiveContractError",
    "Sam2ArchiveDescription",
    "describe_archive",
    "read_declared_sha256s",
    "resolve_package_root",
    "sha256_file",
    "validate_checkpoint_inventory",
    "validate_model_config",
    "verify_declared_provenance",
]
