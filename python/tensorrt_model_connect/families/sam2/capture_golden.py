# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture the exact delivered SAM2 BF16 workload from compatible source.

Invoke this file by absolute path with ``python -I -S``.  It accepts one pinned
public checkout with the seven-file compatible overlay, one exact delivered
package, and one empty evidence destination.  It always performs three
fresh-state runs and cannot relax the source, asset, environment, or workload
contracts.

The source tree is verified before importing ``sam2``.  In particular, the
public base is read from Git objects at the pinned commit while the working
tree must contain exactly the reviewed overlay, its commit receipt, and the
delivered config under a distinct Hydra name.
"""

from __future__ import annotations

import argparse
import atexit
from contextlib import contextmanager
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence


class Sam2GoldenCaptureError(RuntimeError):
    """The source, assets, environment, or exact workload is not authoritative."""


_DIRECT_SCRIPT = __name__ == "__main__" and __package__ in {None, ""}
_RUNNER_PATH = Path(__file__).absolute()
_FAMILY_DIRECTORY = _RUNNER_PATH.parent
_ARCHIVE_CONTRACT_PATH = _FAMILY_DIRECTORY / "archive_contract.py"
_GOLDEN_EVIDENCE_PATH = _FAMILY_DIRECTORY / "golden_evidence.py"
_CANONICAL_RUNNER_NAME = "tensorrt_model_connect.families.sam2.capture_golden"
_EXPECTED_ARCHIVE_CONTRACT_SHA256 = (
    "f0d169032d21157e015eb7e6912b025c39db20c311c67d64df7567cabec8d07a"
)
# Its normalized bytes replace exactly the runner, self-normalized, and later
# reviewed reference-manifest pins with sentinels, keeping the closure acyclic.
_EXPECTED_GOLDEN_EVIDENCE_NORMALIZED_SHA256 = (
    "2cfae7b9c81708221ee5523c52c9dfb706b64a04d6da9bf25e3cae3879d8b689"
)
_REFERENCE_VENV_ROOT = Path("/workspace/ref-work/.venv")
_REFERENCE_VENV_BIN = _REFERENCE_VENV_ROOT / "bin"
_CONTROLLED_SITE_PACKAGES = _REFERENCE_VENV_ROOT / "lib/python3.12/site-packages"
_REFERENCE_PYVENV_CFG_SHA256 = "16529e11b2fe1e50d7bca13c16b18bdd5ff478ae2db7750e483aba6e3733d858"
_TOOL_PIN_PATTERN = re.compile(
    rb'(?m)^AUTHORITATIVE_CAPTURE_TOOL_SHA256: str \| None = \(\n    "([0-9a-f]{64})"\n\)$'
)
_GOLDEN_NORMALIZED_PIN_PATTERN = re.compile(
    rb"(?m)^AUTHORITATIVE_GOLDEN_EVIDENCE_NORMALIZED_SHA256 = \(\n"
    rb'    "([0-9a-f]{64})"\n\)$'
)
_REFERENCE_MANIFEST_PIN_PATTERN = re.compile(
    rb"(?m)^AUTHORITATIVE_REFERENCE_MANIFEST_SHA256: str \| None = \(\n"
    rb'    "([0-9a-f]{64})"\n\)$'
)


def _read_regular_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise Sam2GoldenCaptureError(f"{label} must be a regular non-symlink file: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise Sam2GoldenCaptureError(f"unable to read {label}: {path}") from error


def _raw_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _replace_one_pin(payload: bytes, pattern: re.Pattern[bytes], sentinel: bytes) -> bytes:
    matches = list(pattern.finditer(payload))
    if len(matches) != 1:
        raise Sam2GoldenCaptureError("golden evidence authority pin layout changed")
    start, end = matches[0].span(1)
    return payload[:start] + sentinel + payload[end:]


def _normalized_golden_bytes(payload: bytes) -> tuple[bytes, str, str]:
    tool_matches = list(_TOOL_PIN_PATTERN.finditer(payload))
    normalized_matches = list(_GOLDEN_NORMALIZED_PIN_PATTERN.finditer(payload))
    manifest_matches = list(_REFERENCE_MANIFEST_PIN_PATTERN.finditer(payload))
    if len(tool_matches) != 1 or len(normalized_matches) != 1 or len(manifest_matches) != 1:
        raise Sam2GoldenCaptureError("golden evidence authority pin layout changed")
    tool_pin = tool_matches[0].group(1).decode("ascii")
    normalized_pin = normalized_matches[0].group(1).decode("ascii")
    normalized = _replace_one_pin(payload, _TOOL_PIN_PATTERN, b"R" * 64)
    normalized = _replace_one_pin(
        normalized,
        _GOLDEN_NORMALIZED_PIN_PATTERN,
        b"G" * 64,
    )
    normalized = _replace_one_pin(
        normalized,
        _REFERENCE_MANIFEST_PIN_PATTERN,
        b"M" * 64,
    )
    return normalized, tool_pin, normalized_pin


def _reject_customizers() -> None:
    loaded = [name for name in ("sitecustomize", "usercustomize") if name in sys.modules]
    if loaded:
        raise Sam2GoldenCaptureError(
            "authoritative capture forbids Python customizers: " + ", ".join(loaded)
        )


def _verify_helper_sources() -> tuple[bytes, bytes, bytes, dict[str, str], dict[str, str]]:
    archive_payload = _read_regular_bytes(_ARCHIVE_CONTRACT_PATH, "archive contract helper")
    archive_sha256 = _raw_sha256(archive_payload)
    if archive_sha256 != _EXPECTED_ARCHIVE_CONTRACT_SHA256:
        raise Sam2GoldenCaptureError("archive contract helper hash mismatch")
    golden_payload = _read_regular_bytes(_GOLDEN_EVIDENCE_PATH, "golden evidence helper")
    normalized, tool_pin, normalized_pin = _normalized_golden_bytes(golden_payload)
    normalized_sha256 = _raw_sha256(normalized)
    if normalized_sha256 != _EXPECTED_GOLDEN_EVIDENCE_NORMALIZED_SHA256:
        raise Sam2GoldenCaptureError("golden evidence normalized hash mismatch")
    if normalized_pin != _EXPECTED_GOLDEN_EVIDENCE_NORMALIZED_SHA256:
        raise Sam2GoldenCaptureError("golden evidence normalized self-pin mismatch")
    runner_payload = _read_regular_bytes(_RUNNER_PATH, "capture runner")
    runner_sha256 = _raw_sha256(runner_payload)
    if runner_sha256 != tool_pin:
        raise Sam2GoldenCaptureError("capture runner does not match the verified golden tool pin")
    expected = {
        "tensorrt_model_connect.families.sam2.archive_contract": archive_sha256,
        "tensorrt_model_connect.families.sam2.golden_evidence.normalized": (normalized_sha256),
        _CANONICAL_RUNNER_NAME: runner_sha256,
    }
    raw = {
        "tensorrt_model_connect.families.sam2.archive_contract": archive_sha256,
        "tensorrt_model_connect.families.sam2.golden_evidence": _raw_sha256(golden_payload),
        _CANONICAL_RUNNER_NAME: runner_sha256,
    }
    return archive_payload, golden_payload, runner_payload, expected, raw


def _validate_controlled_venv() -> str:
    if not (
        sys.flags.isolated and sys.flags.safe_path and sys.flags.no_user_site and sys.flags.no_site
    ):
        raise Sam2GoldenCaptureError("authoritative capture requires direct Python -I -S bootstrap")
    if not sys.argv or not Path(sys.argv[0]).is_absolute():
        raise Sam2GoldenCaptureError(
            "authoritative capture requires an absolute capture_golden.py script path"
        )
    if platform.python_version() != "3.12.3":
        raise Sam2GoldenCaptureError("exact Python 3.12.3 is required")
    if (
        _REFERENCE_VENV_ROOT.is_symlink()
        or not _REFERENCE_VENV_ROOT.is_dir()
        or _REFERENCE_VENV_ROOT.resolve() != _REFERENCE_VENV_ROOT
    ):
        raise Sam2GoldenCaptureError("reference venv root is invalid")
    executable = Path(os.path.abspath(sys.executable))
    if executable.parent != _REFERENCE_VENV_BIN:
        raise Sam2GoldenCaptureError(
            f"authoritative interpreter must be under {_REFERENCE_VENV_BIN}, got {executable}"
        )
    config = _REFERENCE_VENV_ROOT / "pyvenv.cfg"
    if _raw_sha256(_read_regular_bytes(config, "reference pyvenv.cfg")) != (
        _REFERENCE_PYVENV_CFG_SHA256
    ):
        raise Sam2GoldenCaptureError("reference pyvenv.cfg hash mismatch")
    if (
        _CONTROLLED_SITE_PACKAGES.is_symlink()
        or not _CONTROLLED_SITE_PACKAGES.is_dir()
        or _CONTROLLED_SITE_PACKAGES.resolve() != _CONTROLLED_SITE_PACKAGES
    ):
        raise Sam2GoldenCaptureError("controlled venv site-packages directory is invalid")
    if any(
        "site-packages" in Path(entry).parts or "dist-packages" in Path(entry).parts
        for entry in sys.path
        if entry
    ):
        raise Sam2GoldenCaptureError("third-party import paths were present before venv bootstrap")
    sys.path.append(str(_CONTROLLED_SITE_PACKAGES))
    return str(config)


def _write_bootstrap_snapshot_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _set_bootstrap_snapshot_read_only(root: Path, read_only: bool) -> None:
    for directory, _directory_names, file_names in os.walk(root):
        current = Path(directory)
        current.chmod(0o500 if read_only else 0o700)
        for name in file_names:
            path = current / name
            if not path.is_symlink():
                path.chmod(0o400 if read_only else 0o600)


def _load_snapshot_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Sam2GoldenCaptureError(f"unable to create import spec for {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _bootstrap_verified_helpers(
    archive_payload: bytes,
    golden_payload: bytes,
    runner_payload: bytes,
    *,
    canonical_package: bool = False,
) -> tuple[Any, Any, tempfile.TemporaryDirectory[str], Path, Path, Path]:
    temporary = tempfile.TemporaryDirectory(prefix="trtmc-sam2-capture-code-")
    root = Path(temporary.name)
    package_name = (
        "tensorrt_model_connect.families.sam2"
        if canonical_package
        else "_trtmc_sam2_capture_closure"
    )
    package_root = root.joinpath(*package_name.split("."))
    archive_path = package_root / "archive_contract.py"
    golden_path = package_root / "golden_evidence.py"
    runner_path = package_root / "capture_golden.py"
    try:
        _write_bootstrap_snapshot_file(archive_path, archive_payload)
        _write_bootstrap_snapshot_file(golden_path, golden_payload)
        _write_bootstrap_snapshot_file(runner_path, runner_payload)
        _set_bootstrap_snapshot_read_only(root, True)
        package_parts = package_name.split(".")
        module_names = {
            *(".".join(package_parts[:index]) for index in range(1, len(package_parts) + 1)),
            f"{package_name}.archive_contract",
            f"{package_name}.golden_evidence",
        }
        preloaded = sorted(module_names.intersection(sys.modules))
        if preloaded:
            raise Sam2GoldenCaptureError(
                "capture helper package was already loaded: " + ", ".join(preloaded)
            )
        for index in range(1, len(package_parts) + 1):
            current_name = ".".join(package_parts[:index])
            current_root = root.joinpath(*package_parts[:index])
            package = types.ModuleType(current_name)
            package.__path__ = [str(current_root)]
            package.__package__ = current_name
            package.__spec__ = importlib.util.spec_from_loader(
                current_name,
                loader=None,
                is_package=True,
            )
            sys.modules[current_name] = package
        previous_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            archive_module = _load_snapshot_module(f"{package_name}.archive_contract", archive_path)
            golden_module = _load_snapshot_module(f"{package_name}.golden_evidence", golden_path)
        finally:
            sys.dont_write_bytecode = previous_bytecode
        return (
            archive_module,
            golden_module,
            temporary,
            archive_path,
            golden_path,
            runner_path,
        )
    except Exception:
        _set_bootstrap_snapshot_read_only(root, False)
        temporary.cleanup()
        raise


os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
if _DIRECT_SCRIPT:
    _reject_customizers()
    (
        _ARCHIVE_PAYLOAD,
        _GOLDEN_PAYLOAD,
        _RUNNER_PAYLOAD,
        _EXPECTED_HELPER_CLOSURE,
        _RAW_HELPER_CLOSURE,
    ) = _verify_helper_sources()
    _PYVENV_CFG_PATH = _validate_controlled_venv()
    (
        _archive_contract,
        _golden_evidence,
        _HELPER_TEMPORARY,
        _SNAPSHOT_ARCHIVE_PATH,
        _SNAPSHOT_GOLDEN_PATH,
        _SNAPSHOT_RUNNER_PATH,
    ) = _bootstrap_verified_helpers(
        _ARCHIVE_PAYLOAD,
        _GOLDEN_PAYLOAD,
        _RUNNER_PAYLOAD,
        canonical_package=True,
    )
else:
    (
        _ARCHIVE_PAYLOAD,
        _GOLDEN_PAYLOAD,
        _RUNNER_PAYLOAD,
        _EXPECTED_HELPER_CLOSURE,
        _RAW_HELPER_CLOSURE,
    ) = _verify_helper_sources()
    _PYVENV_CFG_PATH = "<non-authoritative-import>"
    (
        _archive_contract,
        _golden_evidence,
        _HELPER_TEMPORARY,
        _SNAPSHOT_ARCHIVE_PATH,
        _SNAPSHOT_GOLDEN_PATH,
        _SNAPSHOT_RUNNER_PATH,
    ) = _bootstrap_verified_helpers(
        _ARCHIVE_PAYLOAD,
        _GOLDEN_PAYLOAD,
        _RUNNER_PAYLOAD,
    )


def _cleanup_helper_snapshot() -> None:
    try:
        _set_bootstrap_snapshot_read_only(Path(_HELPER_TEMPORARY.name), False)
    finally:
        _HELPER_TEMPORARY.cleanup()


atexit.register(_cleanup_helper_snapshot)
np = importlib.import_module("numpy")

CHECKPOINT_RELATIVE_PATH = _archive_contract.CHECKPOINT_RELATIVE_PATH
CONFIG_RELATIVE_PATH = _archive_contract.CONFIG_RELATIVE_PATH
REFERENCE_CHECKPOINT_SHA256 = _archive_contract.REFERENCE_CHECKPOINT_SHA256
REFERENCE_CONFIG_SHA256 = _archive_contract.REFERENCE_CONFIG_SHA256
REFERENCE_SHA256SUMS_SHA256 = _archive_contract.REFERENCE_SHA256SUMS_SHA256
SHA256SUMS_RELATIVE_PATH = _archive_contract.SHA256SUMS_RELATIVE_PATH
Sam2ArchiveContractError = _archive_contract.Sam2ArchiveContractError
describe_archive = _archive_contract.describe_archive
sha256_file = _archive_contract.sha256_file

AUTHORITATIVE_CAPTURE_TOOL_SHA256 = _golden_evidence.AUTHORITATIVE_CAPTURE_TOOL_SHA256
COMPATIBLE_SOURCE_COMMIT = _golden_evidence.COMPATIBLE_SOURCE_COMMIT
COMPATIBLE_SOURCE_FILES_SHA256 = _golden_evidence.COMPATIBLE_SOURCE_FILES_SHA256
COMPATIBLE_SOURCE_OVERLAY_FILES_SHA256 = _golden_evidence.COMPATIBLE_SOURCE_OVERLAY_FILES_SHA256
FRAME_INDICES = _golden_evidence.FRAME_INDICES
INPUT_IMAGES_DECODED_RGB_UINT8_SHA256 = _golden_evidence.INPUT_IMAGES_DECODED_RGB_UINT8_SHA256
INPUT_IMAGES_RESIZED_1024_RGB_UINT8_SHA256 = (
    _golden_evidence.INPUT_IMAGES_RESIZED_1024_RGB_UINT8_SHA256
)
INPUT_IMAGES_SHA256 = _golden_evidence.INPUT_IMAGES_SHA256
MODEL_IMAGE_SHAPE_HW = _golden_evidence.MODEL_IMAGE_SHAPE_HW
ORIGINAL_IMAGE_SHAPE_HW = _golden_evidence.ORIGINAL_IMAGE_SHAPE_HW
PUBLIC_SAM2_BASE_COMMIT = _golden_evidence.PUBLIC_SAM2_BASE_COMMIT
PUBLIC_SAM2_BASE_FILES_SHA256 = _golden_evidence.PUBLIC_SAM2_BASE_FILES_SHA256
FrameZeroBBox = _golden_evidence.FrameZeroBBox
Provenance = _golden_evidence.Provenance
WorkloadCapture = _golden_evidence.WorkloadCapture
write_evidence = _golden_evidence.write_evidence

CAPTURE_RUN_COUNT = 3
DELIVERED_CONFIG_NAME = "configs/sam2.1/trtmc_delivery_bbox_59488bb78c7c.yaml"
SOURCE_OVERLAY_COMMIT_RECEIPT = "SOURCE_COMMIT"
_BINARY_MODULE_SUFFIXES = {".dll", ".dylib", ".egg-link", ".pth", ".pyc", ".pyd", ".so"}
_IMPORT_SHADOW_NAMES = {
    "PIL",
    "PIL.py",
    "antlr4",
    "antlr4.py",
    "hydra",
    "hydra.py",
    "iopath",
    "iopath.py",
    "numpy",
    "numpy.py",
    "omegaconf",
    "omegaconf.py",
    "portalocker",
    "portalocker.py",
    "sitecustomize.py",
    "torch",
    "torch.py",
    "torchvision",
    "torchvision.py",
    "tqdm",
    "tqdm.py",
    "usercustomize.py",
    "yaml",
    "yaml.py",
}
_PUBLIC_CONFIG_SYMLINKS = {
    "sam2/sam2_hiera_b+.yaml": "configs/sam2/sam2_hiera_b+.yaml",
    "sam2/sam2_hiera_l.yaml": "configs/sam2/sam2_hiera_l.yaml",
    "sam2/sam2_hiera_s.yaml": "configs/sam2/sam2_hiera_s.yaml",
    "sam2/sam2_hiera_t.yaml": "configs/sam2/sam2_hiera_t.yaml",
}


@dataclass(frozen=True)
class VerifiedCaptureInputs:
    source_root: Path
    package_root: Path
    checkpoint: Path
    image_dir: Path
    staged_config: Path
    source_files_sha256: Mapping[str, str]
    image_files_sha256: Mapping[str, str]
    decoder_environment: Mapping[str, object]
    capture_code_sha256: Mapping[str, str]
    capture_code_raw_sha256: Mapping[str, str]
    tool_sha256: str


@dataclass(frozen=True)
class CaptureSnapshot:
    root: Path
    source_root: Path
    checkpoint: Path
    image_dir: Path
    staged_config: Path


@dataclass(frozen=True)
class CapturedRun:
    workload: WorkloadCapture
    video_res_logits_dtypes: tuple[str, ...]


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_isolated_interpreter() -> None:
    if not (
        sys.flags.isolated and sys.flags.safe_path and sys.flags.no_user_site and sys.flags.no_site
    ):
        raise Sam2GoldenCaptureError(
            "authoritative capture requires direct Python -I -S mode with safe-path, "
            "no-user-site, and no-site"
        )
    if not _DIRECT_SCRIPT:
        raise Sam2GoldenCaptureError(
            "authoritative capture must invoke capture_golden.py by absolute script path"
        )
    _reject_customizers()
    site_paths = [
        Path(entry)
        for entry in sys.path
        if entry and ("site-packages" in Path(entry).parts or "dist-packages" in Path(entry).parts)
    ]
    if site_paths != [_CONTROLLED_SITE_PACKAGES]:
        raise Sam2GoldenCaptureError("controlled venv site-packages path is not isolated")


def _dependency_origin(module: Any, label: str) -> str:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise Sam2GoldenCaptureError(f"{label} has no auditable module origin")
    location = Path(module_file).resolve()
    if not location.is_relative_to(_CONTROLLED_SITE_PACKAGES):
        raise Sam2GoldenCaptureError(
            f"{label} must load from {_CONTROLLED_SITE_PACKAGES}, got {location}"
        )
    return str(location)


def _require_distribution_version(distribution: str, expected: str) -> str:
    discovered = list(
        importlib.metadata.Distribution.discover(
            name=distribution,
            path=[str(_CONTROLLED_SITE_PACKAGES)],
        )
    )
    if len(discovered) != 1:
        raise Sam2GoldenCaptureError(
            f"expected one {distribution} distribution in the controlled venv, "
            f"got {len(discovered)}"
        )
    selected = discovered[0]
    if Path(selected.locate_file("")).resolve() != _CONTROLLED_SITE_PACKAGES:
        raise Sam2GoldenCaptureError(f"{distribution} metadata escaped the controlled venv")
    actual = selected.version
    if actual != expected:
        raise Sam2GoldenCaptureError(f"exact {distribution} {expected} is required, got {actual}")
    return actual


def _capture_code_closure() -> tuple[dict[str, str], dict[str, str]]:
    (
        archive_payload,
        golden_payload,
        runner_payload,
        expected,
        raw,
    ) = _verify_helper_sources()
    if (
        archive_payload != _ARCHIVE_PAYLOAD
        or golden_payload != _GOLDEN_PAYLOAD
        or runner_payload != _RUNNER_PAYLOAD
    ):
        raise Sam2GoldenCaptureError("capture helper originals changed during execution")
    if expected != _EXPECTED_HELPER_CLOSURE or raw != _RAW_HELPER_CLOSURE:
        raise Sam2GoldenCaptureError("capture code closure receipt changed during execution")
    snapshot_root = Path(_HELPER_TEMPORARY.name)
    expected_snapshot_files = {
        _SNAPSHOT_ARCHIVE_PATH,
        _SNAPSHOT_GOLDEN_PATH,
        _SNAPSHOT_RUNNER_PATH,
    }
    actual_snapshot_files = {
        path for path in snapshot_root.rglob("*") if path.is_file() or path.is_symlink()
    }
    if actual_snapshot_files != expected_snapshot_files:
        raise Sam2GoldenCaptureError("capture helper snapshot file set changed")
    for path, payload, label in (
        (_SNAPSHOT_ARCHIVE_PATH, _ARCHIVE_PAYLOAD, "archive contract"),
        (_SNAPSHOT_GOLDEN_PATH, _GOLDEN_PAYLOAD, "golden evidence"),
        (_SNAPSHOT_RUNNER_PATH, _RUNNER_PAYLOAD, "capture runner"),
    ):
        if _read_regular_bytes(path, f"snapshotted {label}") != payload:
            raise Sam2GoldenCaptureError(f"snapshotted {label} changed during execution")
    loaded_origins = {
        Path(_archive_contract.__file__).resolve(),
        Path(_golden_evidence.__file__).resolve(),
    }
    if loaded_origins != {
        _SNAPSHOT_ARCHIVE_PATH.resolve(),
        _SNAPSHOT_GOLDEN_PATH.resolve(),
    }:
        raise Sam2GoldenCaptureError("capture helpers did not load from the private snapshot")

    runner_sha256 = _raw_sha256(runner_payload)
    _normalized, golden_tool_pin, _normalized_pin = _normalized_golden_bytes(golden_payload)
    if runner_sha256 != golden_tool_pin or runner_sha256 != AUTHORITATIVE_CAPTURE_TOOL_SHA256:
        raise Sam2GoldenCaptureError("capture runner does not match the verified golden tool pin")
    return expected, raw


def _run_git(source: Path, *arguments: str) -> bytes:
    command = [
        "git",
        "--no-replace-objects",
        "-c",
        f"safe.directory={source}",
        "-C",
        str(source),
        *arguments,
    ]
    try:
        return subprocess.run(command, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"")
        message = detail.decode(errors="replace").strip() if detail else str(error)
        raise Sam2GoldenCaptureError(f"source Git verification failed: {message}") from error


def _git_paths(source: Path, *arguments: str) -> set[str]:
    payload = _run_git(source, *arguments)
    try:
        return {line for line in payload.decode("utf-8").splitlines() if line}
    except UnicodeDecodeError as error:
        raise Sam2GoldenCaptureError("source Git paths are not UTF-8") from error


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise Sam2GoldenCaptureError(f"{label} must be a regular non-symlink file: {path}")


def _verify_source_tree(
    source_root: str | Path,
    *,
    public_commit: str = PUBLIC_SAM2_BASE_COMMIT,
    overlay_commit: str = COMPATIBLE_SOURCE_COMMIT,
    public_files: Mapping[str, str] = PUBLIC_SAM2_BASE_FILES_SHA256,
    overlay_files: Mapping[str, str] = COMPATIBLE_SOURCE_OVERLAY_FILES_SHA256,
    composed_files: Mapping[str, str] = COMPATIBLE_SOURCE_FILES_SHA256,
    delivered_config_sha256: str = REFERENCE_CONFIG_SHA256,
    public_config_symlinks: Mapping[str, str] = _PUBLIC_CONFIG_SYMLINKS,
) -> tuple[Path, Path, dict[str, str]]:
    source = Path(source_root).expanduser()
    if source.is_symlink() or not source.is_dir():
        raise Sam2GoldenCaptureError("SAM2 source root must be a regular directory")
    source = source.resolve()
    package = source / "sam2"
    if package.is_symlink() or not package.is_dir():
        raise Sam2GoldenCaptureError("SAM2 source package must be a regular directory")

    try:
        revision = _run_git(source, "rev-parse", "HEAD").decode("ascii").strip()
        git_root = Path(
            _run_git(source, "rev-parse", "--show-toplevel").decode("utf-8").strip()
        ).resolve()
    except UnicodeDecodeError as error:
        raise Sam2GoldenCaptureError("public source revision is not ASCII") from error
    if git_root != source:
        raise Sam2GoldenCaptureError(
            f"SAM2 source must be the Git worktree root: expected {source}, got {git_root}"
        )
    if revision != public_commit:
        raise Sam2GoldenCaptureError(
            f"public SAM2 base mismatch: expected {public_commit}, got {revision}"
        )
    for relative, expected in public_files.items():
        actual = _hash_bytes(_run_git(source, "show", f"HEAD:{relative}"))
        if actual != expected:
            raise Sam2GoldenCaptureError(
                f"public base Git object hash mismatch for {relative}: {actual}"
            )
    try:
        tree_records = _run_git(source, "ls-tree", "-r", "HEAD", "--", "sam2").decode("utf-8")
    except UnicodeDecodeError as error:
        raise Sam2GoldenCaptureError("public source tree metadata is not UTF-8") from error
    tracked_symlinks: dict[str, str] = {}
    for record in tree_records.splitlines():
        metadata, relative = record.split("\t", 1)
        if metadata.split()[0] == "120000":
            try:
                tracked_symlinks[relative] = _run_git(source, "show", f"HEAD:{relative}").decode(
                    "utf-8"
                )
            except UnicodeDecodeError as error:
                raise Sam2GoldenCaptureError(
                    f"public source symlink target is not UTF-8: {relative}"
                ) from error
    if tracked_symlinks != dict(public_config_symlinks):
        raise Sam2GoldenCaptureError("public Git-tracked config symlink contract mismatch")

    staged_config = package / DELIVERED_CONFIG_NAME
    receipt = source / SOURCE_OVERLAY_COMMIT_RECEIPT
    _require_regular_file(staged_config, "staged delivered config")
    _require_regular_file(receipt, "source overlay commit receipt")
    if staged_config.stat().st_mode & 0o111:
        raise Sam2GoldenCaptureError("staged delivered config must not be executable")
    if receipt.read_bytes() != f"{overlay_commit}\n".encode("ascii"):
        raise Sam2GoldenCaptureError("source overlay commit receipt mismatch")
    if receipt.stat().st_mode & 0o111:
        raise Sam2GoldenCaptureError("source overlay commit receipt must not be executable")
    if sha256_file(staged_config) != delivered_config_sha256:
        raise Sam2GoldenCaptureError("staged delivered config hash mismatch")

    allowed_changes = set(overlay_files) | {
        f"sam2/{DELIVERED_CONFIG_NAME}",
        SOURCE_OVERLAY_COMMIT_RECEIPT,
    }
    changed = _git_paths(source, "diff", "--name-only", "HEAD", "--")
    changed |= _git_paths(source, "ls-files", "--others", "--exclude-standard", "--")
    ignored = _git_paths(
        source,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
    )
    if ignored:
        raise Sam2GoldenCaptureError(
            "source tree contains ignored artifacts: " + ", ".join(sorted(ignored))
        )
    if changed != allowed_changes:
        missing = sorted(allowed_changes - changed)
        unexpected = sorted(changed - allowed_changes)
        raise Sam2GoldenCaptureError(
            f"source overlay working-tree mismatch; missing={missing}, unexpected={unexpected}"
        )

    python_files: set[str] = set()
    observed_symlinks: dict[str, str] = {}
    for directory, directory_names, file_names in os.walk(package, followlinks=False):
        current = Path(directory)
        for name in [*directory_names, *file_names]:
            path = current / name
            if path.is_symlink():
                relative = path.relative_to(source).as_posix()
                target = os.readlink(path)
                if public_config_symlinks.get(relative) != target:
                    raise Sam2GoldenCaptureError(f"source package contains a symlink: {path}")
                observed_symlinks[relative] = target
        if "__pycache__" in directory_names:
            raise Sam2GoldenCaptureError("source package contains generated __pycache__")
        for name in file_names:
            path = current / name
            relative = path.relative_to(source).as_posix()
            if path.is_symlink():
                continue
            mode = path.stat().st_mode
            if not stat.S_ISREG(mode):
                raise Sam2GoldenCaptureError(f"source artifact is not regular: {relative}")
            if mode & 0o111:
                raise Sam2GoldenCaptureError(f"source artifact is executable: {relative}")
            if any(
                part.lower() == "vendor" or "hoi" in part.lower()
                for part in path.relative_to(package).parts
            ):
                raise Sam2GoldenCaptureError(
                    f"private HOI/vendor artifact is forbidden: {relative}"
                )
            if path.suffix.lower() in _BINARY_MODULE_SUFFIXES:
                raise Sam2GoldenCaptureError(f"binary/import artifact is forbidden: {relative}")
            if path.suffix == ".py":
                python_files.add(relative)

    if observed_symlinks != dict(public_config_symlinks):
        missing = sorted(set(public_config_symlinks) - set(observed_symlinks))
        raise Sam2GoldenCaptureError(f"public config symlink set mismatch; missing={missing}")

    if python_files != set(composed_files):
        missing = sorted(set(composed_files) - python_files)
        unexpected = sorted(python_files - set(composed_files))
        raise Sam2GoldenCaptureError(
            f"composed source Python file set mismatch; missing={missing}, unexpected={unexpected}"
        )
    verified: dict[str, str] = {}
    for relative, expected in composed_files.items():
        path = source / relative
        _require_regular_file(path, "composed source file")
        actual = sha256_file(path)
        if actual != expected:
            raise Sam2GoldenCaptureError(
                f"composed source hash mismatch for {relative}: expected {expected}, got {actual}"
            )
        verified[relative] = actual

    backbones_init = "sam2/modeling/backbones/__init__.py"
    if verified.get(backbones_init) != public_files.get(backbones_init):
        raise Sam2GoldenCaptureError("private backbones __init__.py is forbidden")
    for name in _IMPORT_SHADOW_NAMES:
        if (source / name).exists():
            raise Sam2GoldenCaptureError(f"source root contains import-shadowing artifact {name}")
    return source, staged_config, dict(sorted(verified.items()))


def _verify_input_images(
    image_dir: Path,
    expected: Mapping[str, str] = INPUT_IMAGES_SHA256,
) -> dict[str, str]:
    if image_dir.is_symlink() or not image_dir.is_dir():
        raise Sam2GoldenCaptureError("delivered input directory must be regular")
    names = [path.name for path in sorted(image_dir.iterdir(), key=lambda item: item.name)]
    if names != list(expected):
        raise Sam2GoldenCaptureError(
            f"delivered inputs must be exact numeric order {list(expected)}, got {names}"
        )
    verified: dict[str, str] = {}
    for name, expected_hash in expected.items():
        path = image_dir / name
        _require_regular_file(path, "delivered input image")
        if path.stat().st_mode & 0o111:
            raise Sam2GoldenCaptureError(f"delivered input image is executable: {name}")
        actual = sha256_file(path)
        if actual != expected_hash:
            raise Sam2GoldenCaptureError(f"delivered input hash mismatch for {name}")
        verified[name] = actual
    return verified


def _verify_decoded_input_images(image_dir: Path) -> dict[str, object]:
    """Bind the decoder stack and exact RGB bytes consumed before resize."""

    try:
        pil = importlib.import_module("PIL")
        image_module = importlib.import_module("PIL.Image")
        features = importlib.import_module("PIL.features")
    except ImportError as error:
        raise Sam2GoldenCaptureError("Pillow is required for exact JPEG verification") from error

    _dependency_origin(np, "numpy")
    _dependency_origin(pil, "Pillow")
    _dependency_origin(image_module, "Pillow image module")
    _dependency_origin(features, "Pillow feature module")

    decoder_environment = {
        "numpy": np.__version__,
        "pillow": str(pil.__version__),
        "pillow_jpeg_codec": str(features.version_codec("jpg")),
        "libjpeg_turbo": str(features.version_feature("libjpeg_turbo")),
    }
    expected_environment = {
        "numpy": "2.5.2",
        "pillow": "12.3.0",
        "pillow_jpeg_codec": "6.2",
        "libjpeg_turbo": "3.1.4.1",
    }
    if decoder_environment != expected_environment:
        raise Sam2GoldenCaptureError(
            f"exact JPEG decoder environment mismatch: expected {expected_environment}, "
            f"got {decoder_environment}"
        )

    decoded_hashes: dict[str, str] = {}
    resized_hashes: dict[str, str] = {}
    for name, expected_hash in INPUT_IMAGES_DECODED_RGB_UINT8_SHA256.items():
        try:
            with image_module.open(image_dir / name) as image:
                rgb = image.convert("RGB")
                decoded = np.asarray(rgb)
                resized = np.array(rgb.resize((MODEL_IMAGE_SHAPE_HW[1], MODEL_IMAGE_SHAPE_HW[0])))
        except Exception as error:
            raise Sam2GoldenCaptureError(f"unable to decode exact input image {name}") from error
        expected_shape = (*ORIGINAL_IMAGE_SHAPE_HW, 3)
        if decoded.dtype != np.uint8 or decoded.shape != expected_shape:
            raise Sam2GoldenCaptureError(
                f"decoded input {name} must be RGB uint8 with shape {expected_shape}"
            )
        actual = _hash_bytes(np.ascontiguousarray(decoded).tobytes())
        if actual != expected_hash:
            raise Sam2GoldenCaptureError(
                f"decoded RGB uint8 hash mismatch for {name}: expected {expected_hash}, got {actual}"
            )
        decoded_hashes[name] = actual
        resized_shape = (*MODEL_IMAGE_SHAPE_HW, 3)
        if resized.dtype != np.uint8 or resized.shape != resized_shape:
            raise Sam2GoldenCaptureError(
                f"resized input {name} must be RGB uint8 with shape {resized_shape}"
            )
        resized_actual = _hash_bytes(np.ascontiguousarray(resized).tobytes())
        resized_expected = INPUT_IMAGES_RESIZED_1024_RGB_UINT8_SHA256.get(name)
        if resized_actual != resized_expected:
            raise Sam2GoldenCaptureError(
                f"resized RGB uint8 hash mismatch for {name}: expected "
                f"{resized_expected}, got {resized_actual}"
            )
        resized_hashes[name] = resized_actual
    return {
        **decoder_environment,
        "input_images_decoded_rgb_uint8_sha256": decoded_hashes,
        "input_images_resized_1024_rgb_uint8_sha256": resized_hashes,
    }


def verify_capture_inputs(
    source_root: str | Path, package_dir: str | Path
) -> VerifiedCaptureInputs:
    """Verify all source and delivered bytes before any SAM2 runtime import."""

    source, staged_config, source_hashes = _verify_source_tree(source_root)
    yaml_module = importlib.import_module("yaml")
    _dependency_origin(yaml_module, "PyYAML")
    _require_distribution_version("PyYAML", "6.0.3")
    try:
        description = describe_archive(package_dir, verify_provenance=True)
    except Sam2ArchiveContractError as error:
        raise Sam2GoldenCaptureError(str(error)) from error
    provenance = description.provenance
    if (
        provenance.get("manifest_sha256") != REFERENCE_SHA256SUMS_SHA256
        or provenance.get("checkpoint_sha256") != REFERENCE_CHECKPOINT_SHA256
        or provenance.get("config_sha256") != REFERENCE_CONFIG_SHA256
    ):
        raise Sam2GoldenCaptureError(
            "delivered package provenance does not match the exact archive"
        )
    package = description.root
    checkpoint = package / CHECKPOINT_RELATIVE_PATH
    config = package / CONFIG_RELATIVE_PATH
    if staged_config.read_bytes() != config.read_bytes():
        raise Sam2GoldenCaptureError("staged config is not byte-identical to the delivered config")
    image_hashes = _verify_input_images(package / "inputs")
    decoder_environment = _verify_decoded_input_images(package / "inputs")

    capture_code_sha256, capture_code_raw_sha256 = _capture_code_closure()
    tool_sha256 = capture_code_sha256["tensorrt_model_connect.families.sam2.capture_golden"]
    if AUTHORITATIVE_CAPTURE_TOOL_SHA256 is None:
        raise Sam2GoldenCaptureError("checked-in authoritative capture tool hash is not pinned")
    if tool_sha256 != AUTHORITATIVE_CAPTURE_TOOL_SHA256:
        raise Sam2GoldenCaptureError("running capture tool does not match its checked-in hash pin")
    return VerifiedCaptureInputs(
        source,
        package,
        checkpoint,
        package / "inputs",
        staged_config,
        source_hashes,
        image_hashes,
        decoder_environment,
        capture_code_sha256,
        capture_code_raw_sha256,
        tool_sha256,
    )


def _copy_verified_file(source: Path, destination: Path, expected_sha256: str) -> None:
    _require_regular_file(source, "snapshot source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    if sha256_file(destination) != expected_sha256:
        raise Sam2GoldenCaptureError(f"snapshot copy hash mismatch for {source}")


def _verify_capture_snapshot(snapshot: CaptureSnapshot) -> dict[str, object]:
    expected_files = {
        *(snapshot.source_root / relative for relative in COMPATIBLE_SOURCE_FILES_SHA256),
        snapshot.staged_config,
        snapshot.checkpoint,
        *(snapshot.image_dir / name for name in INPUT_IMAGES_SHA256),
    }
    actual_files: set[Path] = set()
    for directory, directory_names, file_names in os.walk(snapshot.root, followlinks=False):
        current = Path(directory)
        for name in [*directory_names, *file_names]:
            path = current / name
            if path.is_symlink():
                raise Sam2GoldenCaptureError(f"private capture snapshot contains symlink {path}")
        actual_files.update(current / name for name in file_names)
    if actual_files != expected_files:
        raise Sam2GoldenCaptureError("private capture snapshot file set changed")

    for relative, expected in COMPATIBLE_SOURCE_FILES_SHA256.items():
        if sha256_file(snapshot.source_root / relative) != expected:
            raise Sam2GoldenCaptureError(f"private source snapshot changed: {relative}")
    if sha256_file(snapshot.staged_config) != REFERENCE_CONFIG_SHA256:
        raise Sam2GoldenCaptureError("private config snapshot changed")
    if sha256_file(snapshot.checkpoint) != REFERENCE_CHECKPOINT_SHA256:
        raise Sam2GoldenCaptureError("private checkpoint snapshot changed")
    if _verify_input_images(snapshot.image_dir, INPUT_IMAGES_SHA256) != dict(INPUT_IMAGES_SHA256):
        raise Sam2GoldenCaptureError("private input snapshot changed")
    return _verify_decoded_input_images(snapshot.image_dir)


def _set_snapshot_read_only(root: Path, read_only: bool) -> None:
    file_mode = 0o400 if read_only else 0o600
    directory_mode = 0o500 if read_only else 0o700
    for directory, _directory_names, file_names in os.walk(root):
        current = Path(directory)
        current.chmod(directory_mode)
        for name in file_names:
            path = current / name
            if not path.is_symlink():
                path.chmod(file_mode)


@contextmanager
def _private_capture_snapshot(inputs: VerifiedCaptureInputs) -> Iterator[CaptureSnapshot]:
    with tempfile.TemporaryDirectory(prefix="trtmc-sam2-golden-") as temporary:
        root = Path(temporary)
        source = root / "source"
        delivery = root / "delivery"
        for relative, digest in COMPATIBLE_SOURCE_FILES_SHA256.items():
            _copy_verified_file(
                inputs.source_root / relative,
                source / relative,
                digest,
            )
        staged_config = source / "sam2" / DELIVERED_CONFIG_NAME
        _copy_verified_file(inputs.staged_config, staged_config, REFERENCE_CONFIG_SHA256)
        checkpoint = delivery / "checkpoint.pt"
        _copy_verified_file(inputs.checkpoint, checkpoint, REFERENCE_CHECKPOINT_SHA256)
        image_dir = delivery / "inputs"
        for name, digest in INPUT_IMAGES_SHA256.items():
            _copy_verified_file(inputs.image_dir / name, image_dir / name, digest)
        snapshot = CaptureSnapshot(root, source, checkpoint, image_dir, staged_config)
        decoder_environment = _verify_capture_snapshot(snapshot)
        if decoder_environment != inputs.decoder_environment:
            raise Sam2GoldenCaptureError("private snapshot decoder evidence mismatch")
        try:
            _set_snapshot_read_only(root, True)
            yield snapshot
            if _verify_capture_snapshot(snapshot) != inputs.decoder_environment:
                raise Sam2GoldenCaptureError("private capture snapshot changed during execution")
        finally:
            _set_snapshot_read_only(root, False)


def _loaded_sam2_module_mismatch(source: Path) -> tuple[str, str] | None:
    expected = (source / "sam2").resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "sam2" and not name.startswith("sam2."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            return name, "<no file>"
        location = Path(module_file).resolve()
        if not location.is_relative_to(expected):
            return name, str(location)
    return None


def _sam2_import_candidates(source: Path) -> set[Path]:
    candidates: set[Path] = set()
    for entry in sys.path:
        root = Path.cwd() if not entry else Path(entry)
        candidate = root / "sam2"
        if candidate.is_dir():
            candidates.add(candidate.resolve())
    candidates.discard((source / "sam2").resolve())
    return candidates


@contextmanager
def _isolated_runtime_imports(
    source: Path,
) -> Iterator[tuple[Any, Any, Mapping[str, str]]]:
    mismatch = _loaded_sam2_module_mismatch(source)
    if mismatch is not None:
        raise Sam2GoldenCaptureError(
            f"cached SAM2 module {mismatch[0]!r} comes from {mismatch[1]}; start a clean process"
        )
    foreign_candidates = _sam2_import_candidates(source)
    if foreign_candidates:
        raise Sam2GoldenCaptureError(
            "foreign SAM2 import candidates are present: "
            + ", ".join(str(path) for path in sorted(foreign_candidates))
        )

    previous_bytecode = sys.dont_write_bytecode
    previous_env = os.environ.get("PYTHONDONTWRITEBYTECODE")
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    source_inserted = False
    try:
        torch = importlib.import_module("torch")
        torchvision = importlib.import_module("torchvision")
        antlr4_module = importlib.import_module("antlr4")
        hydra = importlib.import_module("hydra")
        global_hydra_module = importlib.import_module("hydra.core.global_hydra")
        omegaconf = importlib.import_module("omegaconf")
        yaml_module = importlib.import_module("yaml")
        tqdm_module = importlib.import_module("tqdm")
        iopath_module = importlib.import_module("iopath")
        portalocker_module = importlib.import_module("portalocker")
        global_hydra = global_hydra_module.GlobalHydra.instance()
        if global_hydra.is_initialized():
            raise Sam2GoldenCaptureError(
                "Hydra was initialized before the exact SAM2 config module import"
            )
        dependency_origins = {
            "antlr4": _dependency_origin(antlr4_module, "antlr4"),
            "numpy": _dependency_origin(np, "numpy"),
            "pillow": _dependency_origin(sys.modules["PIL"], "Pillow"),
            "torch": _dependency_origin(torch, "torch"),
            "torchvision": _dependency_origin(torchvision, "torchvision"),
            "hydra": _dependency_origin(hydra, "hydra"),
            "iopath": _dependency_origin(iopath_module, "iopath"),
            "omegaconf": _dependency_origin(omegaconf, "omegaconf"),
            "portalocker": _dependency_origin(portalocker_module, "portalocker"),
            "pyyaml": _dependency_origin(yaml_module, "PyYAML"),
            "tqdm": _dependency_origin(tqdm_module, "tqdm"),
        }
        sys.path.insert(0, str(source))
        source_inserted = True
        importlib.invalidate_caches()
        sam2 = importlib.import_module("sam2")
        if not global_hydra.is_initialized():
            raise Sam2GoldenCaptureError("SAM2 did not initialize its exact Hydra config module")
        package_paths = {Path(path).resolve() for path in sam2.__path__}
        if package_paths != {(source / "sam2").resolve()}:
            raise Sam2GoldenCaptureError(
                f"SAM2 package search path is not isolated: {package_paths}"
            )
        if importlib.util.find_spec("sam2._C") is not None:
            raise Sam2GoldenCaptureError("optional sam2._C must be absent for the reference run")
        builder_module = importlib.import_module("sam2.build_sam")
        yield torch, builder_module.build_sam2_video_predictor, dependency_origins
        mismatch = _loaded_sam2_module_mismatch(source)
        if mismatch is not None:
            raise Sam2GoldenCaptureError(
                f"loaded SAM2 module {mismatch[0]!r} escaped source root: {mismatch[1]}"
            )
    finally:
        if source_inserted and sys.path and sys.path[0] == str(source):
            sys.path.pop(0)
        elif source_inserted:
            try:
                sys.path.remove(str(source))
            except ValueError:
                pass
        sys.dont_write_bytecode = previous_bytecode
        if previous_env is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = previous_env


def _driver_version() -> str:
    try:
        payload = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Sam2GoldenCaptureError("unable to read the CUDA driver version") from error
    values = [line.strip() for line in payload.splitlines() if line.strip()]
    if values != ["595.58.03"]:
        raise Sam2GoldenCaptureError(f"exact L4 driver 595.58.03 is required, got {values}")
    return values[0]


def _configure_and_record_environment(
    torch: Any,
    decoder_environment: Mapping[str, object],
    dependency_origins: Mapping[str, str],
) -> dict[str, object]:
    if platform.python_version() != "3.12.3":
        raise Sam2GoldenCaptureError("exact Python 3.12.3 is required")
    if (
        _raw_sha256(_read_regular_bytes(Path(_PYVENV_CFG_PATH), "reference pyvenv.cfg"))
        != _REFERENCE_PYVENV_CFG_SHA256
    ):
        raise Sam2GoldenCaptureError("reference pyvenv.cfg changed during capture")
    if str(torch.__version__) != "2.7.1+cu128" or str(torch.version.cuda) != "12.8":
        raise Sam2GoldenCaptureError("exact torch 2.7.1+cu128 with CUDA 12.8 is required")
    versions = {
        "antlr4_python3_runtime": _require_distribution_version("antlr4-python3-runtime", "4.9.3"),
        "hydra_core": _require_distribution_version("hydra-core", "1.3.2"),
        "iopath": _require_distribution_version("iopath", "0.1.10"),
        "omegaconf": _require_distribution_version("omegaconf", "2.3.1"),
        "portalocker": _require_distribution_version("portalocker", "4.1.0"),
        "pyyaml": _require_distribution_version("PyYAML", "6.0.3"),
        "torchvision": _require_distribution_version("torchvision", "0.22.1+cu128"),
        "tqdm": _require_distribution_version("tqdm", "4.67.1"),
    }
    expected_origin_names = {
        "antlr4",
        "hydra",
        "iopath",
        "numpy",
        "omegaconf",
        "pillow",
        "portalocker",
        "pyyaml",
        "torch",
        "torchvision",
        "tqdm",
    }
    if set(dependency_origins) != expected_origin_names:
        raise Sam2GoldenCaptureError("exact dependency origin receipt set mismatch")
    if any(
        not Path(origin).resolve().is_relative_to(_CONTROLLED_SITE_PACKAGES)
        for origin in dependency_origins.values()
    ):
        raise Sam2GoldenCaptureError("dependency origin escaped the controlled venv")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Sam2GoldenCaptureError("the reference capture requires exactly one CUDA GPU")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_name(0) != "NVIDIA L4":
        raise Sam2GoldenCaptureError("the reference capture requires NVIDIA L4")
    if list(torch.cuda.get_device_capability(0)) != [8, 9]:
        raise Sam2GoldenCaptureError("the reference capture requires SM89")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    if torch.backends.cudnn.version() != 90701:
        raise Sam2GoldenCaptureError("exact cuDNN 90701 is required")
    if torch.are_deterministic_algorithms_enabled():
        raise Sam2GoldenCaptureError("deterministic algorithms must be disabled")
    return {
        "python": platform.python_version(),
        **decoder_environment,
        **versions,
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_driver": _driver_version(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "sam2_optional_extension_present": False,
        "autocast": "cuda bfloat16",
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "python_isolated": bool(sys.flags.isolated),
        "python_safe_path": bool(sys.flags.safe_path),
        "python_no_user_site": bool(sys.flags.no_user_site),
        "python_no_site": bool(sys.flags.no_site),
        "controlled_site_packages": str(_CONTROLLED_SITE_PACKAGES),
        "venv_pyvenv_cfg_sha256": _REFERENCE_PYVENV_CFG_SHA256,
        "dependency_origins": dict(dependency_origins),
        "capture_input_isolation": "private_read_only_verified_snapshot_v1",
        "capture_runs": CAPTURE_RUN_COUNT,
        "async_loading_frames": True,
        "apply_postprocessing": True,
        "config_name": DELIVERED_CONFIG_NAME,
    }


def _assert_predictor_contract(predictor: Any) -> None:
    encoder = getattr(predictor, "image_encoder", None)
    if encoder is None or getattr(encoder, "bbox_head", None) is None:
        raise Sam2GoldenCaptureError("predictor does not contain the delivered bbox head")
    if getattr(encoder, "learnable_fpn_module", None) is not None:
        raise Sam2GoldenCaptureError("learnable CSPNeXt FPN is forbidden for the delivery")
    if hasattr(encoder, "hoi_head"):
        raise Sam2GoldenCaptureError("HOI head is forbidden for the bbox delivery")
    if getattr(predictor, "fill_hole_area", None) != 8:
        raise Sam2GoldenCaptureError("source postprocessing fill_hole_area must be 8")
    if getattr(predictor, "binarize_mask_from_pts_for_mem_enc", None) is not True:
        raise Sam2GoldenCaptureError("source prompt-mask postprocessing is not enabled")
    decoder = getattr(predictor, "sam_mask_decoder", None)
    if getattr(decoder, "dynamic_multimask_via_stability", None) is not True:
        raise Sam2GoldenCaptureError("source dynamic multimask postprocessing is not enabled")
    if getattr(predictor, "training", True):
        raise Sam2GoldenCaptureError("source predictor must be in eval mode")


def _numpy(value: Any) -> np.ndarray:
    try:
        tensor = value.detach().cpu().contiguous()
        if str(getattr(tensor, "dtype", "")) == "torch.bfloat16":
            tensor = tensor.float()
        return np.asarray(tensor.numpy())
    except (AttributeError, TypeError, RuntimeError) as error:
        raise Sam2GoldenCaptureError("source returned an unsupported tensor") from error


def _clone(value: Any) -> Any:
    try:
        return value.detach().clone()
    except AttributeError as error:
        raise Sam2GoldenCaptureError("source detection is not a tensor") from error


def _capture_once(predictor: Any, image_dir: Path, torch: Any) -> CapturedRun:
    original_get_det_results = predictor._get_det_results
    pre_rescale: dict[str, Any] = {}

    # This provenance-only interception is deliberately outside every timing
    # boundary; the authoritative artifact makes no performance claim.
    def capture_before_rescale(state: Mapping[str, Any], frame_idx: int):
        if frame_idx == 0 and not pre_rescale:
            try:
                result = state["cached_features"][0][1]["det_results"][0]
            except (KeyError, IndexError, TypeError) as error:
                raise Sam2GoldenCaptureError(
                    "frame-zero cached detector output is malformed"
                ) from error
            if result.get("has_rescaled", False):
                raise Sam2GoldenCaptureError("frame-zero bbox was rescaled before capture")
            pre_rescale.update(
                bboxes=_clone(result["bboxes"]),
                scores=_clone(result["scores"]),
                labels=_clone(result["labels"]),
            )
        return original_get_det_results(state, frame_idx)

    predictor._get_det_results = capture_before_rescale
    state = None
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(
                str(image_dir),
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
                async_loading_frames=True,
                frame_idx=0,
            )
            if state.get("num_frames") != len(FRAME_INDICES):
                raise Sam2GoldenCaptureError("source state does not contain exactly five frames")
            if (state.get("video_height"), state.get("video_width")) != ORIGINAL_IMAGE_SHAPE_HW:
                raise Sam2GoldenCaptureError("source state has unexpected original image geometry")
            try:
                state_paths = [
                    Path(path).resolve(strict=True) for path in state.get("img_paths", ())
                ]
                expected_paths = [
                    (image_dir / name).resolve(strict=True) for name in INPUT_IMAGES_SHA256
                ]
            except (OSError, TypeError) as error:
                raise Sam2GoldenCaptureError(
                    "source loader returned unauditable frame paths"
                ) from error
            if state_paths != expected_paths:
                raise Sam2GoldenCaptureError(
                    "source loader changed the snapshot-bound numeric frame order"
                )
            detection = original_get_det_results(state, 0)
            if not isinstance(detection, Mapping):
                raise Sam2GoldenCaptureError("frame-zero post-NMS detection is missing")

            model_boxes = _numpy(pre_rescale.get("bboxes"))
            original_boxes = _numpy(detection.get("bboxes"))
            scores = _numpy(pre_rescale.get("scores"))
            labels = _numpy(pre_rescale.get("labels"))
            if (
                model_boxes.shape != (1, 4)
                or original_boxes.shape != (1, 4)
                or scores.shape != (1,)
                or labels.shape != (1,)
            ):
                raise Sam2GoldenCaptureError(
                    "frame zero must have exactly one ordered post-NMS detection"
                )
            if not np.array_equal(scores, _numpy(detection.get("scores"))) or not np.array_equal(
                labels, _numpy(detection.get("labels"))
            ):
                raise Sam2GoldenCaptureError("frame-zero detector ordering changed during rescale")
            predictor.add_new_points_or_box(
                state,
                0,
                0,
                box=detection["bboxes"][0],
            )

            masks: list[np.ndarray] = []
            observed_frames: list[int] = []
            video_res_logits_dtypes: list[str] = []
            for result in predictor.propagate_in_video(
                state,
                start_frame_idx=0,
                max_frame_num_to_track=None,
                reverse=False,
            ):
                if not isinstance(result, Sequence) or len(result) != 5:
                    raise Sam2GoldenCaptureError("source propagation result ABI mismatch")
                frame_idx, object_ids, video_res_logits, _detections, _mask_ious = result
                observed_frames.append(int(frame_idx))
                if [int(value) for value in object_ids] != [0]:
                    raise Sam2GoldenCaptureError("source propagation must contain only object 0")
                expected_shape = (1, 1, *ORIGINAL_IMAGE_SHAPE_HW)
                if tuple(getattr(video_res_logits, "shape", ())) != expected_shape:
                    raise Sam2GoldenCaptureError(
                        f"source video-resolution logits must have shape {expected_shape}"
                    )
                logits_dtype = str(getattr(video_res_logits, "dtype", ""))
                if logits_dtype not in {"torch.bfloat16", "torch.float32"}:
                    raise Sam2GoldenCaptureError(
                        "source video-resolution logits must be BF16 or FP32 floating tensors"
                    )
                logits = _numpy(video_res_logits)
                if logits.shape != expected_shape or not np.isfinite(logits).all():
                    raise Sam2GoldenCaptureError(
                        "source video-resolution logits must be exact-shape and finite before threshold"
                    )
                video_res_logits_dtypes.append(logits_dtype)
                binary = _numpy((video_res_logits > 0).to(dtype=torch.uint8))
                if binary.shape != expected_shape:
                    raise Sam2GoldenCaptureError("thresholded source mask shape changed")
                masks.append(np.ascontiguousarray(binary[0], dtype=np.uint8))
            if observed_frames != list(FRAME_INDICES):
                raise Sam2GoldenCaptureError(
                    f"source propagation frame order mismatch: {observed_frames}"
                )

        torch.cuda.synchronize()
        box = FrameZeroBBox(
            original_xyxy=tuple(float(value) for value in original_boxes[0]),
            model_xyxy_1024=tuple(float(value) for value in model_boxes[0]),
            score=float(scores[0]),
            label=int(labels[0]),
        )
        return CapturedRun(
            workload=WorkloadCapture(
                masks=np.stack(masks, axis=0),
                frame_zero_bbox=box,
                post_nms_detection_count=1,
                selected_object_id=0,
            ),
            video_res_logits_dtypes=tuple(video_res_logits_dtypes),
        )
    finally:
        predictor._get_det_results = original_get_det_results
        if state is not None:
            predictor.reset_state(state)


def _capture_three_runs(predictor: Any, image_dir: Path, torch: Any) -> list[CapturedRun]:
    captures = []
    for _run_index in range(CAPTURE_RUN_COUNT):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        captures.append(_capture_once(predictor, image_dir, torch))
        gc.collect()
        torch.cuda.empty_cache()
    return captures


def capture_authoritative_evidence(
    source_root: str | Path,
    package_dir: str | Path,
    destination: str | Path,
) -> dict[str, object]:
    """Verify, run, and write an authoritative-reference candidate."""

    _require_isolated_interpreter()
    _dependency_origin(np, "numpy")
    inputs = verify_capture_inputs(source_root, package_dir)
    with _private_capture_snapshot(inputs) as snapshot:
        with _isolated_runtime_imports(snapshot.source_root) as (
            torch,
            builder,
            dependency_origins,
        ):
            environment = _configure_and_record_environment(
                torch,
                inputs.decoder_environment,
                dependency_origins,
            )
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            predictor = builder(
                config_file=DELIVERED_CONFIG_NAME,
                ckpt_path=str(snapshot.checkpoint),
                device="cuda",
                mode="eval",
                apply_postprocessing=True,
                vos_optimized=False,
            )
            _assert_predictor_contract(predictor)
            torch.cuda.synchronize()
            captures = _capture_three_runs(predictor, snapshot.image_dir, torch)
            environment["video_res_logits_dtypes"] = [
                list(capture.video_res_logits_dtypes) for capture in captures
            ]

    # Rehash source and delivery after execution.  The evidence is withheld if
    # any input changed after the pre-import verification but before capture.
    if verify_capture_inputs(source_root, package_dir) != inputs:
        raise Sam2GoldenCaptureError("capture inputs changed during source execution")

    artifacts = {
        "capture_golden.py": inputs.tool_sha256,
        f"delivery/{SHA256SUMS_RELATIVE_PATH.as_posix()}": sha256_file(
            inputs.package_root / SHA256SUMS_RELATIVE_PATH
        ),
        f"delivery/{CONFIG_RELATIVE_PATH.as_posix()}": REFERENCE_CONFIG_SHA256,
        f"delivery/{CHECKPOINT_RELATIVE_PATH.as_posix()}": REFERENCE_CHECKPOINT_SHA256,
        f"source/{SOURCE_OVERLAY_COMMIT_RECEIPT}": sha256_file(
            inputs.source_root / SOURCE_OVERLAY_COMMIT_RECEIPT
        ),
        f"source/sam2/{DELIVERED_CONFIG_NAME}": REFERENCE_CONFIG_SHA256,
        **{f"delivery/inputs/{name}": digest for name, digest in inputs.image_files_sha256.items()},
        **{
            f"decoded_rgb_uint8/{name}": digest
            for name, digest in INPUT_IMAGES_DECODED_RGB_UINT8_SHA256.items()
        },
        **{
            f"resized_1024_rgb_uint8/{name}": digest
            for name, digest in INPUT_IMAGES_RESIZED_1024_RGB_UINT8_SHA256.items()
        },
        **{f"capture_code/{name}": digest for name, digest in inputs.capture_code_sha256.items()},
        **{
            f"capture_code_raw/{name}": digest
            for name, digest in inputs.capture_code_raw_sha256.items()
        },
    }
    provenance = Provenance(
        source_commit=PUBLIC_SAM2_BASE_COMMIT,
        source_overlay_declared_commit=COMPATIBLE_SOURCE_COMMIT,
        source_files_sha256=inputs.source_files_sha256,
        checkpoint_sha256=REFERENCE_CHECKPOINT_SHA256,
        config_sha256=REFERENCE_CONFIG_SHA256,
        image_files_sha256=inputs.image_files_sha256,
        capture_tool_sha256=inputs.tool_sha256,
        environment=environment,
        artifacts_sha256=artifacts,
    )
    return write_evidence(
        destination,
        capture=captures[0].workload,
        provenance=provenance,
        producer="compatible_source_pytorch_bf16",
        authoritative_source_run=True,
        replay_captures=[capture.workload for capture in captures[1:]],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2-source", type=Path, required=True)
    parser.add_argument("--delivery-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = capture_authoritative_evidence(
        arguments.sam2_source,
        arguments.delivery_root,
        arguments.output,
    )
    print(
        f"wrote SAM2 authoritative-reference candidate to {arguments.output}: "
        f"capture_sha256={manifest['capture_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
