# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit a packaged SAM3 runtime against its TensorRT-only contract.

This tool validates release artifacts, not the developer build environment or
the customer's Python benchmark harness.  Those environments may use PyTorch
to produce a Golden result or synchronize benchmark timing.  A production SAM3
bundle, however, must contain only native TensorRT plans and
tokenizer/configuration assets.  Its inference dependency closure is limited to
Model Connect, TensorRT, CUDA/driver, and standard system libraries; it must not
contain another inference framework or a Python runtime.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

# Keep this pure-data release auditor directly runnable from a clean source
# checkout.  Import the tokenizer contract without initializing the SAM3
# package, whose public ``__init__`` intentionally loads the runtime plugin and
# its third-party build dependencies.
if __package__ in {None, ""}:
    _REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
    _SAM3_PYTHON_ROOT = _REPOSITORY_ROOT / "python" / "tensorrt_model_connect" / "families" / "sam3"
    sys.path.insert(0, str(_SAM3_PYTHON_ROOT))
    _TOKENIZER_CONTRACT_MODULE = "tokenizer_contract"
else:
    _TOKENIZER_CONTRACT_MODULE = "tensorrt_model_connect.families.sam3.tokenizer_contract"

_tokenizer_contract = importlib.import_module(_TOKENIZER_CONTRACT_MODULE)
Sam3TokenizerContractError = _tokenizer_contract.Sam3TokenizerContractError
validate_sam3_tokenizer_json = _tokenizer_contract.validate_sam3_tokenizer_json


_BUNDLE_MAGIC = b"BUNDLE\x01\x00"

SAM3_PLAN_SECTIONS = frozenset(
    {
        "engine_plan",
        "vision_engine_plan",
        "sam3_core_engine_plan",
        "sam3_tracker_init_engine_plan",
        "sam3_tracker_step_engine_plan",
        "sam3_tracker_step_batch2_engine_plan",
        "sam3_tracker_memory_engine_plan",
        "sam3_tracker_memory_batch2_engine_plan",
        "sam3_tracker_hard_memory_engine_plan",
        "sam3_tracker_hard_memory_batch2_engine_plan",
        "sam3_hard_mask_resize_engine_plan",
        "sam3_hard_mask_resize_batch2_engine_plan",
    }
)

SAM3_ASSET_SECTIONS = frozenset(
    {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "processor_config.json",
    }
)

SAM3_BUNDLE_SECTIONS = SAM3_PLAN_SECTIONS | SAM3_ASSET_SECTIONS

_SAM3_PLAN_LOAD_LABELS = {
    "engine_plan": "sam3 text_encoder",
    "vision_engine_plan": "sam3 vision_encoder",
    "sam3_core_engine_plan": "sam3 core_engine",
    "sam3_tracker_init_engine_plan": "sam3 tracker_init_engine",
    "sam3_tracker_step_engine_plan": "sam3 tracker_step_engine",
    "sam3_tracker_step_batch2_engine_plan": "sam3 tracker_step_batch2_engine",
    "sam3_tracker_memory_engine_plan": "sam3 tracker_memory_engine",
    "sam3_tracker_memory_batch2_engine_plan": "sam3 tracker_memory_batch2_engine",
    "sam3_tracker_hard_memory_engine_plan": "sam3 tracker_hard_memory_engine",
    "sam3_tracker_hard_memory_batch2_engine_plan": "sam3 tracker_hard_memory_batch2_engine",
    "sam3_hard_mask_resize_engine_plan": "sam3 hard_mask_resize_engine",
    "sam3_hard_mask_resize_batch2_engine_plan": "sam3 hard_mask_resize_batch2_engine",
}

_JSON_ASSET_SECTIONS = SAM3_ASSET_SECTIONS - {"merges.txt"}

_TENSORRT_PLAN_MAGIC = b"ftrt"

_FORBIDDEN_ASSET_MARKERS = (
    b"aotinductor",
    b"aoti_package",
    b"aotiruntime",
    b"libc10",
    b"libpython",
    b"libtorch",
    b"onnxruntime",
    b"tvmffikernel",
    b"tvmffi",
    b"tvm::ffi",
    b"tvm_ffi",
    b"tvm-ffi",
    b".pt2",
)

_FORBIDDEN_PLAN_MARKERS = _FORBIDDEN_ASSET_MARKERS

_FORBIDDEN_DEPENDENCY = re.compile(
    r"(?:"
    r"lib(?:torch|c10|aten|python)[^\s]*|"
    r"aoti|"
    r"tvm(?:::|[_-])?ffi|"
    r"onnx(?:runtime)?"
    r")",
    re.IGNORECASE,
)

_FORBIDDEN_SYMBOL = re.compile(
    r"(?:"
    r"(?:^|\W)(?:torch|c10|at)::|"
    r"lib(?:torch|c10|aten|python)|"
    r"aoti|"
    r"tvm(?:::|[_-])?ffi|"
    r"onnx(?:runtime)?"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

_FORBIDDEN_PYTHON_SYMBOL = re.compile(
    r"(?:^|\W)_?Py[A-Z_][A-Za-z0-9_]*",
    re.MULTILINE,
)

_FORBIDDEN_BINARY_MARKER = re.compile(
    r"(?:"
    r"lib(?:torch|c10|aten|python)|"
    r"aoti|"
    r"tvm(?:::|[_-])?ffi|"
    r"onnx(?:runtime)?|"
    r"\.pt2"
    r")",
    re.IGNORECASE,
)

_NEEDED_PATTERN = re.compile(r"\(NEEDED\).*?\[([^]]+)\]")
_SONAME_PATTERN = re.compile(r"\(SONAME\).*?\[([^]]+)\]")
_LOAD_TIMING_PATTERN = re.compile(
    r'^\[trtmc\.load_timing\] label="([^"]+)" '
    r"load_deserialize_ms=[0-9]+(?:\.[0-9]+)? plan_bytes=([0-9]+)$",
    re.MULTILINE,
)
_NATIVE_PROBE_PREFIX = "SAM3_NATIVE_LIVE_LOAD="
_TRT_NEEDED_PATTERN = re.compile(r"^libnvinfer(?:_plugin)?\.so(?:\.|$)")
_DSO_ROLE_PATTERNS = {
    "core": re.compile(r"^libtrtmc_core\.so(?:\.[A-Za-z0-9_.+-]+)*$"),
    "tensorrt_backend": re.compile(
        r"^libtrtmc_backend_trt(?:_[A-Za-z0-9_+-]+)?\.so(?:\.[A-Za-z0-9_.+-]+)*$"
    ),
    "sam3_model": re.compile(r"^libtrtmc_model_sam3\.so(?:\.[A-Za-z0-9_.+-]+)*$"),
}
_DSO_ROLE_SYMBOLS = {
    "core": ("trtmc::PipelineRegistry::instance",),
    "tensorrt_backend": ("trtmc_create_backend", "trtmc_backend_abi"),
    "sam3_model": ("trtmc_sam3_video_create", "trtmc_sam3_video_propagate"),
}
_ALLOWED_RUNTIME_LIBRARY = re.compile(
    r"^(?:"
    r"libtrtmc_[A-Za-z0-9_.+-]+\.so(?:\.[A-Za-z0-9_.+-]+)*|"
    r"libnvinfer(?:_[A-Za-z0-9_]+)?\.so(?:\.[A-Za-z0-9_.+-]+)*|"
    r"libcu[A-Za-z0-9_.+-]*\.so(?:\.[A-Za-z0-9_.+-]+)*|"
    r"libnv[A-Za-z0-9_.+-]*\.so(?:\.[A-Za-z0-9_.+-]+)*|"
    r"libnvidia-[A-Za-z0-9_.+-]+\.so(?:\.[A-Za-z0-9_.+-]+)*|"
    r"lib(?:c|m|mvec|stdc\+\+|gcc_s|pthread|dl|rt|util|resolv|atomic|gomp|omp|"
    r"numa|z|zstd|lzma|bz2|crypt|uuid|anl|nsl)\.so(?:\.[A-Za-z0-9_.+-]+)*|"
    r"linux-vdso\.so(?:\.[A-Za-z0-9_.+-]+)*|"
    r"ld-linux-[A-Za-z0-9_.+-]+\.so(?:\.[A-Za-z0-9_.+-]+)*"
    r")$"
)


class Sam3RuntimeContractError(RuntimeError):
    """Raised when a packaged SAM3 artifact violates the runtime contract."""


@dataclass(frozen=True)
class Sam3RuntimeAuditReport:
    """Content identities for a successfully audited SAM3 release artifact."""

    bundle_sha256: str
    cmake_cache_sha256: str
    bundle_sections: tuple[str, ...]
    plan_count: int
    live_load_succeeded: bool
    dso_sha256: dict[str, str]


ToolRunner = Callable[[Sequence[str], str | None], str]
ProbeRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class _DuplicateJsonKeyError(ValueError):
    """Raised by the strict JSON decoder before duplicate keys are overwritten."""


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _strict_json_loads(document: str | bytes, *, label: str) -> object:
    try:
        return json.loads(document, object_pairs_hook=_strict_json_object)
    except _DuplicateJsonKeyError as error:
        raise Sam3RuntimeContractError(
            f"{label} contains duplicate JSON object key {str(error)!r}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Sam3RuntimeContractError(f"{label} is not valid JSON") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bundle_header(path: Path) -> tuple[dict, int, int]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            if handle.read(len(_BUNDLE_MAGIC)) != _BUNDLE_MAGIC:
                raise Sam3RuntimeContractError(f"Not a valid .bundle artifact: {path}")
            encoded_size = handle.read(struct.calcsize("<Q"))
            if len(encoded_size) != struct.calcsize("<Q"):
                raise Sam3RuntimeContractError(f"Truncated .bundle header length: {path}")
            header_size = struct.unpack("<Q", encoded_size)[0]
            header_bytes = handle.read(header_size)
    except OSError as error:
        raise Sam3RuntimeContractError(f"Unable to read SAM3 bundle {path}: {error}") from error

    if len(header_bytes) != header_size:
        raise Sam3RuntimeContractError(f"Truncated .bundle JSON header: {path}")
    header = _strict_json_loads(header_bytes, label=f".bundle JSON header for {path}")
    if not isinstance(header, dict):
        raise Sam3RuntimeContractError("SAM3 bundle header must be a JSON object")
    payload_start = len(_BUNDLE_MAGIC) + struct.calcsize("<Q") + header_size
    return header, payload_start, file_size


def _validated_sections(header: dict, payload_start: int, file_size: int) -> dict[str, dict]:
    sections = header.get("sections")
    if not isinstance(sections, dict):
        raise Sam3RuntimeContractError("SAM3 bundle is missing its section table")
    actual = set(sections)
    if actual != SAM3_BUNDLE_SECTIONS:
        missing = sorted(SAM3_BUNDLE_SECTIONS - actual)
        unexpected = sorted(actual - SAM3_BUNDLE_SECTIONS)
        raise Sam3RuntimeContractError(
            f"SAM3 bundle section contract mismatch: missing={missing}, unexpected={unexpected}"
        )

    spans: list[tuple[int, int, str]] = []
    for name, metadata in sections.items():
        if not isinstance(metadata, dict):
            raise Sam3RuntimeContractError(f"Invalid metadata for bundle section {name!r}")
        try:
            offset = int(metadata["offset"])
            size = int(metadata["size"])
        except (KeyError, TypeError, ValueError) as error:
            raise Sam3RuntimeContractError(
                f"Invalid offset/size for bundle section {name!r}"
            ) from error
        if offset < 0 or size <= 0 or payload_start + offset + size > file_size:
            raise Sam3RuntimeContractError(
                f"Bundle section {name!r} has an invalid payload range: offset={offset}, size={size}"
            )
        spans.append((offset, offset + size, name))

    spans.sort()
    payload_size = file_size - payload_start
    if spans[0][0] != 0:
        raise Sam3RuntimeContractError(
            f"SAM3 bundle has {spans[0][0]} unaccounted payload bytes before its first section"
        )
    for previous, current in zip(spans, spans[1:], strict=False):
        if current[0] < previous[1]:
            raise Sam3RuntimeContractError(
                f"Bundle sections {previous[2]!r} and {current[2]!r} overlap"
            )
        if current[0] != previous[1]:
            raise Sam3RuntimeContractError(
                f"SAM3 bundle has {current[0] - previous[1]} unaccounted payload bytes "
                f"between sections {previous[2]!r} and {current[2]!r}"
            )
    if spans[-1][1] != payload_size:
        raise Sam3RuntimeContractError(
            f"SAM3 bundle has {payload_size - spans[-1][1]} unaccounted payload bytes "
            "after its final section"
        )
    return sections


def _scan_bundle_section_markers(
    path: Path,
    payload_start: int,
    sections: dict[str, dict],
) -> None:
    with path.open("rb") as handle:
        for name in sorted(SAM3_BUNDLE_SECTIONS):
            forbidden_markers = (
                _FORBIDDEN_PLAN_MARKERS if name in SAM3_PLAN_SECTIONS else _FORBIDDEN_ASSET_MARKERS
            )
            longest_marker = max(map(len, forbidden_markers))
            metadata = sections[name]
            remaining = int(metadata["size"])
            handle.seek(payload_start + int(metadata["offset"]))
            if name in SAM3_PLAN_SECTIONS:
                magic = handle.read(len(_TENSORRT_PLAN_MAGIC))
                if magic != _TENSORRT_PLAN_MAGIC:
                    raise Sam3RuntimeContractError(
                        f"Bundle section {name!r} is not a serialized TensorRT plan"
                    )
                remaining -= len(magic)
            carry = b""
            while remaining:
                chunk = handle.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise Sam3RuntimeContractError(f"Truncated TensorRT plan section {name!r}")
                remaining -= len(chunk)
                lowered = carry + chunk.lower()
                marker = next(
                    (candidate for candidate in forbidden_markers if candidate in lowered),
                    None,
                )
                if marker is not None:
                    raise Sam3RuntimeContractError(
                        f"SAM3 bundle section {name!r} contains forbidden marker "
                        f"{marker.decode('ascii', errors='replace')!r}"
                    )
                carry = lowered[-(longest_marker - 1) :] if longest_marker > 1 else b""


def _validate_asset_payloads(
    path: Path,
    payload_start: int,
    sections: dict[str, dict],
) -> None:
    json_documents: dict[str, dict] = {}
    raw_payloads: dict[str, bytes] = {}
    with path.open("rb") as handle:
        for name in sorted(SAM3_ASSET_SECTIONS):
            metadata = sections[name]
            size = int(metadata["size"])
            handle.seek(payload_start + int(metadata["offset"]))
            payload = handle.read(size)
            raw_payloads[name] = payload
            if len(payload) != size:
                raise Sam3RuntimeContractError(f"Truncated SAM3 asset section {name!r}")
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as error:
                raise Sam3RuntimeContractError(
                    f"SAM3 asset section {name!r} is not valid UTF-8"
                ) from error
            if "\x00" in text:
                raise Sam3RuntimeContractError(
                    f"SAM3 asset section {name!r} contains binary NUL bytes"
                )
            if name not in _JSON_ASSET_SECTIONS:
                continue
            document = _strict_json_loads(text, label=f"SAM3 asset section {name!r}")
            if not isinstance(document, dict):
                raise Sam3RuntimeContractError(
                    f"SAM3 JSON asset section {name!r} must contain an object"
                )
            json_documents[name] = document

    config = json_documents["config.json"]
    detector = config.get("detector_config")
    if not isinstance(detector, dict):
        detector = config
    text_config = detector.get("text_config")
    if not isinstance(text_config, dict):
        text_config = config.get("text_config")
    try:
        expected_vocab_size = int(text_config["vocab_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise Sam3RuntimeContractError(
            "SAM3 config.json is missing detector text_config.vocab_size"
        ) from error
    if expected_vocab_size <= 0:
        raise Sam3RuntimeContractError("SAM3 config.json text vocab_size must be positive")
    if config.get("sam3_video_tracking_supported") is not True:
        raise Sam3RuntimeContractError(
            "SAM3 config.json must set sam3_video_tracking_supported to JSON true so the "
            "production live-load smoke deserializes all tracker plans"
        )
    try:
        validate_sam3_tokenizer_json(
            raw_payloads["tokenizer.json"],
            expected_vocab_size=expected_vocab_size,
        )
    except Sam3TokenizerContractError as error:
        raise Sam3RuntimeContractError(
            f"SAM3 tokenizer.json violates the native BPE contract: {error}"
        ) from error


def audit_sam3_bundle(path: str | Path) -> tuple[str, ...]:
    """Require the exact production SAM3 native-plan bundle contract."""

    bundle = Path(path)
    header, payload_start, file_size = _read_bundle_header(bundle)
    if header.get("runtime_strategy") != "sam3_prompted_segmentation":
        raise Sam3RuntimeContractError(
            "SAM3 bundle must select runtime_strategy='sam3_prompted_segmentation'"
        )
    if str(header.get("model_type", "")).replace("-", "_") not in {"sam3", "sam3_video"}:
        raise Sam3RuntimeContractError(
            f"Unexpected SAM3 bundle model_type: {header.get('model_type')!r}"
        )
    sections = _validated_sections(header, payload_start, file_size)
    _scan_bundle_section_markers(bundle, payload_start, sections)
    _validate_asset_payloads(bundle, payload_start, sections)
    return tuple(sections)


def audit_sam3_build_cache(path: str | Path) -> Path:
    """Require the native libraries to be configured without optional bridges."""

    cache = Path(path)
    try:
        text = cache.read_text(encoding="utf-8")
    except OSError as error:
        raise Sam3RuntimeContractError(f"Unable to read CMake cache {cache}: {error}") from error
    except UnicodeDecodeError as error:
        raise Sam3RuntimeContractError(f"CMake cache is not valid UTF-8: {cache}") from error

    lines = text.splitlines()
    if not lines or lines[0] != "# This is the CMakeCache file.":
        raise Sam3RuntimeContractError(f"Not a generated CMakeCache.txt: {cache}")
    if not any(line.startswith("# It was generated by CMake: ") for line in lines[:12]):
        raise Sam3RuntimeContractError(f"CMake generator provenance is missing from {cache}")

    entries: dict[str, tuple[str, str]] = {}
    entry_pattern = re.compile(r"^([^#/:=\s][^:=]*):([^=\s]+)=(.*)$")
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith(("#", "//")):
            continue
        match = entry_pattern.match(line)
        if match is None:
            raise Sam3RuntimeContractError(f"Malformed CMake cache entry at {cache}:{line_number}")
        key, entry_type, value = match.groups()
        if key in entries:
            raise Sam3RuntimeContractError(f"Duplicate CMake cache entry {key!r} in {cache}")
        entries[key] = (entry_type, value)

    required_identity = {
        "CMAKE_PROJECT_NAME": ("STATIC", "tensorrt_model_connect"),
    }
    for key, expected in required_identity.items():
        if entries.get(key) != expected:
            raise Sam3RuntimeContractError(
                f"Unexpected {key} in packaged SAM3 CMake cache: {entries.get(key)!r}"
            )

    for key in ("CMAKE_CACHEFILE_DIR", "CMAKE_HOME_DIRECTORY"):
        entry = entries.get(key)
        if entry is None or entry[0] != "INTERNAL" or not Path(entry[1]).is_absolute():
            raise Sam3RuntimeContractError(
                f"{key}:INTERNAL must contain an absolute path in the packaged CMake cache"
            )
    try:
        declared_cache_dir = Path(entries["CMAKE_CACHEFILE_DIR"][1]).resolve(strict=True)
        actual_cache_dir = cache.resolve(strict=True).parent
    except OSError as error:
        raise Sam3RuntimeContractError(
            f"Unable to resolve packaged CMake cache provenance: {error}"
        ) from error
    if declared_cache_dir != actual_cache_dir:
        raise Sam3RuntimeContractError(
            "CMAKE_CACHEFILE_DIR does not match the directory containing the audited cache: "
            f"declared={declared_cache_dir}, actual={actual_cache_dir}"
        )
    generator = entries.get("CMAKE_GENERATOR")
    if generator is None or generator[0] != "INTERNAL" or not generator[1].strip():
        raise Sam3RuntimeContractError(
            "CMAKE_GENERATOR:INTERNAL is missing from the packaged CMake cache"
        )
    command = entries.get("CMAKE_COMMAND")
    if command is None or command[0] != "INTERNAL" or not Path(command[1]).is_absolute():
        raise Sam3RuntimeContractError(
            "CMAKE_COMMAND:INTERNAL must contain an absolute path in the packaged CMake cache"
        )
    for component in ("MAJOR", "MINOR", "PATCH"):
        key = f"CMAKE_CACHE_{component}_VERSION"
        entry = entries.get(key)
        if entry is None or entry[0] != "INTERNAL" or not entry[1].isdigit():
            raise Sam3RuntimeContractError(
                f"{key}:INTERNAL is missing from the packaged CMake cache"
            )

    for option in ("TRTMC_ENABLE_LIBTORCH_MULTINOMIAL", "TRTMC_ENABLE_TVM_FFI"):
        entry = entries.get(option)
        if entry != ("BOOL", "OFF"):
            raise Sam3RuntimeContractError(
                f"{option}:BOOL must be OFF for the packaged SAM3 runtime; found {entry!r}"
            )
    return actual_cache_dir


def _run_tool(command: Sequence[str], stdin: str | None = None) -> str:
    try:
        completed = subprocess.run(
            list(command),
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as error:
        raise Sam3RuntimeContractError(
            f"Unable to run artifact audit tool {command[0]!r}: {error}"
        ) from error
    if completed.returncode != 0:
        raise Sam3RuntimeContractError(
            f"Artifact audit command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def _require_dso_roles(paths: Sequence[Path]) -> dict[str, Path]:
    roles: dict[str, Path] = {}
    for path in paths:
        matches = [
            role for role, pattern in _DSO_ROLE_PATTERNS.items() if pattern.fullmatch(path.name)
        ]
        if len(matches) != 1:
            raise Sam3RuntimeContractError(
                f"Unexpected direct SAM3 runtime DSO role for {path.name!r}"
            )
        role = matches[0]
        if role in roles:
            raise Sam3RuntimeContractError(
                f"SAM3 runtime audit received multiple DSOs for required role {role!r}"
            )
        roles[role] = path

    missing = sorted(set(_DSO_ROLE_PATTERNS) - set(roles))
    if missing:
        raise Sam3RuntimeContractError(
            f"SAM3 runtime audit is missing required DSO roles: {missing}"
        )
    return roles


def _require_dso_soname(role: str, dynamic: str, path: Path) -> None:
    sonames = _SONAME_PATTERN.findall(dynamic)
    if len(sonames) != 1 or not _DSO_ROLE_PATTERNS[role].fullmatch(sonames[0]):
        raise Sam3RuntimeContractError(
            f"Runtime DSO {path} does not have the expected {role!r} SONAME; found {sonames}"
        )


def _require_dso_role_symbols(role: str, demangled: str, path: Path) -> None:
    missing = [symbol for symbol in _DSO_ROLE_SYMBOLS[role] if symbol not in demangled]
    if missing:
        raise Sam3RuntimeContractError(
            f"Runtime DSO {path} is missing required {role!r} role symbols: {missing}"
        )


def _reject_match(pattern: re.Pattern[str], text: str, label: str) -> None:
    match = pattern.search(text)
    if match is not None:
        line = text.count("\n", 0, match.start()) + 1
        raise Sam3RuntimeContractError(
            f"Forbidden runtime dependency marker {match.group(0)!r} in {label} at line {line}"
        )


def _require_allowed_libraries(names: Sequence[str], label: str) -> None:
    unexpected = sorted({name for name in names if not _ALLOWED_RUNTIME_LIBRARY.match(name)})
    if unexpected:
        raise Sam3RuntimeContractError(f"Non-TensorRT runtime libraries in {label}: {unexpected}")


def _ldd_entries(output: str) -> tuple[tuple[str, Path | None], ...]:
    entries: list[tuple[str, Path | None]] = []
    for line in output.splitlines():
        normalized = re.sub(r"\s+\(0x[0-9a-fA-F]+\)\s*$", "", line.strip())
        if not normalized:
            continue
        if "=>" in normalized:
            raw_name, raw_target = (part.strip() for part in normalized.split("=>", 1))
            name = Path(raw_name).name
            resolved = None if raw_target == "not found" else Path(raw_target)
        elif normalized.startswith("/"):
            resolved = Path(normalized)
            name = resolved.name
        else:
            name = Path(normalized.split(maxsplit=1)[0]).name
            resolved = None
        if ".so" in name:
            entries.append((name, resolved))
    return tuple(entries)


def _require_elf_dso(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
    except OSError as error:
        raise Sam3RuntimeContractError(f"Unable to read runtime DSO {path}: {error}") from error
    if magic != b"\x7fELF":
        raise Sam3RuntimeContractError(f"Runtime artifact is not an ELF DSO: {path}")


def _canonical_path(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise Sam3RuntimeContractError(
            f"Unable to resolve transitive Model Connect runtime DSO {path}: {error}"
        ) from error


def _path_identity(path: Path) -> tuple[Path, tuple[int, int]]:
    canonical = _canonical_path(path)
    try:
        stat = canonical.stat()
    except OSError as error:
        raise Sam3RuntimeContractError(
            f"Unable to stat runtime DSO {canonical}: {error}"
        ) from error
    return canonical, (stat.st_dev, stat.st_ino)


def _mapped_file_backed_dsos() -> dict[tuple[int, int], dict[str, str | int]]:
    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Sam3RuntimeContractError(
            f"Unable to inspect live runtime mappings: {error}"
        ) from error

    mapped: dict[tuple[int, int], dict[str, str | int]] = {}
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path_text = fields[5]
        if path_text.endswith(" (deleted)"):
            continue
        path_text = re.sub(
            r"\\([0-7]{3})",
            lambda match: chr(int(match.group(1), 8)),
            path_text,
        )
        path = Path(path_text)
        try:
            canonical = path.resolve(strict=True)
            stat = canonical.stat()
            device_major, device_minor = fields[3].split(":", 1)
            mapped_identity = (
                os.makedev(int(device_major, 16), int(device_minor, 16)),
                int(fields[4]),
            )
        except (OSError, ValueError):
            continue
        if ".so" not in path.name and ".so" not in canonical.name:
            continue
        if mapped_identity != (stat.st_dev, stat.st_ino):
            raise Sam3RuntimeContractError(
                f"Live-mapped DSO path no longer identifies its mapped inode: {canonical}"
            )
        mapped[mapped_identity] = {
            "path": str(canonical),
            "device": mapped_identity[0],
            "inode": mapped_identity[1],
        }
    return mapped


def _mapped_runtime_roles(
    mapped: dict[tuple[int, int], dict[str, str | int]],
) -> list[dict[str, str | int]]:
    role_entries: list[dict[str, str | int]] = []
    for entry in mapped.values():
        path = Path(str(entry["path"]))
        role_matches = [
            role for role, pattern in _DSO_ROLE_PATTERNS.items() if pattern.fullmatch(path.name)
        ]
        if len(role_matches) == 1:
            role_entries.append({"role": role_matches[0], **entry})
    return role_entries


def _native_live_load_probe(bundle: Path, dso_roles: dict[str, Path]) -> dict[str, object]:
    """Load the bundle through the exact customer-facing SAM3 C ABI."""

    bundle = bundle.resolve(strict=True)
    canonical_roles = {role: path.resolve(strict=True) for role, path in dso_roles.items()}
    model_dso = canonical_roles["sam3_model"]
    backend_dso = canonical_roles["tensorrt_backend"]
    baseline_dsos = _mapped_file_backed_dsos()
    try:
        library = ctypes.CDLL(str(model_dso))
    except OSError as error:
        raise Sam3RuntimeContractError(
            f"Unable to load the requested SAM3 model DSO {model_dso}: {error}"
        ) from error

    try:
        library.trtmc_sam3_video_last_error.argtypes = []
        library.trtmc_sam3_video_last_error.restype = ctypes.c_char_p
        library.trtmc_sam3_video_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        library.trtmc_sam3_video_create.restype = ctypes.c_void_p
        library.trtmc_sam3_video_destroy.argtypes = [ctypes.c_void_p]
        library.trtmc_sam3_video_destroy.restype = None
    except AttributeError as error:
        raise Sam3RuntimeContractError(
            f"Requested SAM3 model DSO is missing the production video C ABI: {model_dso}"
        ) from error

    handle: int | None = None
    with tempfile.TemporaryDirectory(prefix="sam3-runtime-audit-") as temporary_directory:
        probe_bundle = Path(temporary_directory) / "probe.bundle"
        probe_bundle.symlink_to(bundle)
        try:
            handle = library.trtmc_sam3_video_create(
                os.fsencode(probe_bundle),
                os.fsencode(model_dso.parent),
                os.fsencode(backend_dso.parent),
            )
            if not handle:
                encoded_error = library.trtmc_sam3_video_last_error()
                detail = (
                    encoded_error.decode("utf-8", errors="replace")
                    if encoded_error
                    else "unknown native runtime error"
                )
                raise Sam3RuntimeContractError(
                    f"SAM3 production C ABI failed to live-load the bundle: {detail}"
                )
            mapped_after_load = _mapped_file_backed_dsos()
            mapped_dsos = _mapped_runtime_roles(mapped_after_load)
            new_mapped_dsos = [
                entry
                for identity, entry in mapped_after_load.items()
                if identity not in baseline_dsos
            ]
        finally:
            if handle:
                library.trtmc_sam3_video_destroy(handle)

    return {"mapped_dsos": mapped_dsos, "new_mapped_dsos": new_mapped_dsos}


def _native_probe_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--core-dso", required=True)
    parser.add_argument("--backend-dso", required=True)
    parser.add_argument("--model-dso", required=True)
    return parser


def _native_probe_main(argv: Sequence[str]) -> int:
    args = _native_probe_parser().parse_args(argv)
    try:
        payload = _native_live_load_probe(
            Path(args.bundle),
            {
                "core": Path(args.core_dso),
                "tensorrt_backend": Path(args.backend_dso),
                "sam3_model": Path(args.model_dso),
            },
        )
    except (OSError, Sam3RuntimeContractError) as error:
        print(f"SAM3_NATIVE_LIVE_LOAD_FAILED: {error}", file=sys.stderr)
        return 1
    print(f"{_NATIVE_PROBE_PREFIX}{json.dumps(payload, sort_keys=True)}")
    return 0


def _run_probe_subprocess(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as error:
        raise Sam3RuntimeContractError(
            f"Unable to launch SAM3 native live-load probe: {error}"
        ) from error


def _require_live_mapped_dso_identities(
    payload: dict,
    requested_roles: dict[str, Path],
) -> None:
    mapped_dsos = payload.get("mapped_dsos")
    if not isinstance(mapped_dsos, list):
        raise Sam3RuntimeContractError("SAM3 native live-load probe omitted mapped_dsos")

    observed: dict[str, set[tuple[int, int]]] = {role: set() for role in _DSO_ROLE_PATTERNS}
    for entry in mapped_dsos:
        if not isinstance(entry, dict):
            raise Sam3RuntimeContractError("SAM3 native live-load probe returned invalid DSO data")
        role = entry.get("role")
        path_text = entry.get("path")
        device = entry.get("device")
        inode = entry.get("inode")
        if (
            role not in observed
            or not isinstance(path_text, str)
            or not isinstance(device, int)
            or isinstance(device, bool)
            or not isinstance(inode, int)
            or isinstance(inode, bool)
        ):
            raise Sam3RuntimeContractError("SAM3 native live-load probe returned invalid DSO data")
        canonical, actual_identity = _path_identity(Path(path_text))
        reported_identity = (device, inode)
        if actual_identity != reported_identity:
            raise Sam3RuntimeContractError(
                f"Live-mapped DSO identity changed during the audit: {canonical}"
            )
        observed[role].add(actual_identity)

    for role, requested_path in requested_roles.items():
        canonical, expected_identity = _path_identity(requested_path)
        if observed[role] != {expected_identity}:
            raise Sam3RuntimeContractError(
                f"Live SAM3 runtime did not map exactly the audited {role!r} DSO {canonical}; "
                f"observed identities={sorted(observed[role])}"
            )


def _require_live_mapping_delta(
    payload: dict,
    requested_roles: dict[str, Path],
    audited_dso_paths: Sequence[Path],
) -> None:
    new_mapped_dsos = payload.get("new_mapped_dsos")
    if not isinstance(new_mapped_dsos, list):
        raise Sam3RuntimeContractError("SAM3 native live-load probe omitted new_mapped_dsos")

    audited_identities = {_path_identity(path)[1] for path in audited_dso_paths}
    requested_identities = {_path_identity(path)[1] for path in requested_roles.values()}
    observed_identities: set[tuple[int, int]] = set()
    for entry in new_mapped_dsos:
        if not isinstance(entry, dict):
            raise Sam3RuntimeContractError(
                "SAM3 native live-load probe returned invalid mapping-delta data"
            )
        path_text = entry.get("path")
        device = entry.get("device")
        inode = entry.get("inode")
        if (
            not isinstance(path_text, str)
            or not isinstance(device, int)
            or isinstance(device, bool)
            or not isinstance(inode, int)
            or isinstance(inode, bool)
        ):
            raise Sam3RuntimeContractError(
                "SAM3 native live-load probe returned invalid mapping-delta data"
            )
        canonical, actual_identity = _path_identity(Path(path_text))
        reported_identity = (device, inode)
        if actual_identity != reported_identity:
            raise Sam3RuntimeContractError(
                f"Live-mapped DSO identity changed during the audit: {canonical}"
            )
        observed_identities.add(actual_identity)

        library_name = canonical.name
        _reject_match(
            _FORBIDDEN_DEPENDENCY,
            library_name,
            f"native live-load mapping {canonical}",
        )
        _require_allowed_libraries((library_name,), f"native live-load mapping {canonical}")
        if library_name.startswith("libtrtmc_") and actual_identity not in audited_identities:
            raise Sam3RuntimeContractError(
                f"Native live-load mapped an unaudited Model Connect DSO: {canonical}"
            )

    if not requested_identities.issubset(observed_identities):
        raise Sam3RuntimeContractError(
            "SAM3 native live-load mapping delta omitted one or more requested runtime DSOs"
        )


def _audit_native_live_load(
    bundle: Path,
    requested_roles: dict[str, Path],
    plan_sizes: dict[str, int],
    audited_dso_paths: Sequence[Path] | None = None,
    *,
    probe_runner: ProbeRunner,
) -> None:
    command = (
        sys.executable,
        "-I",
        "-S",
        str(Path(__file__).resolve()),
        "--_native-live-load-probe",
        "--bundle",
        str(bundle),
        "--core-dso",
        str(requested_roles["core"]),
        "--backend-dso",
        str(requested_roles["tensorrt_backend"]),
        "--model-dso",
        str(requested_roles["sam3_model"]),
    )
    completed = probe_runner(command)
    if completed.returncode != 0:
        raise Sam3RuntimeContractError(
            "SAM3 production C ABI live-load probe failed "
            f"({completed.returncode}):\n{completed.stderr}"
        )

    probe_lines = [
        line[len(_NATIVE_PROBE_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(_NATIVE_PROBE_PREFIX)
    ]
    if len(probe_lines) != 1:
        raise Sam3RuntimeContractError(
            "SAM3 native live-load probe did not return exactly one identity report"
        )
    payload = _strict_json_loads(probe_lines[0], label="SAM3 native live-load probe report")
    if not isinstance(payload, dict):
        raise Sam3RuntimeContractError("SAM3 native live-load probe report must be a JSON object")
    _require_live_mapped_dso_identities(payload, requested_roles)
    _require_live_mapping_delta(
        payload,
        requested_roles,
        tuple(requested_roles.values()) if audited_dso_paths is None else audited_dso_paths,
    )

    observed_timings: dict[str, list[int]] = {}
    for match in _LOAD_TIMING_PATTERN.finditer(completed.stderr):
        observed_timings.setdefault(match.group(1), []).append(int(match.group(2)))
    if "failed to load optional engine: sam3 tracker_init_engine" in completed.stderr:
        raise Sam3RuntimeContractError(
            "SAM3 native live-load fell back without a usable tracker-init plan"
        )
    expected_labels = set(_SAM3_PLAN_LOAD_LABELS.values())
    unexpected_labels = sorted(set(observed_timings) - expected_labels)
    if unexpected_labels:
        raise Sam3RuntimeContractError(
            f"SAM3 native live-load emitted unexpected engine labels: {unexpected_labels}"
        )
    for section, label in _SAM3_PLAN_LOAD_LABELS.items():
        sizes = observed_timings.get(label, [])
        if not sizes:
            raise Sam3RuntimeContractError(
                f"SAM3 native live-load did not prove plan section {section!r}; "
                f"label {label!r} appeared {len(sizes)} times"
            )
        expected_size = plan_sizes[section]
        if any(size != expected_size for size in sizes):
            raise Sam3RuntimeContractError(
                f"SAM3 native live-load size mismatch for plan section {section!r}: "
                f"expected {expected_size}, observed {sizes}"
            )


def audit_sam3_runtime_dependencies(
    dso_paths: Sequence[str | Path],
    *,
    tool_runner: ToolRunner = _run_tool,
) -> dict[str, str]:
    """Audit direct, transitive, symbol, and embedded-string dependencies."""

    requested_paths = tuple(Path(path) for path in dso_paths)
    requested_roles = _require_dso_roles(requested_paths)

    pending: list[Path] = []
    audited: list[Path] = []
    discovered: set[Path] = set()
    requested_role_by_canonical: dict[Path, str] = {}
    requested_file_identities: dict[tuple[int, int], tuple[str, Path]] = {}
    for role, path in requested_roles.items():
        _require_elf_dso(path)
        canonical = _canonical_path(path)
        try:
            stat = canonical.stat()
        except OSError as error:
            raise Sam3RuntimeContractError(
                f"Unable to stat runtime DSO {canonical}: {error}"
            ) from error
        identity = (stat.st_dev, stat.st_ino)
        if canonical in requested_role_by_canonical or identity in requested_file_identities:
            previous_role, previous_path = requested_file_identities.get(
                identity,
                (requested_role_by_canonical.get(canonical, "unknown"), canonical),
            )
            raise Sam3RuntimeContractError(
                f"Required SAM3 DSO roles {previous_role!r} and {role!r} resolve to the same "
                f"ELF file: {previous_path} and {path}"
            )
        requested_role_by_canonical[canonical] = role
        requested_file_identities[identity] = (role, path)
        discovered.add(canonical)
        pending.append(canonical)

    backend_has_tensorrt = False
    while pending:
        path = pending.pop(0)
        audited.append(path)
        dynamic = tool_runner(("readelf", "-d", str(path)), None)
        _reject_match(_FORBIDDEN_DEPENDENCY, dynamic, f"ELF dynamic dependencies for {path}")
        if "not found" in dynamic.lower():
            raise Sam3RuntimeContractError(
                f"ELF dynamic dependency audit reported an unresolved library for {path}"
            )
        needed = tuple(
            match.group(1)
            for line in dynamic.splitlines()
            if (match := _NEEDED_PATTERN.search(line)) is not None
        )
        _require_allowed_libraries(needed, f"ELF dynamic dependencies for {path}")
        role = requested_role_by_canonical.get(path)
        if role is not None:
            _require_dso_soname(role, dynamic, path)
        if role == "tensorrt_backend":
            backend_has_tensorrt = backend_has_tensorrt or any(
                _TRT_NEEDED_PATTERN.match(name) for name in needed
            )

        closure = tool_runner(("ldd", str(path)), None)
        if "not found" in closure.lower():
            raise Sam3RuntimeContractError(
                f"Runtime dependency closure contains an unresolved library for {path}"
            )
        _reject_match(
            _FORBIDDEN_DEPENDENCY,
            closure,
            f"transitive runtime dependency closure for {path}",
        )
        closure_entries = _ldd_entries(closure)
        _require_allowed_libraries(
            tuple(name for name, _ in closure_entries),
            f"transitive runtime dependency closure for {path}",
        )
        for name, resolved_path in closure_entries:
            if not name.startswith("libtrtmc_"):
                continue
            if resolved_path is None:
                raise Sam3RuntimeContractError(
                    f"Unable to resolve transitive Model Connect runtime DSO {name!r} "
                    f"from dependency closure for {path}"
                )
            _require_elf_dso(resolved_path)
            canonical = _canonical_path(resolved_path)
            if canonical not in discovered:
                discovered.add(canonical)
                pending.append(canonical)

        symbols = tool_runner(("nm", "-D", str(path)), None)
        demangled = tool_runner(("c++filt",), symbols)
        _reject_match(_FORBIDDEN_SYMBOL, demangled, f"dynamic symbol table for {path}")
        _reject_match(
            _FORBIDDEN_PYTHON_SYMBOL,
            demangled,
            f"dynamic symbol table for {path}",
        )
        if role is not None:
            _require_dso_role_symbols(role, demangled, path)

        embedded = tool_runner(("strings", "-a", str(path)), None)
        _reject_match(_FORBIDDEN_BINARY_MARKER, embedded, f"runtime DSO strings for {path}")

    if not backend_has_tensorrt:
        raise Sam3RuntimeContractError("TensorRT backend DSO does not depend on libnvinfer")
    return {str(path): _sha256(path) for path in audited}


def audit_sam3_runtime_artifacts(
    *,
    bundle_path: str | Path,
    dso_paths: Sequence[str | Path],
    cmake_cache: str | Path,
    tool_runner: ToolRunner = _run_tool,
    probe_runner: ProbeRunner = _run_probe_subprocess,
) -> Sam3RuntimeAuditReport:
    """Audit the complete packaged SAM3 bundle and native runtime handoff."""

    bundle = Path(bundle_path)
    cache = Path(cmake_cache)
    requested_roles = _require_dso_roles(tuple(Path(path) for path in dso_paths))
    build_dir = audit_sam3_build_cache(cache)
    for role, path in requested_roles.items():
        canonical, _ = _path_identity(path)
        if not canonical.is_relative_to(build_dir):
            raise Sam3RuntimeContractError(
                f"Audited {role!r} DSO is not from the CMake build tree {build_dir}: {canonical}"
            )
    sections = audit_sam3_bundle(bundle)
    header, payload_start, file_size = _read_bundle_header(bundle)
    section_table = _validated_sections(header, payload_start, file_size)
    plan_sizes = {name: int(section_table[name]["size"]) for name in SAM3_PLAN_SECTIONS}
    dso_sha256 = audit_sam3_runtime_dependencies(dso_paths, tool_runner=tool_runner)
    _audit_native_live_load(
        bundle,
        requested_roles,
        plan_sizes,
        tuple(Path(path) for path in dso_sha256),
        probe_runner=probe_runner,
    )
    return Sam3RuntimeAuditReport(
        bundle_sha256=_sha256(bundle),
        cmake_cache_sha256=_sha256(cache),
        bundle_sections=sections,
        plan_count=len(SAM3_PLAN_SECTIONS),
        live_load_succeeded=True,
        dso_sha256=dso_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a production SAM3 bundle and native TensorRT runtime",
    )
    parser.add_argument("--bundle", required=True, help="SAM3 .bundle artifact")
    parser.add_argument(
        "--dso",
        action="append",
        required=True,
        help=(
            "Runtime DSO to audit; pass libtrtmc_core, the TensorRT backend, "
            "and libtrtmc_model_sam3"
        ),
    )
    parser.add_argument(
        "--cmake-cache",
        required=True,
        help="CMakeCache.txt whose SAM3 runtime bridge settings must both be OFF",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_sam3_runtime_artifacts(
            bundle_path=args.bundle,
            dso_paths=args.dso,
            cmake_cache=args.cmake_cache,
        )
    except Sam3RuntimeContractError as error:
        print(f"SAM3_RUNTIME_CONTRACT_FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    print("SAM3_RUNTIME_CONTRACT_OK")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_native-live-load-probe":
        raise SystemExit(_native_probe_main(sys.argv[2:]))
    raise SystemExit(main())
