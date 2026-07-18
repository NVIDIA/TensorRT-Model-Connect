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
import hashlib
import importlib
import json
import re
import struct
import subprocess
import sys
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


_BUNDLE_MAGIC = b"TRTFB\x00\x01\x00"

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
)

_FORBIDDEN_PLAN_MARKERS = _FORBIDDEN_ASSET_MARKERS + (b".pt2",)

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
_TRT_NEEDED_PATTERN = re.compile(r"^libnvinfer(?:_plugin)?\.so(?:\.|$)")
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
    bundle_sections: tuple[str, ...]
    plan_count: int
    dso_sha256: dict[str, str]


ToolRunner = Callable[[Sequence[str], str | None], str]


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
                raise Sam3RuntimeContractError(f"Not a valid .trtfb bundle: {path}")
            encoded_size = handle.read(struct.calcsize("<Q"))
            if len(encoded_size) != struct.calcsize("<Q"):
                raise Sam3RuntimeContractError(f"Truncated .trtfb header length: {path}")
            header_size = struct.unpack("<Q", encoded_size)[0]
            header_bytes = handle.read(header_size)
    except OSError as error:
        raise Sam3RuntimeContractError(f"Unable to read SAM3 bundle {path}: {error}") from error

    if len(header_bytes) != header_size:
        raise Sam3RuntimeContractError(f"Truncated .trtfb JSON header: {path}")
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Sam3RuntimeContractError(f"Invalid .trtfb JSON header: {path}") from error
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
            try:
                document = json.loads(text)
            except json.JSONDecodeError as error:
                raise Sam3RuntimeContractError(
                    f"SAM3 asset section {name!r} is not valid JSON"
                ) from error
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


def audit_sam3_build_cache(path: str | Path) -> None:
    """Require the native libraries to be configured without optional bridges."""

    cache = Path(path)
    try:
        lines = cache.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise Sam3RuntimeContractError(f"Unable to read CMake cache {cache}: {error}") from error
    values: dict[str, str] = {}
    for line in lines:
        if ":" not in line or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key, _ = key_and_type.split(":", 1)
        values[key] = value.strip().upper()
    for option in ("TRTMC_ENABLE_LIBTORCH_MULTINOMIAL", "TRTMC_ENABLE_TVM_FFI"):
        if values.get(option) != "OFF":
            raise Sam3RuntimeContractError(
                f"{option} must be OFF for the packaged SAM3 runtime; "
                f"found {values.get(option, 'missing')}"
            )


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


def _require_dso_roles(paths: Sequence[Path]) -> None:
    names = {path.name for path in paths}
    if "libtrtmc_core.so" not in names:
        raise Sam3RuntimeContractError("SAM3 runtime audit requires libtrtmc_core.so")
    if "libtrtmc_model_sam3.so" not in names:
        raise Sam3RuntimeContractError("SAM3 runtime audit requires libtrtmc_model_sam3.so")
    if not any(name.startswith("libtrtmc_backend_trt") and ".so" in name for name in names):
        raise Sam3RuntimeContractError("SAM3 runtime audit requires a TensorRT backend DSO")


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


def audit_sam3_runtime_dependencies(
    dso_paths: Sequence[str | Path],
    *,
    tool_runner: ToolRunner = _run_tool,
) -> dict[str, str]:
    """Audit direct, transitive, symbol, and embedded-string dependencies."""

    requested_paths = tuple(Path(path) for path in dso_paths)
    _require_dso_roles(requested_paths)

    pending: list[Path] = []
    audited: list[Path] = []
    discovered: set[Path] = set()
    for path in requested_paths:
        _require_elf_dso(path)
        canonical = _canonical_path(path)
        if canonical not in discovered:
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
        if path.name.startswith("libtrtmc_backend_trt"):
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
) -> Sam3RuntimeAuditReport:
    """Audit the complete packaged SAM3 bundle and native runtime handoff."""

    audit_sam3_build_cache(cmake_cache)
    sections = audit_sam3_bundle(bundle_path)
    dso_sha256 = audit_sam3_runtime_dependencies(dso_paths, tool_runner=tool_runner)
    return Sam3RuntimeAuditReport(
        bundle_sha256=_sha256(Path(bundle_path)),
        bundle_sections=sections,
        plan_count=len(SAM3_PLAN_SECTIONS),
        dso_sha256=dso_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a production SAM3 bundle and native TensorRT runtime",
    )
    parser.add_argument("--bundle", required=True, help="SAM3 .trtfb bundle")
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
    raise SystemExit(main())
