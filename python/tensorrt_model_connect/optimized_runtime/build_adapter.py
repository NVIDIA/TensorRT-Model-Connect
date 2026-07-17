# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON subprocess protocol for capsule-owned build adapters."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .manifest import (
    ImplementationManifest,
    ImplementationRequest,
    manifest_contract_sha256,
    normalized_manifest_contract,
)


PROTOCOL_SCHEMA_VERSION = 1
_PROBE_KEYS = {
    "schema_version",
    "supported",
    "profile_id",
    "reason",
}
_BUILD_KEYS = {"schema_version", "descriptor", "artifacts"}
_BUILD_BINDING_FIELD = "build_binding"
_BUILD_BINDING_KEYS = {
    "schema_version",
    "implementation_id",
    "manifest_sha256",
    "request_sha256",
    "profile_id",
}
_ACTIVE_CUDA_DEVICE_ENV = "TRTMC_INTERNAL_OPTIMIZED_RUNTIME_CUDA_DEVICE"
_ADAPTER_ENVIRONMENT_KEYS = frozenset({_ACTIVE_CUDA_DEVICE_ENV})


class BuildAdapterError(RuntimeError):
    """A capsule build adapter violated or failed the subprocess contract."""


@dataclass(frozen=True)
class ProbeResult:
    supported: bool
    profile_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class BuildArtifact:
    output_directory: Path
    descriptor_path: Path
    descriptor: Mapping[str, Any]
    artifacts_path: Path
    probe: ProbeResult


def _adapter_command(manifest: ImplementationManifest) -> list[str]:
    if manifest.build_entrypoint.suffix == ".py":
        return [sys.executable, str(manifest.build_entrypoint)]
    if not os.access(manifest.build_entrypoint, os.X_OK):
        raise BuildAdapterError(f"Build adapter is not executable: {manifest.build_entrypoint}")
    return [str(manifest.build_entrypoint)]


def _subprocess_environment(overrides: Mapping[str, str] | None) -> dict[str, str]:
    """Create a sanitized adapter environment for transient launch context."""

    environment = os.environ.copy()
    # Never trust or accidentally reuse a caller-provided internal value.
    for key in _ADAPTER_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    if not overrides:
        return environment
    unknown = sorted(set(overrides) - _ADAPTER_ENVIRONMENT_KEYS)
    if unknown:
        raise BuildAdapterError(
            "Unsupported internal adapter environment field(s): " + ", ".join(unknown)
        )
    ordinal = overrides.get(_ACTIVE_CUDA_DEVICE_ENV)
    if (
        not isinstance(ordinal, str)
        or len(ordinal) > 10
        or not ordinal.isascii()
        or not ordinal.isdecimal()
        or str(int(ordinal)) != ordinal
    ):
        raise BuildAdapterError("Internal active CUDA device ordinal must be canonical decimal")
    environment[_ACTIVE_CUDA_DEVICE_ENV] = ordinal
    return environment


def _request_payload(
    manifest: ImplementationManifest,
    request: ImplementationRequest,
    *,
    build_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = request.to_json()
    payload["implementation_id"] = manifest.implementation_id
    if build_binding is not None:
        payload[_BUILD_BINDING_FIELD] = dict(build_binding)
    return payload


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_contract(manifest: ImplementationManifest) -> dict[str, Any]:
    """Return the normalized manifest semantics selected by the generic host."""
    return normalized_manifest_contract(manifest)


def _expected_build_binding(
    manifest: ImplementationManifest,
    request: ImplementationRequest,
    probe: ProbeResult,
) -> dict[str, Any]:
    if probe.supported is not True:
        raise BuildAdapterError("build requires a supported probe result")
    for field, value in (("profile_id", probe.profile_id),):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise BuildAdapterError(
                f"supported probe result {field} must be a non-empty trimmed string"
            )
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "implementation_id": manifest.implementation_id,
        "manifest_sha256": manifest_contract_sha256(manifest),
        "request_sha256": _canonical_digest(_request_payload(manifest, request)),
        "profile_id": probe.profile_id,
    }


def validate_build_descriptor(
    manifest: ImplementationManifest,
    request: ImplementationRequest,
    probe: ProbeResult,
    descriptor: Mapping[str, Any],
) -> None:
    """Validate the strict common binding while leaving capsule metadata opaque."""

    schema_version = descriptor.get("schema_version")
    if type(schema_version) is not int or schema_version != PROTOCOL_SCHEMA_VERSION:
        raise BuildAdapterError(
            f"build descriptor schema_version must be {PROTOCOL_SCHEMA_VERSION}"
        )
    binding = descriptor.get(_BUILD_BINDING_FIELD)
    if not isinstance(binding, Mapping):
        raise BuildAdapterError("build descriptor build_binding must be a JSON object")
    unknown = sorted(set(binding) - _BUILD_BINDING_KEYS)
    missing = sorted(_BUILD_BINDING_KEYS - set(binding))
    if unknown:
        raise BuildAdapterError(
            "build descriptor build_binding contains unknown field(s): " + ", ".join(unknown)
        )
    if missing:
        raise BuildAdapterError(
            "build descriptor build_binding is missing field(s): " + ", ".join(missing)
        )

    binding_schema = binding.get("schema_version")
    if type(binding_schema) is not int or binding_schema != PROTOCOL_SCHEMA_VERSION:
        raise BuildAdapterError(
            f"build descriptor build_binding.schema_version must be {PROTOCOL_SCHEMA_VERSION}"
        )
    expected = _expected_build_binding(manifest, request, probe)
    for field in sorted(_BUILD_BINDING_KEYS - {"schema_version"}):
        actual = binding.get(field)
        required = expected[field]
        if not isinstance(actual, str) or type(actual) is not type(required) or actual != required:
            raise BuildAdapterError(
                f"build descriptor build_binding.{field} does not match the selected "
                "manifest, request, and probe"
            )


def _decode_response(stdout: str, *, operation: str, implementation_id: str) -> dict[str, Any]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise BuildAdapterError(
            f"{implementation_id} {operation} returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise BuildAdapterError(f"{implementation_id} {operation} response must be a JSON object")
    return value


def _terminate_adapter_process_group(process: subprocess.Popen[str]) -> None:
    """Stop the adapter and every inherited exporter or builder child."""

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:  # pragma: no cover - optimized runtimes currently target Linux.
        process.terminate()
    try:
        process.communicate(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:  # pragma: no cover - optimized runtimes currently target Linux.
        process.kill()
    process.communicate()


def _run_adapter(
    manifest: ImplementationManifest,
    request: ImplementationRequest,
    operation: str,
    *,
    output_directory: Path | None = None,
    build_binding: Mapping[str, Any] | None = None,
    timeout_seconds: int | None = None,
    _adapter_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not manifest.matches(request):
        raise BuildAdapterError(
            f"Request does not match implementation {manifest.implementation_id}"
        )
    timeout = timeout_seconds or manifest.build_timeout_seconds
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout_seconds must be a positive integer")

    with tempfile.TemporaryDirectory(prefix="trtmc-optimized-request-") as temp_dir:
        request_path = Path(temp_dir) / "request.json"
        request_path.write_text(
            json.dumps(
                _request_payload(manifest, request, build_binding=build_binding),
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        command = _adapter_command(manifest)
        command.extend([operation, "--request", str(request_path)])
        if output_directory is not None:
            command.extend(["--output", str(output_directory)])
        try:
            process = subprocess.Popen(
                command,
                cwd=manifest.capsule_root,
                env=_subprocess_environment(_adapter_environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=os.name == "posix",
                creationflags=(
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name != "posix" else 0
                ),
            )
        except OSError as exc:
            raise BuildAdapterError(
                f"Could not launch {manifest.implementation_id} {operation}: {exc}"
            ) from exc
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate_adapter_process_group(process)
            raise BuildAdapterError(
                f"{manifest.implementation_id} {operation} timed out after {timeout}s"
            ) from exc
        except BaseException:
            _terminate_adapter_process_group(process)
            raise
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        detail = f": {stderr[:4000]}" if stderr else ""
        raise BuildAdapterError(
            f"{manifest.implementation_id} {operation} failed with exit code "
            f"{result.returncode}{detail}"
        )
    return _decode_response(
        result.stdout,
        operation=operation,
        implementation_id=manifest.implementation_id,
    )


def _validate_response_keys(
    response: Mapping[str, Any], expected: set[str], *, operation: str
) -> None:
    unknown = sorted(set(response) - expected)
    if unknown:
        raise BuildAdapterError(
            f"{operation} response contains unknown field(s): {', '.join(unknown)}"
        )
    schema_version = response.get("schema_version")
    if type(schema_version) is not int or schema_version != PROTOCOL_SCHEMA_VERSION:
        raise BuildAdapterError(
            f"{operation} response schema_version must be {PROTOCOL_SCHEMA_VERSION}"
        )


def _response_string(
    response: Mapping[str, Any], key: str, *, operation: str, required: bool
) -> str:
    value = response.get(key, "")
    if not isinstance(value, str) or (required and not value.strip()):
        requirement = "a non-empty string" if required else "a string"
        raise BuildAdapterError(f"{operation} response {key} must be {requirement}")
    if value != value.strip():
        raise BuildAdapterError(
            f"{operation} response {key} must not contain surrounding whitespace"
        )
    return value


def run_probe(
    manifest: ImplementationManifest,
    request: ImplementationRequest,
    *,
    timeout_seconds: int | None = None,
    _adapter_environment: Mapping[str, str] | None = None,
) -> ProbeResult:
    """Invoke a capsule's side-effect-free ``probe`` operation."""
    response = _run_adapter(
        manifest,
        request,
        "probe",
        timeout_seconds=timeout_seconds,
        _adapter_environment=_adapter_environment,
    )
    _validate_response_keys(response, _PROBE_KEYS, operation="probe")
    supported = response.get("supported")
    if type(supported) is not bool:
        raise BuildAdapterError("probe response supported must be a boolean")
    profile_id = _response_string(response, "profile_id", operation="probe", required=supported)
    reason = _response_string(response, "reason", operation="probe", required=not supported)
    if supported:
        if reason:
            raise BuildAdapterError("supported probe response must not include a non-empty reason")
        return ProbeResult(
            supported=True,
            profile_id=profile_id,
        )

    if profile_id:
        raise BuildAdapterError("unsupported probe response must not include a profile ID")
    return ProbeResult(supported=False, reason=reason)


def _prepare_output_directory(output_directory: str | Path) -> Path:
    path = Path(output_directory)
    if path.is_symlink():
        raise BuildAdapterError(f"Build output directory must not be a symlink: {path}")
    if path.exists():
        if not path.is_dir():
            raise BuildAdapterError(f"Build output path is not a directory: {path}")
        if any(path.iterdir()):
            raise BuildAdapterError(f"Build output directory must be empty: {path}")
    else:
        path.mkdir(parents=True)
    return path.resolve(strict=True)


def _contained_output_path(output: Path, relative_name: Any, *, field: str, kind: str) -> Path:
    if not isinstance(relative_name, str) or not relative_name:
        raise BuildAdapterError(f"build response {field} must be a non-empty string")
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise BuildAdapterError(f"build response {field} must be an output-relative path")
    candidate = output / relative
    if candidate.is_symlink():
        raise BuildAdapterError(f"build response {field} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(output)
    except (FileNotFoundError, ValueError) as exc:
        raise BuildAdapterError(
            f"build response {field} does not resolve inside the output directory"
        ) from exc
    if kind == "file" and not resolved.is_file():
        raise BuildAdapterError(f"build response {field} must resolve to a file")
    if kind == "directory" and not resolved.is_dir():
        raise BuildAdapterError(f"build response {field} must resolve to a directory")
    return resolved


def run_build(
    manifest: ImplementationManifest,
    request: ImplementationRequest,
    output_directory: str | Path,
    *,
    probe: ProbeResult,
    timeout_seconds: int | None = None,
    _adapter_environment: Mapping[str, str] | None = None,
) -> BuildArtifact:
    """Invoke ``build`` and bind its output to the selected probe and request."""
    build_binding = _expected_build_binding(manifest, request, probe)
    output = _prepare_output_directory(output_directory)
    response = _run_adapter(
        manifest,
        request,
        "build",
        output_directory=output,
        build_binding=build_binding,
        timeout_seconds=timeout_seconds,
        _adapter_environment=_adapter_environment,
    )
    _validate_response_keys(response, _BUILD_KEYS, operation="build")
    descriptor_path = _contained_output_path(
        output, response.get("descriptor"), field="descriptor", kind="file"
    )
    artifacts_path = _contained_output_path(
        output, response.get("artifacts"), field="artifacts", kind="directory"
    )
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildAdapterError(f"Invalid build descriptor: {descriptor_path}: {exc}") from exc
    if not isinstance(descriptor, dict):
        raise BuildAdapterError("Build descriptor must contain a JSON object")
    validate_build_descriptor(manifest, request, probe, descriptor)
    return BuildArtifact(
        output_directory=output,
        descriptor_path=descriptor_path,
        descriptor=MappingProxyType(dict(descriptor)),
        artifacts_path=artifacts_path,
        probe=probe,
    )
