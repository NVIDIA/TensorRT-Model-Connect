# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed credential and network checks for the isolated model proof.

The evidence deliberately records only policy names and outcome codes. It never
serializes environment values, token contents, or host-specific credential paths.

Boundary: host/container credential policy and active container egress probes;
model selection, cache preparation, GPU allocation, and model tests stay elsewhere.
"""

from __future__ import annotations

import errno
import json
import os
import socket
from collections.abc import Mapping, Sequence
from pathlib import Path

from .process import CiError


HUGGING_FACE_CREDENTIAL_ENVIRONMENT = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_TOKEN_PATH",
)
HOST_SECURITY_EVIDENCE = "host-security-policy.json"
RUNTIME_SECURITY_EVIDENCE = "runtime-security.json"

_HOST_POLICY = "model-proof-host-no-hugging-face-credentials"
_RUNTIME_POLICY = "model-proof-runtime-no-credentials-no-network"
_DNS_TARGET = "huggingface.co"
_DIRECT_HTTPS_ADDRESS = "1.1.1.1"
_HTTPS_PORT = 443
_DEFAULT_RUNTIME_TOKEN_PATHS = (
    ("one-shot-secret", Path("/run/secrets/hf-token")),
    ("hf-home-token", Path("/work/hf-home/token")),
    ("default-home-token", Path("/tmp/.cache/huggingface/token")),
    ("legacy-home-token", Path("/tmp/.huggingface/token")),
)


def _write_evidence(path: Path, payload: object) -> None:
    """Atomically replace one run-owned evidence file without following it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def host_security_evidence(environment: Mapping[str, str]) -> dict[str, object]:
    """Describe only whether a forbidden credential variable is present."""

    present = [name for name in HUGGING_FACE_CREDENTIAL_ENVIRONMENT if name in environment]
    return {
        "schema_version": 1,
        "policy": _HOST_POLICY,
        "checked_environment_variables": list(HUGGING_FACE_CREDENTIAL_ENVIRONMENT),
        "credentials_absent": not present,
        "present_environment_variables": present,
    }


def enforce_host_security_policy(
    environment: Mapping[str, str], evidence_path: Path
) -> dict[str, object]:
    """Persist host evidence and reject even empty credential variables."""

    evidence = host_security_evidence(environment)
    _write_evidence(evidence_path, evidence)
    present = evidence["present_environment_variables"]
    if present:
        raise CiError(
            "model-proof host policy forbids Hugging Face credential variables: "
            + ", ".join(str(name) for name in present)
        )
    return evidence


def _load_valid_host_evidence(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload == host_security_evidence({})


def _error_evidence(error: OSError) -> dict[str, object]:
    return {
        "error_type": type(error).__name__,
        "error_code": error.errno,
    }


def enforce_runtime_security_policy(
    environment: Mapping[str, str],
    host_evidence_path: Path,
    runtime_evidence_path: Path,
    *,
    token_paths: Sequence[tuple[str, Path]] = _DEFAULT_RUNTIME_TOKEN_PATHS,
) -> dict[str, object]:
    """Actively prove the proof container has neither credentials nor egress.

    DNS and transport are independent checks. The transport probe uses a numeric
    address on the HTTPS port, so a resolver failure or stale DNS cache cannot
    make a network-enabled container look isolated. Only the deterministic Linux
    ``ENETUNREACH`` result is accepted for the direct transport probe; timeouts,
    connection refusals, and TLS failures do not count as isolation.
    """

    issues: list[str] = []
    host_evidence_valid = _load_valid_host_evidence(host_evidence_path)
    if not host_evidence_valid:
        issues.append("host security evidence is missing or invalid")

    present_environment = [
        name for name in HUGGING_FACE_CREDENTIAL_ENVIRONMENT if name in environment
    ]
    if present_environment:
        issues.append(
            "proof container exposes Hugging Face credential variables: "
            + ", ".join(present_environment)
        )

    visible_token_locations: list[str] = []
    for label, path in token_paths:
        try:
            visible = path.exists() or path.is_symlink()
        except OSError:
            visible = True
        if visible:
            visible_token_locations.append(label)
    if visible_token_locations:
        issues.append(
            "proof container exposes Hugging Face credential files: "
            + ", ".join(visible_token_locations)
        )

    try:
        socket.getaddrinfo(_DNS_TARGET, _HTTPS_PORT, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        dns_probe: dict[str, object] = {
            "target": _DNS_TARGET,
            "port": _HTTPS_PORT,
            "outcome": "blocked",
            **_error_evidence(error),
        }
    except OSError as error:
        dns_probe = {
            "target": _DNS_TARGET,
            "port": _HTTPS_PORT,
            "outcome": "unexpected-error",
            **_error_evidence(error),
        }
        issues.append("DNS probe did not fail with a resolver error")
    else:
        dns_probe = {
            "target": _DNS_TARGET,
            "port": _HTTPS_PORT,
            "outcome": "resolved",
        }
        issues.append("DNS resolution unexpectedly succeeded")

    direct_socket = None
    connect_attempted = False
    try:
        direct_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        direct_socket.settimeout(2.0)
        connect_attempted = True
        direct_socket.connect((_DIRECT_HTTPS_ADDRESS, _HTTPS_PORT))
    except OSError as error:
        blocked = connect_attempted and error.errno == errno.ENETUNREACH
        https_probe: dict[str, object] = {
            "numeric_address": _DIRECT_HTTPS_ADDRESS,
            "port": _HTTPS_PORT,
            "phase": "transport-before-tls" if connect_attempted else "socket-setup",
            "outcome": "blocked" if blocked else "unexpected-error",
            **_error_evidence(error),
        }
        if not blocked:
            issues.append("direct HTTPS transport did not fail with ENETUNREACH")
    else:
        https_probe = {
            "numeric_address": _DIRECT_HTTPS_ADDRESS,
            "port": _HTTPS_PORT,
            "phase": "transport-before-tls",
            "outcome": "connected",
        }
        issues.append("direct HTTPS transport unexpectedly connected")
    finally:
        if direct_socket is not None:
            direct_socket.close()

    evidence = {
        "schema_version": 1,
        "policy": _RUNTIME_POLICY,
        "passed": not issues,
        "host_policy": {
            "evidence": HOST_SECURITY_EVIDENCE,
            "valid": host_evidence_valid,
        },
        "credential_environment": {
            "checked_environment_variables": list(HUGGING_FACE_CREDENTIAL_ENVIRONMENT),
            "absent": not present_environment,
            "present_environment_variables": present_environment,
        },
        "credential_files": {
            "checked_locations": [label for label, _path in token_paths],
            "absent": not visible_token_locations,
            "present_locations": visible_token_locations,
        },
        "dns_probe": dns_probe,
        "direct_https_transport_probe": https_probe,
    }
    _write_evidence(runtime_evidence_path, evidence)
    if issues:
        raise CiError("model-proof runtime security policy failed: " + "; ".join(issues))
    return evidence
