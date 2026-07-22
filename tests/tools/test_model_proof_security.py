# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial contracts for model-proof credential and network isolation."""

from __future__ import annotations

import errno
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci import model_proof_security as security
from tools.ci.context import CiContext
from tools.ci.model_proof import ModelProofRequest, ModelProofRunner
from tools.ci.process import CiError


REPO_ROOT = Path(__file__).resolve().parents[2]
PROOF_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "model-proof.yml"
PROOF_INNER = REPO_ROOT / "tools" / "ci" / "model_proof_inner.py"


class _FakeSocket:
    def __init__(self, connect_error: OSError | None):
        self.connect_error = connect_error
        self.timeout: float | None = None
        self.target: tuple[str, int] | None = None
        self.closed = False

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def connect(self, target: tuple[str, int]) -> None:
        self.target = target
        if self.connect_error is not None:
            raise self.connect_error

    def close(self) -> None:
        self.closed = True


def _block_dns(*_args: object, **_kwargs: object) -> list[object]:
    raise socket.gaierror(socket.EAI_AGAIN, "blocked by test network namespace")


def _runtime_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    environment: dict[str, str] | None = None,
    connect_error: OSError | None = OSError(errno.ENETUNREACH, "isolated"),
    dns_resolver: object = _block_dns,
    token_paths: tuple[tuple[str, Path], ...] | None = None,
) -> tuple[dict[str, object], _FakeSocket]:
    host_evidence = tmp_path / security.HOST_SECURITY_EVIDENCE
    runtime_evidence = tmp_path / security.RUNTIME_SECURITY_EVIDENCE
    security.enforce_host_security_policy({}, host_evidence)
    fake_socket = _FakeSocket(connect_error)
    monkeypatch.setattr(security.socket, "getaddrinfo", dns_resolver)
    monkeypatch.setattr(security.socket, "socket", lambda *_args: fake_socket)
    paths = token_paths or (("test-token", tmp_path / "missing-token"),)
    evidence = security.enforce_runtime_security_policy(
        environment or {},
        host_evidence,
        runtime_evidence,
        token_paths=paths,
    )
    assert json.loads(runtime_evidence.read_text(encoding="utf-8")) == evidence
    return evidence, fake_socket


@pytest.mark.parametrize(
    "name,value",
    [
        ("HF_TOKEN", "hf_do_not_serialize_primary"),
        ("HUGGING_FACE_HUB_TOKEN", "hf_do_not_serialize_legacy"),
        ("HF_TOKEN_PATH", "/tmp/do-not-serialize-token-path"),
        ("HF_TOKEN", ""),
    ],
)
def test_host_policy_rejects_credential_presence_before_any_docker_command(
    tmp_path: Path, name: str, value: str
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_marker = tmp_path / "docker-was-called"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'touch "$DOCKER_MARKER"\n'
        "exit 99\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    output = tmp_path / "proof"
    environment = os.environ.copy()
    for credential_name in security.HUGGING_FACE_CREDENTIAL_ENVIRONMENT:
        environment.pop(credential_name, None)
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "DOCKER_MARKER": str(docker_marker),
            name: value,
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.ci",
            "model-proof",
            "--model",
            "fixture-model",
            "--revision",
            "HEAD",
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert name in result.stderr
    if value:
        assert value not in result.stdout
        assert value not in result.stderr
    assert not docker_marker.exists()
    evidence_path = output / "artifacts" / security.HOST_SECURITY_EVIDENCE
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["credentials_absent"] is False
    assert evidence["present_environment_variables"] == [name]
    combined_artifacts = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (output / "artifacts").rglob("*")
        if path.is_file()
    )
    if value:
        assert value not in combined_artifacts


@pytest.mark.parametrize("name", security.HUGGING_FACE_CREDENTIAL_ENVIRONMENT)
def test_rejected_host_credential_is_scrubbed_before_failure_reporting_subprocess(
    tmp_path: Path, name: str
) -> None:
    repository = tmp_path / "repo"
    fallback = repository / ".github" / "scripts" / "write-model-proof-fallback-report.py"
    fallback.parent.mkdir(parents=True)
    fallback.write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        'root = Path(sys.argv[sys.argv.index("--artifacts-dir") + 1])\n'
        f'(root / "fallback-environment.txt").write_text('
        f'"present" if "{name}" in os.environ else "absent", encoding="utf-8")\n',
        encoding="utf-8",
    )
    environment = os.environ.copy()
    for credential_name in security.HUGGING_FACE_CREDENTIAL_ENVIRONMENT:
        environment.pop(credential_name, None)
    environment[name] = "hf_failure_report_must_not_inherit_this"
    output = tmp_path / "proof"
    runner = ModelProofRunner(
        CiContext(repository=repository, env=environment),
        ModelProofRequest(model="fixture-model", output_dir=output),
    )

    with pytest.raises(CiError, match=name):
        runner.run_host()

    assert (
        output / "artifacts" / "fallback-environment.txt"
    ).read_text(encoding="utf-8") == "absent"
    assert name not in runner.context.env
    assert name not in runner.context.commands.env


def test_workflow_rejects_credentials_before_checkout_without_serializing_values() -> None:
    workflow = PROOF_WORKFLOW.read_text(encoding="utf-8")
    host_policy = workflow.split("- name: Validate host proof policy", maxsplit=1)[1].split(
        "- name: Check model proof disk headroom", maxsplit=1
    )[0]

    assert workflow.index("Validate host proof policy") < workflow.index(
        "Check out exact source revision"
    )
    for name in security.HUGGING_FACE_CREDENTIAL_ENVIRONMENT:
        assert name in host_policy
    assert 'if [[ -v "$name" ]]' in host_policy
    assert 'forbidden+=("$name")' in host_policy
    assert "host-security-policy.json" not in host_policy
    assert "${forbidden[*]}" in host_policy


def test_inner_gate_runs_before_proof_setup_and_is_bound_into_final_evidence() -> None:
    source = PROOF_INNER.read_text(encoding="utf-8")
    run = source.split("def run(self)", maxsplit=1)[1].split(
        "def _validate_runtime_security", maxsplit=1
    )[0]
    build = source.split("def _build_and_test", maxsplit=1)[1].split(
        "def _run_task_eval", maxsplit=1
    )[0]

    assert run.index("self._validate_runtime_security()") < run.index("self._prepare()")
    assert '"runtime_security": "host-security-policy.json, runtime-security.json"' in source
    assert "self._revalidate_runtime_security_evidence()" in build
    assert '"runtime_security_evidence": RUNTIME_SECURITY_EVIDENCE' in build
    assert '"host_security_policy_evidence": HOST_SECURITY_EVIDENCE' in build
    assert '"hugging_face_credentials": "absent"' in build


def test_runtime_policy_records_active_dns_and_numeric_https_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence, fake_socket = _runtime_probe(tmp_path, monkeypatch)

    assert evidence["passed"] is True
    assert evidence["credential_environment"]["absent"] is True
    assert evidence["credential_files"]["absent"] is True
    assert evidence["dns_probe"]["outcome"] == "blocked"
    assert evidence["direct_https_transport_probe"] == {
        "numeric_address": "1.1.1.1",
        "port": 443,
        "phase": "transport-before-tls",
        "outcome": "blocked",
        "error_type": "OSError",
        "error_code": errno.ENETUNREACH,
    }
    assert fake_socket.timeout == 2.0
    assert fake_socket.target == ("1.1.1.1", 443)
    assert fake_socket.closed


@pytest.mark.parametrize("name", security.HUGGING_FACE_CREDENTIAL_ENVIRONMENT)
def test_runtime_policy_rejects_credential_variables_without_serializing_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    host_evidence = tmp_path / security.HOST_SECURITY_EVIDENCE
    runtime_evidence = tmp_path / security.RUNTIME_SECURITY_EVIDENCE
    security.enforce_host_security_policy({}, host_evidence)
    fake_socket = _FakeSocket(OSError(errno.ENETUNREACH, "isolated"))
    monkeypatch.setattr(security.socket, "getaddrinfo", _block_dns)
    monkeypatch.setattr(security.socket, "socket", lambda *_args: fake_socket)
    secret = "hf_runtime_secret_must_not_be_serialized"

    with pytest.raises(CiError, match=name) as caught:
        security.enforce_runtime_security_policy(
            {name: secret},
            host_evidence,
            runtime_evidence,
            token_paths=(("test-token", tmp_path / "missing-token"),),
        )

    evidence_text = runtime_evidence.read_text(encoding="utf-8")
    assert secret not in str(caught.value)
    assert secret not in evidence_text
    evidence = json.loads(evidence_text)
    assert evidence["passed"] is False
    assert evidence["credential_environment"]["present_environment_variables"] == [name]


def test_runtime_policy_rejects_visible_token_file_without_reading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text("hf_file_secret_must_not_be_serialized", encoding="utf-8")
    host_evidence = tmp_path / security.HOST_SECURITY_EVIDENCE
    runtime_evidence = tmp_path / security.RUNTIME_SECURITY_EVIDENCE
    security.enforce_host_security_policy({}, host_evidence)
    monkeypatch.setattr(security.socket, "getaddrinfo", _block_dns)
    monkeypatch.setattr(
        security.socket,
        "socket",
        lambda *_args: _FakeSocket(OSError(errno.ENETUNREACH, "isolated")),
    )

    with pytest.raises(CiError, match="credential files: mounted-token"):
        security.enforce_runtime_security_policy(
            {},
            host_evidence,
            runtime_evidence,
            token_paths=(("mounted-token", token),),
        )

    evidence_text = runtime_evidence.read_text(encoding="utf-8")
    assert "hf_file_secret_must_not_be_serialized" not in evidence_text
    evidence = json.loads(evidence_text)
    assert evidence["credential_files"]["present_locations"] == ["mounted-token"]


@pytest.mark.parametrize(
    "connect_error",
    [
        None,
        ConnectionRefusedError(errno.ECONNREFUSED, "refused"),
        TimeoutError("timed out"),
    ],
)
def test_dns_failure_cannot_mask_reachable_or_ambiguous_https_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    connect_error: OSError | None,
) -> None:
    host_evidence = tmp_path / security.HOST_SECURITY_EVIDENCE
    runtime_evidence = tmp_path / security.RUNTIME_SECURITY_EVIDENCE
    security.enforce_host_security_policy({}, host_evidence)
    monkeypatch.setattr(security.socket, "getaddrinfo", _block_dns)
    monkeypatch.setattr(
        security.socket,
        "socket",
        lambda *_args: _FakeSocket(connect_error),
    )

    with pytest.raises(CiError, match="direct HTTPS transport"):
        security.enforce_runtime_security_policy(
            {},
            host_evidence,
            runtime_evidence,
            token_paths=(("test-token", tmp_path / "missing-token"),),
        )

    evidence = json.loads(runtime_evidence.read_text(encoding="utf-8"))
    assert evidence["passed"] is False
    assert evidence["dns_probe"]["outcome"] == "blocked"
    assert evidence["direct_https_transport_probe"]["outcome"] in {
        "connected",
        "unexpected-error",
    }


def test_socket_setup_error_cannot_impersonate_a_blocked_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed_socket_setup(*_args: object) -> _FakeSocket:
        raise OSError(errno.ENETUNREACH, "socket setup failed")

    host_evidence = tmp_path / security.HOST_SECURITY_EVIDENCE
    runtime_evidence = tmp_path / security.RUNTIME_SECURITY_EVIDENCE
    security.enforce_host_security_policy({}, host_evidence)
    monkeypatch.setattr(security.socket, "getaddrinfo", _block_dns)
    monkeypatch.setattr(security.socket, "socket", failed_socket_setup)

    with pytest.raises(CiError, match="direct HTTPS transport"):
        security.enforce_runtime_security_policy(
            {},
            host_evidence,
            runtime_evidence,
            token_paths=(("test-token", tmp_path / "missing-token"),),
        )

    evidence = json.loads(runtime_evidence.read_text(encoding="utf-8"))
    assert evidence["passed"] is False
    assert evidence["direct_https_transport_probe"]["phase"] == "socket-setup"
    assert evidence["direct_https_transport_probe"]["outcome"] == "unexpected-error"


def test_numeric_transport_failure_cannot_mask_successful_dns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def resolved_dns(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.1", 443))]

    host_evidence = tmp_path / security.HOST_SECURITY_EVIDENCE
    runtime_evidence = tmp_path / security.RUNTIME_SECURITY_EVIDENCE
    security.enforce_host_security_policy({}, host_evidence)
    monkeypatch.setattr(security.socket, "getaddrinfo", resolved_dns)
    monkeypatch.setattr(
        security.socket,
        "socket",
        lambda *_args: _FakeSocket(OSError(errno.ENETUNREACH, "isolated")),
    )

    with pytest.raises(CiError, match="DNS resolution unexpectedly succeeded"):
        security.enforce_runtime_security_policy(
            {},
            host_evidence,
            runtime_evidence,
            token_paths=(("test-token", tmp_path / "missing-token"),),
        )

    evidence = json.loads(runtime_evidence.read_text(encoding="utf-8"))
    assert evidence["passed"] is False
    assert evidence["dns_probe"]["outcome"] == "resolved"


def test_runtime_policy_rejects_missing_or_tampered_host_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host_evidence = tmp_path / security.HOST_SECURITY_EVIDENCE
    runtime_evidence = tmp_path / security.RUNTIME_SECURITY_EVIDENCE
    host_evidence.write_text('{"credentials_absent": true}\n', encoding="utf-8")
    monkeypatch.setattr(security.socket, "getaddrinfo", _block_dns)
    monkeypatch.setattr(
        security.socket,
        "socket",
        lambda *_args: _FakeSocket(OSError(errno.ENETUNREACH, "isolated")),
    )

    with pytest.raises(CiError, match="host security evidence is missing or invalid"):
        security.enforce_runtime_security_policy(
            {},
            host_evidence,
            runtime_evidence,
            token_paths=(("test-token", tmp_path / "missing-token"),),
        )

    evidence = json.loads(runtime_evidence.read_text(encoding="utf-8"))
    assert evidence["passed"] is False
    assert evidence["host_policy"]["valid"] is False
