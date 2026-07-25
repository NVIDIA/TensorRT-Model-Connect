# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))
MODULE_PATH = TOOLS_DIR / "qualify_native_dynamic_memory_policies.py"
SPEC = importlib.util.spec_from_file_location(
    "qualify_native_dynamic_memory_policies", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
policies = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policies
SPEC.loader.exec_module(policies)

pytestmark = pytest.mark.dynamic_memory


def _case(name: str, token_ids: list[int]) -> dict:
    return {
        "name": name,
        "selected_token_ids": token_ids,
        "step_top1_token_ids": [11, *token_ids],
    }


def _complete_persisted_warmup_evidence(
    typed_policy: dict,
    *,
    runtime_capacity: int = 777,
) -> dict:
    lifetime_peak = {
        "passed": True,
        "accounted_peak_bytes": 64 * 1024 * 1024,
        "observed_peak_bytes": 64 * 1024 * 1024,
    }
    continuity = {
        "passed": True,
        "process_growth_bytes": 0,
        "device_wide_growth_bytes": 0,
    }
    peak_reconciliation = {
        "schema_version": 2,
        "reconciliation_basis": "cold_start_and_measured_lifetimes",
        "cold_start_reconciliation": lifetime_peak,
        "measured_reconciliation": lifetime_peak,
        "warmup_continuity_reconciliation": continuity,
        "passed": True,
    }
    logits_sha256 = "a" * 64
    return {
        "schema_version": 2,
        "status": "passed",
        "passed": True,
        "lifetime_protocol": {
            "schema_version": 2,
            "execution_order": ["warmup", "measured"],
            "warmup_count": 1,
            "measured_count": 1,
        },
        "typed_policy": typed_policy,
        "runtime_kv_capacity_tokens": runtime_capacity,
        "reconciliation_basis": "cold_start_and_measured_lifetimes",
        "warmup_excluded_from_measured_peak": True,
        "warmup_independently_hard_gated": True,
        "cold_start_output_equivalence": {
            "passed": True,
            "python_full_float32_logits_bitwise_equal": True,
            "python_selected_and_top1_token_ids_equal": True,
            "cold_start_logits_sha256": logits_sha256,
            "measured_logits_sha256": logits_sha256,
        },
        "sampler_identity": {
            "pid": 12_345,
            "cuda_logical_device_index": 0,
            "pci_bus_id": "00000000:01:00.0",
            "gpu_uuid": "GPU-unit-test",
        },
        "lifetime_endpoint_bindings": {
            "warmup": {"passed": True},
            "measured": {"passed": True},
        },
        "cold_start_peak_reconciliation": lifetime_peak,
        "measured_peak_reconciliation": lifetime_peak,
        "peak_memory_reconciliation": peak_reconciliation,
        "cold_start_retention_gate": {"passed": True},
        "cold_start_persistent_driver_gate": {"passed": True},
        "measured_retention_gate": {"passed": True},
        "continuity_reconciliation": continuity,
    }


def test_policy_comparison_requires_exact_tokens_and_full_logits() -> None:
    cases = [_case("auto", [7, 8]), _case("bytes", [7, 8])]
    logits = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    comparisons, passed = policies.compare_policy_outputs(
        cases, {"auto": logits, "bytes": logits.copy()}
    )

    assert passed
    assert comparisons == [
        {
            "reference": "auto",
            "candidate": "bytes",
            "selected_token_ids_equal": True,
            "step_top1_token_ids_equal": True,
            "full_float32_logits_equal": True,
            "passed": True,
        }
    ]


def test_policy_comparison_rejects_one_float32_bit_or_token_change() -> None:
    reference = np.asarray([[1.0, 2.0]], dtype=np.float32)
    changed = reference.copy()
    changed.view(np.uint32)[0, 0] += 1

    comparisons, passed = policies.compare_policy_outputs(
        [_case("auto", [7]), _case("fraction", [8]), _case("bytes", [7])],
        {
            "auto": reference,
            "fraction": reference.copy(),
            "bytes": changed,
        },
    )

    assert not passed
    assert not comparisons[0]["selected_token_ids_equal"]
    assert not comparisons[1]["full_float32_logits_equal"]


def test_source_state_gate_is_fail_closed() -> None:
    pre = {"source_state_sha256": "a" * 64, "git_head": "b" * 40}
    post = {"source_state_sha256": "a" * 64, "git_head": "b" * 40}
    report = {"passed": True}

    assert policies.apply_source_state_gate(report, pre, post)
    assert report["source_state_pre"] is pre
    assert report["source_state_post"] is post
    assert report["source_state_unchanged"] is True
    assert report["passed"] is True

    changed = {"source_state_sha256": "c" * 64, "git_head": "b" * 40}
    report = {"passed": True}
    assert not policies.apply_source_state_gate(report, pre, changed)
    assert report["source_state_unchanged"] is False
    assert report["passed"] is False

    report = {"passed": False}
    assert policies.apply_source_state_gate(report, pre, post)
    assert report["passed"] is False


def test_source_snapshot_excludes_artifact_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, str]] = []

    def snapshot(
        repo_root: Path,
        tool_path: Path,
        artifact_dir: Path,
        *,
        label: str,
    ) -> dict:
        assert repo_root == policies.REPO_ROOT
        assert tool_path == Path(policies.__file__)
        calls.append((artifact_dir, label))
        return {"source_state_sha256": "a" * 64, "git_head": "b" * 40}

    monkeypatch.setattr(
        policies.boundary, "source_state_provenance", snapshot
    )
    external = tmp_path / "proof"
    policies._source_state_snapshot(external, label="pre")
    artifact = policies.REPO_ROOT / "artifacts" / "unit-policy-proof"
    policies._source_state_snapshot(artifact, label="post")

    assert calls == [
        (external.resolve(), "pre"),
        (artifact.resolve(), "post"),
    ]
    with pytest.raises(ValueError, match="source snapshots exclude it"):
        policies._source_state_snapshot(
            policies.REPO_ROOT / "unit-policy-proof",
            label="post",
        )


def test_policy_runner_uses_warmup_and_preserves_typed_request_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_capacity = 777
    policy_cases = (
        (
            policies.PolicyCase("auto", "auto"),
            (),
            {"kind": "auto"},
        ),
        (
            policies.PolicyCase("fraction", "fraction", 0.8),
            ("--kv-cache-fraction", "0.8"),
            {"kind": "fraction", "requested_fraction": 0.8},
        ),
        (
            policies.PolicyCase("bytes", "bytes", 12_345_678),
            ("--kv-cache-bytes", "12345678"),
            {"kind": "bytes", "requested_bytes": 12_345_678},
        ),
        (
            policies.PolicyCase(
                "max-sequence",
                "max_sequence_length",
                512,
            ),
            ("--max-sequence-length", "512"),
            {"kind": "max_sequence_length", "requested_tokens": 512},
        ),
    )
    commands: dict[str, list[str]] = {}
    replayed_policies: dict[str, dict] = {}
    sampler = policies.boundary.SamplerTrustAnchor(
        pid=12_345,
        cuda_logical_device_index=0,
        physical_device_index=0,
        pci_bus_id="0000:01:00.0",
        gpu_uuid="GPU-01234567-89ab-cdef-0123-456789abcdef",
    )

    def run(command: list[str]):
        name = Path(command[command.index("--tokens") + 1]).parent.name
        commands[name] = command
        Path(command[command.index("--logits") + 1]).write_bytes(
            b"mock-float32-logits"
        )
        trace = {
            "status": "ok",
            "runtime_memory_receipt": {
                "runtime_kv_capacity_tokens": runtime_capacity,
            },
            "effective_request_limit": runtime_capacity,
            "selected_token_ids": [41, 42],
            "step_top1_token_ids": [40, 41, 42],
        }
        return (
            subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(trace),
                stderr="",
            ),
            sampler.pid,
        )

    def replay_capture(
        evidence_dir: Path,
        **kwargs: object,
    ) -> dict:
        case = kwargs["case"]
        expected_policy = kwargs["expected_lifetime_policy"]
        assert isinstance(case, policies.boundary.Case)
        assert evidence_dir.name == case.name
        assert kwargs["expected_sampler"] == sampler
        assert kwargs["expected_returncode"] == 0
        assert isinstance(expected_policy, dict)
        warmup_evidence = _complete_persisted_warmup_evidence(
            expected_policy,
            runtime_capacity=runtime_capacity,
        )
        return {
            "logits": np.zeros((3, 4), dtype=np.float32),
            "validation_evidence": {
                "cold_start_evidence": warmup_evidence,
                "warmup_evidence": warmup_evidence,
                "peak_memory_reconciliation": warmup_evidence[
                    "peak_memory_reconciliation"
                ],
            },
        }

    def replay_warmup_evidence(
        evidence: object,
        *,
        trace: object,
        case: object,
        trusted_geometry: object,
        expected_sampler: object,
        expected_lifetime_policy: object,
    ) -> bool:
        assert trusted_geometry == policies.boundary.trusted_runtime_geometry(
            policies.boundary.SPECS[
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            ]
        )
        assert expected_sampler == sampler
        assert isinstance(trace, dict) and trace["status"] == "ok"
        assert isinstance(case, policies.boundary.Case)
        assert case.prompt_tokens == 127
        assert case.decode_tokens == 2
        assert isinstance(expected_lifetime_policy, dict)
        assert evidence == _complete_persisted_warmup_evidence(
            expected_lifetime_policy,
            runtime_capacity=runtime_capacity,
        )
        replayed_policies[case.name] = expected_lifetime_policy
        return True

    monkeypatch.setattr(policies.boundary, "_run_captured_command", run)
    monkeypatch.setattr(
        policies.boundary,
        "_sampler_trust_anchor",
        lambda **kwargs: sampler,
    )
    monkeypatch.setattr(
        policies.boundary,
        "_write_runner_capture_manifest",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        policies.boundary,
        "replay_runner_capture",
        replay_capture,
    )
    monkeypatch.setattr(
        policies.boundary,
        "context_shape_sweep",
        lambda trace: [{"trace_status": trace["status"]}],
    )
    monkeypatch.setattr(
        policies.boundary,
        "_persisted_case_warmup_evidence_passed",
        replay_warmup_evidence,
    )

    spec = policies.boundary.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    tokens = np.arange(127, dtype=np.int32)
    for policy, expected_arguments, expected_policy in policy_cases:
        case_report, logits, actual_sampler = policies._run_policy(
            policy=policy,
            runner=tmp_path / "runner",
            bundle=tmp_path / "bundle.trtmc",
            tokens=tokens,
            model_spec=spec,
            output_dir=tmp_path,
            backend_dirs=[tmp_path / "backends"],
            model_plugin_dirs=[tmp_path / "models"],
        )

        command = commands[policy.name]
        evidence_dir = tmp_path / "runner-evidence" / policy.name
        assert command[:9] == [
            str(tmp_path / "runner"),
            "--bundle",
            str(tmp_path / "bundle.trtmc"),
            "--tokens",
            str(evidence_dir / "tokens.txt"),
            "--logits",
            str(evidence_dir / "runner-logits.bin"),
            "--max-new-tokens",
            "2",
        ]
        warmup_index = command.index("--warmup-load-cycle")
        assert tuple(command[warmup_index + 1 : warmup_index + 3]) == (
            expected_arguments
            if expected_arguments
            else ("--backend-dir", str(tmp_path / "backends"))
        )
        assert command.count("--warmup-load-cycle") == 1
        for argument in expected_arguments:
            assert argument in command
        assert case_report["expected_lifetime_policy"] == expected_policy
        assert case_report["cold_start_evidence"] is case_report[
            "warmup_evidence"
        ]
        assert (
            case_report["warmup_evidence"]["typed_policy"] == expected_policy
        )
        assert case_report["peak_memory_reconciliation"]["passed"] is True
        assert case_report["memory_evidence_passed"] is True
        assert case_report["runner_evidence"] == str(evidence_dir)
        assert logits.shape == (3, 4)
        assert actual_sampler == sampler

    assert replayed_policies == {
        policy.name: expected_policy
        for policy, _, expected_policy in policy_cases
    }
    assert replayed_policies["max-sequence"]["requested_tokens"] == 512
    assert replayed_policies["max-sequence"]["requested_tokens"] != runtime_capacity


@pytest.mark.parametrize(
    ("evidence_state", "expected_passed"),
    (
        ("complete", True),
        ("missing", False),
        ("incomplete", False),
        ("self-policy-tamper", False),
    ),
)
def test_policy_final_aggregate_requires_complete_persisted_memory_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence_state: str,
    expected_passed: bool,
) -> None:
    bundle = tmp_path / "model.trtmc"
    runner = tmp_path / "runner"
    output_dir = tmp_path / "proof"
    bundle.write_bytes(b"bundle")
    runner.write_bytes(b"runner")

    spec = policies.boundary.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    source_state = {
        "source_state_sha256": "b" * 64,
        "git_head": "c" * 40,
    }
    monkeypatch.setattr(
        policies,
        "_source_state_snapshot",
        lambda artifact_dir, *, label: dict(source_state),
    )
    monkeypatch.setattr(
        policies.boundary,
        "_read_bundle_header",
        lambda path: {
            "runtime_memory": {"kv_bytes_per_token": 1_024},
            "vocab_size": 32_000,
        },
    )
    monkeypatch.setattr(policies.boundary, "_resolve_spec", lambda header: spec)

    sampler = policies.boundary.SamplerTrustAnchor(
        pid=12_345,
        cuda_logical_device_index=0,
        physical_device_index=0,
        pci_bus_id="0000:01:00.0",
        gpu_uuid="GPU-01234567-89ab-cdef-0123-456789abcdef",
    )

    def run_policy(
        **kwargs: object,
    ) -> tuple[dict, np.ndarray, policies.boundary.SamplerTrustAnchor]:
        policy = kwargs["policy"]
        assert isinstance(policy, policies.PolicyCase)
        capacity = 777 if policy.kind in {"auto", "fraction"} else 512
        evidence = _complete_persisted_warmup_evidence(
            policy.expected_lifetime_policy,
            runtime_capacity=capacity,
        )
        case_report = {
            "name": policy.name,
            "expected_lifetime_policy": policy.expected_lifetime_policy,
            "runtime_kv_capacity_tokens": capacity,
            "selected_token_ids": [41, 42],
            "step_top1_token_ids": [40, 41, 42],
            "memory_evidence_passed": True,
            "warmup_evidence": evidence,
            "trace": {"policy_name": policy.name},
            "runner_evidence": str(
                output_dir / "runner-evidence" / policy.name
            ),
        }
        if policy.name == "auto":
            if evidence_state == "missing":
                del case_report["warmup_evidence"]
            elif evidence_state == "incomplete":
                case_report["warmup_evidence"] = {
                    "status": "passed",
                    "passed": True,
                }
            elif evidence_state == "self-policy-tamper":
                case_report["expected_lifetime_policy"] = {
                    "kind": "bytes",
                    "requested_bytes": 1,
                }
        return (
            case_report,
            np.zeros((3, 4), dtype=np.float32),
            sampler,
        )

    def replay_capture(
        evidence_dir: Path,
        **kwargs: object,
    ) -> dict:
        policy = kwargs["expected_lifetime_policy"]
        capacity = (
            777
            if policy["kind"] in {"auto", "fraction"}
            else 512
        )
        return {
            "logits": np.zeros((3, 4), dtype=np.float32),
            "validation_evidence": {
                "warmup_evidence": _complete_persisted_warmup_evidence(
                    policy,
                    runtime_capacity=capacity,
                ),
            },
        }

    def replay_warmup_evidence(
        evidence: object,
        *,
        trace: object,
        case: object,
        trusted_geometry: object,
        expected_sampler: object,
        expected_lifetime_policy: object,
    ) -> bool:
        assert trusted_geometry == policies.boundary.trusted_runtime_geometry(
            spec
        )
        assert expected_sampler == sampler
        assert isinstance(trace, dict)
        assert isinstance(case, policies.boundary.Case)
        assert trace["policy_name"] == case.name
        assert case.prompt_tokens == 127
        assert case.decode_tokens == 2
        return bool(
            isinstance(evidence, dict)
            and evidence.get("schema_version") == 2
            and evidence.get("typed_policy") == expected_lifetime_policy
            and evidence.get("passed") is True
        )

    monkeypatch.setattr(policies, "_run_policy", run_policy)
    monkeypatch.setattr(
        policies.boundary,
        "replay_runner_capture",
        replay_capture,
    )
    monkeypatch.setattr(
        policies.boundary,
        "_persisted_case_warmup_evidence_passed",
        replay_warmup_evidence,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--bundle",
            str(bundle),
            "--runner",
            str(runner),
            "--output-dir",
            str(output_dir),
        ],
    )

    returncode = policies.main()
    report = json.loads(
        (output_dir / "policy-equivalence-report.json").read_text(
            encoding="utf-8"
        )
    )

    assert "all_memory_evidence_passed" in report
    assert report["all_memory_evidence_passed"] is expected_passed
    assert report["passed"] is expected_passed
    assert returncode == (0 if expected_passed else 1)
