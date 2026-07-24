#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove one dynamic-memory bundle is token/logit invariant across policies.

This qualification intentionally uses the private native runner: the product
CLI returns user text, while UX-04 requires exact generated token IDs and full
float32 logits.  The request crosses the 128-token decode bucket so the same
receipt also records an actual-shape A/T sweep at two runtime capacities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import qualify_native_dynamic_memory as boundary


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PolicyCase:
    name: str
    kind: str
    requested_value: int | float | None = None

    def __post_init__(self) -> None:
        if self.kind == "auto":
            if self.requested_value is not None:
                raise ValueError("auto policy must not have a requested value")
            return
        if self.kind == "fraction":
            if (
                type(self.requested_value) is not float
                or not 0.0 < self.requested_value <= 1.0
            ):
                raise ValueError("fraction policy requires a float in (0, 1]")
            return
        if self.kind in {"bytes", "max_sequence_length"}:
            if (
                type(self.requested_value) is not int
                or self.requested_value <= 0
            ):
                raise ValueError(
                    f"{self.kind} policy requires a positive integer"
                )
            return
        raise ValueError(f"unknown runtime-memory policy kind: {self.kind!r}")

    @property
    def arguments(self) -> tuple[str, ...]:
        if self.kind == "auto":
            return ()
        assert self.requested_value is not None
        flag = {
            "fraction": "--kv-cache-fraction",
            "bytes": "--kv-cache-bytes",
            "max_sequence_length": "--max-sequence-length",
        }[self.kind]
        return (flag, str(self.requested_value))

    @property
    def expected_lifetime_policy(self) -> dict[str, Any]:
        if self.kind == "auto":
            return {"kind": "auto"}
        assert self.requested_value is not None
        value_name = {
            "fraction": "requested_fraction",
            "bytes": "requested_bytes",
            "max_sequence_length": "requested_tokens",
        }[self.kind]
        return {
            "kind": self.kind,
            value_name: self.requested_value,
        }


def compare_policy_outputs(
    cases: list[dict[str, Any]],
    logits_by_policy: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], bool]:
    if not cases:
        raise ValueError("policy comparison requires at least one case")
    reference = cases[0]
    reference_logits = logits_by_policy[reference["name"]]
    comparisons: list[dict[str, Any]] = []
    all_equal = True
    for case in cases[1:]:
        token_ids_equal = (
            case["selected_token_ids"] == reference["selected_token_ids"]
        )
        top1_equal = (
            case["step_top1_token_ids"] == reference["step_top1_token_ids"]
        )
        logits_equal = bool(
            np.array_equal(logits_by_policy[case["name"]], reference_logits)
        )
        passed = token_ids_equal and top1_equal and logits_equal
        comparisons.append(
            {
                "reference": reference["name"],
                "candidate": case["name"],
                "selected_token_ids_equal": token_ids_equal,
                "step_top1_token_ids_equal": top1_equal,
                "full_float32_logits_equal": logits_equal,
                "passed": passed,
            }
        )
        all_equal = all_equal and passed
    return comparisons, all_equal


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path),
    }


def _source_state_snapshot(
    artifact_dir: Path, *, label: str
) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    try:
        relative = artifact_dir.relative_to(REPO_ROOT)
    except ValueError:
        relative = None
    if relative is not None:
        top_level = relative.parts[0] if relative.parts else ""
        if not (
            top_level == "artifacts"
            or top_level == "build"
            or top_level.startswith("build-")
        ):
            raise ValueError(
                "qualification output inside the repository must be under "
                "artifacts/, build/, or build-* so source snapshots exclude it"
            )
    return boundary.source_state_provenance(
        REPO_ROOT,
        Path(__file__),
        artifact_dir,
        label=label,
    )


def apply_source_state_gate(
    report: dict[str, Any],
    source_state_pre: dict[str, Any],
    source_state_post: dict[str, Any],
) -> bool:
    pre_sha = source_state_pre.get("source_state_sha256")
    post_sha = source_state_post.get("source_state_sha256")
    unchanged = bool(
        isinstance(pre_sha, str)
        and pre_sha
        and pre_sha == post_sha
        and source_state_pre.get("git_head")
        == source_state_post.get("git_head")
    )
    report["source_state_pre"] = source_state_pre
    report["source_state_post"] = source_state_post
    report["source_state_unchanged"] = unchanged
    report["passed"] = bool(report.get("passed") is True and unchanged)
    return unchanged


def _policy_command(
    *,
    policy: PolicyCase,
    runner: Path,
    bundle: Path,
    token_file: Path,
    logits_file: Path,
    backend_dirs: list[Path],
    model_plugin_dirs: list[Path],
) -> list[str]:
    command = [
        str(runner),
        "--bundle",
        str(bundle),
        "--tokens",
        str(token_file),
        "--logits",
        str(logits_file),
        "--max-new-tokens",
        "2",
        "--warmup-load-cycle",
        *policy.arguments,
    ]
    for directory in backend_dirs:
        command.extend(["--backend-dir", str(directory)])
    for directory in model_plugin_dirs:
        command.extend(["--model-plugin-dir", str(directory)])
    return command


def _run_policy(
    *,
    policy: PolicyCase,
    runner: Path,
    bundle: Path,
    tokens: np.ndarray,
    model_spec: boundary.ModelSpec,
    output_dir: Path,
    backend_dirs: list[Path],
    model_plugin_dirs: list[Path],
) -> tuple[
    dict[str, Any],
    np.ndarray,
    boundary.SamplerTrustAnchor,
]:
    evidence_dir = output_dir / "runner-evidence" / policy.name
    evidence_dir.mkdir(parents=True, exist_ok=False)
    token_file = evidence_dir / "tokens.txt"
    logits_file = evidence_dir / "runner-logits.bin"
    boundary._write_tokens(token_file, tokens)
    command = _policy_command(
        policy=policy,
        runner=runner,
        bundle=bundle,
        token_file=token_file,
        logits_file=logits_file,
        backend_dirs=backend_dirs,
        model_plugin_dirs=model_plugin_dirs,
    )

    (evidence_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n",
        encoding="utf-8",
    )
    completed, child_pid = boundary._run_captured_command(command)
    sampler_anchor = boundary._sampler_trust_anchor(
        child_pid=child_pid,
        cuda_logical_device_index=0,
    )
    stdout_path = evidence_dir / "runner.stdout.log"
    stderr_path = evidence_dir / "runner.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    (evidence_dir / "returncode.txt").write_text(
        f"{completed.returncode}\n",
        encoding="utf-8",
    )
    trace = boundary._parse_runner_json(completed.stdout)
    if completed.returncode != 0 or trace.get("status") != "ok":
        raise RuntimeError(
            f"{policy.name}: runner failed ({completed.returncode}); "
            f"trace={trace}; stderr={completed.stderr[-4000:]}"
        )
    (evidence_dir / "runner-trace.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    boundary._write_runner_capture_manifest(
        evidence_dir,
        child_pid=child_pid,
        sampler_anchor=sampler_anchor,
        include_logits=True,
    )
    runtime_capacity = int(
        trace["runtime_memory_receipt"]["runtime_kv_capacity_tokens"]
    )
    qualification_case = boundary.Case(policy.name, 127, 2)
    trusted_geometry = boundary.trusted_runtime_geometry(model_spec)
    replay = boundary.replay_runner_capture(
        evidence_dir,
        expected_command=command,
        expected_tokens=tokens,
        expected_returncode=0,
        expected_trace=trace,
        case=qualification_case,
        model_spec=model_spec,
        trusted_geometry=trusted_geometry,
        expected_sampler=sampler_anchor,
        expected_lifetime_policy=policy.expected_lifetime_policy,
    )
    logits = replay["logits"]
    validation_evidence = replay["validation_evidence"]
    if not isinstance(logits, np.ndarray):
        raise RuntimeError(f"{policy.name}: replay returned no logits")
    if validation_evidence is None:
        raise RuntimeError(
            f"{policy.name}: trace validation did not return memory evidence"
        )
    warmup_evidence = validation_evidence["warmup_evidence"]
    if not boundary._persisted_case_warmup_evidence_passed(
        warmup_evidence,
        trace=trace,
        case=qualification_case,
        trusted_geometry=trusted_geometry,
        expected_sampler=sampler_anchor,
        expected_lifetime_policy=policy.expected_lifetime_policy,
    ):
        raise RuntimeError(
            f"{policy.name}: persisted cold/measured memory evidence is incomplete"
        )
    return (
        {
            "name": policy.name,
            "command": command,
            "expected_lifetime_policy": policy.expected_lifetime_policy,
            "returncode": completed.returncode,
            "runtime_kv_capacity_tokens": runtime_capacity,
            "effective_request_limit": trace["effective_request_limit"],
            "selected_token_ids": trace["selected_token_ids"],
            "step_top1_token_ids": trace["step_top1_token_ids"],
            "logits_artifact": str(logits_file),
            "logits_sha256": _sha256(logits_file),
            "trace": trace,
            "actual_shape_context_sweep": boundary.context_shape_sweep(trace),
            "cold_start_evidence": validation_evidence[
                "cold_start_evidence"
            ],
            "warmup_evidence": warmup_evidence,
            "memory_evidence_passed": True,
            "peak_memory_reconciliation": validation_evidence[
                "peak_memory_reconciliation"
            ],
            "runner_stderr": str(stderr_path),
            "runner_stdout": str(stdout_path),
            "runner_evidence": str(evidence_dir),
        },
        logits,
        sampler_anchor,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--model-plugin-dir", type=Path, action="append", default=[]
    )
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    runner = args.runner.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_state_pre = _source_state_snapshot(output_dir, label="pre")

    header = boundary._read_bundle_header(bundle)
    spec = boundary._resolve_spec(header)
    contract = header["runtime_memory"]
    vocab_size = int(header["vocab_size"])
    bytes_per_token = int(contract["kv_bytes_per_token"])
    small_capacity = min(512, spec.context_limit)
    if small_capacity < 129:
        raise ValueError("UX-04 policy sweep requires capacity for 129 tokens")
    tokens = boundary.deterministic_token_ids(127, vocab_size)

    policies = (
        PolicyCase("auto", "auto"),
        PolicyCase("fraction-80pct", "fraction", 0.8),
        PolicyCase(
            "bytes-512-rows",
            "bytes",
            small_capacity * bytes_per_token,
        ),
        PolicyCase(
            "max-sequence-512",
            "max_sequence_length",
            small_capacity,
        ),
    )
    bundle_before = _bundle_identity(bundle)
    cases: list[dict[str, Any]] = []
    logits_by_policy: dict[str, np.ndarray] = {}
    sampler_anchors: dict[str, boundary.SamplerTrustAnchor] = {}
    backend_dirs = [path.resolve() for path in args.backend_dir]
    model_plugin_dirs = [
        path.resolve() for path in args.model_plugin_dir
    ]
    for policy in policies:
        print(f"[policy-equivalence] {policy.name}", file=sys.stderr, flush=True)
        case_report, logits, sampler_anchor = _run_policy(
            policy=policy,
            runner=runner,
            bundle=bundle,
            tokens=tokens,
            model_spec=spec,
            output_dir=output_dir,
            backend_dirs=backend_dirs,
            model_plugin_dirs=model_plugin_dirs,
        )
        cases.append(case_report)
        logits_by_policy[policy.name] = logits
        sampler_anchors[policy.name] = sampler_anchor

    bundle_after = _bundle_identity(bundle)
    bundle_unchanged = bundle_before == bundle_after
    observed_capacities = sorted(
        {int(case["runtime_kv_capacity_tokens"]) for case in cases}
    )
    capacity_sweep_passed = len(observed_capacities) >= 2
    trusted_geometry = boundary.trusted_runtime_geometry(spec)
    replayed_logits: dict[str, np.ndarray] = {}
    replay_states: dict[str, str] = {}
    all_memory_evidence_passed = len(cases) == len(policies)
    for policy, case in zip(policies, cases):
        try:
            if (
                case.get("name") != policy.name
                or case.get("expected_lifetime_policy")
                != policy.expected_lifetime_policy
                or case.get("memory_evidence_passed") is not True
            ):
                raise RuntimeError(
                    f"{policy.name}: report does not bind the fixed policy matrix"
                )
            evidence_dir = (
                output_dir / "runner-evidence" / policy.name
            ).resolve()
            if case.get("runner_evidence") != str(evidence_dir):
                raise RuntimeError(
                    f"{policy.name}: report does not bind its trusted raw capture"
                )
            replay = boundary.replay_runner_capture(
                evidence_dir,
                expected_command=_policy_command(
                    policy=policy,
                    runner=runner,
                    bundle=bundle,
                    token_file=evidence_dir / "tokens.txt",
                    logits_file=evidence_dir / "runner-logits.bin",
                    backend_dirs=backend_dirs,
                    model_plugin_dirs=model_plugin_dirs,
                ),
                expected_tokens=tokens,
                expected_returncode=0,
                expected_trace=case.get("trace"),
                case=boundary.Case(policy.name, 127, 2),
                model_spec=spec,
                trusted_geometry=trusted_geometry,
                expected_sampler=sampler_anchors[policy.name],
                expected_lifetime_policy=policy.expected_lifetime_policy,
            )
            logits = replay.get("logits")
            validation = replay.get("validation_evidence")
            if (
                not isinstance(logits, np.ndarray)
                or not isinstance(validation, dict)
                or case.get("warmup_evidence")
                != validation.get("warmup_evidence")
                or not boundary._persisted_case_warmup_evidence_passed(
                    case.get("warmup_evidence"),
                    trace=case.get("trace"),
                    case=boundary.Case(policy.name, 127, 2),
                    trusted_geometry=trusted_geometry,
                    expected_sampler=sampler_anchors[policy.name],
                    expected_lifetime_policy=(
                        policy.expected_lifetime_policy
                    ),
                )
            ):
                raise RuntimeError(
                    f"{policy.name}: replayed policy evidence is incomplete"
                )
            replayed_logits[policy.name] = logits
            replay_states[policy.name] = "passed"
        except (KeyError, RuntimeError, TypeError, ValueError, OSError):
            replay_states[policy.name] = "failed"
            all_memory_evidence_passed = False
    comparisons, all_equal = compare_policy_outputs(
        cases,
        replayed_logits if all_memory_evidence_passed else logits_by_policy,
    )
    report = {
        "schema_version": 1,
        "gate": "UX-04",
        "model_id": spec.model_id,
        "runner": str(runner),
        "runner_sha256": _sha256(runner),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "bundle_before": bundle_before,
        "bundle_after": bundle_after,
        "bundle_unchanged": bundle_unchanged,
        "prompt_tokens": 127,
        "decode_tokens": 2,
        "input_token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        "observed_runtime_capacities": observed_capacities,
        "capacity_sweep_passed": capacity_sweep_passed,
        "all_memory_evidence_passed": all_memory_evidence_passed,
        "raw_runner_replay": {
            "passed": all_memory_evidence_passed,
            "case_states": replay_states,
        },
        "cases": cases,
        "comparisons": comparisons,
        "passed": bool(
            all_equal
            and bundle_unchanged
            and capacity_sweep_passed
            and all_memory_evidence_passed
        ),
    }
    source_state_post = _source_state_snapshot(output_dir, label="post")
    apply_source_state_gate(report, source_state_pre, source_state_post)
    report_path = output_dir / "policy-equivalence-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "report": str(report_path),
                "bundle_sha256": bundle_before["sha256"],
                "runtime_capacities": observed_capacities,
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
