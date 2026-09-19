# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native request-local decoding sessions compared with complete HSTU execution."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from families.hstu.tests.fixtures import (
    ACTION_IDS, ITEM_IDS, SEED, context_length, make_checkpoint, sample_request,
)


SESSION_CASES = (
    "hstu-session-tiny-fp32", "hstu-session-tiny-fp16", "hstu-session-tiny-bf16",
    "hstu-session-hundreds-bf16", "hstu-session-retrieval-fp32",
    "hstu-session-default-scaling-fp32",
)


def _session_binary() -> Path:
    explicit = os.environ.get("TRTMC_HSTU_SESSION_BINARY")
    build = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    generic = os.environ.get("TRTMC_BINARY")
    if explicit:
        binary = Path(explicit)
    elif build:
        binary = Path(build) / "hstu_session_runner"
    elif generic:
        binary = Path(generic).with_name("hstu_session_runner")
    else:
        raise AssertionError("selected session E2E requires TRTMC_NATIVE_BUILD_DIR or TRTMC_BINARY")
    assert binary.is_file() and os.access(binary, os.X_OK), f"native session driver is missing: {binary}"
    return binary


def _history_tokens(sequence: dict) -> int:
    return (context_length(sequence) + len(sequence["history_item_ids"])
            + len(sequence.get("history_action_ids", ())))


def _check_result(actual: dict, expected: dict, thresholds: dict, label: str) -> list[dict]:
    errors = []
    for field in ("candidate_item_ids", "num_candidates", "embedding_dim", "output_dim", "sequence_length"):
        assert actual[field] == expected[field], (label, field)
    for field in ("logits", "scores", "embeddings", "sequence_embeddings"):
        left, right = np.asarray(actual[field]), np.asarray(expected.get(field, []))
        assert left.shape == right.shape and np.isfinite(left).all(), (label, field)
        np.testing.assert_allclose(left, right, **thresholds, err_msg=f"{label}/{field}")
        errors.append({"comparison": label, "field": field,
                       "max_absolute_error": float(np.abs(left - right).max()) if left.size else 0.0})
    return errors


def _verify_session_transitions(audit: dict, initial: dict, candidates: list[int], config: dict) -> None:
    rows = {step["name"]: step for step in audit["steps"]}
    assert len(rows) == len(audit["steps"]) == 31
    assert audit["persistent_history_unchanged"] and audit["session_outlived_task"]
    assert audit["invalid_append"]["error"]
    assert audit["invalid_score_error"]
    assert audit["budget_rejection_error"]
    assert audit["persistent_after_session_creation"]["publications"] == 1
    assert audit["persistent_before_final_probe"]["publications"] == 1
    scale_changes = config["scaling_seqlen"] == -1
    assert audit["persistent_final"]["publications"] == (2 if scale_changes else 1)
    candidate_tokens = len(candidates) if config["mode"] == "ranking" else 0
    initial_tokens = _history_tokens(initial)
    stride = 2 if initial.get("history_action_ids") else 1
    previous = initial_tokens
    for step in range(21):
        name = "initial" if step == 0 else f"step-{step:02d}"
        row = rows[name]
        request = row["request"]["sequences"][0]
        report = row["actual"]["sequences"][0]["cache"]
        assert request["candidate_item_ids"] == candidates
        reused = 0 if scale_changes else previous
        assert report["source"] == "request_local"
        assert report["reused_history_tokens"] == reused, (name, report)
        assert report["computed_tokens"] == initial_tokens + step * stride - reused + candidate_tokens
        reason = "exact_history" if step == 0 else "append_history"
        assert report["reason"] == ("scale_changed" if scale_changes else reason)
        assert report["history_tokens"] == initial_tokens + step * stride
        if step:
            assert request["history_item_ids"][-1] == row["selected_item_id"]
        previous = report["history_tokens"]
    parent = rows["step-20"]["request"]["sequences"][0]
    left = rows["fork-left"]["request"]["sequences"][0]
    right = rows["fork-right"]["request"]["sequences"][0]
    assert left["history_item_ids"][:-1] == right["history_item_ids"][:-1] == parent["history_item_ids"]
    assert left["history_item_ids"][-1] != right["history_item_ids"][-1]
    for name in ("fork-left", "fork-right", "fork-after-pending-append", "pending-parent"):
        report = rows[name]["actual"]["sequences"][0]["cache"]
        reused = 0 if scale_changes else previous
        assert report["reused_history_tokens"] == reused
        assert report["computed_tokens"] == previous + stride - reused + candidate_tokens
    assert rows["fork-after-pending-append"]["request"] == rows["pending-parent"]["request"]
    for name in ("parent-after-forks", "after-invalid-append", "after-task-destruction"):
        assert rows[name]["request"] == rows["step-20"]["request"]
        report = rows[name]["actual"]["sequences"][0]["cache"]
        assert report["reused_history_tokens"] == previous
        assert report["computed_tokens"] == candidate_tokens
    recovery = rows["after-invalid-score"]
    assert recovery["request"] == rows["step-20"]["request"]
    assert recovery["actual"]["sequences"][0]["cache"]["reused_history_tokens"] == 0
    assert recovery["actual"]["sequences"][0]["cache"]["computed_tokens"] == previous + candidate_tokens
    persistent = rows["persistent-after"]["actual"]["sequences"][0]["cache"]
    assert persistent["history_tokens"] == initial_tokens
    reused = 0 if scale_changes else initial_tokens
    assert persistent["reused_history_tokens"] == reused
    assert persistent["computed_tokens"] == initial_tokens - reused + candidate_tokens


def run_session_case(case_name: str, tmp_path: Path, *, decode_steps: int | None = None) -> None:
    """Run one explicitly selected family manifest case without another selector."""
    if case_name not in SESSION_CASES:
        raise ValueError(f"unknown HSTU session testcase: {case_name}")
    import torch
    from safetensors.numpy import load_file
    from tensorrt_model_connect import BuildRequest, build
    from families.hstu.tests.environment import reference_source
    from families.hstu.tests.reference import reference_receipt, run_reference
    from tools.e2e_evidence import evidence_stage, record_evidence

    assert torch.cuda.is_available(), "selected HSTU session case requires CUDA"
    test_root = Path(__file__).parent
    manifest = json.loads((test_root / "manifests" / f"{case_name}.json").read_text())
    case = next(entry for entry in manifest["testcases"] if entry["name"] == case_name)
    if decode_steps is None:
        decode_steps = case["decode_steps"]
    assert decode_steps == case["decode_steps"] == 20
    precision = manifest["precision"]
    checkpoint = tmp_path / "checkpoint"
    config = make_checkpoint(checkpoint, **case["config_overrides"])
    assert config["enable_history_cache"]
    initial = sample_request(config, **case.get("fixture", {}))["sequences"][0]
    candidates = initial.pop("candidate_item_ids")
    initial["candidate_item_ids"] = []
    initial["cache"] = {"subject_id": "persistent-user", "feature_version": "features-v1",
                        "history_epoch": "history-v1"}
    threshold_path = test_root / "thresholds" / f"{case_name}.json"
    thresholds = json.loads(threshold_path.read_text())["threshold_overrides"]
    record_evidence("thresholds", thresholds)
    source = reference_source()
    runtime_root = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    binary = _session_binary()
    bundle = tmp_path / manifest["bundle"]
    arrays_path = checkpoint / "checkpoint-arrays.npz"
    np.savez_compressed(arrays_path, **load_file(str(checkpoint / "model.safetensors")))
    record_evidence("reference_source", reference_receipt(source))
    record_evidence("checkpoint", {"seed": SEED, "config": config,
                                   "canonical_arrays": arrays_path,
                                   "weights": checkpoint / "model.safetensors",
                                   "qualification": "seeded native session parity; no trained accuracy claim"})
    with evidence_stage("build"):
        build(BuildRequest(model_dir=checkpoint, output_path=bundle, family="hstu",
                           task=manifest["task"], precision=precision,
                           max_batch_size=manifest["max_batch_size"],
                           tensor_parallel_size=manifest["tensor_parallel_size"],
                           max_sequence_length=config["max_sequence_length"]))
    settings = {"initial_history": initial, "candidate_item_ids": candidates,
                "fork_item_ids": [ITEM_IDS[5], ITEM_IDS[6]],
                "num_steps": decode_steps, "cache_max_bytes": 64 * 1024 * 1024, **thresholds}
    if any(table["role"] == "action" for table in config["embedding_tables"]):
        settings["append_action_id"] = ACTION_IDS[4]
    input_path, output_path = tmp_path / "session-trace.json", tmp_path / "session-audit.json"
    input_path.write_text(json.dumps(settings, indent=2) + "\n")
    command = [str(binary), "--bundle", str(bundle), "--runtime-root", str(runtime_root),
               "--input-json", str(input_path), "--output-json", str(output_path)]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(filter(None, (
        str(runtime_root), environment.get("LD_LIBRARY_PATH", ""),
    )))
    with evidence_stage("native"):
        completed = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=300)
    log = tmp_path / "session-native.log"
    log.write_text(completed.stdout + completed.stderr)
    record_evidence("commands", {"argv": command, "returncode": completed.returncode})
    record_evidence("inputs", {"trace": input_path})
    record_evidence("native", {"audit": output_path, "log": log})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    audit = json.loads(output_path.read_text())
    assert "error" not in audit, audit.get("error")
    receipt = {"reference": reference_receipt(source), "case": case_name,
               "thresholds": thresholds, "comparisons": []}
    with evidence_stage("reference"):
        for row in audit["steps"]:
            request = deepcopy(row["request"])
            reference = run_reference(checkpoint, request, upstream_root=source, precision=precision)
            assert len(reference["sequences"]) == 1
            expected = reference["sequences"][0]
            for path in ("actual", "baseline"):
                receipt["comparisons"].extend(_check_result(
                    row[path]["sequences"][0], expected, thresholds, f"{row['name']}/{path}"))
            assert row["baseline"]["sequences"][0]["cache"]["source"] == "disabled"
            _check_result(row["actual"]["sequences"][0], row["baseline"]["sequences"][0],
                          thresholds, f"{row['name']}/session-vs-full")
    with evidence_stage("compare"):
        _verify_session_transitions(audit, initial, candidates, config)
    parity_path = tmp_path / "session-parity.json"
    parity_path.write_text(json.dumps(receipt, indent=2) + "\n")
    record_evidence("reference", {"receipt": parity_path})
