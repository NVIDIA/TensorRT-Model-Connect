# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeated native requests qualify history reuse against the original oracle."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from families.hstu.tests.fixtures import (
    ACTION_IDS, ITEM_IDS, SEED, context_length, make_checkpoint, sample_request, token_count,
)


def _identity(subject="user-a", **changes):
    return {"subject_id": subject, "feature_version": "features-v1", "history_epoch": "epoch-1", **changes}


def _keyed(sequence, subject="user-a", **changes):
    result = deepcopy(sequence)
    result["cache"] = _identity(subject, **changes)
    return result


def _history(sequence):
    return context_length(sequence) + len(sequence["history_item_ids"]) + len(sequence.get("history_action_ids", ()))


def _append(sequence):
    result = deepcopy(sequence)
    result["history_item_ids"].append(ITEM_IDS[5])
    if "history_action_ids" in result:
        result["history_action_ids"].append(ACTION_IDS[4])
    return result


def _request_step(name, *sequences):
    return {"name": name, "request": {"sequences": deepcopy(list(sequences))}}


def _trace_binary():
    explicit = os.environ.get("TRTMC_HSTU_CACHE_BINARY")
    build = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    generic = os.environ.get("TRTMC_BINARY")
    if explicit:
        binary = Path(explicit)
    elif build:
        binary = Path(build) / "hstu_cache_sequence_runner"
    elif generic:
        binary = Path(generic).with_name("hstu_cache_sequence_runner")
    else:
        raise AssertionError("selected cache E2E requires TRTMC_NATIVE_BUILD_DIR or TRTMC_BINARY")
    assert binary.is_file() and os.access(binary, os.X_OK), f"native cache trace driver is missing: {binary}"
    return binary


def _run_trace(tmp_path, manifest, case, steps):
    import torch
    from safetensors.numpy import load_file
    from tensorrt_model_connect import BuildRequest, build
    from families.hstu.tests.environment import reference_source
    from families.hstu.tests.reference import reference_receipt, run_reference
    from tools.e2e_evidence import evidence_stage, record_evidence

    assert torch.cuda.is_available(), "selected cache E2E requires CUDA"
    runtime_root = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    binary = _trace_binary()
    precision = manifest["precision"]
    source = reference_source()
    checkpoint = tmp_path / "checkpoint"
    actual_config = make_checkpoint(checkpoint, **case["config_overrides"])
    assert actual_config["enable_history_cache"]
    arrays_path = checkpoint / "checkpoint-arrays.npz"
    np.savez_compressed(arrays_path, **load_file(str(checkpoint / "model.safetensors")))
    record_evidence("checkpoint", {"config": actual_config, "seed": SEED,
                                   "canonical_arrays": arrays_path,
                                   "weights": checkpoint / "model.safetensors",
                                   "qualification": "seeded cache correctness; not trained model quality"})
    record_evidence("reference_source", reference_receipt(source))
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(BuildRequest(
            model_dir=checkpoint, output_path=bundle, family="hstu", task=manifest["task"],
            precision=precision, max_batch_size=manifest["max_batch_size"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
            max_sequence_length=actual_config["max_sequence_length"],
        ))
    trace = {
        "cache": {"max_bytes": 32 * 1024 * 1024, "max_entries": 32,
                  "storage_max_bytes": 32 * 1024 * 1024, "write_through": True},
        "steps": steps,
    }
    input_path, output_path = tmp_path / "trace.json", tmp_path / "audit.json"
    input_path.write_text(json.dumps(trace, indent=2) + "\n")
    command = [str(binary), "--bundle", str(bundle), "--runtime-root", str(runtime_root),
               "--input-json", str(input_path), "--output-json", str(output_path)]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(filter(None, (
        str(runtime_root), environment.get("LD_LIBRARY_PATH", ""),
    )))
    with evidence_stage("native"):
        completed = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=180)
    (tmp_path / "native.log").write_text(completed.stdout + completed.stderr)
    record_evidence("commands", {"argv": command, "returncode": completed.returncode})
    record_evidence("inputs", {"trace": input_path})
    record_evidence("native", {"log": tmp_path / "native.log", "audit": output_path})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    audit = json.loads(output_path.read_text())
    threshold_path = Path(__file__).parent / "thresholds" / f"{case['name']}.json"
    thresholds = json.loads(threshold_path.read_text())["threshold_overrides"]
    record_evidence("thresholds", thresholds)
    receipt = {"source": reference_receipt(source), "command": command,
               "thresholds": thresholds, "comparisons": []}
    assert len(audit["steps"]) == len(steps)
    for step, result in zip(steps, audit["steps"]):
        assert result["name"] == step["name"]
        if "action" in step:
            continue
        if step.get("expect_error"):
            assert result.get("error"), result
            assert result.get("baseline_error"), result
            assert result["stats_after"]["publications"] == result["stats_before"]["publications"]
            continue
        assert "error" not in result and "baseline_error" not in result, result
        reference = run_reference(checkpoint, step["request"], upstream_root=source, precision=precision)
        for path in ("actual", "baseline"):
            assert len(result[path]["sequences"]) == len(reference["sequences"])
            for index, (actual, expected) in enumerate(zip(result[path]["sequences"], reference["sequences"])):
                for field in ("candidate_item_ids", "num_candidates", "embedding_dim", "output_dim", "sequence_length"):
                    assert actual[field] == expected[field], (step["name"], path, index, field)
                for field in ("logits", "scores", "embeddings", "sequence_embeddings"):
                    left, right = np.asarray(actual[field]), np.asarray(expected.get(field, []))
                    assert left.shape == right.shape and np.isfinite(left).all()
                    np.testing.assert_allclose(left, right, **thresholds,
                                               err_msg=f"{step['name']}/{path}/{index}/{field}")
                    receipt["comparisons"].append({
                        "step": step["name"], "path": path, "sequence": index, "field": field,
                        "max_absolute_error": float(np.abs(left - right).max()) if left.size else 0.0,
                    })
        for actual, baseline in zip(result["actual"]["sequences"], result["baseline"]["sequences"]):
            assert baseline["cache"]["source"] == "disabled"
            for field in ("logits", "scores", "embeddings", "sequence_embeddings"):
                np.testing.assert_allclose(actual[field], baseline[field], **thresholds,
                                           err_msg=f"{step['name']}/cached-vs-disabled/{field}")
    (tmp_path / "parity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    record_evidence("reference", {"receipt": tmp_path / "parity.json"})
    return {row["name"]: row for row in audit["steps"]}, actual_config


def _cache(rows, name, index=0):
    return rows[name]["actual"]["sequences"][index]["cache"]


def _cached_history_trace(tmp_path, manifest, case):
    from families.hstu.tests.fixtures import tiny_config

    config = case["config_overrides"]
    sample = sample_request(tiny_config(**config))["sequences"]
    first, second = _keyed(sample[0]), _keyed(sample[1], "user-b")
    changed_candidates = deepcopy(first)
    changed_candidates["candidate_item_ids"][-1] = ITEM_IDS[-1]
    appended = _append(first)
    corrected = deepcopy(appended)
    corrected["history_item_ids"][0] = ITEM_IDS[-2]
    truncated = deepcopy(corrected)
    truncated["history_item_ids"].pop()
    truncated["history_action_ids"].pop()
    epoch = _keyed(truncated, history_epoch="epoch-2")
    feature = _keyed(epoch, feature_version="features-v2", history_epoch="epoch-2")
    readonly = _keyed(first, "read-only", read_only=True)
    invalid = _keyed(second, "invalid")
    invalid["history_item_ids"][0] = 2**62
    steps = [
        _request_step("cold", first), _request_step("exact", first),
        _request_step("candidate-change", changed_candidates), _request_step("candidate-restored", first),
        _request_step("append", appended), _request_step("mixed-hit-miss", appended, second),
        _request_step("mixed-exact", appended, second),
        _request_step("read-only", readonly), _request_step("read-only-again", readonly),
        _request_step("correction", corrected), _request_step("truncation", truncated),
        _request_step("epoch", epoch), _request_step("feature", feature),
        {"name": "clear-memory", "action": "clear_memory"},
        _request_step("storage", feature),
        {"name": "invalidate", "action": "invalidate", "identity": feature["cache"]},
        _request_step("invalidated", feature),
        {**_request_step("failure", _append(feature), invalid), "expect_error": True},
        _request_step("after-failure", feature),
    ]
    rows, _ = _run_trace(tmp_path, manifest, case, steps)
    assert _cache(rows, "cold")["source"] == "miss"
    assert _cache(rows, "cold")["published"]
    for name in ("exact", "candidate-change", "candidate-restored"):
        report = _cache(rows, name)
        assert report["reused_history_tokens"] == _history(first)
        assert report["computed_tokens"] == len(first["candidate_item_ids"])
    assert _cache(rows, "append")["reused_history_tokens"] == _history(first)
    assert _cache(rows, "append")["reason"] == "append_history"
    assert _cache(rows, "mixed-hit-miss")["reused_history_tokens"] == _history(appended)
    assert _cache(rows, "mixed-hit-miss", 1)["reused_history_tokens"] == 0
    assert _cache(rows, "mixed-exact", 1)["reused_history_tokens"] == _history(second)
    for name in ("read-only", "read-only-again"):
        assert not _cache(rows, name)["published"]
        assert _cache(rows, name)["reused_history_tokens"] == 0
    for name in ("correction", "truncation", "epoch", "feature", "invalidated"):
        assert _cache(rows, name)["reused_history_tokens"] == 0
    assert _cache(rows, "correction")["reason"] == "history_changed"
    assert _cache(rows, "truncation")["reason"] == "history_truncated"
    assert _cache(rows, "storage")["source"] == "storage"
    assert _cache(rows, "storage")["reused_history_tokens"] == _history(feature)
    assert _cache(rows, "after-failure")["reused_history_tokens"] == _history(feature)
    assert rows["after-failure"]["stats_after"]["storage_hits"] >= 1


def _cache_reuse_respects_attention_semantics(tmp_path, manifest, case):
    from families.hstu.tests.fixtures import tiny_config

    variant = case["cache_scenario"]
    overrides = case["config_overrides"]
    config = tiny_config(**overrides)
    first = _keyed(sample_request(config)["sequences"][0])
    appended = _append(first)
    if config["time_buckets"]:
        appended["token_timestamps"] = [1_700_000_000 + 71 * i * i for i in range(token_count(appended))]
    steps = [_request_step("cold", first), _request_step("exact", first), _request_step("append", appended)]
    rows, _ = _run_trace(tmp_path, manifest, case, steps)
    if variant == "retrieval":
        assert _cache(rows, "exact")["computed_tokens"] == 0
        assert _cache(rows, "exact")["reused_history_tokens"] == _history(first)
        assert _cache(rows, "append")["reused_history_tokens"] == _history(first)
    else:
        assert _cache(rows, "append")["reused_history_tokens"] == 0
        if variant != "noncausal":
            assert _cache(rows, "exact")["reused_history_tokens"] == _history(first)
        else:
            assert not _cache(rows, "cold")["published"]
            assert not _cache(rows, "exact")["published"]


def _cached_history_hundreds_and_256_candidates(tmp_path, manifest, case):
    from families.hstu.tests.fixtures import tiny_config

    overrides = case["config_overrides"]
    sample = sample_request(tiny_config(**overrides), **case["fixture"])["sequences"]
    first, second = _keyed(sample[0]), _keyed(sample[1], "user-b")
    appended = _append(first)
    candidates = deepcopy(first)
    candidates["candidate_item_ids"] = list(reversed(candidates["candidate_item_ids"]))
    steps = [_request_step("cold", first, second), _request_step("exact", first, second),
             _request_step("candidates", candidates, second), _request_step("append", appended, second)]
    rows, _ = _run_trace(tmp_path, manifest, case, steps)
    for index, sequence in enumerate((first, second)):
        assert _cache(rows, "exact", index)["reused_history_tokens"] == _history(sequence)
        assert _cache(rows, "exact", index)["computed_tokens"] == 256
    assert _cache(rows, "append")["reused_history_tokens"] == _history(first)


CACHE_CASES = (
    "hstu-cache-lifecycle-fp32", "hstu-cache-lifecycle-fp16", "hstu-cache-lifecycle-bf16",
    "hstu-cache-contextual-fp32", "hstu-cache-timestamp-fp32", "hstu-cache-default-scaling-fp32",
    "hstu-cache-noncausal-fp32", "hstu-cache-retrieval-fp32", "hstu-cache-hundreds-bf16",
)


def run_cache_case(case_name: str, tmp_path: Path) -> None:
    """Run one explicitly selected family manifest case without a second selector."""
    if case_name not in CACHE_CASES:
        raise ValueError(f"unknown HSTU cache testcase: {case_name}")
    manifest_path = Path(__file__).parent / "manifests" / f"{case_name}.json"
    manifest = json.loads(manifest_path.read_text())
    case = next(value for value in manifest["testcases"] if value["name"] == case_name)
    if case["cache_scenario"] == "lifecycle":
        _cached_history_trace(tmp_path, manifest, case)
    elif case["cache_scenario"] == "hundreds":
        _cached_history_hundreds_and_256_candidates(tmp_path, manifest, case)
    else:
        _cache_reuse_respects_attention_semantics(tmp_path, manifest, case)
