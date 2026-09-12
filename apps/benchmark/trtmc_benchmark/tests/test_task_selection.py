# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from trtmc_benchmark import cli, worker
from trtmc_benchmark.catalog import ManifestCatalog, resolve_case
from trtmc_benchmark.types import BenchmarkError


def model(tmp_path):
    manifest = tmp_path / "families/example/tests/manifests/example.json"
    manifest.parent.mkdir(parents=True)
    payload = {
        "name": "example", "hf_id": "example/checkpoint", "bundle": "example.bundle",
        "family": "example", "task": "text_to_pooled_features", "precision": "fp32",
        "testcases": [
            {"name": "pooled", "prompt": "hello"},
            {"name": "tokens", "selected_task": "text_to_token_features",
             "inputs": {"token_ids": [7, 9]}, "config": {"custom_scale": 0.0}},
            {"name": "embedding", "selected_task": "text_to_embedding", "prompt": "hello"},
        ],
    }
    manifest.write_text(json.dumps(payload))
    return ManifestCatalog(tmp_path / "families").resolve(str(manifest))


def test_family_owned_testcase_selects_task_without_changing_bundle_identity(tmp_path):
    descriptor = model(tmp_path)
    bundle = tmp_path / "example.bundle"
    default = resolve_case(descriptor, bundle)
    tokens = resolve_case(descriptor, bundle, case_name="tokens").with_values(runtime_root=tmp_path)
    assert default.selected_task is None
    assert default.effective_task == descriptor.task == "text_to_pooled_features"
    assert tokens.operation == "encode"
    assert tokens.effective_task == "text_to_token_features"
    assert tokens.request == {"token_ids": [7, 9], "config": {"custom_scale": 0.0}}
    wire = tokens.worker_request()
    assert wire["expected_task"] == "text_to_pooled_features"
    assert wire["selected_task"] == "text_to_token_features"
    assert wire["expected_family"] == "example"
    assert "selected_task" not in wire["request"]
    assert tokens.to_json()["model"]["task"] == "text_to_pooled_features"
    assert tokens.to_json()["selected_task"] == "text_to_token_features"
    assert tokens.with_values(name="renamed").selected_task == tokens.selected_task


def test_explicit_selection_uses_selected_default_operation_and_rejects_alias(tmp_path):
    descriptor = model(tmp_path)
    selected = resolve_case(descriptor, tmp_path / "bundle", selected_task="text_to_embedding")
    assert selected.operation == "embed"
    assert selected.request == {"prompt": "hello"}
    with pytest.raises(BenchmarkError, match="explicitly selected"):
        resolve_case(descriptor, tmp_path / "bundle", selected_task="text_to_embedding", operation="encode")
    embedded = replace(descriptor, task="text_to_embedding")
    encoded = resolve_case(embedded, tmp_path / "bundle", operation="encode")
    assert encoded.selected_task == "text_to_pooled_features"
    assert encoded.model.task == "text_to_embedding"


@pytest.mark.parametrize("selection", [None, "", " ", " text_to_embedding", 1, False, [], "unknown_task"])
def test_bad_family_task_declaration_fails_before_execution(tmp_path, selection):
    descriptor = model(tmp_path)
    changed = replace(descriptor, testcases=({"name": "bad", "prompt": "x", "selected_task": selection},))
    with pytest.raises(BenchmarkError):
        resolve_case(changed, tmp_path / "bundle")


def test_catalog_uses_default_case_selection_but_retains_physical_primary(tmp_path):
    descriptor = model(tmp_path)
    payload = json.loads(descriptor.manifest_path.read_text())
    payload["task"] = "image_text_to_embedding"
    payload["testcases"][0]["selected_task"] = "text_to_pooled_features"
    descriptor.manifest_path.write_text(json.dumps(payload))
    entry, = ManifestCatalog(tmp_path / "families").entries()
    assert entry.status == "ready" and entry.operation == "encode"
    assert entry.model.task == "image_text_to_embedding"


def test_cli_task_flag_reaches_dry_run_without_entering_config(tmp_path, capsys):
    descriptor = model(tmp_path)
    bundle = tmp_path / "example.bundle"
    bundle.write_bytes(b"not loaded during dry run")
    assert cli.main(["run", "--model", str(descriptor.manifest_path), "--bundle", str(bundle),
                     "--manifest-root", str(tmp_path / "families"),
                     "--task", "text_to_embedding", "--dry-run"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["selected_task"] == "text_to_embedding"
    assert rows[0]["operation"] == "embed"
    assert rows[0]["model"]["task"] == "text_to_pooled_features"
    assert "selected_task" not in rows[0]["request"]


@pytest.mark.parametrize("returned", [None, "text_to_pooled_features", "text_to_token_features"])
def test_worker_receipt_must_match_explicit_task(tmp_path, monkeypatch, returned):
    descriptor = model(tmp_path)
    case = resolve_case(descriptor, tmp_path / "bundle", case_name="tokens").with_values(runtime_root=tmp_path)

    def invoke(command, **_kwargs):
        request = json.loads(Path(command[command.index("--request") + 1]).read_text())
        assert request["selected_task"] == "text_to_token_features"
        result = {
            "schema_version": "trtmc.benchmark-worker-result/v2", "status": "completed",
            "case_name": case.name, "operation": case.operation,
            "timing_scope": case.measurement.timing_scope,
            "asset_loading_included": case.measurement.asset_loading_included,
            "observations": [{} for _ in range(case.measurement.iterations)],
        }
        if returned is not None:
            result["selected_task"] = returned
        Path(command[command.index("--output") + 1]).write_text(json.dumps(result))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(worker.subprocess, "run", invoke)
    if returned == "text_to_token_features":
        assert worker.run_worker(case, tmp_path, Path("unused-worker"))["selected_task"] == returned
    else:
        with pytest.raises(BenchmarkError, match="selected_task"):
            worker.run_worker(case, tmp_path, Path("unused-worker"))
