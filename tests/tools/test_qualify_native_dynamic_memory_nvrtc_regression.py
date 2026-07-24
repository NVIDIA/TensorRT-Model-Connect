# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "qualify_native_dynamic_memory_nvrtc_regression.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_nvrtc_regression_qualifier", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


qualifier = _load_module()
pytestmark = pytest.mark.dynamic_memory


def _runtime(pid: int = 101) -> dict:
    return {
        "pid": pid,
        "device": 0,
        "device_name": "NVIDIA GB300",
        "sm": 103,
        "cuda_runtime_version": 13030,
        "cuda_driver_api_version": 13030,
        "cudnn_backend_version": 92000,
        "cudnn_frontend_revision": qualifier.FRONTEND_REVISION,
        "nvrtc_major": 13,
        "nvrtc_minor": 0,
        "nvrtc_dladdr_path": "/forced/libnvrtc.so.13",
    }


def _legacy_payload(pid: int = 101) -> dict:
    return {
        "schema_version": qualifier.PROBE_SCHEMA,
        "mode": "legacy",
        "probe_passed": True,
        "shape": dict(qualifier.EXPECTED_SHAPE),
        "runtime": _runtime(pid),
        "graph_contract": {
            "generate_stats": False,
            "optional_logit_max": True,
            "optional_score_sum_exp": True,
            "output_context": True,
            "output_log_sum_exp": False,
            "serialized_contract_bytes": 100,
            "serialized_contains_legacy_logit_max": True,
            "serialized_contains_legacy_score_sum_exp": True,
            "serialized_contains_log_sum_exp": False,
        },
        "result": {
            "plan_count": 3,
            "candidate_index": 0,
            "candidate_plan": qualifier.EXPECTED_LEGACY_PLAN,
            "candidate_build_succeeded": False,
            "candidate_build_error_code": 6,
            "candidate_build_message": (
                "compilationResult != NVRTC_SUCCESS "
                "CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED"
            ),
            "expected_nvrtc_failure_observed": True,
            "fallback_plan_selected": False,
            "graph_executed": False,
        },
    }


def _lse_payload(pid: int = 102) -> dict:
    return {
        "schema_version": qualifier.PROBE_SCHEMA,
        "mode": "lse",
        "probe_passed": True,
        "shape": dict(qualifier.EXPECTED_SHAPE),
        "runtime": _runtime(pid),
        "graph_contract": {
            "generate_stats": True,
            "optional_logit_max": False,
            "optional_score_sum_exp": False,
            "output_context": True,
            "output_log_sum_exp": True,
            "serialized_contract_bytes": 90,
            "serialized_contains_legacy_logit_max": False,
            "serialized_contains_legacy_score_sum_exp": False,
            "serialized_contains_log_sum_exp": True,
        },
        "result": {
            "selected_plan": "eng10_k24=7",
            "workspace_bytes": 0,
            "graph_build_succeeded": True,
            "graph_executed": True,
            "device_synchronize_succeeded": True,
            "finite_lse_observed": True,
            "legacy_optional_outputs_bound": False,
        },
    }


def test_exact_legacy_and_lse_payloads_validate() -> None:
    assert qualifier.validate_probe_payload(
        _legacy_payload(), mode="legacy"
    ) == {
        "graph_contract": True,
        "expected_failure": True,
        "no_fallback": True,
    }
    assert qualifier.validate_probe_payload(
        _lse_payload(), mode="lse"
    ) == {
        "graph_contract": True,
        "build_and_execute": True,
        "legacy_outputs_absent": True,
    }


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda value: value["runtime"].update(nvrtc_minor=3),
            "runtime.nvrtc_minor",
        ),
        (
            lambda value: value["shape"].update(history_rows=1024),
            "geometry drifted",
        ),
        (
            lambda value: value["result"].update(
                candidate_build_succeeded=True
            ),
            "exact optional-output",
        ),
        (
            lambda value: value["result"].update(
                candidate_plan="eng1_k24=35"
            ),
            "exact optional-output",
        ),
        (
            lambda value: value["result"].update(
                candidate_build_message="CUDNN_STATUS_NOT_SUPPORTED"
            ),
            "exact optional-output",
        ),
        (
            lambda value: value["result"].update(
                fallback_plan_selected=True
            ),
            "exact optional-output",
        ),
    ],
)
def test_legacy_payload_fails_closed_on_evidence_drift(
    mutation, error: str
) -> None:
    payload = _legacy_payload()
    mutation(payload)
    with pytest.raises(qualifier.QualificationError, match=error):
        qualifier.validate_probe_payload(payload, mode="legacy")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["graph_contract"].update(
            optional_logit_max=True
        ),
        lambda value: value["graph_contract"].update(
            serialized_contains_legacy_score_sum_exp=True
        ),
        lambda value: value["graph_contract"].update(
            serialized_contains_log_sum_exp=False
        ),
        lambda value: value["result"].update(graph_executed=False),
        lambda value: value["result"].update(
            legacy_optional_outputs_bound=True
        ),
        lambda value: value["result"].update(finite_lse_observed=False),
    ],
)
def test_lse_payload_rejects_legacy_outputs_or_nonexecution(
    mutation,
) -> None:
    payload = _lse_payload()
    mutation(payload)
    with pytest.raises(
        qualifier.QualificationError, match="standard-LSE"
    ):
        qualifier.validate_probe_payload(payload, mode="lse")


def _mapping_rows() -> tuple[list[dict], dict, dict]:
    nvrtc = {
        "device_major": 0,
        "device_minor": 84,
        "inode": 11,
    }
    builtins = {
        "device_major": 0,
        "device_minor": 84,
        "inode": 12,
    }

    def row(
        path: str,
        inode: int,
        *,
        permissions: str = "r-xp",
        device_minor: int = 84,
    ) -> dict:
        return {
            "address": f"{inode:x}-{inode + 1:x}",
            "permissions": permissions,
            "offset": "00000000",
            "device_major": 0,
            "device_minor": device_minor,
            "inode": inode,
            "path": path,
        }

    rows = [
        row("/forced/libnvrtc.so.13", 11),
        row(
            "/forced/libnvrtc.so.13",
            11,
            permissions="r--p",
        ),
        row("/forced/libnvrtc-builtins.so.13.0", 12),
        row("/cuda/libcudart.so.13.3.33", 13),
        row("/cudnn/libcudnn.so.9.20.0", 14),
        row("/driver/libcuda.so.580.105.08", 15),
    ]
    return rows, nvrtc, builtins


def test_live_mapping_validation_pins_pair_and_complete_stack() -> None:
    rows, nvrtc, builtins = _mapping_rows()
    result = qualifier.validate_runtime_mapping_set(
        rows, nvrtc_identity=nvrtc, builtins_identity=builtins
    )
    assert result["pinned_nvrtc"]["inode"] == 11
    assert result["pinned_nvrtc_builtins"]["inode"] == 12
    assert all(
        len(result["runtime_libraries"][label]) == 1
        for label in (
            "cuda_runtime",
            "cudnn",
            "cuda_driver",
            "nvrtc",
            "nvrtc_builtins",
        )
    )


def test_live_mapping_validation_rejects_competing_nvrtc() -> None:
    rows, nvrtc, builtins = _mapping_rows()
    rows.append(
        {
            "address": "aa-bb",
            "permissions": "r-xp",
            "offset": "00000000",
            "device_major": 0,
            "device_minor": 84,
            "inode": 99,
            "path": "/other/libnvrtc.so.13.3.33",
        }
    )
    with pytest.raises(
        qualifier.QualificationError, match="competing or missing NVRTC"
    ):
        qualifier.validate_runtime_mapping_set(
            rows, nvrtc_identity=nvrtc, builtins_identity=builtins
        )


def test_live_mapping_validation_rejects_missing_builtins() -> None:
    rows, nvrtc, builtins = _mapping_rows()
    rows = [
        row
        for row in rows
        if "libnvrtc-builtins" not in row["path"]
    ]
    with pytest.raises(
        qualifier.QualificationError,
        match="exactly one pinned NVRTC builtins",
    ):
        qualifier.validate_runtime_mapping_set(
            rows, nvrtc_identity=nvrtc, builtins_identity=builtins
        )


def test_runtime_tuple_comparison_ignores_pid_but_not_stack() -> None:
    left = _runtime(101)
    right = _runtime(202)
    assert qualifier._runtime_comparison(left, right)
    right["cuda_runtime_version"] = 13020
    assert not qualifier._runtime_comparison(left, right)


def _graph_artifact(mode: str) -> dict:
    tensors = {
        "1": {"name": "query", "dim": [1, 16, 1, 128]},
        "2": {
            "name": "key_history_token_major",
            "dim": [1, 8, 512, 128],
        },
        "3": {
            "name": "value_history_token_major",
            "dim": [1, 8, 512, 128],
        },
        "4": {"name": "context", "dim": [1, 16, 1, 128]},
        "5": {"name": "sequence_length_q", "dim": [1, 1, 1, 1]},
        "6": {
            "name": "sequence_length_history",
            "dim": [1, 1, 1, 1],
        },
    }
    if mode == "legacy":
        tensors.update(
            {
                "7": {
                    "name": "legacy_logit_max",
                    "dim": [1, 16, 1, 1],
                },
                "8": {
                    "name": "legacy_score_sum_exp",
                    "dim": [1, 16, 1, 1],
                },
            }
        )
        node = {
            "tag": "SDPA",
            "name": "trtmc_legacy_optional_output_history_sdpa",
            "generate_stats": False,
            "padding_mask": True,
            "inputs": {
                "Q": 1,
                "K": 2,
                "V": 3,
                "SEQ_LEN_Q": 5,
                "SEQ_LEN_KV": 6,
            },
            "outputs": {"Max": 7, "O": 4, "Sum_exp": 8},
        }
    else:
        tensors["9"] = {
            "name": "log_sum_exp",
            "dim": [1, 16, 1, 1],
        }
        node = {
            "tag": "SDPA",
            "name": "trtmc_standard_lse_history_sdpa",
            "generate_stats": True,
            "padding_mask": True,
            "inputs": {
                "Q": 1,
                "K": 2,
                "V": 3,
                "SEQ_LEN_Q": 5,
                "SEQ_LEN_KV": 6,
            },
            "outputs": {"O": 4, "Stats": 9},
        }
    return {
        "json_version": "1.0",
        "cudnn_backend_version": "9.20.0",
        "cudnn_frontend_version": 12100,
        "nodes": [node],
        "tensors": tensors,
    }


def test_graph_artifacts_are_independently_validated(
    tmp_path: Path,
) -> None:
    for mode in ("legacy", "lse"):
        path = tmp_path / f"{mode}.json"
        path.write_text(
            json.dumps(_graph_artifact(mode)), encoding="utf-8"
        )
        result = qualifier.validate_graph_artifact(path, mode=mode)
        assert result["independent_contract_validation"] is True
    assert result["legacy_outputs_absent"] is True


def test_lse_graph_artifact_rejects_hidden_legacy_tensor(
    tmp_path: Path,
) -> None:
    graph = _graph_artifact("lse")
    graph["tensors"]["7"] = {
        "name": "legacy_logit_max",
        "dim": [1, 16, 1, 1],
    }
    path = tmp_path / "lse.json"
    path.write_text(json.dumps(graph), encoding="utf-8")
    with pytest.raises(
        qualifier.QualificationError, match="retained legacy"
    ):
        qualifier.validate_graph_artifact(path, mode="lse")


def test_legacy_graph_artifact_requires_exact_optional_outputs(
    tmp_path: Path,
) -> None:
    graph = _graph_artifact("legacy")
    graph["nodes"][0]["outputs"] = {"O": 4, "Stats": 7}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(graph), encoding="utf-8")
    with pytest.raises(
        qualifier.QualificationError, match="exactly Max/O/Sum_exp"
    ):
        qualifier.validate_graph_artifact(path, mode="legacy")


def test_cache_snapshot_rejects_symlink(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "escape").symlink_to(tmp_path)
    with pytest.raises(
        qualifier.QualificationError, match="contains a symlink"
    ):
        qualifier._cache_snapshot(cache)


def _write_elf(path: Path) -> None:
    path.write_bytes(b"\x7fELF" + b"\0" * 64)


def _qualify_args(tmp_path: Path) -> Namespace:
    probe = tmp_path / "probe"
    nvrtc = tmp_path / "libnvrtc.so.13"
    builtins = tmp_path / "libnvrtc-builtins.so.13.0"
    for path in (probe, nvrtc, builtins):
        _write_elf(path)
    return Namespace(
        probe=probe,
        nvrtc=nvrtc,
        nvrtc_builtins=builtins,
        output=tmp_path / "qualification-receipt.json",
        timeout_seconds=10.0,
    )


def _mock_run(mode: str) -> dict:
    return {
        "probe_receipt": (
            _legacy_payload(101)
            if mode == "legacy"
            else _lse_payload(202)
        )
    }


def test_receipt_passes_component_but_not_dirty_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {
        "git_head": "a" * 40,
        "source_state_sha256": "b" * 64,
        "exact_head_gate_satisfied": False,
    }
    monkeypatch.setattr(
        qualifier, "_source_snapshot", lambda *_args, **_kwargs: dict(source)
    )
    monkeypatch.setattr(
        qualifier,
        "_run_mode",
        lambda *, mode, **_kwargs: _mock_run(mode),
    )
    monkeypatch.setattr(
        qualifier,
        "_driver_version",
        lambda: {"version": "580.105.08"},
    )
    args = _qualify_args(tmp_path)

    report = qualifier.qualify(args)

    assert report["passed"] is True
    assert report["promotion_eligible"] is False
    assert report["qualification_gates"]["clean_exact_head"] is False
    persisted = json.loads(args.output.read_text(encoding="utf-8"))
    assert persisted["status"] == "completed"
    assert persisted["receipt_sha256"]


def test_receipt_fails_closed_and_persists_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {
        "git_head": "a" * 40,
        "source_state_sha256": "b" * 64,
        "exact_head_gate_satisfied": True,
    }
    changed = {
        **source,
        "source_state_sha256": "c" * 64,
    }
    snapshots = iter((source, changed))
    monkeypatch.setattr(
        qualifier,
        "_source_snapshot",
        lambda *_args, **_kwargs: copy.deepcopy(next(snapshots)),
    )
    monkeypatch.setattr(
        qualifier,
        "_run_mode",
        lambda *, mode, **_kwargs: _mock_run(mode),
    )
    monkeypatch.setattr(
        qualifier,
        "_driver_version",
        lambda: {"version": "580.105.08"},
    )
    args = _qualify_args(tmp_path)

    with pytest.raises(
        qualifier.QualificationError, match="mandatory NVRTC"
    ):
        qualifier.qualify(args)

    persisted = json.loads(args.output.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["passed"] is False
    assert persisted["source_state_unchanged"] is False


def test_runtime_failure_persists_a_failed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {
        "git_head": "a" * 40,
        "source_state_sha256": "b" * 64,
        "exact_head_gate_satisfied": True,
    }
    monkeypatch.setattr(
        qualifier, "_source_snapshot", lambda *_args, **_kwargs: dict(source)
    )
    monkeypatch.setattr(
        qualifier,
        "_run_mode",
        lambda **_kwargs: (_ for _ in ()).throw(
            qualifier.QualificationError("legacy execution drift")
        ),
    )
    args = _qualify_args(tmp_path)

    with pytest.raises(
        qualifier.QualificationError, match="execution drift"
    ):
        qualifier.qualify(args)

    persisted = json.loads(args.output.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["passed"] is False
    assert persisted["error"] == "legacy execution drift"
