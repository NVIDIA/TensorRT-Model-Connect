# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "qualify_native_dynamic_memory_perf.py"
SPEC = importlib.util.spec_from_file_location(
    "qualify_native_dynamic_memory_perf", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
perf = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = perf
SPEC.loader.exec_module(perf)

pytestmark = [pytest.mark.unit, pytest.mark.dynamic_memory]
SOURCE_SHA = "1" * 64
GIT_HEAD = "2" * 40
TOOLCHAIN_SHA = "3" * 64
ENVIRONMENT_SHA = "4" * 64
SHORT_REQUEST_SHA = "5" * 64
MEDIUM_REQUEST_SHA = "6" * 64
RUNTIME_STACK = {
    "schema": 1,
    "sm": "sm103",
    "tensorrt": "11.2.0.113",
    "cuda_runtime": "13.3",
    "cudnn_backend": "9.20.0",
    "cudnn_frontend_revision":
        "7b9b711c22b6823e87150213ecd8449260db8610",
    "nvrtc": "13.3",
    "driver": "580.105.08",
}
RUNTIME_PLAN = {
    "schema": 1,
    "device": 0,
    "role": "history",
    "hq": 16,
    "hkv": 8,
    "d": 128,
    "C": 128,
    "Sq": 1,
    "T": 512,
    "stats": "lse",
    "heur": "A",
    "plan": "eng10_k24=7",
    "workspace_bytes": 0,
    "cudnn_version": 92000,
}
TOKENIZER_CONTRACT = {
    "schema_version": perf.TOKENIZER_CONTRACT_SCHEMA,
    "tokenizer_json_sha256": "8" * 64,
    "tokenizer_json_bytes": 4096,
    "tokenizer_add_special_tokens": False,
    "tokenizer_special_prefix_ids": [],
    "tokenizer_special_suffix_ids": [],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cuda_jit_cache() -> dict:
    empty_sha = perf._canonical_sha([])
    return {
        "path": "/tmp/trtmc-cuda-cache",
        "path_source": "CUDA_CACHE_PATH",
        "cuda_cache_path_env": "/tmp/trtmc-cuda-cache",
        "cuda_cache_disable_env": None,
        "enabled": True,
        "initial_state": "cold",
        "worker_started_ns": 200,
        "worker_finished_ns": 300,
        "before": {
            "captured_at_ns": 100,
            "exists": False,
            "is_directory": False,
            "entry_count": 0,
            "file_count": 0,
            "total_bytes": 0,
            "metadata_sha256": empty_sha,
        },
        "after": {
            "captured_at_ns": 400,
            "exists": True,
            "is_directory": True,
            "entry_count": 1,
            "file_count": 1,
            "total_bytes": 4096,
            "metadata_sha256": "7" * 64,
        },
    }


def _runtime_libraries(root: Path) -> dict:
    directory = root / "runtime-libraries"
    directory.mkdir(parents=True, exist_ok=True)
    nvrtc = directory / "libnvrtc.so.13.3.33"
    builtins = directory / "libnvrtc-builtins.so.13.3"
    nvrtc.write_bytes(b"nvrtc")
    builtins.write_bytes(b"nvrtc-builtins")
    return {
        "directory": str(directory.resolve()),
        "live_nvrtc_version": "13.3",
        "nvrtc": {
            "path": str(nvrtc.resolve()),
            "basename": nvrtc.name,
            "sha256": _sha256(nvrtc),
            "size_bytes": nvrtc.stat().st_size,
        },
        "nvrtc_builtins": {
            "path": str(builtins.resolve()),
            "basename": builtins.name,
            "sha256": _sha256(builtins),
            "size_bytes": builtins.stat().st_size,
        },
    }


def _write_result(
    path: Path,
    *,
    role: str,
    bundle: Path,
    prompt_kind: str,
    prefill_ms: float = 10.0,
    decode_ms: float = 100.0,
    output_tokens: int = 64,
    serialized_plan_bytes: int = 1_000,
    resident_weight_bytes: int = 2_000,
    resident_weight_copy_count: int = 2,
    weight_streaming_active: bool = False,
    fresh_build: bool = True,
    artifact_reused: bool = False,
) -> dict:
    request_sha = (
        SHORT_REQUEST_SHA if prompt_kind == "short" else MEDIUM_REQUEST_SHA
    )
    runtime_attention_plans = (
        [] if role == "exact-head-static-split" else [dict(RUNTIME_PLAN)]
    )
    runtime_stack = (
        None if role == "exact-head-static-split" else dict(RUNTIME_STACK)
    )
    runtime_libraries = (
        None
        if role == "exact-head-static-split"
        else _runtime_libraries(path.parent)
    )
    cuda_jit_cache = _cuda_jit_cache()
    generated_stream = list(range(output_tokens))
    structural_identity = {
        "operation": "generate",
        "prompt_sha256": (
            "9" * 64 if prompt_kind == "short" else "a" * 64
        ),
        "prompt_utf8_bytes": 16 if prompt_kind == "short" else 128,
        "generation": {
            "generation_mode": "ar",
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "num_samples": 1,
            "eos_token_id": 2147483647,
            "use_chat_template": False,
            "stop_on_boxed_answer": False,
            "capture_generated_token_ids": True,
            "max_new_tokens": output_tokens,
        },
        "measurement": {"warmup": 2, "iterations": 3},
    }
    generation_workload = {
        "schema_version": perf.GENERATION_WORKLOAD_SCHEMA,
        "kind": "fixed_length_greedy_ar",
        "structural_identity": structural_identity,
        "structural_identity_sha256": perf._canonical_sha(
            structural_identity
        ),
        "measured_generated_token_ids": [
            generated_stream,
            generated_stream,
            generated_stream,
        ],
        "measured_generated_token_ids_sha256": perf._canonical_sha(
            [generated_stream, generated_stream, generated_stream]
        ),
        "token_stream_repeatable_within_case": True,
    }
    payload = {
        "schema_version": perf.RESULT_SCHEMA,
        "status": "completed",
        "case_name": f"{role}-{prompt_kind}",
        "case_digest": hashlib.sha256(
            f"{role}-{prompt_kind}".encode()
        ).hexdigest(),
        "model_id": "example/model",
        "operation": "generate",
        "timing_scope": "public_pipeline_call_wall",
        "warmup": 2,
        "iterations": 3,
        "observations": [
            {
                "iteration": index,
                "runtime_e2e_wall_ms": prefill_ms + decode_ms,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "output_tokens": output_tokens,
            }
            for index in range(3)
        ],
        "qualification_provenance": {
            "git_head": GIT_HEAD,
            "source_state_sha256": SOURCE_SHA,
            "prebuild_source_state_sha256": SOURCE_SHA,
            "postbuild_source_state_sha256": SOURCE_SHA,
            "bundle_sha256": _sha256(bundle),
            "request_sha256": request_sha,
            "model_revision": "revision-pinned",
            "precision": "bf16",
            "target": "sm103",
            "toolchain_sha256": TOOLCHAIN_SHA,
            "benchmark_environment_sha256": ENVIRONMENT_SHA,
            "bundle_build_id": (
                "static-build"
                if role == "exact-head-static-split"
                else "dynamic-build"
            ),
            "artifact_role": role,
            "fresh_build": fresh_build,
            "artifact_reused": artifact_reused,
            "runtime_attention_plans_sha256": perf._canonical_sha(
                runtime_attention_plans
            ),
            "runtime_stack_sha256": perf._canonical_sha(runtime_stack),
            "runtime_libraries_sha256": perf._canonical_sha(
                runtime_libraries
            ),
            "cuda_jit_cache_sha256": perf._canonical_sha(cuda_jit_cache),
            "generation_workload_sha256": perf._canonical_sha(
                generation_workload
            ),
            "tokenizer_contract_sha256": perf._canonical_sha(
                TOKENIZER_CONTRACT
            ),
        },
        "runtime_memory_receipt": {
            "serialized_plan_bytes": serialized_plan_bytes,
            "resident_weight_bytes": resident_weight_bytes,
            "resident_weight_copy_count": resident_weight_copy_count,
            "weight_streaming_active": weight_streaming_active,
            "measurement_sources": dict(perf._MEASUREMENT_SOURCES),
        },
        "runtime_attention_plans": runtime_attention_plans,
        "runtime_stack": runtime_stack,
        "runtime_libraries": runtime_libraries,
        "cuda_jit_cache": cuda_jit_cache,
        "generation_workload": generation_workload,
        "tokenizer_contract": dict(TOKENIZER_CONTRACT),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _evidence(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    static_bundle = tmp_path / "static.trtfb"
    dynamic_bundle = tmp_path / "dynamic.trtfb"
    static_bundle.write_bytes(b"s" * 1_000)
    dynamic_bundle.write_bytes(b"d" * 1_040)
    paths = {
        "static_bundle": static_bundle,
        "dynamic_bundle": dynamic_bundle,
    }
    for prompt_kind in ("short", "medium"):
        for kind, role, bundle in (
            ("static", "exact-head-static-split", static_bundle),
            ("dynamic", "native-dynamic", dynamic_bundle),
        ):
            path = tmp_path / f"{kind}-{prompt_kind}.json"
            _write_result(
                path,
                role=role,
                bundle=bundle,
                prompt_kind=prompt_kind,
                prefill_ms=10.0 if kind == "static" else 10.9,
                decode_ms=100.0 if kind == "static" else 104.0,
                serialized_plan_bytes=(
                    1_000 if kind == "static" else 1_040
                ),
                resident_weight_bytes=(
                    2_000 if kind == "static" else 2_080
                ),
            )
            paths[f"{kind}_{prompt_kind}"] = path
    return paths


def _qualify(paths: dict[str, Path]) -> dict:
    return perf.qualify(
        static_short=paths["static_short"],
        dynamic_short=paths["dynamic_short"],
        static_medium=paths["static_medium"],
        dynamic_medium=paths["dynamic_medium"],
        static_bundle=paths["static_bundle"],
        dynamic_bundle=paths["dynamic_bundle"],
    )


def _edit(path: Path, callback) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    callback(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _refresh_evidence_hash(payload: dict, field: str) -> None:
    payload["qualification_provenance"][f"{field}_sha256"] = (
        perf._canonical_sha(payload[field])
    )


def test_passes_all_performance_packaging_and_provenance_gates(
    tmp_path: Path,
) -> None:
    report = _qualify(_evidence(tmp_path))

    assert report["status"] == "passed"
    assert report["gates"]["performance"]["short"][
        "decode_throughput_gte_95_percent_static"
    ]
    assert report["gates"]["performance"]["medium"][
        "prefill_proxy_regression_lte_10_percent"
    ]
    assert report["gates"]["packaging"][
        "bundle_bytes_lte_105_percent_static"
    ]
    assert report["gates"]["runtime_evidence_consistency"][
        "dynamic_runtime_attention_plans_present"
    ]
    assert report["gates"]["provenance"]["one_tokenizer_contract"]
    assert report["gates"]["performance"]["short"][
        "same_fixed_length_structural_workload"
    ]
    assert report["diagnostics"]["token_stream_equivalence"]["short"][
        "exact_generated_token_ids_match"
    ]
    assert not report["diagnostics"]["runtime_attention_plan_scope"][
        "per_invocation_H_A_profile_plan_proved"
    ]


@pytest.mark.parametrize(
    ("field", "error"),
    [
        (
            "runtime_attention_plans",
            "missing field: dynamic-short.runtime_attention_plans",
        ),
        ("runtime_stack", "missing field: dynamic-short.runtime_stack"),
        (
            "runtime_libraries",
            "missing field: dynamic-short.runtime_libraries",
        ),
        ("cuda_jit_cache", "missing field: dynamic-short.cuda_jit_cache"),
        (
            "generation_workload",
            "missing field: dynamic-short.generation_workload",
        ),
        (
            "tokenizer_contract",
            "missing field: dynamic-short.tokenizer_contract",
        ),
    ],
)
def test_requires_capture_produced_runtime_evidence(
    tmp_path: Path, field: str, error: str
) -> None:
    paths = _evidence(tmp_path)
    _edit(paths["dynamic_short"], lambda payload: payload.pop(field))

    with pytest.raises(perf.QualificationError, match=error):
        _qualify(paths)


@pytest.mark.parametrize(
    "field",
    [
        "runtime_attention_plans_sha256",
        "runtime_stack_sha256",
        "runtime_libraries_sha256",
        "cuda_jit_cache_sha256",
        "generation_workload_sha256",
        "tokenizer_contract_sha256",
    ],
)
def test_requires_runtime_evidence_provenance_hashes(
    tmp_path: Path, field: str
) -> None:
    paths = _evidence(tmp_path)
    _edit(
        paths["dynamic_short"],
        lambda payload: payload["qualification_provenance"].pop(field),
    )

    with pytest.raises(
        perf.QualificationError,
        match=(
            "missing field: "
            f"dynamic-short.qualification_provenance.{field}"
        ),
    ):
        _qualify(paths)


def test_rejects_runtime_evidence_hash_mismatch(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    _edit(
        paths["dynamic_short"],
        lambda payload: payload["runtime_attention_plans"][0].__setitem__(
            "T", 1024
        ),
    )

    with pytest.raises(
        perf.QualificationError,
        match="runtime_attention_plans_sha256 does not match captured evidence",
    ):
        _qualify(paths)


@pytest.mark.parametrize(
    ("case", "field", "replacement", "error"),
    [
        (
            "dynamic_short",
            "runtime_attention_plans",
            [],
            "must contain at least one dynamic LSE plan",
        ),
        (
            "static_short",
            "runtime_attention_plans",
            [RUNTIME_PLAN],
            r"must be \[\] for a static baseline",
        ),
        (
            "static_short",
            "runtime_stack",
            RUNTIME_STACK,
            "must be null for a static baseline",
        ),
        (
            "static_short",
            "runtime_libraries",
            {},
            "must be null for a static baseline",
        ),
    ],
)
def test_enforces_dynamic_and_static_runtime_evidence_roles(
    tmp_path: Path,
    case: str,
    field: str,
    replacement: object,
    error: str,
) -> None:
    paths = _evidence(tmp_path)

    def replace(payload: dict) -> None:
        payload[field] = replacement
        _refresh_evidence_hash(payload, field)

    _edit(paths[case], replace)
    with pytest.raises(perf.QualificationError, match=error):
        _qualify(paths)


def test_requires_lse_plan_and_cudnn_stack_coherence(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)

    def replace_stats(payload: dict) -> None:
        payload["runtime_attention_plans"][0]["stats"] = "max_sum_exp"
        _refresh_evidence_hash(payload, "runtime_attention_plans")

    _edit(paths["dynamic_short"], replace_stats)
    with pytest.raises(
        perf.QualificationError,
        match=r"runtime_attention_plans\[0\]\.stats must be 'lse'",
    ):
        _qualify(paths)

    paths = _evidence(tmp_path / "cudnn")

    def disagree(payload: dict) -> None:
        payload["runtime_attention_plans"][0]["cudnn_version"] = 91900
        _refresh_evidence_hash(payload, "runtime_attention_plans")

    _edit(paths["dynamic_short"], disagree)
    with pytest.raises(
        perf.QualificationError,
        match="cudnn_version disagrees with runtime_stack.cudnn_backend",
    ):
        _qualify(paths)


def test_requires_complete_live_runtime_stack(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)

    def remove_nvrtc(payload: dict) -> None:
        payload["runtime_stack"].pop("nvrtc")
        _refresh_evidence_hash(payload, "runtime_stack")

    _edit(paths["dynamic_short"], remove_nvrtc)
    with pytest.raises(
        perf.QualificationError,
        match=r"runtime_stack fields must match capture schema:.*nvrtc",
    ):
        _qualify(paths)


def test_requires_runtime_library_paths_and_hashes_to_match_files(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)

    def corrupt_file_hash(payload: dict) -> None:
        payload["runtime_libraries"]["nvrtc"]["sha256"] = "0" * 64
        _refresh_evidence_hash(payload, "runtime_libraries")

    _edit(paths["dynamic_short"], corrupt_file_hash)
    with pytest.raises(
        perf.QualificationError,
        match="runtime_libraries.nvrtc does not match the captured file",
    ):
        _qualify(paths)


def test_requires_coherent_cuda_jit_cache_evidence(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)

    def claim_warm(payload: dict) -> None:
        payload["cuda_jit_cache"]["initial_state"] = "warm"
        _refresh_evidence_hash(payload, "cuda_jit_cache")

    _edit(paths["dynamic_short"], claim_warm)
    with pytest.raises(
        perf.QualificationError,
        match="cuda_jit_cache.initial_state must be 'cold'",
    ):
        _qualify(paths)


def test_fails_when_dynamic_live_stacks_differ_across_prompts(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)

    def change_driver(payload: dict) -> None:
        payload["runtime_stack"]["driver"] = "580.105.09"
        _refresh_evidence_hash(payload, "runtime_stack")

    _edit(paths["dynamic_medium"], change_driver)
    report = _qualify(paths)

    assert report["status"] == "failed"
    assert not report["gates"]["runtime_evidence_consistency"][
        "dynamic_runtime_stack_matches_across_prompts"
    ]


def test_fails_when_dynamic_runtime_libraries_differ_across_prompts(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    alternate = _runtime_libraries(tmp_path / "alternate")

    def change_libraries(payload: dict) -> None:
        payload["runtime_libraries"] = alternate
        _refresh_evidence_hash(payload, "runtime_libraries")

    _edit(paths["dynamic_medium"], change_libraries)
    report = _qualify(paths)

    assert report["status"] == "failed"
    assert not report["gates"]["runtime_evidence_consistency"][
        "dynamic_runtime_libraries_match_across_prompts"
    ]


def test_generated_token_divergence_is_explicit_but_not_a_false_failure(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)

    def diverge(payload: dict) -> None:
        workload = payload["generation_workload"]
        for stream in workload["measured_generated_token_ids"]:
            stream[5] = 999
        workload["measured_generated_token_ids_sha256"] = (
            perf._canonical_sha(workload["measured_generated_token_ids"])
        )
        _refresh_evidence_hash(payload, "generation_workload")

    _edit(paths["dynamic_medium"], diverge)
    report = _qualify(paths)

    assert report["status"] == "passed"
    diagnostic = report["diagnostics"]["token_stream_equivalence"]["medium"]
    assert diagnostic["exact_generated_token_ids_match"] is False
    assert diagnostic["common_prefix_tokens"] == 5
    assert "diagnostic_only" in diagnostic["gate_effect"]


def test_structural_workload_mismatch_is_a_hard_failure(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)

    def change_prompt(payload: dict) -> None:
        workload = payload["generation_workload"]
        workload["structural_identity"]["prompt_sha256"] = "b" * 64
        workload["structural_identity_sha256"] = perf._canonical_sha(
            workload["structural_identity"]
        )
        _refresh_evidence_hash(payload, "generation_workload")

    _edit(paths["dynamic_medium"], change_prompt)
    report = _qualify(paths)

    assert report["status"] == "failed"
    assert not report["gates"]["performance"]["medium"][
        "same_fixed_length_structural_workload"
    ]


def test_rejects_non_repeatable_token_stream_inside_one_case(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)

    def change_one_iteration(payload: dict) -> None:
        workload = payload["generation_workload"]
        workload["measured_generated_token_ids"][1][0] = 999
        workload["measured_generated_token_ids_sha256"] = (
            perf._canonical_sha(workload["measured_generated_token_ids"])
        )
        workload["token_stream_repeatable_within_case"] = False
        _refresh_evidence_hash(payload, "generation_workload")

    _edit(paths["dynamic_short"], change_one_iteration)
    with pytest.raises(
        perf.QualificationError,
        match="does not prove a repeatable greedy token stream",
    ):
        _qualify(paths)


def test_fails_when_static_and_dynamic_tokenizer_contracts_differ(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)

    for prompt_kind in ("short", "medium"):
        def change_tokenizer(payload: dict) -> None:
            payload["tokenizer_contract"][
                "tokenizer_json_sha256"
            ] = "c" * 64
            _refresh_evidence_hash(payload, "tokenizer_contract")

        _edit(paths[f"dynamic_{prompt_kind}"], change_tokenizer)

    report = _qualify(paths)
    assert report["status"] == "failed"
    assert not report["gates"]["provenance"]["one_tokenizer_contract"]


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    [
        ("decode_ms", 106.0, "decode_throughput_gte_95_percent_static"),
        ("prefill_ms", 11.1, "prefill_proxy_regression_lte_10_percent"),
    ],
)
def test_fails_short_or_medium_performance_threshold(
    tmp_path: Path,
    field: str,
    value: float,
    gate: str,
) -> None:
    paths = _evidence(tmp_path)

    def regress(payload: dict) -> None:
        for observation in payload["observations"]:
            observation[field] = value

    _edit(paths["dynamic_medium"], regress)
    report = _qualify(paths)

    assert report["status"] == "failed"
    assert not report["gates"]["performance"]["medium"][gate]


@pytest.mark.parametrize(
    ("receipt_field", "value", "gate"),
    [
        (
            "serialized_plan_bytes",
            1_051,
            "serialized_plan_bytes_lte_105_percent_static",
        ),
        (
            "resident_weight_bytes",
            2_101,
            "resident_weight_bytes_lte_105_percent_static",
        ),
    ],
)
def test_fails_mem13_size_thresholds(
    tmp_path: Path,
    receipt_field: str,
    value: int,
    gate: str,
) -> None:
    paths = _evidence(tmp_path)
    for prompt_kind in ("short", "medium"):
        _edit(
            paths[f"dynamic_{prompt_kind}"],
            lambda payload: payload["runtime_memory_receipt"].__setitem__(
                receipt_field, value
            ),
        )

    report = _qualify(paths)

    assert report["status"] == "failed"
    assert not report["gates"]["packaging"][gate]


def test_fails_weight_copy_streaming_and_bundle_size_gates(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    paths["dynamic_bundle"].write_bytes(b"d" * 1_051)
    dynamic_sha = _sha256(paths["dynamic_bundle"])
    for prompt_kind in ("short", "medium"):
        def break_receipt(payload: dict) -> None:
            payload["qualification_provenance"]["bundle_sha256"] = dynamic_sha
            payload["runtime_memory_receipt"][
                "resident_weight_copy_count"
            ] = 3
            payload["runtime_memory_receipt"]["weight_streaming_active"] = True

        _edit(paths[f"dynamic_{prompt_kind}"], break_receipt)

    report = _qualify(paths)

    assert report["status"] == "failed"
    packaging = report["gates"]["packaging"]
    assert not packaging["bundle_bytes_lte_105_percent_static"]
    assert not packaging["resident_weight_copy_count_lte_2"]["dynamic"]
    assert not packaging["weight_streaming_disabled"]["dynamic"]


def test_fails_closed_on_old_worker_result_with_named_missing_field(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    _edit(paths["dynamic_short"], lambda payload: payload.pop(
        "qualification_provenance"
    ))
    output = tmp_path / "report.json"

    returncode = perf.main(
        [
            "--static-short",
            str(paths["static_short"]),
            "--dynamic-short",
            str(paths["dynamic_short"]),
            "--static-medium",
            str(paths["static_medium"]),
            "--dynamic-medium",
            str(paths["dynamic_medium"]),
            "--static-bundle",
            str(paths["static_bundle"]),
            "--dynamic-bundle",
            str(paths["dynamic_bundle"]),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert returncode == 1
    assert report["status"] == "failed"
    assert report["errors"] == [
        "missing field: dynamic-short.qualification_provenance"
    ]


def test_fails_source_request_bundle_and_freshness_provenance(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)

    def corrupt(payload: dict) -> None:
        provenance = payload["qualification_provenance"]
        provenance["source_state_sha256"] = "a" * 64
        provenance["request_sha256"] = "b" * 64
        provenance["bundle_sha256"] = "c" * 64
        provenance["fresh_build"] = False
        provenance["artifact_reused"] = True

    _edit(paths["dynamic_short"], corrupt)
    report = _qualify(paths)

    assert report["status"] == "failed"
    gates = report["gates"]["provenance"]
    assert not gates["shared_fields_match"]["source_state_sha256"]
    assert not gates["source_stable_prebuild_to_postbuild"]["dynamic-short"]
    assert not gates["dynamic_bundle_sha_matches_file"]
    assert not gates["short_request_sha_matches"]
    assert not gates["all_bundles_declared_fresh"]
    assert not gates["no_reused_artifacts"]


def test_fails_closed_on_unavailable_or_unproven_memory_measurement(
    tmp_path: Path,
) -> None:
    paths = _evidence(tmp_path)
    _edit(
        paths["dynamic_short"],
        lambda payload: payload["runtime_memory_receipt"].__setitem__(
            "resident_weight_bytes", None
        ),
    )
    with pytest.raises(
        perf.QualificationError,
        match=(
            "dynamic-short.runtime_memory_receipt.resident_weight_bytes "
            "must be a positive integer"
        ),
    ):
        _qualify(paths)

    paths = _evidence(tmp_path / "second")
    _edit(
        paths["dynamic_short"],
        lambda payload: payload["runtime_memory_receipt"][
            "measurement_sources"
        ].pop("resident_weight_bytes"),
    )
    with pytest.raises(
        perf.QualificationError,
        match="missing field: .*measurement_sources.resident_weight_bytes",
    ):
        _qualify(paths)
