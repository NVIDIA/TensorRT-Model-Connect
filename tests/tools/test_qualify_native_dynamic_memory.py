# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "qualify_native_dynamic_memory.py"
SPEC = importlib.util.spec_from_file_location("qualify_native_dynamic_memory", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
qualify = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualify
SPEC.loader.exec_module(qualify)

pytestmark = pytest.mark.dynamic_memory


def test_deterministic_token_ids_are_prefix_stable_and_in_vocab() -> None:
    short = qualify.deterministic_token_ids(2_047, 32_000)
    long = qualify.deterministic_token_ids(2_049, 32_000)

    np.testing.assert_array_equal(short, long[: short.size])
    assert short.dtype == np.int32
    assert int(short.min()) >= 1
    assert int(long.max()) < 32_000


def test_dirty_source_provenance_captures_both_patches_and_untracked_hashes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    staged = repo / "staged.txt"
    unstaged = repo / "unstaged.txt"
    staged.write_text("base staged\n", encoding="utf-8")
    unstaged.write_text("base unstaged\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "staged.txt", "unstaged.txt"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=TRTMC Test",
            "-c",
            "user.email=trtmc@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        cwd=repo,
        check=True,
    )

    staged.write_text("changed and staged\n", encoding="utf-8")
    subprocess.run(["git", "add", "staged.txt"], cwd=repo, check=True)
    unstaged.write_text("changed but unstaged\n", encoding="utf-8")
    untracked = repo / "new-source.cpp"
    untracked.write_text("int value = 1;\n", encoding="utf-8")
    artifact_dir = repo / "artifacts" / "qualification"

    before = qualify.source_state_provenance(
        repo,
        MODULE_PATH,
        artifact_dir,
        label="pre",
    )

    assert before["git_dirty"]
    assert not before["exact_head_gate_satisfied"]
    assert before["artifacts"]["staged_patch"]["size_bytes"] > 0
    assert before["artifacts"]["unstaged_patch"]["size_bytes"] > 0
    assert before["untracked_files"] == [
        {
            "path": "new-source.cpp",
            "kind": "file",
            "size_bytes": untracked.stat().st_size,
            "sha256": hashlib.sha256(untracked.read_bytes()).hexdigest(),
        }
    ]
    manifest = json.loads(
        Path(before["artifacts"]["untracked_manifest"]["path"]).read_text(encoding="utf-8")
    )
    assert manifest == before["untracked_files"]

    untracked.write_text("int value = 2;\n", encoding="utf-8")
    after = qualify.source_state_provenance(
        repo,
        MODULE_PATH,
        artifact_dir,
        label="post",
    )
    assert after["source_state_sha256"] != before["source_state_sha256"]


def test_qwen_matrix_contains_exact_chunk_and_model_boundaries() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    cases = {case.name: case for case in qualify._cases_for(spec)}

    assert (
        cases["c-minus-1"].prompt_tokens,
        cases["c"].prompt_tokens,
        cases["c-plus-1"].prompt_tokens,
    ) == (1_023, 1_024, 1_025)
    assert cases["two-c-plus-17"].prompt_tokens == 2_065
    assert (cases["total-32768"].prompt_tokens, cases["total-32768"].decode_tokens) == (32_760, 8)
    assert (cases["total-model-limit"].prompt_tokens, cases["total-model-limit"].decode_tokens) == (
        40_952,
        8,
    )
    assert cases["prefill-last-position"].prompt_tokens == 40_960
    assert cases["model-limit-plus-1"].prompt_tokens == 40_961
    assert cases["model-limit-plus-1"].expect_admission_rejection
    for bucket in (128, 256, 512, 1_024, 2_048, 8_192, 32_768):
        crossing = cases[f"profile-crossing-{bucket}"]
        assert (crossing.prompt_tokens, crossing.decode_tokens) == (bucket, 2)
    for index, bucket in enumerate(spec.buckets[:-1]):
        expected = (
            ("p-minus-1", bucket - 1, index),
            ("p", bucket, index),
            ("p-plus-1", bucket + 1, index + 1),
        )
        for label, prompt_tokens, profile_id in expected:
            boundary = cases[f"decode-bucket-{bucket}-{label}"]
            assert (boundary.prompt_tokens, boundary.decode_tokens) == (
                prompt_tokens,
                1,
            )
            assert boundary.expected_decode_profile_ids == (profile_id,)
            assert boundary.expected_decode_bucket_limits == (spec.buckets[profile_id],)


def test_tiny_matrix_covers_every_bucket_neighbor_and_m_plus_one() -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    cases = qualify._cases_for(spec)
    lengths = {case.prompt_tokens for case in cases}

    for bucket in spec.buckets:
        assert {bucket - 1, bucket, bucket + 1}.issubset(lengths)
    rejected = [case for case in cases if case.expect_admission_rejection]
    assert [(case.prompt_tokens, case.decode_tokens) for case in rejected] == [(2_049, 0)]
    assert any(case.prompt_tokens == 2_040 and case.decode_tokens == 8 for case in cases)
    assert any(case.prompt_tokens == 2_048 and case.decode_tokens == 0 for case in cases)
    by_name = {case.name: case for case in cases}
    for bucket in (128, 256, 512):
        assert (
            by_name[f"profile-crossing-{bucket}"].prompt_tokens,
            by_name[f"profile-crossing-{bucket}"].decode_tokens,
        ) == (bucket, 2)
    for index, bucket in enumerate(spec.buckets[:-1]):
        for label, prompt_tokens, profile_id in (
            ("p-minus-1", bucket - 1, index),
            ("p", bucket, index),
            ("p-plus-1", bucket + 1, index + 1),
        ):
            boundary = by_name[f"decode-bucket-{bucket}-{label}"]
            assert (boundary.prompt_tokens, boundary.decode_tokens) == (
                prompt_tokens,
                1,
            )
            assert boundary.expected_decode_profile_ids == (profile_id,)
            assert boundary.expected_decode_bucket_limits == (spec.buckets[profile_id],)


def _qualified_engine_graph_evidence(spec) -> dict:
    num_layers = spec.num_layers
    width = spec.kv_bytes_per_token // (2 * num_layers * qualify._KV_DTYPE_BYTES[spec.kv_dtype])

    def section(role: str) -> dict:
        is_prefill = role == "prefill"
        profile_count = 1 if is_prefill else len(spec.buckets)
        token_profiles = (
            [
                {
                    "min": [1],
                    "opt": [spec.chunk_limit],
                    "max": [spec.chunk_limit],
                }
            ]
            if is_prefill
            else [{"min": [1], "opt": [1], "max": [1]} for _ in spec.buckets]
        )
        inputs = {
            "token_id": {
                "shape": [-1] if is_prefill else [1],
                "profiles": token_profiles,
            },
            "position_id": {
                "shape": [-1] if is_prefill else [1],
                "profiles": copy.deepcopy(token_profiles),
            },
            "history_length": {
                "shape": [1],
                "profiles": [{"min": [1], "opt": [1], "max": [1]} for _ in range(profile_count)],
            },
        }
        outputs = {
            "logits": {
                "shape": [1, spec.vocab_size],
            },
        }
        for layer in range(num_layers):
            for value_name in ("k", "v"):
                cache_profiles = (
                    [
                        {
                            "min": [1, width],
                            "opt": [spec.chunk_limit, width],
                            "max": [spec.context_limit, width],
                        }
                    ]
                    if is_prefill
                    else [
                        {
                            "min": [1, width],
                            "opt": [bucket, width],
                            "max": [bucket, width],
                        }
                        for bucket in spec.buckets
                    ]
                )
                inputs[f"cache_{value_name}_{layer}"] = {
                    "shape": [-1, width],
                    "profiles": cache_profiles,
                }
                outputs[f"present_{value_name}_{layer}"] = {
                    "shape": [-1 if is_prefill else 1, width],
                }
        return {
            "engine_sha256": ("a" * 64 if is_prefill else "b" * 64),
            "num_optimization_profiles": profile_count,
            "inputs": inputs,
            "outputs": outputs,
            "native_contiguous_attention_layer_indices": list(range(num_layers)),
            "dense_attention_layers": [],
            "cache_concat_layers": [],
            "inspector_path": f"{role}.engine-inspector.json",
            "inspector_size_bytes": 128,
            "inspector_sha256": ("c" * 64 if is_prefill else "d" * 64),
        }

    return {
        "runtime_stack": {
            "sm": "sm103",
            "tensorrt": "11.2.0.113",
            "cuda_runtime": "13.3",
            "driver": "580.105.08",
        },
        "model_contract": {
            "model_context_limit": spec.context_limit,
            "prefill_chunk_limit": spec.chunk_limit,
            "active_kv_profile_limits": list(spec.buckets),
            "num_layers": num_layers,
            "vocab_size": spec.vocab_size,
            "kv_dtype": spec.kv_dtype,
            "kv_bytes_per_token": spec.kv_bytes_per_token,
            "kv_width": width,
        },
        "engine_sections": {
            "prefill_engine_plan": section("prefill"),
            "engine_plan": section("decode"),
        },
    }


@pytest.mark.parametrize(
    ("model_id", "expected_layers", "expected_width"),
    (
        ("Qwen/Qwen3-0.6B", 28, 1_024),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 22, 256),
    ),
)
def test_full_model_engine_graph_gate_accepts_live_io_contract(
    model_id: str,
    expected_layers: int,
    expected_width: int,
) -> None:
    spec = qualify.SPECS[model_id]
    evidence = _qualified_engine_graph_evidence(spec)

    result = qualify._validate_qualified_engine_graph_evidence(
        evidence,
        spec,
        num_layers=spec.num_layers,
        expected_runtime_stack=evidence["runtime_stack"],
    )

    assert result["passed"]
    assert all(result["gates"].values())
    assert result["runtime_stack"] == {
        "sm": "sm103",
        "tensorrt": "11.2.0.113",
        "cuda_runtime": "13.3",
        "driver": "580.105.08",
    }
    assert result["model_contract"] == evidence["model_contract"]
    assert result["model_contract"]["num_layers"] == expected_layers
    assert result["model_contract"]["kv_width"] == expected_width
    assert result["engine_sections"]["prefill_engine_plan"]["outputs"]["logits"]["shape"] == [
        1,
        spec.vocab_size,
    ]


def test_graph_model_contract_is_derived_from_bundle_header() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    header = {
        "num_layers": spec.num_layers,
        "vocab_size": spec.vocab_size,
        "runtime_memory": {
            "model_context_limit": spec.context_limit,
            "prefill_chunk_limit": spec.chunk_limit,
            "active_kv_profile_limits": list(spec.buckets),
            "kv_dtype": spec.kv_dtype,
            "kv_bytes_per_token": spec.kv_bytes_per_token,
        },
    }

    assert qualify._graph_model_contract_from_bundle_header(header) == {
        "model_context_limit": spec.context_limit,
        "prefill_chunk_limit": spec.chunk_limit,
        "active_kv_profile_limits": list(spec.buckets),
        "num_layers": 28,
        "vocab_size": 151_936,
        "kv_dtype": "bfloat16",
        "kv_bytes_per_token": 114_688,
        "kv_width": 1_024,
    }


def test_full_model_engine_graph_gate_fails_closed_on_runtime_stack_tamper() -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    evidence = _qualified_engine_graph_evidence(spec)
    expected_stack = copy.deepcopy(evidence["runtime_stack"])
    evidence["runtime_stack"]["driver"] = "tampered"

    with pytest.raises(RuntimeError, match="runtime stack"):
        qualify._validate_qualified_engine_graph_evidence(
            evidence,
            spec,
            num_layers=spec.num_layers,
            expected_runtime_stack=expected_stack,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("missing-runtime-stack", "runtime stack"),
        ("attention-mask-input", "forbidden attention_mask"),
        ("missing-native-layer", "NativeContiguousAttentionV2"),
        ("dense-attention-path", "dense attention mask/score"),
        ("cache-concat", "full-history cache concat"),
        ("full-history-present", "current-row output"),
        ("same-plan", "same serialized engine identity"),
    ),
)
def test_full_model_engine_graph_gate_fails_closed(
    mutation: str,
    error: str,
) -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    evidence = _qualified_engine_graph_evidence(spec)
    expected_stack = copy.deepcopy(evidence["runtime_stack"])
    prefill = evidence["engine_sections"]["prefill_engine_plan"]
    decode = evidence["engine_sections"]["engine_plan"]
    if mutation == "missing-runtime-stack":
        evidence.pop("runtime_stack")
    elif mutation == "attention-mask-input":
        decode["inputs"]["attention_mask"] = {
            "shape": [1, -1],
            "profiles": [],
        }
    elif mutation == "missing-native-layer":
        decode["native_contiguous_attention_layer_indices"] = [0]
    elif mutation == "dense-attention-path":
        prefill["dense_attention_layers"] = ["layer.1.attn.attention_scores"]
    elif mutation == "cache-concat":
        decode["cache_concat_layers"] = ["layer.0.cache_concat"]
    elif mutation == "full-history-present":
        decode["outputs"]["present_k_0"]["shape"] = [
            -1,
            evidence["model_contract"]["kv_width"],
        ]
    elif mutation == "same-plan":
        decode["engine_sha256"] = prefill["engine_sha256"]
    else:  # pragma: no cover - keeps additions to the table explicit.
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=error):
        qualify._validate_qualified_engine_graph_evidence(
            evidence,
            spec,
            num_layers=spec.num_layers,
            expected_runtime_stack=expected_stack,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("token-shape", "token_id shape"),
        ("position-shape", "position_id shape"),
        ("token-position-profile", "profiles are not identical"),
        ("token-profile", "prefill profile does not cover"),
        ("history-shape", "history_length is not a scalar"),
        ("history-profile", "history_length profile"),
        ("logits-row-shape", "logits shape"),
        ("logits-vocab", "logits shape"),
        ("cache-width", "source-bound KV width"),
        ("present-width", "source-bound KV width"),
        ("cache-profile-width", "does not bind bucket"),
        ("derived-width", "KV width does not match"),
        ("nondivisible-b", "not exactly divisible"),
        ("qualified-b-mismatch", "model_contract mismatch"),
        ("unknown-dtype", "unsupported KV dtype"),
    ),
)
def test_full_model_engine_graph_gate_rejects_io_or_kv_geometry_mismatch(
    mutation: str,
    error: str,
) -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    evidence = _qualified_engine_graph_evidence(spec)
    expected_stack = copy.deepcopy(evidence["runtime_stack"])
    prefill = evidence["engine_sections"]["prefill_engine_plan"]
    decode = evidence["engine_sections"]["engine_plan"]
    model_contract = evidence["model_contract"]
    width = model_contract["kv_width"]
    if mutation == "token-shape":
        prefill["inputs"]["token_id"]["shape"] = [1]
    elif mutation == "position-shape":
        decode["inputs"]["position_id"]["shape"] = [-1]
    elif mutation == "token-position-profile":
        prefill["inputs"]["position_id"]["profiles"][0]["opt"] = [1]
    elif mutation == "token-profile":
        for name in ("token_id", "position_id"):
            prefill["inputs"][name]["profiles"][0]["max"] = [spec.chunk_limit - 1]
    elif mutation == "history-shape":
        decode["inputs"]["history_length"]["shape"] = [-1]
    elif mutation == "history-profile":
        decode["inputs"]["history_length"]["profiles"][0]["max"] = [2]
    elif mutation == "logits-row-shape":
        prefill["outputs"]["logits"]["shape"] = [-1, spec.vocab_size]
    elif mutation == "logits-vocab":
        decode["outputs"]["logits"]["shape"] = [1, spec.vocab_size + 1]
    elif mutation == "cache-width":
        decode["inputs"]["cache_v_1"]["shape"] = [-1, width + 1]
    elif mutation == "present-width":
        decode["outputs"]["present_k_1"]["shape"] = [1, width + 1]
    elif mutation == "cache-profile-width":
        decode["inputs"]["cache_k_0"]["profiles"][0]["opt"] = [
            spec.buckets[0],
            width + 1,
        ]
    elif mutation == "derived-width":
        model_contract["kv_width"] = width + 1
    elif mutation == "nondivisible-b":
        model_contract["kv_bytes_per_token"] += 1
    elif mutation == "qualified-b-mismatch":
        model_contract["kv_bytes_per_token"] *= 2
        model_contract["kv_width"] *= 2
    elif mutation == "unknown-dtype":
        model_contract["kv_dtype"] = "int8"
    else:  # pragma: no cover - keeps additions to the table explicit.
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=error):
        qualify._validate_qualified_engine_graph_evidence(
            evidence,
            spec,
            num_layers=spec.num_layers,
            expected_runtime_stack=expected_stack,
        )


@pytest.fixture
def qualification_outcome_inputs() -> dict:
    canonical_cases = (
        qualify.Case("runtime-case", 128, 1),
        qualify.Case(
            "admission-case",
            2_049,
            0,
            expect_admission_rejection=True,
        ),
    )
    case_reports = (
        {
            "name": "runtime-case",
            "execution_passed": True,
            "passed": True,
            "parity": {
                "status": "passed",
                "passed": True,
            },
        },
        {
            "name": "admission-case",
            "execution_passed": True,
            "passed": True,
            "parity": {
                "status": "not_applicable",
            },
        },
    )
    clean_source = {
        "git_head": "a" * 40,
        "git_dirty": False,
        "source_state_sha256": "b" * 64,
        "exact_head_gate_satisfied": True,
    }
    return {
        "canonical_cases": canonical_cases,
        "selected_cases": canonical_cases,
        "case_reports": case_reports,
        "skip_hf": False,
        "case_filter_used": False,
        "source_state_pre": clean_source,
        "source_state_post": copy.deepcopy(clean_source),
        "context_memory_envelope": {
            "status": "passed",
            "passed": True,
            "coverage_required": True,
            "gates": {
                "all_points_within_o_c_times_a_envelope": True,
                "all_points_below_materialized_score_bound": True,
                "coverage": {
                    "has_prefill_and_decode": True,
                    "reaches_model_context_limit": True,
                    "has_at_least_three_active_lengths": True,
                },
            },
        },
        "qualified_engine_graph": {
            "passed": True,
            "runtime_stack": {
                "sm": "sm103",
                "tensorrt": "11.2.0.113",
            },
            "gates": {
                "actual_split_engine_sections": True,
                "native_segmented_attention_covers_full_model": True,
            },
        },
    }


def test_qualification_outcome_promotes_only_full_green_matrix(
    qualification_outcome_inputs: dict,
) -> None:
    result = qualify.evaluate_qualification_outcome(**qualification_outcome_inputs)

    assert result["passed"]
    assert result["diagnostic_passed"]
    assert result["execution_passed"]
    assert result["status"] == "passed"
    assert all(result["qualification_gates"].values())
    assert result["qualification_blockers"] == []


def test_qualification_outcome_marks_skip_hf_as_diagnostic_only(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    inputs["skip_hf"] = True
    inputs["case_reports"][0]["parity"] = {
        "status": "not_run",
        "reason": "--skip-hf was requested",
    }
    # A stale case-level flag must never promote a skipped parity run.
    inputs["case_reports"][0]["passed"] = True

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert not result["passed"]
    assert result["diagnostic_passed"]
    assert result["status"] == "diagnostic_passed"
    assert not result["qualification_gates"]["hf_parity_executed_and_passed"]
    assert result["parity_execution"]["runtime-case"] == "not_run"


def test_qualification_outcome_marks_any_case_filter_as_diagnostic_only(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    # Even an explicit filter that happens to name the complete matrix is not
    # a canonical unfiltered qualification invocation.
    inputs["case_filter_used"] = True

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert result["qualification_gates"]["canonical_matrix_complete"]
    assert not result["qualification_gates"]["case_filter_not_used"]
    assert not result["passed"]
    assert result["diagnostic_passed"]
    assert result["status"] == "diagnostic_passed"


def test_qualification_outcome_requires_full_context_memory_coverage(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    envelope = inputs["context_memory_envelope"]
    envelope["coverage_required"] = False
    envelope["gates"]["coverage"]["reaches_model_context_limit"] = False

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert not result["qualification_gates"]["full_context_memory_coverage"]
    assert not result["passed"]
    assert result["diagnostic_passed"]
    assert result["status"] == "diagnostic_passed"


def test_qualification_outcome_marks_dirty_unchanged_source_diagnostic_only(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    for snapshot in (
        inputs["source_state_pre"],
        inputs["source_state_post"],
    ):
        snapshot["git_dirty"] = True
        snapshot["exact_head_gate_satisfied"] = False

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert result["qualification_gates"]["source_state_unchanged"]
    assert not result["qualification_gates"]["source_clean_exact_head"]
    assert not result["passed"]
    assert result["diagnostic_passed"]
    assert result["status"] == "diagnostic_passed"


def test_qualification_outcome_rejects_source_drift(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    inputs["source_state_post"]["source_state_sha256"] = "c" * 64

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert not result["qualification_gates"]["source_state_unchanged"]
    assert not result["passed"]
    assert not result["diagnostic_passed"]
    assert result["status"] == "failed"


def test_qualification_outcome_rejects_false_graph_gate(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    inputs["qualified_engine_graph"]["passed"] = False
    inputs["qualified_engine_graph"]["gates"]["native_segmented_attention_covers_full_model"] = (
        False
    )

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert not result["qualification_gates"]["qualified_engine_graph_passed"]
    assert not result["passed"]
    assert not result["diagnostic_passed"]
    assert result["status"] == "failed"


def test_inspector_layer_classification_ignores_container_text() -> None:
    inspector = {
        "Metadata": "container mentions concat cache_k_0 attention_scores",
        "Layers": [
            {
                "Name": "layer.0.cache_concat",
                "LayerType": "Concatenation",
                "Inputs": [{"Name": "cache_k_0"}],
            },
            {
                "Name": "layer.1.attn.attention_scores",
                "LayerType": "MatrixMultiply",
            },
        ],
    }

    assert qualify._cache_concat_layers(inspector) == ["layer.0.cache_concat"]
    assert qualify._dense_attention_layers(inspector) == ["layer.1.attn.attention_scores"]


def test_c_div_2_variant_may_change_only_internal_chunk_policy() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    contract = {
        "contract_version": 1,
        "qualified_model_id": spec.model_id,
        "qualified_model_revision": "1" * 40,
        "qualified_config_sha256": "2" * 64,
        "qualified_target": "gb300-trt-11.2",
        "qualified_runtime_stack": {
            "sm": "sm103",
            "tensorrt": "11.2.0.113",
            "cuda_runtime": "13.3",
            "cudnn_backend": "9.20.0",
            "cudnn_frontend_revision": "7b9b711c22b6823e87150213ecd8449260db8610",
            "nvrtc": "13.3",
            "driver": "580.105.08",
        },
        "native_kv_plugin_abi": 2,
        "model_context_limit": spec.context_limit,
        "prefill_chunk_limit": spec.chunk_limit,
        "kv_layout": "contiguous_runtime_v1",
        "kv_dtype": "bfloat16",
        "kv_bytes_per_token": 114_688,
        "active_kv_profile_limits": list(spec.buckets),
        "runtime_owned": True,
    }
    base = {"vocab_size": 151_936, "runtime_memory": contract}
    variant_contract = dict(contract)
    variant_contract["prefill_chunk_limit"] = spec.chunk_limit // 2
    variant_contract["active_kv_profile_limits"] = list(spec.buckets)
    variant = {
        "vocab_size": base["vocab_size"],
        "runtime_memory": variant_contract,
    }

    assert qualify._validate_chunk_variant(base, variant, spec) == spec.chunk_limit // 2

    variant_contract["qualified_model_revision"] = "3" * 40
    with pytest.raises(ValueError, match="changes qualified model facts"):
        qualify._validate_chunk_variant(base, variant, spec)


def _qwen_chunk_variant_headers() -> tuple[dict, dict]:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    contract = {
        "contract_version": 1,
        "qualified_model_id": spec.model_id,
        "qualified_model_revision": "1" * 40,
        "qualified_config_sha256": "2" * 64,
        "qualified_target": "gb300-trt-11.2",
        "qualified_runtime_stack": {
            "sm": "sm103",
            "tensorrt": "11.2.0.113",
            "cuda_runtime": "13.3",
            "cudnn_backend": "9.20.0",
            "cudnn_frontend_revision": "7b9b711c22b6823e87150213ecd8449260db8610",
            "nvrtc": "13.3",
            "driver": "580.105.08",
        },
        "native_kv_plugin_abi": 2,
        "model_context_limit": spec.context_limit,
        "prefill_chunk_limit": spec.chunk_limit,
        "kv_layout": "contiguous_runtime_v1",
        "kv_dtype": "bfloat16",
        "kv_bytes_per_token": 114_688,
        "active_kv_profile_limits": list(spec.buckets),
        "runtime_owned": True,
    }
    variant_contract = dict(contract)
    variant_contract["prefill_chunk_limit"] = spec.chunk_limit // 2
    variant_contract["active_kv_profile_limits"] = sorted({*spec.buckets, spec.chunk_limit // 2})
    return (
        {"vocab_size": 151_936, "runtime_memory": contract},
        {"vocab_size": 151_936, "runtime_memory": variant_contract},
    )


def _file_identity(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_chunk_variant_receipt(
    tmp_path: Path,
    *,
    source_sha: str = "b" * 64,
) -> tuple[Path, Path, dict, dict, dict]:
    base, variant = _qwen_chunk_variant_headers()
    bundle = tmp_path / "variant.trtfb"
    bundle.write_bytes(b"variant-bundle")
    timing = tmp_path / "timing.json"
    timing.write_text('{"schema_version": 1}\n', encoding="utf-8")
    producer_path = REPO_ROOT / "tools" / "build_native_dynamic_memory_chunk_variant.py"
    source_state = {
        "git_head": "a" * 40,
        "source_state_sha256": source_sha,
    }
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    receipt = {
        "schema_version": qualify.CHUNK_VARIANT_BUILD_SCHEMA,
        "developer_only": True,
        "fresh_build": True,
        "artifact_reused": False,
        "source_state_unchanged": True,
        "opt_in": {
            "environment": qualify.DEVELOPER_CHUNK_VARIANT_ENV,
            "value": qualify.DEVELOPER_CHUNK_VARIANT_VALUE,
        },
        "builder_entrypoint": (
            "tensorrt_model_connect.engine_builder._build_native_impl_qualified"
        ),
        "qualified_model": {
            "model_id": spec.model_id,
            "revision": variant["runtime_memory"]["qualified_model_revision"],
            "config_sha256": variant["runtime_memory"]["qualified_config_sha256"],
            "target": variant["runtime_memory"]["qualified_target"],
            "model_dir": str(tmp_path / "model"),
        },
        "default_policy": {
            "prefill_chunk_limit": spec.chunk_limit,
            "active_kv_profile_limits": list(spec.buckets),
        },
        "variant_policy": {
            "prefill_chunk_limit": spec.chunk_limit // 2,
            "active_kv_profile_limits": sorted({*spec.buckets, spec.chunk_limit // 2}),
        },
        "bundle": _file_identity(bundle),
        "build_timing": _file_identity(timing),
        "producer": _file_identity(producer_path),
        "runtime_memory": variant["runtime_memory"],
        "source_state_pre": source_state,
        "source_state_post": dict(source_state),
    }
    receipt_path = tmp_path / "variant.receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt_path, bundle, base, variant, source_state


def test_chunk_variant_qualification_consumes_source_bound_build_receipt(
    tmp_path: Path,
) -> None:
    receipt, bundle, base, variant, source_state = _write_chunk_variant_receipt(tmp_path)

    validated = qualify._validate_chunk_variant_build_receipt(
        receipt_path=receipt,
        variant_bundle=bundle,
        base_header=base,
        variant_header=variant,
        spec=qualify.SPECS["Qwen/Qwen3-0.6B"],
        source_state=source_state,
    )

    assert validated["sha256"] == qualify._sha256(receipt)
    assert validated["bundle"]["sha256"] == qualify._sha256(bundle)
    assert validated["source_state_sha256"] == source_state["source_state_sha256"]


def test_chunk_variant_receipt_fails_closed_on_source_or_bundle_drift(
    tmp_path: Path,
) -> None:
    receipt, bundle, base, variant, source_state = _write_chunk_variant_receipt(tmp_path)
    changed_source = dict(source_state)
    changed_source["source_state_sha256"] = "c" * 64

    with pytest.raises(ValueError, match="does not match qualification source"):
        qualify._validate_chunk_variant_build_receipt(
            receipt_path=receipt,
            variant_bundle=bundle,
            base_header=base,
            variant_header=variant,
            spec=qualify.SPECS["Qwen/Qwen3-0.6B"],
            source_state=changed_source,
        )

    bundle.write_bytes(b"changed")
    with pytest.raises(ValueError, match="size identity mismatch"):
        qualify._validate_chunk_variant_build_receipt(
            receipt_path=receipt,
            variant_bundle=bundle,
            base_header=base,
            variant_header=variant,
            spec=qualify.SPECS["Qwen/Qwen3-0.6B"],
            source_state=source_state,
        )


def _hf_contract(*, revision: str, config_sha256: str) -> dict:
    return {
        "qualified_model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "qualified_model_revision": revision,
        "qualified_config_sha256": config_sha256,
    }


def test_hf_reference_rejects_wrong_snapshot_revision(tmp_path: Path) -> None:
    config = b'{"model_type":"llama"}\n'
    expected_revision = "a" * 40
    snapshot = tmp_path / "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0" / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(config)
    contract = _hf_contract(
        revision=expected_revision,
        config_sha256=hashlib.sha256(config).hexdigest(),
    )

    with pytest.raises(ValueError, match="exact qualified cache snapshot"):
        qualify.verify_hf_reference(str(snapshot), contract, remote_revision=None)


def test_hf_reference_rejects_wrong_config_fingerprint(tmp_path: Path) -> None:
    expected_revision = "a" * 40
    snapshot = (
        tmp_path / "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0" / "snapshots" / expected_revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"tampered"}\n', encoding="utf-8")
    contract = _hf_contract(
        revision=expected_revision,
        config_sha256=hashlib.sha256(b'{"model_type":"llama"}\n').hexdigest(),
    )

    with pytest.raises(ValueError, match="config fingerprint mismatch"):
        qualify.verify_hf_reference(str(snapshot), contract, remote_revision=None)


def test_remote_hf_reference_requires_exact_immutable_revision() -> None:
    contract = _hf_contract(
        revision="a" * 40,
        config_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="explicit immutable"):
        qualify.verify_hf_reference(
            contract["qualified_model_id"],
            contract,
            remote_revision=None,
        )


def test_logits_artifact_reader_checks_version_shape_and_payload(tmp_path: Path) -> None:
    path = tmp_path / "logits.bin"
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    path.write_bytes(
        qualify.LOGITS_HEADER.pack(qualify.LOGITS_MAGIC, 1, 1, values.shape[0], values.shape[1])
        + values.astype("<f4").tobytes()
    )

    np.testing.assert_array_equal(qualify.read_logits_artifact(path), values)

    path.write_bytes(
        struct.pack("<8sIIQQ", qualify.LOGITS_MAGIC, 1, 1, 3, 4)
        + values.astype("<f4").tobytes()[:-1]
    )
    with pytest.raises(ValueError, match="payload size mismatch"):
        qualify.read_logits_artifact(path)


def test_exact_logits_pass_all_existing_family_gates() -> None:
    logits = np.asarray(
        [[-1.0, 0.2, 2.0, 0.0], [0.1, 4.0, -2.0, 0.5]],
        dtype=np.float32,
    )
    thresholds = {
        "logit_atol": 0.001,
        "logit_cosine_p5": 0.99,
        "logit_rel_l2_p95": 0.05,
        "stable_margin": 0.1,
        "stable_top1_match_rate": 0.9,
        "token_agreement_rate": 0.8,
        "unstable_topk_hit_rate": 0.8,
    }

    result = qualify.compare_logits(logits, logits.copy(), [2], thresholds)

    assert result["passed"]
    assert all(result["gates"].values())


def test_stable_top1_divergence_fails_without_weakening_gate() -> None:
    hf = np.asarray([[0.0, 4.0, -2.0]], dtype=np.float32)
    trt = np.asarray([[5.0, 0.0, -2.0]], dtype=np.float32)
    thresholds = {
        "logit_atol": 10.0,
        "logit_cosine_p5": -1.0,
        "logit_rel_l2_p95": 10.0,
        "stable_margin": 0.1,
        "stable_top1_match_rate": 0.9,
        "token_agreement_rate": 0.8,
        "unstable_topk_hit_rate": 0.8,
    }

    result = qualify.compare_logits(trt, hf, [], thresholds)

    assert not result["passed"]
    assert not result["gates"]["stable_top1_match_rate"]
    assert not result["gates"]["token_agreement_rate"]


def test_family_composite_does_not_promote_atol_to_hard_gate() -> None:
    hf = np.asarray([[0.0, 4.0, -2.0]], dtype=np.float32)
    trt = np.asarray([[0.02, 4.02, -1.98]], dtype=np.float32)
    thresholds = {
        "logit_atol": 0.001,
        "logit_cosine_p5": 0.99,
        "logit_rel_l2_p95": 0.05,
        "stable_margin": 0.1,
        "stable_top1_match_rate": 0.9,
        "token_agreement_rate": 0.8,
        "unstable_topk_hit_rate": 0.8,
    }

    result = qualify.compare_logits(trt, hf, [1], thresholds)

    assert not result["gates"]["logit_atol"]
    assert result["composite_gates"] == {
        "numerical": True,
        "token_level": True,
    }
    assert result["passed"]


def test_family_composite_fails_when_cosine_and_relative_l2_both_fail() -> None:
    hf = np.asarray([[0.0, 4.0, -2.0]], dtype=np.float32)
    trt = np.asarray([[0.0, 0.1, 4.0]], dtype=np.float32)
    thresholds = {
        "logit_atol": 10.0,
        "logit_cosine_p5": 0.99,
        "logit_rel_l2_p95": 0.05,
        "stable_margin": 10.0,
        "stable_top1_match_rate": 0.0,
        "token_agreement_rate": 0.0,
        "unstable_topk_hit_rate": 0.0,
    }

    result = qualify.compare_logits(trt, hf, [], thresholds)

    assert not result["gates"]["logit_cosine_p5"]
    assert not result["gates"]["logit_rel_l2_p95"]
    assert not result["composite_gates"]["numerical"]
    assert not result["passed"]


def test_unstable_tie_uses_family_top5_fallback() -> None:
    # HF has a zero-margin tie between IDs 0 and 1; TRT selecting the other
    # tied ID must fail exact agreement but pass the family top-k fallback.
    hf = np.asarray([[4.0, 4.0, 3.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    trt = np.asarray([[4.01, 4.02, 3.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    thresholds = {
        "logit_atol": 0.001,
        "logit_cosine_p5": 0.99,
        "logit_rel_l2_p95": 0.05,
        "stable_margin": 0.1,
        "stable_top1_match_rate": 0.9,
        "token_agreement_rate": 0.8,
        "unstable_topk_hit_rate": 0.8,
    }

    result = qualify.compare_logits(trt, hf, [1], thresholds)

    assert result["metrics"]["hf_top1_margins"] == [0.0]
    assert not result["gates"]["token_agreement_rate"]
    assert result["gates"]["unstable_topk_hit_rate"]
    assert result["composite_gates"]["token_level"]
    assert result["passed"]


def _sampled_peak_receipt(context_bytes: int = 4096) -> dict:
    return {
        "context_device_memory_bytes": context_bytes,
        "peak_device_bytes": 8192,
        "peak_device_bytes_scope": "device_wide",
        "peak_device_sample_count": 2,
        "peak_device_sample_boundaries": [
            "after_runtime_kv_allocation",
            "after_successful_request_completion",
        ],
    }


def _plan_id(role: str) -> str:
    if role == "prefill":
        return "prefill_engine_plan@engine=0x1000"
    if role == "decode":
        return "engine_plan@engine=0x2000"
    raise ValueError(role)


def test_trace_validation_requires_exact_launch_formula_and_allocation_id() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    case = qualify.Case("c-plus-1", 1_025, 0)
    trace = {
        "prompt_tokens": 1_025,
        "prefill_chunk_limit": 1_024,
        "prefill_launches": 2,
        "decode_launches": 0,
        "final_kv_position": 1_025,
        "effective_request_limit": 40_960,
        "runtime_memory_receipt": {
            **_sampled_peak_receipt(),
            "kv_allocation_id": 7,
            "runtime_kv_capacity_tokens": 40_960,
            "kv_bytes_per_token": 114_688,
        },
        "invocations": [
            {
                "invocation_index": 0,
                "role": "prefill",
                "plan_id": _plan_id("prefill"),
                "profile_id": 6,
                "chunk_range": [0, 1_024],
                "launch_count": 1,
                "kv_allocation_id": 7,
                "kv_base_address": 4096,
                "H": 0,
                "A": 1_024,
                "T": 1,
                "R": 40_960,
                "context_device_memory_bytes": 2048,
                "cuda_graph_status": "uncaptured",
                "kv_device_to_host_bytes": 0,
                "kv_append_bytes": 1_024 * 114_688,
                "full_history_device_to_device_bytes": 0,
            },
            {
                "invocation_index": 1,
                "role": "prefill",
                "plan_id": _plan_id("prefill"),
                "profile_id": 6,
                "chunk_range": [1_024, 1_025],
                "launch_count": 1,
                "kv_allocation_id": 7,
                "kv_base_address": 4096,
                "H": 1_024,
                "A": 1_025,
                "T": 1_024,
                "R": 40_960,
                "context_device_memory_bytes": 4096,
                "cuda_graph_status": "uncaptured",
                "kv_device_to_host_bytes": 0,
                "kv_append_bytes": 114_688,
                "full_history_device_to_device_bytes": 0,
            },
        ],
    }

    qualify._validate_trace(case, spec, trace, np.zeros((1, 8), dtype=np.float32))
    trace["prefill_launches"] = 1
    with pytest.raises(RuntimeError, match="prefill_launches"):
        qualify._validate_trace(case, spec, trace, np.zeros((1, 8), dtype=np.float32))


def test_trace_validation_rejects_full_history_copy_traffic() -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    case = qualify.Case("one", 1, 0)
    trace = {
        "prompt_tokens": 1,
        "prefill_chunk_limit": 512,
        "prefill_launches": 1,
        "decode_launches": 0,
        "final_kv_position": 1,
        "effective_request_limit": 2_048,
        "runtime_memory_receipt": {
            **_sampled_peak_receipt(),
            "kv_allocation_id": 3,
            "runtime_kv_capacity_tokens": 2_048,
            "kv_bytes_per_token": 22_528,
        },
        "invocations": [
            {
                "invocation_index": 0,
                "role": "prefill",
                "plan_id": _plan_id("prefill"),
                "profile_id": 3,
                "chunk_range": [0, 1],
                "launch_count": 1,
                "kv_allocation_id": 3,
                "kv_base_address": 8192,
                "H": 0,
                "A": 1,
                "T": 1,
                "R": 2_048,
                "context_device_memory_bytes": 1024,
                "cuda_graph_status": "uncaptured",
                "kv_device_to_host_bytes": 0,
                "kv_append_bytes": 22_528,
                "full_history_device_to_device_bytes": 22_528,
            }
        ],
    }

    with pytest.raises(RuntimeError, match="full_history_device_to_device_bytes"):
        qualify._validate_trace(case, spec, trace, np.zeros((1, 8), dtype=np.float32))


def test_profile_crossing_trace_switches_decode_profile_without_reprefill() -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    case = qualify.Case(
        "profile-crossing-128",
        128,
        2,
        expected_decode_profile_ids=(0, 1),
        expected_decode_bucket_limits=(128, 256),
    )
    b = 22_528
    common = {
        "launch_count": 1,
        "kv_allocation_id": 5,
        "kv_base_address": 12_288,
        "R": 2_048,
        "context_device_memory_bytes": 2048,
        "cuda_graph_status": "uncaptured",
        "kv_device_to_host_bytes": 0,
        "full_history_device_to_device_bytes": 0,
    }
    invocations = [
        {
            **common,
            "invocation_index": 0,
            "role": "prefill",
            "plan_id": _plan_id("prefill"),
            "profile_id": 3,
            "chunk_range": [0, 128],
            "H": 0,
            "A": 128,
            "T": 1,
            "kv_append_bytes": 128 * b,
        },
        {
            **common,
            "invocation_index": 1,
            "role": "decode",
            "plan_id": _plan_id("decode"),
            "profile_id": 0,
            "chunk_range": [128, 129],
            "H": 128,
            "A": 129,
            "T": 128,
            "kv_append_bytes": b,
        },
        {
            **common,
            "invocation_index": 2,
            "role": "decode",
            "plan_id": _plan_id("decode"),
            "profile_id": 1,
            "chunk_range": [129, 130],
            "H": 129,
            "A": 130,
            "T": 256,
            "kv_append_bytes": b,
        },
    ]
    trace = {
        "prompt_tokens": 128,
        "prefill_chunk_limit": 512,
        "prefill_launches": 1,
        "decode_launches": 2,
        "final_kv_position": 130,
        "effective_request_limit": 2_048,
        "runtime_memory_receipt": {
            **_sampled_peak_receipt(),
            "kv_allocation_id": 5,
            "runtime_kv_capacity_tokens": 2_048,
            "kv_bytes_per_token": b,
        },
        "invocations": invocations,
    }
    qualify._validate_trace(case, spec, trace, np.zeros((3, 8), dtype=np.float32))
    assert qualify.context_shape_sweep(trace) == [
        {
            "role": "prefill",
            "Sq": 128,
            "H": 0,
            "A": 128,
            "T": 1,
            "R": 2_048,
            "context_device_memory_bytes": 2048,
        },
        {
            "role": "decode",
            "Sq": 1,
            "H": 128,
            "A": 129,
            "T": 128,
            "R": 2_048,
            "context_device_memory_bytes": 2048,
        },
        {
            "role": "decode",
            "Sq": 1,
            "H": 129,
            "A": 130,
            "T": 256,
            "R": 2_048,
            "context_device_memory_bytes": 2048,
        },
    ]

    invocations[-1]["profile_id"] = 0
    with pytest.raises(RuntimeError, match="decode profiles"):
        qualify._validate_trace(case, spec, trace, np.zeros((3, 8), dtype=np.float32))
    invocations[-1]["profile_id"] = 1

    invocations[0]["plan_id"] = "engine_plan:prefill"
    with pytest.raises(RuntimeError, match="invalid prefill plan identity"):
        qualify._validate_trace(case, spec, trace, np.zeros((3, 8), dtype=np.float32))
    invocations[0]["plan_id"] = _plan_id("prefill")

    invocations[0]["plan_id"] = "prefill_engine_plan@engine=0x0"
    with pytest.raises(RuntimeError, match="invalid prefill plan identity"):
        qualify._validate_trace(case, spec, trace, np.zeros((3, 8), dtype=np.float32))
    invocations[0]["plan_id"] = _plan_id("prefill")

    invocations[-1]["plan_id"] = "engine_plan@engine=0x3000"
    with pytest.raises(RuntimeError, match="share one engine identity"):
        qualify._validate_trace(case, spec, trace, np.zeros((3, 8), dtype=np.float32))
    invocations[-1]["plan_id"] = _plan_id("decode")

    for invocation in invocations[1:]:
        invocation["plan_id"] = "engine_plan@engine=0x1000"
    with pytest.raises(RuntimeError, match="same engine identity"):
        qualify._validate_trace(case, spec, trace, np.zeros((3, 8), dtype=np.float32))


def _context_memory_case(
    name: str,
    *,
    active_tokens: tuple[int, ...],
    context_bytes: tuple[int, ...],
    capacity_tokens: int = 40_960,
) -> dict[str, object]:
    assert len(active_tokens) == len(context_bytes)
    sweep: list[dict[str, object]] = []
    for role in ("prefill", "decode"):
        for active, measured_bytes in zip(
            active_tokens,
            context_bytes,
            strict=True,
        ):
            sweep.append(
                {
                    "role": role,
                    "Sq": 128 if role == "prefill" else 1,
                    "H": max(0, active - 1),
                    "A": active,
                    "T": active,
                    "R": capacity_tokens,
                    "context_device_memory_bytes": measured_bytes,
                }
            )
    return {
        "name": name,
        "actual_shape_context_sweep": sweep,
    }


def test_context_memory_envelope_accepts_linear_full_context_sweep() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    active_tokens = (128, 8_192, spec.context_limit)
    base_bytes = 2 * 1024 * 1024
    context_bytes = tuple(
        base_bytes + (spec.chunk_limit * (active - active_tokens[0]) * spec.num_query_heads)
        for active in active_tokens
    )

    result = qualify.validate_context_memory_envelope(
        spec,
        [
            _context_memory_case(
                "full-context",
                active_tokens=active_tokens,
                context_bytes=context_bytes,
            )
        ],
        require_full_coverage=True,
    )

    assert result["passed"]
    assert result["status"] == "passed"
    assert result["gates"]["all_points_within_o_c_times_a_envelope"]
    assert result["gates"]["all_points_below_materialized_score_bound"]
    assert result["gates"]["coverage"] == {
        "has_prefill_and_decode": True,
        "reaches_model_context_limit": True,
        "has_at_least_three_active_lengths": True,
    }


def test_context_memory_envelope_accepts_tinyllama_workspace_scaling() -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    result = qualify.validate_context_memory_envelope(
        spec,
        [
            _context_memory_case(
                "tinyllama-full-context",
                active_tokens=(128, 512, spec.context_limit),
                context_bytes=(11_223_552, 23_920_640, 23_920_640),
                capacity_tokens=spec.context_limit,
            )
        ],
        require_full_coverage=True,
    )

    assert result["passed"]
    assert result["max_score_equivalent_surfaces"] == 2


def test_context_memory_envelope_rejects_quadratic_growth() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    active_tokens = (128, 8_192, spec.context_limit)
    quadratic_bytes = tuple(spec.num_query_heads * active * active * 2 for active in active_tokens)

    result = qualify.validate_context_memory_envelope(
        spec,
        [
            _context_memory_case(
                "quadratic-full-score",
                active_tokens=active_tokens,
                context_bytes=quadratic_bytes,
            )
        ],
        require_full_coverage=True,
    )

    assert not result["passed"]
    assert result["status"] == "failed"
    assert not result["gates"]["all_points_within_o_c_times_a_envelope"]
    assert not result["gates"]["all_points_below_materialized_score_bound"]


def test_context_memory_envelope_requires_full_model_coverage() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    active_tokens = (128, 512, 8_192)
    context_bytes = (2_000_000, 3_000_000, 8_000_000)

    result = qualify.validate_context_memory_envelope(
        spec,
        [
            _context_memory_case(
                "short-only",
                active_tokens=active_tokens,
                context_bytes=context_bytes,
            )
        ],
        require_full_coverage=True,
    )

    assert not result["passed"]
    assert not result["gates"]["coverage"]["reaches_model_context_limit"]


def test_context_memory_envelope_allows_partial_case_qualification() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    result = qualify.validate_context_memory_envelope(
        spec,
        [
            _context_memory_case(
                "one-selected-case",
                active_tokens=(128,),
                context_bytes=(2_000_000,),
            )
        ],
        require_full_coverage=False,
    )

    assert result["passed"]
    assert not result["gates"]["coverage"]["reaches_model_context_limit"]


def test_trace_validation_rejects_kv_base_change_across_bucket() -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    case = qualify.Case(
        "profile-crossing-128",
        128,
        2,
        expected_decode_profile_ids=(0, 1),
        expected_decode_bucket_limits=(128, 256),
    )
    b = 22_528
    receipt = {
        **_sampled_peak_receipt(),
        "kv_allocation_id": 5,
        "runtime_kv_capacity_tokens": 2_048,
        "kv_bytes_per_token": b,
    }
    invocations = []
    for index, (role, begin, end, bound) in enumerate(
        (
            ("prefill", 0, 128, 1),
            ("decode", 128, 129, 128),
            ("decode", 129, 130, 256),
        )
    ):
        invocations.append(
            {
                "invocation_index": index,
                "role": role,
                "plan_id": _plan_id(role),
                "profile_id": index,
                "chunk_range": [begin, end],
                "launch_count": 1,
                "kv_allocation_id": 5,
                "kv_base_address": 4096 if index < 2 else 8192,
                "H": begin,
                "A": end,
                "T": bound,
                "R": 2_048,
                "context_device_memory_bytes": 2048,
                "cuda_graph_status": "uncaptured",
                "kv_device_to_host_bytes": 0,
                "kv_append_bytes": (end - begin) * b,
                "full_history_device_to_device_bytes": 0,
            }
        )
    trace = {
        "prompt_tokens": 128,
        "prefill_chunk_limit": 512,
        "prefill_launches": 1,
        "decode_launches": 2,
        "final_kv_position": 130,
        "effective_request_limit": 2_048,
        "runtime_memory_receipt": receipt,
        "invocations": invocations,
    }

    with pytest.raises(RuntimeError, match="replaced the KV base address"):
        qualify._validate_trace(case, spec, trace, np.zeros((3, 8), dtype=np.float32))


def _attributed_phase_sample(
    *,
    phase: str,
    cuda_free: int,
    current_process: int,
    other_process: int,
    nvml_used: int,
    post_nvml_free: int | None = None,
) -> dict:
    nvml_reserved = 100_000_000
    nvml_total = 1_100_000_000
    return {
        "phase": phase,
        "free_bytes": cuda_free,
        "total_bytes": 1_000_000_000,
        "process_used_bytes": current_process,
        "all_compute_process_used_bytes": current_process + other_process,
        "other_compute_process_used_bytes": other_process,
        "nvml_device_total_bytes": nvml_total,
        "nvml_device_reserved_bytes": nvml_reserved,
        "nvml_device_free_bytes": nvml_total - nvml_reserved - nvml_used,
        "nvml_device_used_bytes": nvml_used,
        "post_nvml_free_bytes": (cuda_free if post_nvml_free is None else post_nvml_free),
        "post_nvml_total_bytes": 1_000_000_000,
        "compute_processes": [
            {"pid": 123, "used_bytes": current_process},
            {"pid": 456, "used_bytes": other_process},
        ],
    }


def _attributed_peak_trace() -> dict:
    return {
        "memory_sampler": {
            "source": "nvmlDeviceGetComputeRunningProcesses_v3",
            "pid": 123,
            "captures_all_compute_processes": True,
            "device_memory_source": "nvmlDeviceGetMemoryInfo_v2",
        },
        "runtime_memory_receipt": {
            "peak_device_bytes": 100_000_000,
            "pre_load_total_bytes": 1_000_000_000,
        },
        "load_cycles": [
            {
                "runtime_phase_memory_samples": [
                    _attributed_phase_sample(
                        phase="before runtime-memory Qwen engine deserialization",
                        cuda_free=800_000_000,
                        current_process=100_000_000,
                        other_process=50_000_000,
                        nvml_used=200_000_000,
                    ),
                    _attributed_phase_sample(
                        phase="before runtime KV planning",
                        cuda_free=740_000_000,
                        current_process=160_000_000,
                        other_process=50_000_000,
                        nvml_used=260_000_000,
                    ),
                    _attributed_phase_sample(
                        phase="after runtime KV allocation",
                        cuda_free=700_000_000,
                        current_process=198_000_000,
                        other_process=50_000_000,
                        nvml_used=298_000_000,
                    ),
                    _attributed_phase_sample(
                        phase="after successful runtime-memory request completion",
                        cuda_free=705_000_000,
                        current_process=190_000_000,
                        other_process=55_000_000,
                        nvml_used=295_000_000,
                    ),
                ],
            }
        ],
    }


def test_peak_reconciliation_uses_independent_nvml_process_samples() -> None:
    trace = _attributed_peak_trace()

    result = qualify.reconcile_device_peak_with_nvml(trace)

    assert result["passed"]
    assert result["nvml_process_peak_bytes"] == 98_000_000
    assert result["absolute_difference_bytes"] == 2_000_000
    assert result["tolerance_bytes"] == 64 * 1024 * 1024
    assert result["synchronized_cuda_peak_bytes"] == 100_000_000

    trace["runtime_memory_receipt"]["peak_device_bytes"] = 200_000_000
    with pytest.raises(RuntimeError, match="does not match synchronized"):
        qualify.reconcile_device_peak_with_nvml(trace)


def test_peak_reconciliation_accepts_signed_visible_external_growth() -> None:
    trace = _attributed_peak_trace()
    allocation = trace["load_cycles"][0]["runtime_phase_memory_samples"][2]
    allocation.update(
        _attributed_phase_sample(
            phase="after runtime KV allocation",
            cuda_free=500_000_000,
            current_process=198_000_000,
            other_process=250_000_000,
            nvml_used=498_000_000,
        )
    )
    trace["runtime_memory_receipt"]["peak_device_bytes"] = 300_000_000

    result = qualify.reconcile_device_peak_with_nvml(trace)

    assert result["passed"]
    allocation_row = result["boundary_reconciliation"][0]
    assert allocation_row["nvml_visible_other_process_growth_bytes"] == 200_000_000
    assert allocation_row["nvml_non_current_device_growth_bytes"] == 200_000_000
    assert allocation_row["unexplained_growth_bytes"] == 2_000_000

    completion = trace["load_cycles"][0]["runtime_phase_memory_samples"][3]
    completion.update(
        _attributed_phase_sample(
            phase="after successful runtime-memory request completion",
            cuda_free=760_000_000,
            current_process=190_000_000,
            other_process=0,
            nvml_used=240_000_000,
        )
    )
    result = qualify.reconcile_device_peak_with_nvml(trace)
    assert result["passed"]
    assert (
        result["boundary_reconciliation"][1]["nvml_visible_other_process_growth_bytes"]
        == -50_000_000
    )


def test_peak_reconciliation_rejects_unexplained_or_unlisted_growth() -> None:
    trace = _attributed_peak_trace()
    trace["load_cycles"][0]["runtime_phase_memory_samples"][2]["free_bytes"] = 600_000_000
    trace["runtime_memory_receipt"]["peak_device_bytes"] = 200_000_000
    with pytest.raises(RuntimeError, match="external attribution"):
        qualify.reconcile_device_peak_with_nvml(trace)

    trace = _attributed_peak_trace()
    allocation = trace["load_cycles"][0]["runtime_phase_memory_samples"][2]
    allocation["nvml_device_used_bytes"] += 100_000_000
    allocation["nvml_device_free_bytes"] -= 100_000_000
    with pytest.raises(RuntimeError, match="external attribution"):
        qualify.reconcile_device_peak_with_nvml(trace)

    trace = _attributed_peak_trace()
    trace["load_cycles"][0]["runtime_phase_memory_samples"][2]["post_nvml_free_bytes"] -= (
        100_000_000
    )
    with pytest.raises(RuntimeError, match="external attribution"):
        qualify.reconcile_device_peak_with_nvml(trace)

    trace = _attributed_peak_trace()
    del trace["load_cycles"][0]["runtime_phase_memory_samples"][2]["all_compute_process_used_bytes"]
    with pytest.raises(RuntimeError, match="sample is invalid"):
        qualify.reconcile_device_peak_with_nvml(trace)


def test_peak_reconciliation_rejects_duplicate_required_boundary() -> None:
    trace = _attributed_peak_trace()
    samples = trace["load_cycles"][0]["runtime_phase_memory_samples"]
    samples.append(copy.deepcopy(samples[2]))

    with pytest.raises(RuntimeError, match="exactly one sample"):
        qualify.reconcile_device_peak_with_nvml(trace)


def test_peak_reconciliation_rejects_unsynchronized_lifetime_samples() -> None:
    trace = {
        "memory_sampler": {
            "source": "nvmlDeviceGetComputeRunningProcesses_v3",
            "pid": 123,
            "captures_all_compute_processes": True,
            "device_memory_source": "nvmlDeviceGetMemoryInfo_v2",
        },
        "runtime_memory_receipt": {
            "peak_device_bytes": 100_000_000,
            "pre_load_total_bytes": 1_000_000_000,
        },
        "load_cycles": [
            {
                "before_load": {"process_used_bytes": 100_000_000},
                "after_requests": {"process_used_bytes": 198_000_000},
            }
        ],
    }

    with pytest.raises(RuntimeError, match="no synchronized runtime phase"):
        qualify.reconcile_device_peak_with_nvml(trace)


def test_failure_checkpoint_persists_first_case_and_source_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_state = {
        "git_head": "a" * 40,
        "source_state_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        qualify,
        "source_state_provenance",
        lambda *_args, **_kwargs: dict(source_state),
    )
    report_path = tmp_path / "qualification-report.json"
    report = {
        "source_state_pre": dict(source_state),
        "status": "running",
        "passed": False,
        "cases": [],
    }

    with qualify.qualification_failure_checkpoint(
        report=report,
        report_path=report_path,
        repo_root=tmp_path,
        output_dir=tmp_path,
    ):
        report["cases"].append(
            {
                "name": "first-case",
                "status": "running",
                "stage": "chunk_variant_validation",
                "runner_evidence": {
                    "base": str(tmp_path / "runner-evidence" / "first-case" / "base")
                },
            }
        )
        raise RuntimeError("injected attribution failure")

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["source_state_unchanged"] is True
    assert persisted["failure"]["type"] == "RuntimeError"
    assert persisted["failure"]["stage"] == "chunk_variant_validation"
    assert persisted["cases"][0]["status"] == "failed"
    assert persisted["cases"][0]["execution_passed"] is False
    assert persisted["cases"][0]["failure"]["message"] == ("injected attribution failure")


def test_failure_checkpoint_persists_post_case_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_state = {
        "git_head": "a" * 40,
        "source_state_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        qualify,
        "source_state_provenance",
        lambda *_args, **_kwargs: dict(source_state),
    )
    report_path = tmp_path / "qualification-report.json"
    report = {
        "source_state_pre": dict(source_state),
        "status": "running",
        "stage": "context_memory_envelope",
        "passed": False,
        "cases": [{"name": "last-case", "status": "passed"}],
    }

    with qualify.qualification_failure_checkpoint(
        report=report,
        report_path=report_path,
        repo_root=tmp_path,
        output_dir=tmp_path,
    ):
        raise RuntimeError("injected finalization failure")

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["failure"]["stage"] == "context_memory_envelope"
    assert persisted["cases"][0]["status"] == "passed"
