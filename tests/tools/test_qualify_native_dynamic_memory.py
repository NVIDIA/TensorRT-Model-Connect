# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
SPEC = importlib.util.spec_from_file_location(
    "qualify_native_dynamic_memory", MODULE_PATH
)
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
        Path(
            before["artifacts"]["untracked_manifest"]["path"]
        ).read_text(encoding="utf-8")
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

    assert (cases["c-minus-1"].prompt_tokens,
            cases["c"].prompt_tokens,
            cases["c-plus-1"].prompt_tokens) == (2_047, 2_048, 2_049)
    assert cases["two-c-plus-17"].prompt_tokens == 4_113
    assert (cases["total-32768"].prompt_tokens,
            cases["total-32768"].decode_tokens) == (32_760, 8)
    assert (cases["total-model-limit"].prompt_tokens,
            cases["total-model-limit"].decode_tokens) == (40_952, 8)
    assert cases["prefill-last-position"].prompt_tokens == 40_960
    assert cases["model-limit-plus-1"].prompt_tokens == 40_961
    assert cases["model-limit-plus-1"].expect_admission_rejection
    for bucket in (128, 512, 2_048, 8_192, 32_768):
        crossing = cases[f"profile-crossing-{bucket}"]
        assert (crossing.prompt_tokens, crossing.decode_tokens) == (bucket, 2)


def test_tiny_matrix_covers_every_bucket_neighbor_and_m_plus_one() -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    cases = qualify._cases_for(spec)
    lengths = {case.prompt_tokens for case in cases}

    for bucket in spec.buckets:
        assert {bucket - 1, bucket, bucket + 1}.issubset(lengths)
    rejected = [case for case in cases if case.expect_admission_rejection]
    assert [(case.prompt_tokens, case.decode_tokens) for case in rejected] == [(2_049, 0)]
    assert any(
        case.prompt_tokens == 2_040 and case.decode_tokens == 8
        for case in cases
    )
    assert any(
        case.prompt_tokens == 2_048 and case.decode_tokens == 0
        for case in cases
    )
    by_name = {case.name: case for case in cases}
    for bucket in (128, 512):
        assert (
            by_name[f"profile-crossing-{bucket}"].prompt_tokens,
            by_name[f"profile-crossing-{bucket}"].decode_tokens,
        ) == (bucket, 2)


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
            "cudnn_frontend_revision":
                "7b9b711c22b6823e87150213ecd8449260db8610",
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
    variant_contract["active_kv_profile_limits"] = [
        128, 512, spec.chunk_limit // 2, spec.chunk_limit,
        8_192, 32_768, spec.context_limit,
    ]
    variant = {
        "vocab_size": base["vocab_size"],
        "runtime_memory": variant_contract,
    }

    assert qualify._validate_chunk_variant(
        base, variant, spec
    ) == spec.chunk_limit // 2

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
            "cudnn_frontend_revision":
                "7b9b711c22b6823e87150213ecd8449260db8610",
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
    variant_contract["active_kv_profile_limits"] = sorted(
        {*spec.buckets, spec.chunk_limit // 2}
    )
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
    producer_path = (
        REPO_ROOT / "tools" / "build_native_dynamic_memory_chunk_variant.py"
    )
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
            "tensorrt_model_connect.engine_builder."
            "_build_native_impl_qualified"
        ),
        "qualified_model": {
            "model_id": spec.model_id,
            "revision": variant["runtime_memory"][
                "qualified_model_revision"
            ],
            "config_sha256": variant["runtime_memory"][
                "qualified_config_sha256"
            ],
            "target": variant["runtime_memory"]["qualified_target"],
            "model_dir": str(tmp_path / "model"),
        },
        "default_policy": {
            "prefill_chunk_limit": spec.chunk_limit,
            "active_kv_profile_limits": list(spec.buckets),
        },
        "variant_policy": {
            "prefill_chunk_limit": spec.chunk_limit // 2,
            "active_kv_profile_limits": sorted(
                {*spec.buckets, spec.chunk_limit // 2}
            ),
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
    receipt, bundle, base, variant, source_state = (
        _write_chunk_variant_receipt(tmp_path)
    )

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
    assert validated["source_state_sha256"] == source_state[
        "source_state_sha256"
    ]


def test_chunk_variant_receipt_fails_closed_on_source_or_bundle_drift(
    tmp_path: Path,
) -> None:
    receipt, bundle, base, variant, source_state = (
        _write_chunk_variant_receipt(tmp_path)
    )
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
    snapshot = (
        tmp_path
        / "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
        / "snapshots"
        / ("b" * 40)
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(config)
    contract = _hf_contract(
        revision=expected_revision,
        config_sha256=hashlib.sha256(config).hexdigest(),
    )

    with pytest.raises(ValueError, match="exact qualified cache snapshot"):
        qualify.verify_hf_reference(
            str(snapshot), contract, remote_revision=None
        )


def test_hf_reference_rejects_wrong_config_fingerprint(tmp_path: Path) -> None:
    expected_revision = "a" * 40
    snapshot = (
        tmp_path
        / "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
        / "snapshots"
        / expected_revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        '{"model_type":"tampered"}\n', encoding="utf-8"
    )
    contract = _hf_contract(
        revision=expected_revision,
        config_sha256=hashlib.sha256(b'{"model_type":"llama"}\n').hexdigest(),
    )

    with pytest.raises(ValueError, match="config fingerprint mismatch"):
        qualify.verify_hf_reference(
            str(snapshot), contract, remote_revision=None
        )


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
        qualify.LOGITS_HEADER.pack(
            qualify.LOGITS_MAGIC, 1, 1, values.shape[0], values.shape[1]
        )
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


def test_trace_validation_requires_exact_launch_formula_and_allocation_id() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    case = qualify.Case("c-plus-1", 2_049, 0)
    trace = {
        "prompt_tokens": 2_049,
        "prefill_chunk_limit": 2_048,
        "prefill_launches": 2,
        "decode_launches": 0,
        "final_kv_position": 2_049,
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
                "plan_id": "engine_plan:prefill",
                "profile_id": 6,
                "chunk_range": [0, 2_048],
                "launch_count": 1,
                "kv_allocation_id": 7,
                "kv_base_address": 4096,
                "H": 0,
                "A": 2_048,
                "T": 1,
                "R": 40_960,
                "context_device_memory_bytes": 2048,
                "cuda_graph_status": "uncaptured",
                "kv_device_to_host_bytes": 0,
                "kv_append_bytes": 2_048 * 114_688,
                "full_history_device_to_device_bytes": 0,
            },
            {
                "invocation_index": 1,
                "role": "prefill",
                "plan_id": "engine_plan:prefill",
                "profile_id": 6,
                "chunk_range": [2_048, 2_049],
                "launch_count": 1,
                "kv_allocation_id": 7,
                "kv_base_address": 4096,
                "H": 2_048,
                "A": 2_049,
                "T": 2_048,
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
        qualify._validate_trace(
            case, spec, trace, np.zeros((1, 8), dtype=np.float32)
        )


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
                "plan_id": "engine_plan:prefill",
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
    case = qualify.Case("profile-crossing-128", 128, 2)
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
            "plan_id": "engine_plan:prefill",
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
            "plan_id": "engine_plan:decode",
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
            "plan_id": "engine_plan:decode",
            "profile_id": 1,
            "chunk_range": [129, 130],
            "H": 129,
            "A": 130,
            "T": 512,
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
            "T": 512,
            "R": 2_048,
            "context_device_memory_bytes": 2048,
        },
    ]

    invocations[-1]["profile_id"] = 0
    with pytest.raises(RuntimeError, match="profile did not switch"):
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
        base_bytes
        + (
            spec.chunk_limit
            * (active - active_tokens[0])
            * spec.num_query_heads
        )
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
    quadratic_bytes = tuple(
        spec.num_query_heads * active * active * 2
        for active in active_tokens
    )

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
    case = qualify.Case("profile-crossing-128", 128, 2)
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
            ("decode", 129, 130, 512),
        )
    ):
        invocations.append(
            {
                "invocation_index": index,
                "role": role,
                "plan_id": f"engine_plan:{role}",
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
        qualify._validate_trace(
            case, spec, trace, np.zeros((3, 8), dtype=np.float32)
        )


def test_peak_reconciliation_uses_independent_nvml_process_samples() -> None:
    trace = {
        "memory_sampler": {
            "source": "nvmlDeviceGetComputeRunningProcesses_v3",
            "pid": 123,
        },
        "runtime_memory_receipt": {
            "peak_device_bytes": 100_000_000,
            "pre_load_total_bytes": 1_000_000_000,
        },
        "load_cycles": [
            {
                "runtime_phase_memory_samples": [
                    {
                        "phase": (
                            "before runtime-memory Qwen engine deserialization"
                        ),
                        "free_bytes": 800_000_000,
                        "total_bytes": 1_000_000_000,
                        "process_used_bytes": 100_000_000,
                    },
                    {
                        "phase": "before runtime KV planning",
                        "free_bytes": 740_000_000,
                        "total_bytes": 1_000_000_000,
                        "process_used_bytes": 160_000_000,
                    },
                    {
                        "phase": "after runtime KV allocation",
                        "free_bytes": 700_000_000,
                        "total_bytes": 1_000_000_000,
                        "process_used_bytes": 198_000_000,
                    },
                    {
                        "phase": (
                            "after successful runtime-memory request completion"
                        ),
                        "free_bytes": 705_000_000,
                        "total_bytes": 1_000_000_000,
                        "process_used_bytes": 190_000_000,
                    },
                ],
            }
        ],
    }

    result = qualify.reconcile_device_peak_with_nvml(trace)

    assert result["passed"]
    assert result["nvml_process_peak_bytes"] == 98_000_000
    assert result["absolute_difference_bytes"] == 2_000_000
    assert result["tolerance_bytes"] == 64 * 1024 * 1024
    assert result["synchronized_cuda_peak_bytes"] == 100_000_000

    trace["runtime_memory_receipt"]["peak_device_bytes"] = 200_000_000
    with pytest.raises(RuntimeError, match="does not match synchronized"):
        qualify.reconcile_device_peak_with_nvml(trace)

    trace["load_cycles"][0]["runtime_phase_memory_samples"][2][
        "free_bytes"
    ] = 600_000_000
    with pytest.raises(RuntimeError, match="do not reconcile"):
        qualify.reconcile_device_peak_with_nvml(trace)


def test_peak_reconciliation_rejects_unsynchronized_lifetime_samples() -> None:
    trace = {
        "memory_sampler": {
            "source": "nvmlDeviceGetComputeRunningProcesses_v3",
            "pid": 123,
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
