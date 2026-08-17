# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any, Callable

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools/sam2_native_qualification/generate_record.py"
SPEC = importlib.util.spec_from_file_location("sam2_qualification_generator", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compact(value: dict[str, Any], *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (json.dumps(value, separators=(",", ":")) + suffix).encode()


def _attention() -> dict[str, Any]:
    return {
        "implementation": "tensorrt_iattention_v2",
        "operator": "IAttention",
        "api": "addAttentionV2",
        "block_count": 16,
        "head_dimension": 96,
        "query_form": "padded_bhnd",
        "key_value_form": "padded_bhnd",
        "output_form": "padded_bhnd",
        "normalization": "softmax",
        "causal_mask": "none",
        "decomposable": False,
        "fused_kernel_intent": True,
        "metadata_prefix": "trtmc.sam2.iattention.block.",
        "metadata_index_width": 2,
        "q_scale_formula": "1/sqrt(head_dimension)",
        "k_scale_formula": "none",
        "effective_score_scale": "1/sqrt(head_dimension)",
        "scale_dtype": "bf16",
    }


def _config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "family": tool.FAMILY,
        "model_id": tool.MODEL_ID,
        "engine_contract_version": tool.ENGINE_CONTRACT_VERSION,
        "runtime_strategy": tool.STRATEGY,
        "precision": tool.PRECISION,
        "checkpoint_sha256": tool.CHECKPOINT_SHA256,
        "source_config_sha256": tool.CONFIG_SHA256,
        "golden_manifest_sha256": tool.GOLDEN_MANIFEST_SHA256,
        "frame_count": 5,
        "selected_object_count": 1,
        "model_image_size": 1024,
        "original_image_height": 1280,
        "original_image_width": 1088,
        "plan_sections": list(tool.PLAN_SECTIONS),
        "qualification": "unqualified",
        "runtime_eligible": False,
    }


def _write_bundle(
    path: Path,
    *,
    salt: bytes = b"",
    mutate_build_receipt: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    plans = [f"plan-{index}".encode() + salt for index in range(6)]
    config = _compact(_config())
    graph_rows = []
    for index, name in enumerate(tool.PLAN_SECTIONS):
        row = {
            "section": name,
            "kind": tool.GRAPH_KINDS[index],
            "history_frames": tool.GRAPH_HISTORY_FRAMES[index],
            "inputs": tool.GRAPH_INPUTS[index],
            "outputs": tool.GRAPH_OUTPUTS[index],
            "layers": tool.GRAPH_LAYERS[index],
        }
        if index == 0:
            row.update(tool.IMAGE_GRAPH_LAYER_COUNTS)
        row.update(
            {
                "referenced_checkpoint_tensors": tool.GRAPH_REFERENCED_TENSORS[index],
                "serialized_bytes": len(plans[index]),
                "serialized_sha256": _sha(plans[index]),
                "graph_complete": True,
            }
        )
        graph_rows.append(row)
    build_receipt = {
        "schema_version": tool.BUILD_RECEIPT_SCHEMA_VERSION,
        "family": tool.FAMILY,
        "model_id": tool.MODEL_ID,
        "qualification": {
            "state": "unqualified",
            "runtime_eligible": False,
            "golden_parity_verified": False,
        },
        "assets": {
            "checkpoint_sha256": tool.CHECKPOINT_SHA256,
            "source_config_sha256": tool.CONFIG_SHA256,
            "golden_manifest_sha256": tool.GOLDEN_MANIFEST_SHA256,
            "embedded_config_sha256": _sha(config),
        },
        "build": {
            "created_at_utc": "2026-08-17T00:00:00Z",
            "workspace_bytes": 1,
            "network_mode": "strongly_typed",
            "tf32_enabled": False,
            "builder_optimization_level": tool.BUILDER_OPTIMIZATION_LEVEL,
            "plan_profiling_verbosity": "detailed",
            "tensorrt_version": tool.TENSORRT_VERSION,
            "tensorrt_abi": tool.TENSORRT_ABI,
            "cuda_runtime_version": "13.3.0",
            "cuda_driver_version": "13.0.0",
            "gpu": {
                "device": 0,
                "name": tool.GPU_NAME,
                "compute_capability": tool.COMPUTE_CAPABILITY,
                "global_memory_bytes": 1,
            },
        },
        "image_attention": _attention(),
        "graphs": graph_rows,
    }
    if mutate_build_receipt is not None:
        mutate_build_receipt(build_receipt)
    receipt = _compact(build_receipt)
    payloads = plans + [config, receipt]
    sections: dict[str, Any] = {}
    offset = 0
    for name, payload in zip(tool.ALL_SECTIONS, payloads, strict=True):
        sections[name] = {"offset": offset, "size": len(payload), "sha256": _sha(payload)}
        offset += len(payload)
    header = {
        "model_id": tool.MODEL_ID,
        "model_type": "sam2_video_tracking",
        "family": tool.FAMILY,
        "precision": tool.PRECISION,
        "trt_version": tool.TENSORRT_VERSION,
        "trt_abi": tool.TENSORRT_ABI,
        "gpu_name": tool.GPU_NAME,
        "created_at": "2026-08-17T00:00:00Z",
        "runtime_strategy": tool.STRATEGY,
        "sections": sections,
    }
    header_bytes = _compact(header)
    bundle_bytes = (
        tool.BUNDLE_MAGIC + struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(payloads)
    )
    path.write_bytes(bundle_bytes)
    return {
        "sha256": _sha(bundle_bytes),
        "size": len(bundle_bytes),
        "section_sha256": {name: sections[name]["sha256"] for name in tool.ALL_SECTIONS},
    }


def _runtime(end: str) -> dict[str, Any]:
    return {
        "gpu_device": 0,
        "gpu_name": tool.GPU_NAME,
        "compute_capability": tool.COMPUTE_CAPABILITY,
        "global_memory_bytes": 1,
        "tensorrt_version": tool.TENSORRT_VERSION,
        "tensorrt_abi": tool.TENSORRT_ABI,
        "cuda_runtime_version": "13.3.0",
        "cuda_driver_version": "13.0.0",
        "hostname": "ipp2-2249",
        "started_at_utc": "2026-08-17T00:01:00Z",
        "ended_at_utc": end,
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "pci_bus_id": "0000:01:00.0",
        "cxx_compiler_id": "GNU",
        "cxx_compiler_version": "13.3.0",
        "cxx_language_standard": 201703,
        "engine_profiling_verbosity": "detailed",
        "execution_context_nvtx_verbosity": "none",
    }


def _replay(index: int) -> dict[str, Any]:
    return {
        "index": index,
        "mask_sha256": "a" * 64,
        "bbox_sha256": "b" * 64,
        "foreground_pixels": [1, 1, 1, 1, 1],
        "frame_iou": [0.999] * 5,
        "macro_iou": 0.999,
        "global_iou": 0.999,
        "bbox_iou": 1.0,
        "bbox_max_coordinate_error": 0.0,
        "bbox_score_error": 0.001,
        "bbox_label_exact": True,
        "candidate_bbox": {"label": 1, "score": 0.9, "original_image_xyxy": [1.0] * 4},
        "reference_bbox": {"label": 1, "score": 0.9, "original_image_xyxy": [1.0] * 4},
        "passes": True,
    }


def _common_assets(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_sha256": tool.CHECKPOINT_SHA256,
        "source_config_sha256": tool.CONFIG_SHA256,
        "golden_manifest_sha256": tool.GOLDEN_MANIFEST_SHA256,
        "golden_masks_sha256": tool.GOLDEN_MASKS_SHA256,
        "encoded_jpeg_sha256": list(tool.ENCODED_JPEG_SHA256),
        "decoded_rgb_sha256": list(tool.DECODED_JPEG_SHA256),
        "native_bundle_sha256": bundle["sha256"],
        "native_build_receipt_sha256": bundle["section_sha256"]["sam2_build_receipt.json"],
        "native_plans": [
            {"section": name, "sha256": bundle["section_sha256"][name]}
            for name in tool.PLAN_SECTIONS
        ],
        "benchmark_executable_sha256": "c" * 64,
        "benchmark_source_manifest_sha256": "d" * 64,
        "benchmark_source_closure_sha256": "e" * 64,
        "benchmark_source_closure_role": tool.BENCHMARK_SOURCE_CLOSURE_ROLE,
    }


def _accuracy(key: str, replays: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "thresholds": copy.deepcopy(tool.THRESHOLDS),
        "repeat_hashes_exact": False,
        "foreground_counts_exact": False,
        "repeat_contract": tool.REPEAT_CONTRACT,
        key: replays,
    }


def _write_receipts(directory: Path, bundle: dict[str, Any]) -> tuple[Path, Path]:
    status_q3 = {
        "accuracy_qualified_for_this_diagnostic": True,
        "runtime_eligible": False,
        "performance_claim": False,
        "timing_performed": False,
        "outlier_filtering": False,
    }
    replays = [_replay(index) for index in range(3)]
    q3 = {
        "schema_version": tool.BENCHMARK_RECEIPT_SCHEMA_VERSION,
        "family": tool.FAMILY,
        "workload": tool.WORKLOAD,
        "mode": "accuracy_only",
        "accuracy_only": True,
        "timing_performed": False,
        "status": status_q3,
        "process_model": copy.deepcopy(tool.Q3_PROCESS_MODEL),
        "sequence": {
            "accuracy_replays": 3,
            "frames_per_replay": 5,
            "reset_before_each_replay": True,
            "order": "Q3_only",
            "warmup_rows": 0,
            "measurement_rows": 0,
            "postqualification_replays": 0,
        },
        "assets": _common_assets(bundle),
        "runtime": _runtime("2026-08-17T00:02:00Z"),
        "image_attention": _attention(),
        "accuracy": _accuracy("replays", replays),
    }
    q3_path = directory / "q3.json"
    q3_bytes = _compact(q3, newline=True)
    q3_path.write_bytes(q3_bytes)

    def row(index: int, *, warmup: bool = False) -> dict[str, Any]:
        base = 1_000_000 if warmup else 20_000_000
        prefill = base + index * 1_000
        tracker = base * 6 + index * 2_000
        total = prefill + tracker
        return {
            "index": index,
            "native_prefill_ns": prefill,
            "native_tracker_ns": tracker,
            "closest_envelope_total_ns": total,
            "native_prefill_ms": prefill / 1_000_000,
            "native_tracker_ms": tracker / 1_000_000,
            "closest_envelope_total_ms": total / 1_000_000,
        }

    warmup_rows = [row(index, warmup=True) for index in range(3)]
    measurement_rows = [row(index) for index in range(100)]

    def metric_summary(key: str) -> dict[str, float]:
        values = sorted(item[key] for item in measurement_rows)
        return {
            "mean_ms": round(sum(values) / len(values) / 1_000_000, 6),
            "median_ms": round((values[49] + values[50]) / 2 / 1_000_000, 6),
            "p90_ms": round(values[89] / 1_000_000, 6),
            "min_ms": round(values[0] / 1_000_000, 6),
            "max_ms": round(values[-1] / 1_000_000, 6),
        }

    regular_assets = _common_assets(bundle)
    regular_assets.update(
        {
            "baseline_receipt_sha256": tool.BASELINE_RECEIPT_SHA256,
            "baseline_capture_script_sha256": tool.BASELINE_CAPTURE_SHA256,
            "q3_receipt_sha256": _sha(q3_bytes),
            "q3_receipt_size_bytes": len(q3_bytes),
            "q3_receipt_role": "exclusive same-process same-bundle Q3 receipt published before W3",
        }
    )
    regular = {
        "schema_version": tool.BENCHMARK_RECEIPT_SCHEMA_VERSION,
        "family": tool.FAMILY,
        "workload": tool.WORKLOAD,
        "mode": "diagnostic_benchmark",
        "accuracy_only": False,
        "timing_performed": True,
        "status": {**status_q3, "timing_performed": True},
        "process_model": copy.deepcopy(tool.REGULAR_PROCESS_MODEL),
        "sequence": {
            "prequalification_replays": 3,
            "warmup_rows": 3,
            "measurement_rows": 100,
            "postqualification_replays": 1,
            "order": "Q3_then_W3_then_N100_then_Q1",
            "accuracy_materialization_between_timing_rows": False,
        },
        "timing_boundaries": copy.deepcopy(tool.TIMING_BOUNDARIES),
        "assets": regular_assets,
        "runtime": _runtime("2026-08-17T00:03:00Z"),
        "image_attention": _attention(),
        "accuracy": {
            **_accuracy("prequalification", replays),
            "postqualification": [_replay(0)],
        },
        "timing": {
            "sample_count": 100,
            "excluded_rows": 0,
            "p90_method": tool.P90_METHOD,
            "outlier_removal": False,
            "warmup_rows": warmup_rows,
            "measurement_rows": measurement_rows,
            "summary": {
                "native_prefill": metric_summary("native_prefill_ns"),
                "native_tracker": metric_summary("native_tracker_ns"),
                "closest_envelope_total": metric_summary("closest_envelope_total_ns"),
            },
        },
        "delivered_baseline_reference": copy.deepcopy(tool.DELIVERED_BASELINE_REFERENCE),
    }
    regular_path = directory / "regular.json"
    regular_path.write_bytes(_compact(regular, newline=True))
    return q3_path, regular_path


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    bundle_path = tmp_path / "model.bundle"
    bundle = _write_bundle(bundle_path)
    q3, regular = _write_receipts(tmp_path, bundle)
    return bundle_path, q3, regular


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(_compact(value, newline=True))


def _rebind_regular_to_q3(q3_path: Path, regular_path: Path) -> None:
    regular = _read_json(regular_path)
    q3_bytes = q3_path.read_bytes()
    regular["assets"]["q3_receipt_sha256"] = _sha(q3_bytes)
    regular["assets"]["q3_receipt_size_bytes"] = len(q3_bytes)
    _write_json(regular_path, regular)


def _generate(tmp_path: Path, bundle: Path, q3: Path, regular: Path) -> tuple[bytes, bytes]:
    return tool.generate(
        bundle_path=bundle,
        q3_receipt_path=q3,
        regular_receipt_path=regular,
        record_output=tmp_path / "record.json",
        audit_output=tmp_path / "audit.json",
        authority_id="sam2-l4-trt11.1-contract5-0001",
        authority_serial=1,
        generated_at_utc="2026-08-17T00:04:00Z",
    )


def _publication_temps(directory: Path) -> list[Path]:
    return list(directory.glob(".sam2-qualification-*.tmp"))


def test_generator_emits_non_authorizing_record_and_audit(tmp_path: Path) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    record_bytes, audit_bytes = _generate(tmp_path, bundle, q3, regular)
    record = json.loads(record_bytes)
    audit = json.loads(audit_bytes)
    assert record["schema_version"] == 2
    assert record["self_authorizing"] is False
    assert record["accuracy_evidence"]["regular_receipt_sha256"] == _sha(regular.read_bytes())
    assert audit["self_authorizing"] is False
    assert audit["pin_mutation_supported"] is False
    assert audit["derived_gates"]["q1_all_semantic_gates_passed"] is True
    assert record_bytes == (tmp_path / "record.json").read_bytes()
    assert audit_bytes == (tmp_path / "audit.json").read_bytes()
    assert _publication_temps(tmp_path) == []


def test_exclusive_writer_publishes_only_complete_file_and_fsyncs_without_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "record.json"
    payload = b'{"complete":true}\n'
    original_link = tool.os.link
    original_fsync = tool.os.fsync
    observed_link = False
    directory_fsyncs = 0

    def inspect_link(source: str, target: str, **kwargs: Any) -> None:
        nonlocal observed_link
        assert source.startswith(".sam2-qualification-")
        assert target == output.name
        assert kwargs["follow_symlinks"] is False
        source_dir = Path(os.readlink(f"/proc/self/fd/{kwargs['src_dir_fd']}"))
        assert (source_dir / source).read_bytes() == payload
        assert not output.exists()
        original_link(source, target, **kwargs)
        assert output.read_bytes() == payload
        observed_link = True

    def inspect_fsync(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            assert _publication_temps(tmp_path) == []
            directory_fsyncs += 1
        original_fsync(descriptor)

    monkeypatch.setattr(tool.os, "link", inspect_link)
    monkeypatch.setattr(tool.os, "fsync", inspect_fsync)
    publication = tool._write_exclusive(output, payload, "record output")
    publication.close()
    assert observed_link
    assert directory_fsyncs == 1
    assert output.read_bytes() == payload


@pytest.mark.parametrize("fault", ["write", "file_fsync", "link", "temp_unlink", "dir_fsync"])
def test_exclusive_writer_faults_remove_only_owned_artifacts_and_fsync_parent_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    output = tmp_path / "record.json"
    original_write = tool.os.write
    original_fsync = tool.os.fsync
    original_link = tool.os.link
    original_unlink = tool.os.unlink
    fired = False
    directory_fsync_attempts = 0

    def fail_write(descriptor: int, data: bytes) -> int:
        nonlocal fired
        if fault == "write" and not fired:
            fired = True
            raise OSError(errno.EIO, "injected write failure")
        return original_write(descriptor, data)

    def fail_fsync(descriptor: int) -> None:
        nonlocal fired, directory_fsync_attempts
        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        if is_directory:
            assert _publication_temps(tmp_path) == []
            directory_fsync_attempts += 1
        if fault == "file_fsync" and not is_directory and not fired:
            fired = True
            raise OSError(errno.EIO, "injected file fsync failure")
        if fault == "dir_fsync" and is_directory and not fired:
            fired = True
            raise OSError(errno.EIO, "injected directory fsync failure")
        original_fsync(descriptor)

    def fail_link(source: str, target: str, **kwargs: Any) -> None:
        nonlocal fired
        if fault == "link" and not fired:
            fired = True
            raise OSError(errno.EIO, "injected link failure")
        original_link(source, target, **kwargs)

    def fail_unlink(path: str, **kwargs: Any) -> None:
        nonlocal fired
        if fault == "temp_unlink" and path.startswith(".sam2-qualification-") and not fired:
            fired = True
            raise OSError(errno.EIO, "injected temp unlink failure")
        original_unlink(path, **kwargs)

    monkeypatch.setattr(tool.os, "write", fail_write)
    monkeypatch.setattr(tool.os, "fsync", fail_fsync)
    monkeypatch.setattr(tool.os, "link", fail_link)
    monkeypatch.setattr(tool.os, "unlink", fail_unlink)
    with pytest.raises(tool.EvidenceError, match=r"cleanup: .*parent-fsync=ok"):
        tool._write_exclusive(output, b"complete bytes\n", "record output")
    assert fired
    assert not output.exists()
    assert _publication_temps(tmp_path) == []
    assert directory_fsync_attempts == (2 if fault == "dir_fsync" else 1)


@pytest.mark.parametrize("target_kind", ["regular", "symlink"])
def test_exclusive_writer_never_replaces_or_removes_existing_target(
    tmp_path: Path, target_kind: str
) -> None:
    output = tmp_path / "record.json"
    expected = b"existing target\n"
    if target_kind == "regular":
        output.write_bytes(expected)
    else:
        victim = tmp_path / "victim.json"
        victim.write_bytes(expected)
        output.symlink_to(victim.name)
    with pytest.raises(tool.EvidenceError, match="cannot publish exclusive"):
        tool._write_exclusive(output, b"replacement\n", "record output")
    if target_kind == "regular":
        assert output.read_bytes() == expected
    else:
        assert output.is_symlink()
        assert output.readlink() == Path("victim.json")
        assert (tmp_path / "victim.json").read_bytes() == expected
    assert _publication_temps(tmp_path) == []


def test_exclusive_writer_uses_exclusive_nofollow_temp_and_nofollow_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "record.json"
    original_open = tool.os.open
    observed_flags: list[int] = []

    def inspect_open(path: str | os.PathLike[str], flags: int, *args: Any, **kwargs: Any) -> int:
        if isinstance(path, str) and path.startswith(".sam2-qualification-"):
            observed_flags.append(flags)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(tool.os, "open", inspect_open)
    publication = tool._write_exclusive(output, b"complete\n", "record output")
    publication.close()
    assert len(observed_flags) == 1
    assert observed_flags[0] & os.O_EXCL
    assert observed_flags[0] & getattr(os, "O_NOFOLLOW", 0)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(tool.EvidenceError, match="cannot open no-follow"):
        tool._write_exclusive(symlink_parent / "audit.json", b"complete\n", "audit output")
    assert not (real_parent / "audit.json").exists()


def test_audit_publish_failure_rolls_back_durable_record_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    original_write_exclusive = tool._write_exclusive

    def fail_audit(path: Path, data: bytes, label: str) -> Any:
        if label == "audit output":
            tool._fail("injected audit publication failure")
        return original_write_exclusive(path, data, label)

    monkeypatch.setattr(tool, "_write_exclusive", fail_audit)
    with pytest.raises(
        tool.EvidenceError,
        match=r"injected audit publication failure.*record rollback: record=removed, "
        r"parent-fsync=ok",
    ):
        _generate(tmp_path, bundle, q3, regular)
    assert not (tmp_path / "record.json").exists()
    assert not (tmp_path / "audit.json").exists()
    assert _publication_temps(tmp_path) == []


def test_audit_failure_retries_authenticated_record_rollback_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    original_write_exclusive = tool._write_exclusive
    original_unlink = tool.os.unlink
    failed_record_unlink = False

    def fail_audit(path: Path, data: bytes, label: str) -> Any:
        if label == "audit output":
            tool._fail("injected audit publication failure")
        return original_write_exclusive(path, data, label)

    def fail_first_record_unlink(path: str, **kwargs: Any) -> None:
        nonlocal failed_record_unlink
        if path == "record.json" and not failed_record_unlink:
            failed_record_unlink = True
            raise OSError(errno.EIO, "injected record rollback unlink failure")
        original_unlink(path, **kwargs)

    monkeypatch.setattr(tool, "_write_exclusive", fail_audit)
    monkeypatch.setattr(tool.os, "unlink", fail_first_record_unlink)
    with pytest.raises(
        tool.EvidenceError,
        match=r"record=unlink-error.*record-retry=removed, parent-fsync=ok",
    ):
        _generate(tmp_path, bundle, q3, regular)
    assert failed_record_unlink
    assert not (tmp_path / "record.json").exists()
    assert not (tmp_path / "audit.json").exists()


def test_audit_directory_fsync_failure_rolls_back_both_owned_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    original_fsync = tool.os.fsync
    directory_fsyncs = 0

    def fail_second_directory_fsync(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            assert _publication_temps(tmp_path) == []
            directory_fsyncs += 1
            if directory_fsyncs == 2:
                raise OSError(errno.EIO, "injected audit directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(tool.os, "fsync", fail_second_directory_fsync)
    with pytest.raises(
        tool.EvidenceError,
        match=r"audit directory fsync failure.*final=removed.*record rollback: "
        r"record=removed, parent-fsync=ok",
    ):
        _generate(tmp_path, bundle, q3, regular)
    # record publish, failed audit fsync, audit cleanup fsync, record rollback fsync
    assert directory_fsyncs == 4
    assert not (tmp_path / "record.json").exists()
    assert not (tmp_path / "audit.json").exists()
    assert _publication_temps(tmp_path) == []


def test_audit_publish_failure_preserves_concurrent_record_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    record_output = tmp_path / "record.json"
    replacement = b"concurrent replacement\n"
    original_write_exclusive = tool._write_exclusive

    def replace_before_audit_failure(path: Path, data: bytes, label: str) -> Any:
        if label == "audit output":
            replacement_path = tmp_path / "replacement.tmp"
            replacement_path.write_bytes(replacement)
            os.replace(replacement_path, record_output)
            tool._fail("injected audit publication failure")
        return original_write_exclusive(path, data, label)

    monkeypatch.setattr(tool, "_write_exclusive", replace_before_audit_failure)
    with pytest.raises(tool.EvidenceError, match="record=preserved-nonowned"):
        _generate(tmp_path, bundle, q3, regular)
    assert record_output.read_bytes() == replacement
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("existing", ["record", "audit"])
def test_generator_existing_output_preflight_never_creates_or_replaces_pair_member(
    tmp_path: Path, existing: str
) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    existing_path = tmp_path / f"{existing}.json"
    existing_bytes = b"existing output\n"
    existing_path.write_bytes(existing_bytes)
    with pytest.raises(tool.EvidenceError, match=f"{existing} output must be absent"):
        _generate(tmp_path, bundle, q3, regular)
    assert existing_path.read_bytes() == existing_bytes
    other = "audit" if existing == "record" else "record"
    assert not (tmp_path / f"{other}.json").exists()
    assert _publication_temps(tmp_path) == []


def test_crash_window_record_is_non_authorizing_and_production_pin_stays_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SimulatedCrash(BaseException):
        pass

    bundle, q3, regular = _artifacts(tmp_path)
    pins_path = ROOT / "src/runtime/models/sam2/sam2_production_qualification_pins.cpp"
    pins_before = pins_path.read_bytes()
    original_write_exclusive = tool._write_exclusive

    def crash_before_audit(path: Path, data: bytes, label: str) -> Any:
        if label == "audit output":
            raise SimulatedCrash
        return original_write_exclusive(path, data, label)

    monkeypatch.setattr(tool, "_write_exclusive", crash_before_audit)
    with pytest.raises(SimulatedCrash):
        _generate(tmp_path, bundle, q3, regular)
    record = _read_json(tmp_path / "record.json")
    assert record["self_authorizing"] is False
    assert not (tmp_path / "audit.json").exists()
    assert b"std::array<NativeQualificationStaticPin, 0>" in pins_before
    assert pins_path.read_bytes() == pins_before


def test_summary_rounding_matches_cpp_positive_half_away_from_zero() -> None:
    summary = tool._metric_summary([1] * 50 + [2] * 50)
    assert summary == {
        "mean_ms": 0.000002,
        "median_ms": 0.000002,
        "p90_ms": 0.000002,
        "min_ms": 0.000001,
        "max_ms": 0.000002,
    }


def test_public_record_omits_raw_infrastructure_provenance(tmp_path: Path) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    record_bytes, _ = _generate(tmp_path, bundle, q3, regular)
    record = json.loads(record_bytes)
    private_infrastructure_fields = {
        "qualification_environment",
        "gpu_uuid",
        "pci_bus_id",
        "cuda_runtime_version",
        "cuda_driver_version",
        "hostname",
    }
    assert private_infrastructure_fields.isdisjoint(record["accuracy_evidence"])
    assert not any(field.encode() in record_bytes for field in private_infrastructure_fields)
    assert private_infrastructure_fields - {"qualification_environment"} <= set(
        json.loads(q3.read_bytes())["runtime"]
    )


def test_generator_accepts_cpp_equivalent_float_lexeme(tmp_path: Path) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    original = b'"bbox_score_error":0.001'
    cpp_spelling = b'"bbox_score_error":0.0010000000000000000'
    q3_bytes = q3.read_bytes()
    assert original in q3_bytes
    q3.write_bytes(q3_bytes.replace(original, cpp_spelling, 1))
    _rebind_regular_to_q3(q3, regular)

    record_bytes, _ = _generate(tmp_path, bundle, q3, regular)

    assert json.loads(record_bytes)["accuracy_evidence"]["receipt_sha256"] == _sha(q3.read_bytes())


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1, "b":2}\n',
        b'{"a":1,"a":1}\n',
        b'{"a":01}\n',
        b'{"a":NaN}\n',
        b'{"a":1e400}\n',
        b'{"a":1}\ntrailing',
    ],
    ids=[
        "whitespace",
        "duplicate-key",
        "leading-zero",
        "nan",
        "overflow-to-infinity",
        "trailing-bytes",
    ],
)
def test_raw_canonical_json_gate_rejects_malformed_or_noncanonical_bytes(
    payload: bytes,
) -> None:
    with pytest.raises(tool.EvidenceError):
        tool._parse_json(payload, "test receipt", trailing_newline=True)


def test_raw_canonical_json_gate_preserves_valid_number_lexemes() -> None:
    payload = b'{"a":0.33722800000000003,"b":[-0,1.00,1e+06]}\n'

    value = tool._parse_json(payload, "test receipt", trailing_newline=True)

    assert value == {"a": 0.337228, "b": [0, 1.0, 1_000_000.0]}


@pytest.mark.parametrize(
    "attack", ["duplicate", "noncanonical", "root_key_order", "nested_key_order", "metrics"]
)
def test_generator_rejects_receipt_attacks(tmp_path: Path, attack: str) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    if attack == "duplicate":
        q3.write_bytes(
            q3.read_bytes().replace(
                b'{"schema_version":2', b'{"schema_version":2,"schema_version":2', 1
            )
        )
    elif attack == "noncanonical":
        q3.write_bytes(b" " + q3.read_bytes())
    elif attack == "root_key_order":
        q3.write_bytes(
            q3.read_bytes().replace(
                b'{"schema_version":2,"family":"sam2"',
                b'{"family":"sam2","schema_version":2',
                1,
            )
        )
        _rebind_regular_to_q3(q3, regular)
    elif attack == "nested_key_order":
        q3.write_bytes(
            q3.read_bytes().replace(
                b'"process_model":{"tensorrt_iattention_v2_image_attention":true,'
                b'"external_attention_dso_loaded":false',
                b'"process_model":{"external_attention_dso_loaded":false,'
                b'"tensorrt_iattention_v2_image_attention":true',
                1,
            )
        )
        _rebind_regular_to_q3(q3, regular)
    else:
        value = json.loads(q3.read_bytes())
        value["accuracy"]["replays"][1]["frame_iou"][0] = 0.979
        q3.write_bytes(_compact(value, newline=True))
    with pytest.raises(tool.EvidenceError):
        _generate(tmp_path, bundle, q3, regular)


@pytest.mark.parametrize("mutation", ["extra", "missing", "tamper", "wrong_mode_fields"])
def test_generator_rejects_exact_process_model_mutations(tmp_path: Path, mutation: str) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    target = regular if mutation == "wrong_mode_fields" else q3
    value = _read_json(target)
    process_model = value["process_model"]
    if mutation == "extra":
        process_model["unreviewed"] = True
    elif mutation == "missing":
        process_model.pop("processor")
    elif mutation == "tamper":
        process_model["engine_deserialization_count"] = 5
    else:
        process_model.pop("checkpoint_graph_build_outside_timing")
        process_model["checkpoint_graph_build_before_replays"] = True
    _write_json(target, value)
    with pytest.raises(tool.EvidenceError, match="process model"):
        _generate(tmp_path, bundle, q3, regular)


@pytest.mark.parametrize("mutation", ["extra", "missing", "repeat_contract", "label_mismatch"])
def test_generator_rejects_accuracy_contract_mutations(tmp_path: Path, mutation: str) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    value = _read_json(q3)
    if mutation == "extra":
        value["accuracy"]["unreviewed"] = True
    elif mutation == "missing":
        value["accuracy"].pop("foreground_counts_exact")
    elif mutation == "repeat_contract":
        value["accuracy"]["repeat_contract"] = "semantic only"
    else:
        value["accuracy"]["replays"][0]["candidate_bbox"]["label"] = 2
    _write_json(q3, value)
    with pytest.raises(tool.EvidenceError):
        _generate(tmp_path, bundle, q3, regular)


@pytest.mark.parametrize(
    "mutation",
    ["closure_role", "q3_reverse", "regular_reverse", "cross_order", "different_process"],
)
def test_generator_rejects_provenance_and_runtime_order_mutations(
    tmp_path: Path, mutation: str
) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    if mutation in {"closure_role", "q3_reverse", "cross_order"}:
        value = _read_json(q3)
        if mutation == "closure_role":
            value["assets"]["benchmark_source_closure_role"] = "test closure"
        elif mutation == "q3_reverse":
            value["runtime"]["ended_at_utc"] = "2026-08-17T00:00:59Z"
        else:
            value["runtime"]["ended_at_utc"] = "2026-08-17T00:04:00Z"
        _write_json(q3, value)
        if mutation == "cross_order":
            _rebind_regular_to_q3(q3, regular)
    else:
        value = _read_json(regular)
        if mutation == "regular_reverse":
            value["runtime"]["ended_at_utc"] = "2026-08-17T00:00:59Z"
        else:
            value["runtime"]["hostname"] = "different-process"
        _write_json(regular, value)
    with pytest.raises(tool.EvidenceError):
        _generate(tmp_path, bundle, q3, regular)


@pytest.mark.parametrize(
    "mutation",
    [
        "boundaries_extra",
        "boundaries_missing",
        "boundaries_tamper",
        "p90",
        "baseline_extra",
        "baseline_missing",
        "baseline_tamper",
    ],
)
def test_generator_rejects_regular_timing_contract_mutations(tmp_path: Path, mutation: str) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    value = _read_json(regular)
    if mutation == "boundaries_extra":
        value["timing_boundaries"]["unreviewed"] = True
    elif mutation == "boundaries_missing":
        value["timing_boundaries"].pop("clock")
    elif mutation == "boundaries_tamper":
        value["timing_boundaries"]["comparison_scope"] = "diagnostic"
    elif mutation == "p90":
        value["timing"]["p90_method"] = "nearest rank"
    elif mutation == "baseline_extra":
        value["delivered_baseline_reference"]["speedup"] = 1.5
    elif mutation == "baseline_missing":
        value["delivered_baseline_reference"].pop("comparison_warning")
    else:
        value["delivered_baseline_reference"]["total_mean_ms"] += 0.001
    _write_json(regular, value)
    with pytest.raises(tool.EvidenceError):
        _generate(tmp_path, bundle, q3, regular)


@pytest.mark.parametrize("mutation", ["extra", "missing", "milliseconds", "sum"])
def test_generator_rejects_timing_row_mutations(tmp_path: Path, mutation: str) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    value = _read_json(regular)
    row = value["timing"]["measurement_rows"][0]
    if mutation == "extra":
        row["unreviewed"] = 0
    elif mutation == "missing":
        row.pop("native_prefill_ms")
    elif mutation == "milliseconds":
        row["native_prefill_ms"] += 0.001
    else:
        row["closest_envelope_total_ns"] += 1
        row["closest_envelope_total_ms"] = row["closest_envelope_total_ns"] / 1_000_000
    _write_json(regular, value)
    with pytest.raises(tool.EvidenceError):
        _generate(tmp_path, bundle, q3, regular)


@pytest.mark.parametrize("mutation", ["extra", "missing", "tamper"])
def test_generator_rejects_independently_recomputed_summary_mutations(
    tmp_path: Path, mutation: str
) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    value = _read_json(regular)
    metric = value["timing"]["summary"]["native_prefill"]
    if mutation == "extra":
        metric["stddev_ms"] = 0.0
    elif mutation == "missing":
        metric.pop("min_ms")
    else:
        metric["mean_ms"] += 0.000001
    _write_json(regular, value)
    with pytest.raises(tool.EvidenceError):
        _generate(tmp_path, bundle, q3, regular)


@pytest.mark.parametrize(
    "mutation",
    [
        "image_extra",
        "image_missing",
        "image_tamper",
        "tracker_extra",
        "tracker_missing",
        "tracker_tamper",
    ],
)
def test_generator_rejects_build_graph_schema_mutations(tmp_path: Path, mutation: str) -> None:
    def mutate(receipt: dict[str, Any]) -> None:
        graph = receipt["graphs"][0 if mutation.startswith("image") else 2]
        if mutation.endswith("extra"):
            graph["unreviewed"] = 0
        elif mutation.endswith("missing"):
            graph.pop("attention_output_layers" if mutation.startswith("image") else "kind")
        elif mutation.startswith("image"):
            graph["matrix_multiply_layers"] += 1
        else:
            graph["history_frames"] += 1

    bundle_path = tmp_path / "model.bundle"
    bundle_facts = _write_bundle(bundle_path, mutate_build_receipt=mutate)
    q3, regular = _write_receipts(tmp_path, bundle_facts)
    with pytest.raises(tool.EvidenceError, match="build receipt graph"):
        _generate(tmp_path, bundle_path, q3, regular)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("legacy_schema", "build receipt schema version"),
        ("missing_level", "build facts field set drifted"),
        ("wrong_level", "builder optimization level drifted"),
    ],
)
def test_generator_rejects_build_reproducibility_mutations(
    tmp_path: Path, mutation: str, message: str
) -> None:
    def mutate(receipt: dict[str, Any]) -> None:
        if mutation == "legacy_schema":
            receipt["schema_version"] = 1
        elif mutation == "missing_level":
            receipt["build"].pop("builder_optimization_level")
        else:
            receipt["build"]["builder_optimization_level"] = tool.BUILDER_OPTIMIZATION_LEVEL + 1

    bundle_path = tmp_path / "model.bundle"
    bundle_facts = _write_bundle(bundle_path, mutate_build_receipt=mutate)
    q3, regular = _write_receipts(tmp_path, bundle_facts)
    with pytest.raises(tool.EvidenceError, match=message):
        _generate(tmp_path, bundle_path, q3, regular)


def test_generator_rejects_sibling_bundle_receipts(tmp_path: Path) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    sibling = tmp_path / "sibling.bundle"
    _write_bundle(sibling, salt=b"-different")
    with pytest.raises(tool.EvidenceError, match="native_bundle_sha256"):
        _generate(tmp_path, sibling, q3, regular)


def test_path_swap_after_snapshot_cannot_change_bound_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, q3, regular = _artifacts(tmp_path)
    original_q3 = q3.read_bytes()
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b'{"not":"evidence"}\n')
    original_snapshot = tool._snapshot_regular_file
    swapped = False

    def snapshot(path: Path, maximum_size: int, label: str) -> Any:
        nonlocal swapped
        result = original_snapshot(path, maximum_size, label)
        if label == "Q3 receipt" and not swapped:
            replacement.replace(q3)
            swapped = True
        return result

    monkeypatch.setattr(tool, "_snapshot_regular_file", snapshot)
    record_bytes, _ = _generate(tmp_path, bundle, q3, regular)
    assert json.loads(record_bytes)["accuracy_evidence"]["receipt_sha256"] == _sha(original_q3)
    assert q3.read_bytes() != original_q3
