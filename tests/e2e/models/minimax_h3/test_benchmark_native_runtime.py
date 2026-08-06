# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("benchmark_native_runtime.py")
SPEC = importlib.util.spec_from_file_location("minimax_h3_benchmark_native_runtime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_resolve_trt_backend_dso_matches_runtime_candidate_order(tmp_path: Path) -> None:
    executable = tmp_path / "trtmc_benchmark_worker"
    executable.write_bytes(b"binary")
    unversioned = tmp_path / "libtrtmc_backend_trt.so"
    versioned = tmp_path / "libtrtmc_backend_trt_11_2.so"
    unversioned.write_bytes(b"unversioned")
    versioned.write_bytes(b"versioned")
    config = {"engine_backend": "trt", "trt_abi": "11.2"}

    assert MODULE.resolve_trt_backend_dso(executable, config) == versioned.resolve()
    versioned.unlink()
    assert MODULE.resolve_trt_backend_dso(executable, config) == unversioned.resolve()


def test_resolve_trt_backend_dso_fails_closed(tmp_path: Path) -> None:
    executable = tmp_path / "trtmc_benchmark_worker"
    executable.write_bytes(b"binary")

    with pytest.raises(ValueError, match="engine_backend=trt"):
        MODULE.resolve_trt_backend_dso(executable, {"engine_backend": "rtx", "trt_abi": "11.2"})
    with pytest.raises(ValueError, match="invalid TensorRT ABI"):
        MODULE.resolve_trt_backend_dso(executable, {"engine_backend": "trt", "trt_abi": "latest"})
    with pytest.raises(FileNotFoundError, match="adjacent TensorRT backend"):
        MODULE.resolve_trt_backend_dso(executable, {"engine_backend": "trt", "trt_abi": "11.2"})


def test_file_page_eviction_is_portable_when_advice_constant_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(MODULE.os, "POSIX_FADV_DONTNEED", raising=False)

    def fail_open(*_args, **_kwargs) -> int:
        raise AssertionError("file should not be opened without POSIX_FADV_DONTNEED")

    monkeypatch.setattr(MODULE.os, "open", fail_open)

    assert MODULE.evict_file_pages(Path("bundle.trtfb")) == {
        "supported": False,
        "attempted": False,
        "succeeded": False,
    }


def test_file_page_eviction_reports_open_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_open(*_args, **_kwargs) -> int:
        raise OSError("open rejected")

    monkeypatch.setattr(MODULE.os, "posix_fadvise", lambda *_args: None, raising=False)
    monkeypatch.setattr(MODULE.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(MODULE.os, "open", fail_open)

    assert MODULE.evict_file_pages(Path("bundle.trtfb")) == {
        "supported": True,
        "attempted": True,
        "succeeded": False,
        "error": "OSError: open rejected",
    }


def _request_log(*, requests: int, cuda_graph_failure: bool = False, resident: bool = False) -> str:
    lines: list[str] = []
    for index in range(requests):
        if not resident:
            for stage_index in range(4):
                lines.append(
                    '[trtmc.engine_timing] label="engine" '
                    f"execute_ms={10 + index + stage_index:.6f} launches={stage_index + 1}"
                )
        cache_markers = (
            f" text_cache_hit={int(index > 0)} adaln_cache_hit={int(index > 0)}"
            f" denoiser_resident_hit={int(index > 0)} vae_resident_hit={int(index > 0)}"
            " cache_threshold=0.08 full_steps=40 skipped_steps=9"
            if resident
            else ""
        )
        lines.append(
            "[minimax-h3.perf] "
            f"text_encoder_ms={21 + index:.3f} "
            f"adaln_ms={22 + index:.3f} "
            f"denoiser_ms={23 + index:.3f} "
            f"vae_decoder_ms={24 + index:.3f} total_ms={100 + index:.3f}{cache_markers}"
        )
    if resident:
        for stage_index in range(4):
            lines.append(
                '[trtmc.engine_timing] label="engine" '
                f"execute_ms={30 + stage_index:.6f} launches={requests * (stage_index + 1)}"
            )
    if cuda_graph_failure:
        lines.append("[cuda_graph] Capture failed, disabling CUDA Graphs")
    return "\n".join(lines)


def _worker_result(*, warmup: int, iterations: int) -> dict:
    return {
        "status": "completed",
        "operation": "generate_image",
        "warmup": warmup,
        "iterations": iterations,
        "load_ms": 7.5,
        "observations": [
            {"iteration": index, "measured_wall_ms": 110.0 + index} for index in range(iterations)
        ],
        "output_summary": {
            "height": 768,
            "width": 1344,
            "num_frames": 124,
        },
    }


def test_worker_request_forwards_cuda_graphs_and_fixed_workload(tmp_path: Path) -> None:
    request = MODULE.build_worker_request(
        bundle=tmp_path / "h3.trtfb",
        plugin_dir=tmp_path / "plugins",
        prompt="prompt",
        seed=0,
        source_revision="a" * 40,
        cuda_graphs=True,
        warmup=1,
        iterations=2,
    )

    assert request["operation"] == "generate_image"
    assert request["runtime"] == {
        "cuda_graphs": True,
        "model_plugin_search_paths": [str(tmp_path / "plugins")],
    }
    assert request["request"]["video_num_frames"] == 124
    assert request["request"]["num_inference_steps"] == 50
    assert request["measurement"]["warmup"] == 1
    assert request["measurement"]["iterations"] == 2


def test_worker_request_forwards_optional_cache_threshold(tmp_path: Path) -> None:
    request = MODULE.build_worker_request(
        bundle=tmp_path / "h3.trtfb",
        plugin_dir=tmp_path / "plugins",
        prompt="prompt",
        seed=0,
        source_revision="a" * 40,
        cuda_graphs=False,
        warmup=1,
        iterations=2,
        cache_threshold=0.05,
    )

    assert request["runtime"]["config"] == {"minimax_h3.first_block_cache_threshold": 0.05}

    default_request = MODULE.build_worker_request(
        bundle=tmp_path / "h3.trtfb",
        plugin_dir=tmp_path / "plugins",
        prompt="prompt",
        seed=0,
        source_revision="a" * 40,
        cuda_graphs=False,
        warmup=1,
        iterations=2,
    )
    assert "config" not in default_request["runtime"]
    assert request["case_digest"] != default_request["case_digest"]


def test_parse_worker_evidence_separates_pipeline_and_engine_boundaries() -> None:
    backend = "libtrtmc_backend_trt_11_2.so"
    evidence = MODULE.parse_worker_evidence(
        _worker_result(warmup=1, iterations=2),
        f"[trtmc] Backend loaded: TensorRT ({backend})\n" + _request_log(requests=3),
        warmup=1,
        iterations=2,
        cuda_graphs=False,
        expected_backend_dso_name=backend,
    )

    assert evidence["pipeline_factory_load_ms"] == 7.5
    assert evidence["pipeline_factory_load_includes_plan_deserialization"] is False
    assert evidence["loaded_backend_dso"] == backend
    assert evidence["measured_requests_are_pipeline_warm"] is True
    assert evidence["first_request"]["pipeline_total_ms"] == 100.0
    assert [item["pipeline_total_ms"] for item in evidence["requests"]] == [101.0, 102.0]
    assert evidence["requests"][0]["public_pipeline_call_wall_ms"] == 110.0
    assert evidence["requests"][0]["engine_execute"]["total_ms"] == 50.0
    assert evidence["requests"][0]["stage_wall"]["total_ms"] == 94.0
    assert evidence["requests"][0]["stage_non_engine"]["total_ms"] == 44.0
    assert evidence["requests"][0]["host_prepost_and_output_assembly_ms"] == 7.0
    assert evidence["summary"]["samples"] == 2
    assert evidence["cold_start"]["plan_deserialization_separately_instrumented"] is False
    assert evidence["cold_start"]["plan_deserialization_included_in"] == (
        "first_request.stage_wall"
    )


@pytest.mark.parametrize(
    "backend_log",
    [
        "",
        "[trtmc] Backend loaded: TensorRT (libtrtmc_backend_trt.so)\n",
        (
            "[trtmc] Backend loaded: TensorRT (libtrtmc_backend_trt_11_2.so)\n"
            "[trtmc] Backend loaded: TensorRT (libtrtmc_backend_trt_11_2.so)\n"
        ),
    ],
)
def test_parse_worker_evidence_rejects_unbound_backend_log(backend_log: str) -> None:
    with pytest.raises(ValueError, match="provenance-bound TensorRT backend"):
        MODULE.parse_worker_evidence(
            _worker_result(warmup=0, iterations=1),
            backend_log + _request_log(requests=1),
            warmup=0,
            iterations=1,
            cuda_graphs=False,
            expected_backend_dso_name="libtrtmc_backend_trt_11_2.so",
        )


def test_parse_worker_evidence_rejects_cuda_graph_fallback() -> None:
    with pytest.raises(ValueError, match="CUDA graph capture failed"):
        MODULE.parse_worker_evidence(
            _worker_result(warmup=0, iterations=1),
            _request_log(requests=1, cuda_graph_failure=True),
            warmup=0,
            iterations=1,
            cuda_graphs=True,
        )


def test_parse_worker_evidence_preserves_resident_engine_aggregate() -> None:
    evidence = MODULE.parse_worker_evidence(
        _worker_result(warmup=1, iterations=2),
        _request_log(requests=3, resident=True),
        warmup=1,
        iterations=2,
        cuda_graphs=False,
        cache_threshold=0.08,
    )

    assert evidence["engine_timing_scope"] == "variant_aggregate"
    assert evidence["aggregate_engine_execute"] == {
        "entries": [
            {"execute_ms": 30.0, "launches": 3},
            {"execute_ms": 31.0, "launches": 6},
            {"execute_ms": 32.0, "launches": 9},
            {"execute_ms": 33.0, "launches": 12},
        ],
        "total_ms": 126.0,
        "total_launches": 30,
        "requests_included": 3,
        "per_request_values_available": False,
    }
    assert evidence["cache_markers_available"] is True
    assert evidence["measured_requests_hit_all_resident_caches"] is True
    assert evidence["cache_threshold_override"] == 0.08
    assert "engine_execute" not in evidence["requests"][0]
    assert evidence["requests"][0]["runtime_annotations"] == {
        "text_cache_hit": True,
        "adaln_cache_hit": True,
        "denoiser_resident_hit": True,
        "vae_resident_hit": True,
        "cache_threshold": 0.08,
        "full_steps": 40,
        "skipped_steps": 9,
    }


def test_receipt_rejects_unapplied_cache_threshold_override() -> None:
    with pytest.raises(ValueError, match="does not match the requested override"):
        MODULE.parse_worker_evidence(
            _worker_result(warmup=1, iterations=1),
            _request_log(requests=2, resident=True),
            warmup=1,
            iterations=1,
            cuda_graphs=False,
            cache_threshold=0.05,
        )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("off", (False,)), ("on", (True,)), ("both", (False, True))],
)
def test_cuda_graph_variant_selection(mode: str, expected: tuple[bool, ...]) -> None:
    assert MODULE._variants(mode) == expected


def test_partial_engine_timings_remain_a_variant_aggregate() -> None:
    lines = _request_log(requests=2).splitlines()
    malformed_log = "\n".join(lines[1:])
    evidence = MODULE.parse_worker_evidence(
        _worker_result(warmup=1, iterations=1),
        malformed_log,
        warmup=1,
        iterations=1,
        cuda_graphs=False,
    )

    assert evidence["engine_timing_scope"] == "variant_aggregate"
    assert evidence["aggregate_engine_execute"]["per_request_values_available"] is False
