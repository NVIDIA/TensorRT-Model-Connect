# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the Fast Foundation Stereo Nsight Compute analyzer."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.models.fast_foundation_stereo import analyze_ncu


_FIELDS = ["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"]


def _write_csv(path: Path, rows: list[dict[str, str]], *, preamble: bool = False) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if preamble:
            handle.write("==PROF== Connected to process 123\n")
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _raw_rows(
    launch_id: str,
    kernel: str,
    duration: str,
    compute: str,
    memory: str,
) -> list[dict[str, str]]:
    return [
        {
            "ID": launch_id,
            "Kernel Name": kernel,
            "Metric Name": "gpu__time_duration.sum",
            "Metric Unit": "nsecond",
            "Metric Value": duration,
        },
        {
            "ID": launch_id,
            "Kernel Name": kernel,
            "Metric Name": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "Metric Unit": "%",
            "Metric Value": compute,
        },
        {
            "ID": launch_id,
            "Kernel Name": kernel,
            "Metric Name": ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
            "Metric Unit": "%",
            "Metric Value": memory,
        },
    ]


def test_load_samples_parses_raw_metric_ids_and_units(tmp_path: Path) -> None:
    report = tmp_path / "raw.csv"
    _write_csv(
        report,
        _raw_rows("1", "tensor_core", "12,500", "91.5", "32.0"),
        preamble=True,
    )
    with report.open("a", encoding="utf-8") as handle:
        handle.write("==PROF== Disconnected from process 123\n")

    samples = analyze_ncu._load_samples([report])

    assert samples == [
        {
            "id": "1",
            "kernel": "tensor_core",
            "source": str(report.resolve()),
            "duration_us": 12.5,
            "compute_pct": 91.5,
            "memory_pct": 32.0,
            "limiter_pct": 91.5,
            "limiter": "compute",
        }
    ]


def test_load_samples_accepts_legacy_names_with_embedded_units(tmp_path: Path) -> None:
    report = tmp_path / "legacy.csv"
    rows = [
        {
            "ID": "7",
            "Kernel Name": "copy",
            "Metric Name": "Duration (ms)",
            "Metric Unit": "",
            "Metric Value": "0.25",
        },
        {
            "ID": "7",
            "Kernel Name": "copy",
            "Metric Name": "Compute (SM) Throughput (%)",
            "Metric Unit": "",
            "Metric Value": "10",
        },
        {
            "ID": "7",
            "Kernel Name": "copy",
            "Metric Name": "Memory Throughput (%)",
            "Metric Unit": "",
            "Metric Value": "80",
        },
    ]
    _write_csv(report, rows)

    sample = analyze_ncu._load_samples([report])[0]

    assert sample["duration_us"] == 250.0
    assert sample["limiter"] == "memory"


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        (_raw_rows("1", "kernel", "0", "50", "50"), "duration must be positive"),
        (_raw_rows("1", "kernel", "1000", "101", "50"), "percentage must be"),
        (_raw_rows("1", "kernel", "nan", "50", "50"), "must be finite"),
        (_raw_rows("1", "kernel", "1000", "50", "N/A"), "invalid numeric"),
        (_raw_rows("1", "kernel", "1000", "50", "50")[:-1], "missing metrics"),
    ],
)
def test_load_samples_rejects_incomplete_or_invalid_launches(
    tmp_path: Path, rows: list[dict[str, str]], error: str
) -> None:
    report = tmp_path / "invalid.csv"
    _write_csv(report, rows)

    with pytest.raises(ValueError, match=error):
        analyze_ncu._load_samples([report])


def test_load_samples_rejects_empty_and_duplicate_launch_metrics(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    duplicate = tmp_path / "duplicate.csv"
    _write_csv(empty, [])
    rows = _raw_rows("1", "kernel", "1000", "50", "50")
    _write_csv(duplicate, [*rows, rows[0]])

    with pytest.raises(ValueError, match="contains no kernel launches"):
        analyze_ncu._load_samples([empty])
    with pytest.raises(ValueError, match="duplicate duration_us metric"):
        analyze_ncu._load_samples([duplicate])


def test_summary_uses_duration_weighted_limiter_and_near_ceiling_fraction() -> None:
    samples = [
        {
            "duration_us": 10.0,
            "compute_pct": 90.0,
            "memory_pct": 20.0,
            "limiter_pct": 90.0,
            "limiter": "compute",
            "kernel": "compute_kernel",
        },
        {
            "duration_us": 30.0,
            "compute_pct": 20.0,
            "memory_pct": 70.0,
            "limiter_pct": 70.0,
            "limiter": "memory",
            "kernel": "memory_kernel",
        },
    ]

    summary = analyze_ncu.summarize_samples(samples, near_ceiling_pct=80.0, top=1)

    assert summary["duration_weighted_compute_pct"] == 37.5
    assert summary["duration_weighted_memory_pct"] == 57.5
    assert summary["duration_weighted_limiter_pct"] == 75.0
    assert summary["near_ceiling_duration_fraction"] == 0.25
    assert summary["compute_limited_duration_fraction"] == 0.25
    assert summary["memory_limited_duration_fraction"] == 0.75
    assert "max_compute_pct" not in summary
    assert summary["diagnostics"]["max_compute_pct"] == 90.0
    assert summary["diagnostics"]["top_duration_launches"][0]["kernel"] == "memory_kernel"


@pytest.mark.parametrize(("compute_pct", "expected_pass"), ((90, True), (79, False)))
def test_main_writes_json_summary_and_enforces_exit_status(
    tmp_path: Path, monkeypatch, capsys, compute_pct: int, expected_pass: bool
) -> None:
    details_fixture = tmp_path / "raw.csv"
    output = tmp_path / "results" / "summary.json"
    _write_csv(
        details_fixture,
        _raw_rows("0", "kernel", "1000", str(compute_pct), "10"),
    )
    digest = hashlib.sha256(b"artifact").hexdigest()
    profile_script = Path(analyze_ncu.__file__).with_name("profile_ncu.py").resolve()
    profile_digest = hashlib.sha256(profile_script.read_bytes()).hexdigest()
    benchmark_script = profile_script.with_name("benchmark.py")
    benchmark_digest = hashlib.sha256(benchmark_script.read_bytes()).hexdigest()
    runner_script = profile_script.with_name("trt_runner.py")
    runner_digest = hashlib.sha256(runner_script.read_bytes()).hexdigest()
    python_executable = Path(sys.executable).resolve()
    python_digest = hashlib.sha256(python_executable.read_bytes()).hexdigest()
    monkeypatch.setattr(analyze_ncu, "_TRUSTED_NCU_TARGET_PATH", str(python_executable))
    monkeypatch.setattr(analyze_ncu, "_TRUSTED_NCU_TARGET_SHA256", python_digest)
    monkeypatch.setattr(analyze_ncu, "_TRUSTED_NCU_TARGET_VERSION", "test-ncu")
    canonical_inputs = {
        "pair.png": {"left": digest, "right": digest},
        "pair-2.png": {"left": digest, "right": digest},
        "pair-3.png": {"left": digest, "right": digest},
        "pair-4.png": {"left": digest, "right": digest},
        "pair-5.png": {"left": digest, "right": digest},
    }
    monkeypatch.setattr(analyze_ncu, "_CANONICAL_INPUT_SHA256", canonical_inputs)
    monkeypatch.setattr(analyze_ncu, "_CANONICAL_ACCURACY_REFERENCE_SHA256", digest)
    model_root = (tmp_path / "model").resolve()
    input_root = (tmp_path / "inputs").resolve()
    feature_engine = (tmp_path / "feature.plan").resolve()
    post_engine = (tmp_path / "post.plan").resolve()
    plugin_library = (tmp_path / "plugin.so").resolve()
    manifest = tmp_path / "profile-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "profile_scope": "native_feature_and_post",
                "nvtx_ranges": ["ffs_full", "ffs_feature", "ffs_post"],
                "required_profiler_contract": {
                    "tool": "ncu",
                    "config_file": "off",
                    "metric_set": "roofline",
                    "nvtx_include": "ffs_full/",
                    "replay_mode": "kernel",
                    "launch_skip": 0,
                    "launch_count": "all",
                },
                "model_root": str(model_root),
                "input_root": str(input_root),
                "pair": {
                    "name": "pair.png",
                    "left_sha256": digest,
                    "right_sha256": digest,
                    "scale": 1.0,
                },
                "engines": {
                    "feature": {"path": str(feature_engine), "sha256": digest},
                    "post": {"path": str(post_engine), "sha256": digest},
                },
                "plugin_artifacts": [{"path": str(plugin_library), "sha256": digest}],
                "source_artifacts": {
                    "python_executable": {
                        "path": str(python_executable),
                        "sha256": python_digest,
                    },
                    "profile_ncu": {
                        "path": str(profile_script),
                        "sha256": profile_digest,
                    },
                    "benchmark": {
                        "path": str(benchmark_script),
                        "sha256": benchmark_digest,
                    },
                    "trt_runner": {
                        "path": str(runner_script),
                        "sha256": runner_digest,
                    },
                },
                "cuda_graphs": False,
                "requested_warmup_enqueues": 5,
                "environment": {
                    "torch": "2.6.0+cu124",
                    "tensorrt": "10.8.0.43",
                    "cuda_runtime": "12.4",
                    "gpu_name": "NVIDIA L4",
                    "gpu_capability": [8, 9],
                },
            }
        ),
        encoding="utf-8",
    )
    benchmark = tmp_path / "benchmark-result.json"
    benchmark.write_text(
        json.dumps(
            {
                "backend": "trt",
                "num_pairs": 5,
                "start_index": 0,
                "scale": 1.0,
                "warmup_iters": 20,
                "iters": 100,
                "valid_iters": 8,
                "max_disp": 192,
                "cuda_graphs": True,
                "environment": {
                    "torch": "2.6.0+cu124",
                    "tensorrt": "10.8.0.43",
                    "cuda_runtime": "12.4",
                    "gpu_name": "NVIDIA L4",
                    "gpu_capability": [8, 9],
                },
                "minimum_cosine": 0.999,
                "maximum_mean_abs_error": 0.5,
                "maximum_bad_2px_fraction": 0.02,
                "accuracy_passed": True,
                "accuracy": {
                    "global_cosine": 1.0,
                    "mean_abs_error": 0.0,
                    "bad_2px_fraction": 0.0,
                },
                "samples_ms": {
                    "preprocess": [1.0] * 100,
                    "inference": [2.0] * 100,
                    "total": [3.0] * 100,
                },
                "infer_5_pairs": {
                    "count": 100,
                    "mean_ms": 2.0,
                    "median_ms": 2.0,
                    "p90_ms": 2.0,
                },
                "artifacts": {
                    "feature_engine": {"sha256": digest},
                    "post_engine": {"sha256": digest},
                    "benchmark_tool": {"sha256": benchmark_digest},
                    "runner_tool": {"sha256": runner_digest},
                    "plugin_libraries": [{"sha256": digest}],
                    "accuracy_reference": {"sha256": digest},
                    "inputs": [
                        {
                            "name": name,
                            "left": {"sha256": hashes["left"]},
                            "right": {"sha256": hashes["right"]},
                        }
                        for name, hashes in canonical_inputs.items()
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    ncu_report = tmp_path / "profile.ncu-rep"
    ncu_report.write_bytes(b"NVR\0report")
    ncu_session = tmp_path / "profile-session.csv"
    ncu_session.write_text(
        "Launch Settings\n"
        '"Launch Attribute","Value"\n'
        f'"Profiler Command Line","{python_executable} --config-file off '
        "--set roofline --nvtx "
        "--nvtx-include ffs_full/ --replay-mode kernel "
        f"--export {ncu_report} --force-overwrite {python_executable} "
        f"{profile_script} "
        f"{model_root} {feature_engine} {post_engine} "
        f"--plugin-library {plugin_library} --input-root {input_root} "
        f'--pair-name pair.png --warmup 5 --manifest {manifest.resolve()}"\n'
        "\nSession Info\n"
        '"Session Attribute","Value"\n'
        '"Nsight Compute Target","test-ncu"\n'
        "\nProcesses\n"
        '"Process Id","Process Name"\n'
        f'"123","{python_executable}"\n'
        "\nDevice Attributes\n"
        '"Device Attribute","Device 0"\n'
        '"display_name","NVIDIA L4"\n'
        '"compute_capability_major","8"\n'
        '"compute_capability_minor","9"\n',
        encoding="utf-8",
    )

    def export_ncu_page(_binary, _report, *, page, output):
        source = details_fixture if page == "details" else ncu_session
        output.write_bytes(source.read_bytes())
        return ["ncu", "--import", str(_report), "--page", page]

    monkeypatch.setattr(analyze_ncu, "_export_ncu_page", export_ncu_page)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_ncu.py",
            "--manifest",
            str(manifest),
            "--benchmark-result",
            str(benchmark),
            "--ncu-report",
            str(ncu_report),
            "--output",
            str(output),
        ],
    )

    if expected_pass:
        analyze_ncu.main()
    else:
        with pytest.raises(SystemExit) as error:
            analyze_ncu.main()
        assert error.value.code == 2

    written = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert written == printed
    assert written["near_ceiling_duration_fraction"] == float(expected_pass)
    assert written["profile_manifest"]["pair_name"] == "pair.png"
    assert written["profile_manifest"]["cuda_graphs"] is False
    assert written["coverage"]["exhaustive"] is True
    assert written["benchmark_receipt"]["accuracy_passed"] is True
    assert written["qualification_gate"]["passed"] is expected_pass
    assert written["qualification_gate"]["beats_recorded_baseline"] is True
    assert written["qualification_gate"]["roofline_passed"] is expected_pass
    assert written["ncu_report"]["sha256"] == hashlib.sha256(b"NVR\0report").hexdigest()
    assert written["ncu_session"]["launch_filters"] == []
    assert written["ncu_session"]["device"] == {
        "name": "NVIDIA L4",
        "compute_capability": [8, 9],
    }
    assert (
        written["ncu_exports"]["details"]["sha256"]
        == hashlib.sha256(details_fixture.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("value", (True, False, float("nan"), float("inf"), "1.0", None))
def test_finite_number_schema_rejects_non_numeric_values(value: object) -> None:
    assert analyze_ncu._is_finite_number(value) is False


@pytest.mark.parametrize("page", ("details", "session"))
def test_export_ncu_page_disables_config_files(tmp_path: Path, monkeypatch, page: str) -> None:
    report = tmp_path / "profile.ncu-rep"
    report.write_bytes(b"NVR\0report")
    ncu_binary = tmp_path / "ncu"
    ncu_binary.write_bytes(b"trusted-ncu")
    monkeypatch.setattr(analyze_ncu, "_TRUSTED_NCU_IMPORTER_PATH", str(ncu_binary))
    monkeypatch.setattr(
        analyze_ncu,
        "_TRUSTED_NCU_IMPORTER_SHA256",
        hashlib.sha256(ncu_binary.read_bytes()).hexdigest(),
    )
    output = tmp_path / f"{page}.csv"
    captured: list[list[str]] = []

    def fake_run(command, **_kwargs):
        captured.append(command)
        output.write_text("export", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(analyze_ncu.subprocess, "run", fake_run)

    command = analyze_ncu._export_ncu_page(str(ncu_binary), report, page=page, output=output)

    assert command == captured[0]
    assert command[1:3] == ["--config-file", "off"]

    evil_binary = tmp_path / "fake-ncu"
    evil_binary.write_bytes(b"fake")
    with pytest.raises(ValueError, match="not the pinned qualification binary"):
        analyze_ncu._export_ncu_page(str(evil_binary), report, page=page, output=output)


def test_coverage_rejects_gaps_overlaps_and_partial_capture() -> None:
    def sample(launch_id: str, source: str) -> dict[str, object]:
        return {"id": launch_id, "source": source}

    with pytest.raises(ValueError, match="begin at launch 0"):
        analyze_ncu._validate_exhaustive_coverage([sample("1", "a.csv")])
    with pytest.raises(ValueError, match="gaps"):
        analyze_ncu._validate_exhaustive_coverage([sample("0", "a.csv"), sample("2", "a.csv")])
    with pytest.raises(ValueError, match="appears in both"):
        analyze_ncu._validate_exhaustive_coverage([sample("0", "a.csv"), sample("0", "b.csv")])


@pytest.mark.parametrize(
    "launch_filter",
    (
        "--launch-count 10",
        "--launch-c=1",
        "-c 10",
        "-s 10",
        "-k tensor_core",
        "--range-filter 1:2",
        "--nvtx-exclude ffs_post/",
        "--nvtx-include ffs_post/",
        "@filters.rsp",
    ),
)
def test_ncu_session_rejects_launch_filters(
    tmp_path: Path, monkeypatch, launch_filter: str
) -> None:
    report = tmp_path / "profile.ncu-rep"
    report.write_bytes(b"NVR\0report")
    ncu_target = tmp_path / "ncu-target"
    ncu_target.write_bytes(b"trusted-ncu-target")
    monkeypatch.setattr(analyze_ncu, "_TRUSTED_NCU_TARGET_PATH", str(ncu_target))
    monkeypatch.setattr(
        analyze_ncu,
        "_TRUSTED_NCU_TARGET_SHA256",
        hashlib.sha256(ncu_target.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(analyze_ncu, "_TRUSTED_NCU_TARGET_VERSION", "test-ncu")
    session = tmp_path / "session.csv"
    session.write_text(
        f'"Profiler Command Line","{ncu_target} --config-file off --set roofline --nvtx '
        "--nvtx-include ffs_full/ "
        f"--replay-mode kernel {launch_filter} --export {report} --force-overwrite "
        'python profile_ncu.py"\n'
        "\nSession Info\n"
        '"Session Attribute","Value"\n'
        '"Nsight Compute Target","test-ncu"\n'
        "\nDevice Attributes\n"
        '"Device Attribute","Device 0"\n'
        '"display_name","NVIDIA L4"\n'
        '"compute_capability_major","8"\n'
        '"compute_capability_minor","9"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not allowlisted|duplicate|response files"):
        analyze_ncu._load_ncu_session(session, report=report, profile_manifest={})


def test_profile_manifest_must_cover_full_native_scope(tmp_path: Path) -> None:
    manifest = tmp_path / "profile-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "profile_scope": "post_only",
                "nvtx_ranges": ["ffs_post"],
                "pair": {},
                "engines": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not cover native feature and post"):
        analyze_ncu._load_profile_manifest(manifest)
