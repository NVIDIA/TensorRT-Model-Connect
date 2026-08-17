# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strictly summarize Nsight Compute SpeedOfLight CSV exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


_REQUIRED_COLUMNS = {"ID", "Kernel Name", "Metric Name", "Metric Value"}
_DURATION_METRICS = {
    "duration",
    "gpu__time_duration.sum",
}
_COMPUTE_METRICS = {
    "compute (sm) throughput",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
}
_MEMORY_METRICS = {
    "memory throughput",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
}
_DURATION_UNIT_TO_US = {
    "ns": 1.0e-3,
    "nsec": 1.0e-3,
    "nsecond": 1.0e-3,
    "nseconds": 1.0e-3,
    "us": 1.0,
    "usec": 1.0,
    "usecond": 1.0,
    "useconds": 1.0,
    "ms": 1.0e3,
    "msec": 1.0e3,
    "msecond": 1.0e3,
    "mseconds": 1.0e3,
    "s": 1.0e6,
    "sec": 1.0e6,
    "second": 1.0e6,
    "seconds": 1.0e6,
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MIN_COSINE = 0.999
_MAX_MEAN_ABS_ERROR = 0.5
_MAX_BAD_2PX_FRACTION = 0.02
_ROOFLINE_THRESHOLD_PCT = 80.0
_ROOFLINE_MINIMUM_DURATION_FRACTION = 0.80
_TRUSTED_NCU_IMPORTER_PATH = "/usr/local/cuda-12.4/bin/ncu"
_TRUSTED_NCU_IMPORTER_SHA256 = "02fffd2174c647582a6faa3829bcab06ca66dfb026c7b0e8f863fbbfb0cca877"
_TRUSTED_NCU_TARGET_PATH = (
    "/opt/nvidia/nsight-compute/2024.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu"
)
_TRUSTED_NCU_TARGET_SHA256 = "e2c04f5b1ebe7cec85e5f4b48c73cacc4e01dcb58a765fd54d9c362498f609d0"
_TRUSTED_NCU_TARGET_VERSION = "2024.1.1.0 (build 33998838) (public-release)"
_RECORDED_BASELINE_INFER_5_PAIRS_MEAN_MS = 631.8328293785453
_CANONICAL_ACCURACY_REFERENCE_SHA256 = (
    "998029a0b3089bc31b65aab1a7422a21820e8849337d04dbd82aa613ace74b30"
)
_CANONICAL_INPUT_SHA256 = {
    "1783934428009999872.png": {
        "left": "86af91c8a59225bbca46eb70daa0924fffe84770cd0b21c2cccd8cd5dfd7a292",
        "right": "a7b696ba30b8a8afaff043bfce8ddcac13399e39b0e258de132d17c0ef1c5959",
    },
    "1783934428076000000.png": {
        "left": "87a217aee7132e659ef8ad80f905e835c15578a0370dd7b1eff532bde87f3cb0",
        "right": "7c29a50d7115b4402a7f1307ff1fab78dce603a2cf68b8d44d419c7438050289",
    },
    "1783934428143000064.png": {
        "left": "23977bbde7440ec84f131bf5fcbed2fa0d35e65b7cb3ff9e3024dcd0966ffa28",
        "right": "0ea1dc08389dad2f3fa05bd7b2436bad79c001edefb765ba2033ba74dc319737",
    },
    "1783934428209999872.png": {
        "left": "f0f3770bdff0eabe80fb943f103f0195ff029d5c8fc7c2d2237c835f7bd8d0ec",
        "right": "478378f0a3b289fd24acec863e34cf43cbfb980de8f82f0a1ef228f4569ae047",
    },
    "1783934428276000000.png": {
        "left": "9a8d8ccf4567a00d871c77a865138392cbac806b262522ede21e702387cbbceb",
        "right": "4d6b4cb2d239da7deed002826097da0ea9fdc5fe12f7c631cbbd53e0c87000d8",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_unit(unit: str) -> str:
    return unit.strip().lower().replace("μ", "u").replace("µ", "u").replace(" ", "")


def _metric_and_embedded_unit(name: str) -> tuple[str, str | None]:
    stripped = name.strip()
    match = re.search(r"\s*\(([^()]*)\)\s*$", stripped)
    if match is None:
        return stripped.lower(), None
    unit = match.group(1).strip()
    metric = stripped[: match.start()].strip()
    # ``Compute (SM) Throughput`` contains semantic parentheses, not a unit.
    if metric.lower() == "compute" and unit.lower() == "sm":
        return stripped.lower(), None
    return metric.lower(), unit


def _parse_number(value: str, *, context: str) -> float:
    try:
        parsed = float(value.strip().replace(",", ""))
    except ValueError as exc:
        raise ValueError(f"{context}: invalid numeric value {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{context}: value must be finite, got {value!r}")
    return parsed


def _is_finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _parse_duration_us(value: str, unit: str, *, context: str) -> float:
    normalized = _normalized_unit(unit)
    try:
        multiplier = _DURATION_UNIT_TO_US[normalized]
    except KeyError as exc:
        raise ValueError(f"{context}: unsupported duration unit {unit!r}") from exc
    duration_us = _parse_number(value, context=context) * multiplier
    if duration_us <= 0:
        raise ValueError(f"{context}: duration must be positive, got {duration_us} us")
    return duration_us


def _parse_percentage(value: str, unit: str, *, context: str) -> float:
    if _normalized_unit(unit) not in {"%", "percent", "percentage"}:
        raise ValueError(f"{context}: unsupported throughput unit {unit!r}")
    percentage = _parse_number(value, context=context)
    if not 0.0 <= percentage <= 100.0:
        raise ValueError(f"{context}: percentage must be in [0, 100], got {percentage}")
    return percentage


def _csv_reader(path: Path) -> csv.DictReader[str]:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        try:
            fields = next(csv.reader([line]))
        except csv.Error:
            continue
        if _REQUIRED_COLUMNS <= {field.strip() for field in fields}:
            header_index = index
            break
    if header_index is None:
        raise ValueError(
            f"{path}: no Nsight Compute CSV header containing {sorted(_REQUIRED_COLUMNS)}"
        )
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    fieldnames = {field.strip() for field in (reader.fieldnames or [])}
    missing = _REQUIRED_COLUMNS - fieldnames
    if missing:
        raise ValueError(f"{path}: CSV is missing columns {sorted(missing)}")
    return reader


def _store_metric(launch: dict[str, Any], field: str, value: float, *, context: str) -> None:
    if field in launch:
        raise ValueError(f"{context}: duplicate {field} metric")
    launch[field] = value


def _load_samples(paths: list[Path]) -> list[dict[str, object]]:
    if not paths:
        raise ValueError("at least one Nsight Compute CSV is required")

    samples: list[dict[str, object]] = []
    required_metrics = {"duration_us", "memory_pct", "compute_pct"}
    for input_path in paths:
        path = input_path.resolve()
        launches: dict[str, dict[str, object]] = {}
        for row_number, raw_row in enumerate(_csv_reader(path), start=2):
            row = {(key or "").strip(): (value or "").strip() for key, value in raw_row.items()}
            if row.get("ID") == "ID":
                continue
            launch_id = row.get("ID", "")
            kernel = row.get("Kernel Name", "")
            if launch_id.startswith(("==PROF==", "==WARNING==", "==ERROR==")) and not kernel:
                continue
            if not launch_id or not kernel:
                raise ValueError(f"{path}:{row_number}: launch ID and kernel name are required")
            launch = launches.setdefault(
                launch_id,
                {
                    "id": launch_id,
                    "kernel": kernel,
                    "source": str(path),
                },
            )
            if launch["kernel"] != kernel:
                raise ValueError(
                    f"{path}:{row_number}: launch {launch_id} has conflicting kernel names"
                )

            metric_name = row.get("Metric Name", "")
            metric, embedded_unit = _metric_and_embedded_unit(metric_name)
            unit = row.get("Metric Unit", "") or embedded_unit or ""
            value = row.get("Metric Value", "")
            context = f"{path}:{row_number} launch {launch_id} metric {metric_name!r}"
            if metric in _DURATION_METRICS:
                _store_metric(
                    launch,
                    "duration_us",
                    _parse_duration_us(value, unit, context=context),
                    context=context,
                )
            elif metric in _COMPUTE_METRICS:
                _store_metric(
                    launch,
                    "compute_pct",
                    _parse_percentage(value, unit, context=context),
                    context=context,
                )
            elif metric in _MEMORY_METRICS:
                _store_metric(
                    launch,
                    "memory_pct",
                    _parse_percentage(value, unit, context=context),
                    context=context,
                )

        if not launches:
            raise ValueError(f"{path}: CSV contains no kernel launches")
        for launch_id, launch in launches.items():
            missing = required_metrics - launch.keys()
            if missing:
                raise ValueError(
                    f"{path}: launch {launch_id} ({launch['kernel']}) is missing "
                    f"metrics {sorted(missing)}"
                )
            launch["limiter_pct"] = max(float(launch["compute_pct"]), float(launch["memory_pct"]))
            launch["limiter"] = (
                "compute"
                if float(launch["compute_pct"]) >= float(launch["memory_pct"])
                else "memory"
            )
            samples.append(launch)
    return samples


def _validate_exhaustive_coverage(samples: list[dict[str, object]]) -> dict[str, object]:
    launch_sources: dict[int, str] = {}
    for sample in samples:
        raw_id = str(sample.get("id", ""))
        try:
            launch_id = int(raw_id)
        except ValueError as exc:
            raise ValueError(f"launch ID must be a nonnegative integer, got {raw_id!r}") from exc
        if launch_id < 0:
            raise ValueError(f"launch ID must be nonnegative, got {launch_id}")
        source = str(sample.get("source", ""))
        if launch_id in launch_sources:
            raise ValueError(
                f"launch {launch_id} appears in both {launch_sources[launch_id]} and {source}"
            )
        launch_sources[launch_id] = source

    launch_ids = sorted(launch_sources)
    if not launch_ids or launch_ids[0] != 0:
        first = launch_ids[0] if launch_ids else None
        raise ValueError(f"exhaustive capture must begin at launch 0, got {first}")
    missing = sorted(set(range(launch_ids[-1] + 1)) - set(launch_ids))
    if missing:
        preview = missing[:20]
        raise ValueError(f"launch coverage has gaps at {preview}")
    return {
        "exhaustive": True,
        "first_launch_id": 0,
        "last_launch_id": launch_ids[-1],
        "launch_count": len(launch_ids),
        "gaps": [],
        "overlaps": [],
        "incomplete_launches": 0,
    }


def _load_profile_manifest(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{resolved}: invalid profile manifest JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{resolved}: profile manifest root must be an object")
    if manifest.get("profile_scope") != "native_feature_and_post":
        raise ValueError(f"{resolved}: manifest does not cover native feature and post")
    required_ranges = {"ffs_full", "ffs_feature", "ffs_post"}
    ranges = manifest.get("nvtx_ranges")
    if not isinstance(ranges, list) or not required_ranges <= set(ranges):
        raise ValueError(f"{resolved}: manifest is missing required NVTX ranges")
    expected_contract = {
        "tool": "ncu",
        "config_file": "off",
        "metric_set": "roofline",
        "nvtx_include": "ffs_full/",
        "replay_mode": "kernel",
        "launch_skip": 0,
        "launch_count": "all",
    }
    if manifest.get("required_profiler_contract") != expected_contract:
        raise ValueError(f"{resolved}: manifest is missing the exhaustive profiler contract")
    if manifest.get("cuda_graphs") is not False:
        raise ValueError(f"{resolved}: roofline qualification requires eager kernel capture")

    pair = manifest.get("pair")
    engines = manifest.get("engines")
    if not isinstance(pair, dict) or not isinstance(engines, dict):
        raise ValueError(f"{resolved}: manifest is missing pair or engine provenance")
    left_image = pair.get("left_image")
    right_image = pair.get("right_image")
    digests: dict[str, object] = {
        "left_image": (
            left_image.get("sha256") if isinstance(left_image, dict) else pair.get("left_sha256")
        ),
        "right_image": (
            right_image.get("sha256") if isinstance(right_image, dict) else pair.get("right_sha256")
        ),
        "feature_engine": (
            engines.get("feature", {}).get("sha256")
            if isinstance(engines.get("feature"), dict)
            else None
        ),
        "post_engine": (
            engines.get("post", {}).get("sha256") if isinstance(engines.get("post"), dict) else None
        ),
    }
    plugin_artifacts = manifest.get("plugin_artifacts", [])
    if not isinstance(plugin_artifacts, list) or not plugin_artifacts:
        raise ValueError(f"{resolved}: manifest is missing native plugin provenance")
    for index, artifact in enumerate(plugin_artifacts):
        digests[f"plugin_{index}"] = artifact.get("sha256") if isinstance(artifact, dict) else None
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        raise ValueError(f"{resolved}: manifest is missing profiler source provenance")
    for name in ("python_executable", "profile_ncu", "benchmark", "trt_runner"):
        artifact = source_artifacts.get(name)
        digests[f"source_{name}"] = artifact.get("sha256") if isinstance(artifact, dict) else None
    invalid = [
        name
        for name, digest in digests.items()
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
    ]
    if invalid:
        raise ValueError(f"{resolved}: invalid or missing SHA-256 fields {invalid}")
    canonical_sources = {
        "profile_ncu": Path(__file__).with_name("profile_ncu.py").resolve(),
        "benchmark": Path(__file__).with_name("benchmark.py").resolve(),
        "trt_runner": Path(__file__).with_name("trt_runner.py").resolve(),
    }
    for name, canonical_path in canonical_sources.items():
        artifact = source_artifacts.get(name)
        artifact_path = artifact.get("path") if isinstance(artifact, dict) else None
        if (
            Path(str(artifact_path)).resolve() != canonical_path
            or not canonical_path.is_file()
            or digests[f"source_{name}"] != _sha256(canonical_path)
        ):
            raise ValueError(f"{resolved}: source artifact {name} is not current")
    pair_name = pair.get("name")
    expected_pair = _CANONICAL_INPUT_SHA256.get(pair_name)
    if expected_pair is None:
        raise ValueError(f"{resolved}: profile pair is not in the canonical L4 input set")
    if pair.get("scale") != 1.0:
        raise ValueError(f"{resolved}: roofline qualification requires input scale 1.0")
    if (
        digests["left_image"] != expected_pair["left"]
        or digests["right_image"] != expected_pair["right"]
    ):
        raise ValueError(f"{resolved}: profile pair hashes do not match the canonical L4 input")
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        raise ValueError(f"{resolved}: manifest is missing profiler environment")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "profile_scope": manifest["profile_scope"],
        "nvtx_ranges": ranges,
        "pair_name": pair_name,
        "artifact_sha256": digests,
        "cuda_graphs": manifest.get("cuda_graphs"),
        "environment": environment,
        "command_binding": {
            "model_root": manifest.get("model_root"),
            "input_root": manifest.get("input_root"),
            "feature_engine": (
                engines.get("feature", {}).get("path")
                if isinstance(engines.get("feature"), dict)
                else None
            ),
            "post_engine": (
                engines.get("post", {}).get("path")
                if isinstance(engines.get("post"), dict)
                else None
            ),
            "plugin_libraries": [
                artifact.get("path") for artifact in plugin_artifacts if isinstance(artifact, dict)
            ],
            "profile_script": (
                source_artifacts.get("profile_ncu", {}).get("path")
                if isinstance(source_artifacts.get("profile_ncu"), dict)
                else None
            ),
            "python_executable": (
                source_artifacts.get("python_executable", {}).get("path")
                if isinstance(source_artifacts.get("python_executable"), dict)
                else None
            ),
            "pair_name": pair_name,
            "warmup": manifest.get("requested_warmup_enqueues"),
        },
    }


def _artifact_digest(value: object, *, context: str) -> str:
    if not isinstance(value, dict):
        raise ValueError(f"benchmark receipt is missing {context}")
    digest = value.get("sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"benchmark receipt has invalid {context} SHA-256")
    return digest


def _load_benchmark_receipt(
    path: Path,
    *,
    profile_manifest: dict[str, object],
) -> dict[str, object]:
    resolved = path.resolve()
    try:
        receipt = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{resolved}: invalid benchmark receipt JSON") from exc
    if not isinstance(receipt, dict) or receipt.get("accuracy_passed") is not True:
        raise ValueError(f"{resolved}: benchmark receipt did not pass the accuracy gate")
    environment = receipt.get("environment")
    profile_environment = profile_manifest.get("environment")
    required_environment_keys = (
        "torch",
        "tensorrt",
        "cuda_runtime",
        "gpu_name",
        "gpu_capability",
    )
    if not isinstance(environment, dict) or not isinstance(profile_environment, dict):
        raise ValueError(f"{resolved}: benchmark or profile environment is missing")
    environment_mismatches = {
        name: {"profile": profile_environment.get(name), "benchmark": environment.get(name)}
        for name in required_environment_keys
        if environment.get(name) != profile_environment.get(name)
    }
    if environment_mismatches:
        raise ValueError(
            f"{resolved}: benchmark and profile environments do not match: {environment_mismatches}"
        )
    if environment.get("gpu_name") != "NVIDIA L4" or environment.get("gpu_capability") != [8, 9]:
        raise ValueError(
            f"{resolved}: qualification requires an NVIDIA L4 (compute capability 8.9)"
        )
    required_protocol = {
        "backend": "trt",
        "num_pairs": 5,
        "start_index": 0,
        "scale": 1.0,
        "valid_iters": 8,
        "max_disp": 192,
        "cuda_graphs": True,
    }

    def protocol_matches(actual: object, expected: object) -> bool:
        if type(expected) is bool:
            return type(actual) is bool and actual is expected
        if type(expected) is int:
            return type(actual) is int and actual == expected
        if type(expected) is float:
            return _is_finite_number(actual) and float(actual) == expected
        return type(actual) is type(expected) and actual == expected

    protocol_mismatches = {
        name: {"expected": expected_value, "actual": receipt.get(name)}
        for name, expected_value in required_protocol.items()
        if not protocol_matches(receipt.get(name), expected_value)
    }
    if protocol_mismatches:
        raise ValueError(f"{resolved}: benchmark protocol mismatch: {protocol_mismatches}")
    warmup = receipt.get("warmup_iters")
    iterations = receipt.get("iters")
    if type(warmup) is not int or warmup < 20:
        raise ValueError(f"{resolved}: benchmark requires at least 20 warmup iterations")
    if type(iterations) is not int or iterations < 100:
        raise ValueError(f"{resolved}: benchmark requires at least 100 timed iterations")

    thresholds = {
        "minimum_cosine": (_MIN_COSINE, lambda value, limit: value >= limit),
        "maximum_mean_abs_error": (
            _MAX_MEAN_ABS_ERROR,
            lambda value, limit: value <= limit,
        ),
        "maximum_bad_2px_fraction": (
            _MAX_BAD_2PX_FRACTION,
            lambda value, limit: value <= limit,
        ),
    }
    for name, (limit, accepted) in thresholds.items():
        value = receipt.get(name)
        if not _is_finite_number(value):
            raise ValueError(f"{resolved}: benchmark has invalid threshold {name}")
        if not accepted(float(value), limit):
            raise ValueError(f"{resolved}: benchmark weakened required threshold {name}")

    accuracy = receipt.get("accuracy")
    if not isinstance(accuracy, dict):
        raise ValueError(f"{resolved}: benchmark is missing accuracy metrics")
    accuracy_checks = {
        "global_cosine": (lambda value: value >= _MIN_COSINE),
        "mean_abs_error": (lambda value: value <= _MAX_MEAN_ABS_ERROR),
        "bad_2px_fraction": (lambda value: value <= _MAX_BAD_2PX_FRACTION),
    }
    for name, accepted in accuracy_checks.items():
        value = accuracy.get(name)
        if not _is_finite_number(value):
            raise ValueError(f"{resolved}: benchmark has invalid accuracy metric {name}")
        if not accepted(float(value)):
            raise ValueError(f"{resolved}: benchmark fails required accuracy metric {name}")

    samples = receipt.get("samples_ms")
    if not isinstance(samples, dict):
        raise ValueError(f"{resolved}: benchmark is missing raw timing samples")
    for name in ("preprocess", "inference", "total"):
        values = samples.get(name)
        if not isinstance(values, list) or len(values) != iterations:
            raise ValueError(f"{resolved}: benchmark {name} sample count does not match iters")
        if any(not _is_finite_number(value) or float(value) <= 0 for value in values):
            raise ValueError(f"{resolved}: benchmark {name} samples must be finite and positive")
    inference_values = [float(value) for value in samples["inference"]]
    inference_summary = receipt.get("infer_5_pairs")
    if not isinstance(inference_summary, dict):
        raise ValueError(f"{resolved}: benchmark is missing inference summary")
    recomputed = {
        "count": iterations,
        "mean_ms": float(sum(inference_values) / iterations),
        "median_ms": float(np.median(np.asarray(inference_values, dtype=np.float64))),
        "p90_ms": float(np.percentile(np.asarray(inference_values, dtype=np.float64), 90)),
    }
    for name, value in recomputed.items():
        recorded = inference_summary.get(name)
        if not _is_finite_number(recorded) or not math.isclose(
            float(recorded), float(value), rel_tol=1.0e-9, abs_tol=1.0e-9
        ):
            raise ValueError(f"{resolved}: benchmark inference summary does not match raw samples")
    inference_mean_ms = recomputed["mean_ms"]
    if inference_mean_ms >= _RECORDED_BASELINE_INFER_5_PAIRS_MEAN_MS:
        raise ValueError(
            f"{resolved}: benchmark mean {inference_mean_ms} ms does not beat the recorded "
            f"baseline {_RECORDED_BASELINE_INFER_5_PAIRS_MEAN_MS} ms"
        )
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"{resolved}: benchmark receipt is missing artifact provenance")

    expected = profile_manifest["artifact_sha256"]
    if not isinstance(expected, dict):
        raise ValueError("profile manifest artifact provenance is invalid")
    actual = {
        "feature_engine": _artifact_digest(
            artifacts.get("feature_engine"), context="feature engine"
        ),
        "post_engine": _artifact_digest(artifacts.get("post_engine"), context="post engine"),
        "source_benchmark": _artifact_digest(
            artifacts.get("benchmark_tool"), context="benchmark tool"
        ),
        "source_trt_runner": _artifact_digest(artifacts.get("runner_tool"), context="runner tool"),
    }
    plugins = artifacts.get("plugin_libraries")
    if not isinstance(plugins, list) or not plugins:
        raise ValueError(f"{resolved}: benchmark receipt is missing native plugin artifacts")
    plugin_digests = {
        _artifact_digest(plugin, context=f"plugin {index}") for index, plugin in enumerate(plugins)
    }
    expected_plugins = {
        digest for name, digest in expected.items() if str(name).startswith("plugin_")
    }
    if plugin_digests != expected_plugins:
        raise ValueError("benchmark and profile native plugin hashes do not match")

    inputs = artifacts.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError(f"{resolved}: benchmark receipt is missing input artifacts")
    input_digests: dict[str, dict[str, str]] = {}
    for index, item in enumerate(inputs):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"{resolved}: benchmark input {index} is invalid")
        name = item["name"]
        if name in input_digests:
            raise ValueError(f"{resolved}: benchmark input {name!r} is duplicated")
        input_digests[name] = {
            "left": _artifact_digest(item.get("left"), context=f"input {name} left image"),
            "right": _artifact_digest(item.get("right"), context=f"input {name} right image"),
        }
    if input_digests != _CANONICAL_INPUT_SHA256:
        raise ValueError(f"{resolved}: benchmark inputs do not match the canonical five-pair set")
    pair_name = profile_manifest.get("pair_name")
    if not isinstance(pair_name, str) or pair_name not in input_digests:
        raise ValueError(f"{resolved}: benchmark receipt does not contain the profile pair")
    actual["left_image"] = input_digests[pair_name]["left"]
    actual["right_image"] = input_digests[pair_name]["right"]

    mismatches = {
        name: {"profile": expected.get(name), "benchmark": digest}
        for name, digest in actual.items()
        if expected.get(name) != digest
    }
    if mismatches:
        raise ValueError(f"benchmark and profile artifact hashes do not match: {mismatches}")
    reference_digest = _artifact_digest(
        artifacts.get("accuracy_reference"), context="accuracy reference"
    )
    if reference_digest != _CANONICAL_ACCURACY_REFERENCE_SHA256:
        raise ValueError(f"{resolved}: accuracy reference is not the canonical PyTorch output")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "accuracy_passed": True,
        "accuracy": receipt.get("accuracy"),
        "protocol": {
            **required_protocol,
            "warmup_iters": warmup,
            "iters": iterations,
        },
        "infer_5_pairs": inference_summary,
        "recorded_baseline_infer_5_pairs_mean_ms": (_RECORDED_BASELINE_INFER_5_PAIRS_MEAN_MS),
        "inference_throughput_speedup": (
            _RECORDED_BASELINE_INFER_5_PAIRS_MEAN_MS / inference_mean_ms
        ),
        "beats_recorded_baseline": True,
        "environment": environment,
        "accuracy_reference_sha256": reference_digest,
        "artifact_sha256": actual,
    }


def _load_ncu_session(
    path: Path,
    *,
    report: Path,
    profile_manifest: dict[str, object],
) -> dict[str, object]:
    resolved = path.resolve()
    command_line = None
    device_header_seen = False
    device_attributes: dict[str, str] = {}
    process_header_seen = False
    process_names: list[str] = []
    in_process_table = False
    ncu_target_version = None
    with resolved.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            stripped = [value.strip() for value in row]
            if len(stripped) >= 2 and stripped[0] == "Profiler Command Line":
                if command_line is not None:
                    raise ValueError(f"{resolved}: duplicate profiler command line")
                command_line = stripped[1]
            if len(stripped) >= 2 and stripped[0] == "Nsight Compute Target":
                if ncu_target_version is not None:
                    raise ValueError(f"{resolved}: duplicate Nsight Compute target version")
                ncu_target_version = stripped[1]
            if stripped == ["Process Id", "Process Name"]:
                if process_header_seen:
                    raise ValueError(f"{resolved}: duplicate process table")
                process_header_seen = True
                in_process_table = True
                continue
            if in_process_table:
                if len(stripped) == 2 and stripped[0].isdigit():
                    process_names.append(stripped[1])
                    continue
                in_process_table = False
            if stripped and stripped[0] == "Device Attribute":
                if device_header_seen or stripped != ["Device Attribute", "Device 0"]:
                    raise ValueError(f"{resolved}: expected exactly one profiled CUDA device")
                device_header_seen = True
            if stripped and stripped[0] in {
                "display_name",
                "compute_capability_major",
                "compute_capability_minor",
            }:
                if len(stripped) != 2 or stripped[0] in device_attributes:
                    raise ValueError(f"{resolved}: invalid or duplicate CUDA device attribute")
                device_attributes[stripped[0]] = stripped[1]
    if not command_line:
        raise ValueError(f"{resolved}: missing Nsight Compute profiler command line")
    expected_device = {
        "display_name": "NVIDIA L4",
        "compute_capability_major": "8",
        "compute_capability_minor": "9",
    }
    if not device_header_seen or device_attributes != expected_device:
        raise ValueError(f"{resolved}: Nsight Compute report is not from one NVIDIA L4 device")
    arguments = shlex.split(command_line)
    if any(argument.startswith("@") for argument in arguments):
        raise ValueError(f"{resolved}: response files are forbidden in profiler commands")
    trusted_target = Path(_TRUSTED_NCU_TARGET_PATH).resolve()
    if (
        not arguments
        or Path(arguments[0]).resolve() != trusted_target
        or not trusted_target.is_file()
        or _sha256(trusted_target) != _TRUSTED_NCU_TARGET_SHA256
        or ncu_target_version != _TRUSTED_NCU_TARGET_VERSION
    ):
        raise ValueError(f"{resolved}: profiler command is not from the trusted Nsight Compute")

    value_options = {
        "--config-file": "off",
        "--set": "roofline",
        "--nvtx-include": "ffs_full/",
        "--replay-mode": "kernel",
        "--export": None,
    }
    flag_options = {"--nvtx", "--force-overwrite"}
    seen_values: dict[str, str] = {}
    seen_flags: set[str] = set()
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument in value_options:
            if argument in seen_values or index + 1 >= len(arguments):
                raise ValueError(f"{resolved}: duplicate or incomplete profiler option {argument}")
            seen_values[argument] = arguments[index + 1]
            index += 2
        elif argument in flag_options:
            if argument in seen_flags:
                raise ValueError(f"{resolved}: duplicate profiler flag {argument}")
            seen_flags.add(argument)
            index += 1
        elif argument.startswith("-"):
            raise ValueError(f"{resolved}: profiler option is not allowlisted: {argument}")
        else:
            break
    if seen_flags != flag_options or set(seen_values) != set(value_options):
        raise ValueError(f"{resolved}: profiler command is missing required exact options")
    for option, expected in value_options.items():
        if expected is not None and seen_values[option] != expected:
            raise ValueError(f"{resolved}: profiler option {option} must be {expected!r}")

    export = seen_values["--export"]
    expected_report = Path(export)
    if expected_report.suffix != ".ncu-rep":
        expected_report = expected_report.with_suffix(".ncu-rep")
    if expected_report.resolve() != report.resolve():
        raise ValueError(
            f"{resolved}: profiler export {expected_report} does not match report {report}"
        )

    target = arguments[index:]
    if len(target) < 5:
        raise ValueError(f"{resolved}: profiler target must be Python profile_ncu.py")
    binding = profile_manifest.get("command_binding")
    if not isinstance(binding, dict):
        raise ValueError(f"{resolved}: profile manifest has no command binding")
    artifact_sha256 = profile_manifest.get("artifact_sha256")
    if not isinstance(artifact_sha256, dict):
        raise ValueError(f"{resolved}: profile manifest has no source provenance")
    analyzer_python = Path(sys.executable).resolve()
    canonical_python = Path(str(binding.get("python_executable"))).resolve()
    if (
        canonical_python != analyzer_python
        or Path(target[0]).resolve() != canonical_python
        or not canonical_python.is_file()
        or _sha256(canonical_python) != _sha256(analyzer_python)
        or artifact_sha256.get("source_python_executable") != _sha256(canonical_python)
    ):
        raise ValueError(f"{resolved}: profiler interpreter does not match the manifest")
    if (
        not process_header_seen
        or len(process_names) != 1
        or Path(process_names[0]).resolve() != canonical_python
    ):
        raise ValueError(f"{resolved}: profiled process is not the bound Python interpreter")
    canonical_profile_script = Path(__file__).with_name("profile_ncu.py").resolve()
    if Path(target[1]).resolve() != canonical_profile_script:
        raise ValueError(f"{resolved}: profiler target is not the canonical profile_ncu.py")
    if Path(
        str(binding.get("profile_script"))
    ).resolve() != canonical_profile_script or artifact_sha256.get("source_profile_ncu") != _sha256(
        canonical_profile_script
    ):
        raise ValueError(f"{resolved}: profiler target source does not match the manifest")
    expected_positionals = [
        binding.get("model_root"),
        binding.get("feature_engine"),
        binding.get("post_engine"),
    ]
    if target[2:5] != expected_positionals:
        raise ValueError(f"{resolved}: profiler target engine paths do not match the manifest")
    target_value_options = {
        "--plugin-library": None,
        "--input-root": binding.get("input_root"),
        "--pair-name": binding.get("pair_name"),
        "--warmup": str(binding.get("warmup")),
        "--manifest": profile_manifest.get("path"),
    }
    target_values: dict[str, str] = {}
    index = 5
    while index < len(target):
        argument = target[index]
        if argument not in target_value_options or argument in target_values:
            raise ValueError(f"{resolved}: profiler target option is not allowlisted: {argument}")
        if index + 1 >= len(target):
            raise ValueError(f"{resolved}: profiler target option has no value: {argument}")
        target_values[argument] = target[index + 1]
        index += 2
    if set(target_values) != set(target_value_options):
        raise ValueError(f"{resolved}: profiler target is missing required exact options")
    expected_plugins = binding.get("plugin_libraries")
    if (
        not isinstance(expected_plugins, list)
        or target_values["--plugin-library"] not in expected_plugins
    ):
        raise ValueError(f"{resolved}: profiler plugin path does not match the manifest")
    for option, expected in target_value_options.items():
        if option != "--plugin-library" and target_values[option] != expected:
            raise ValueError(f"{resolved}: profiler target {option} does not match the manifest")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "profiler_command_line": command_line,
        "launch_filters": [],
        "device": {
            "name": device_attributes["display_name"],
            "compute_capability": [
                int(device_attributes["compute_capability_major"]),
                int(device_attributes["compute_capability_minor"]),
            ],
        },
    }


def _export_ncu_page(
    ncu_binary: str,
    report: Path,
    *,
    page: str,
    output: Path,
) -> list[str]:
    resolved_binary = Path(ncu_binary).resolve()
    trusted_binary = Path(_TRUSTED_NCU_IMPORTER_PATH).resolve()
    if (
        resolved_binary != trusted_binary
        or not trusted_binary.is_file()
        or _sha256(trusted_binary) != _TRUSTED_NCU_IMPORTER_SHA256
    ):
        raise ValueError("Nsight Compute importer is not the pinned qualification binary")
    command = [
        str(trusted_binary),
        "--config-file",
        "off",
        "--import",
        str(report),
        "--page",
        page,
        "--csv",
    ]
    if page == "details":
        command.extend(("--print-metric-name", "name", "--print-units", "base"))
    command.extend(("--log-file", str(output)))
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Nsight Compute failed to export {page} from {report}:\n{completed.stdout}"
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Nsight Compute did not produce the {page} export: {output}")
    return command


def summarize_samples(
    samples: list[dict[str, object]], *, near_ceiling_pct: float = 80.0, top: int = 10
) -> dict[str, object]:
    if not samples:
        raise ValueError("cannot summarize an empty sample set")
    if not 0.0 < near_ceiling_pct <= 100.0:
        raise ValueError("near-ceiling-pct must be in (0, 100]")
    if top < 0:
        raise ValueError("top cannot be negative")

    total_duration = sum(float(sample["duration_us"]) for sample in samples)
    if not math.isfinite(total_duration) or total_duration <= 0:
        raise ValueError("sampled duration must be finite and positive")

    def weighted(field: str) -> float:
        return (
            sum(float(sample["duration_us"]) * float(sample[field]) for sample in samples)
            / total_duration
        )

    compute_limited_duration = sum(
        float(sample["duration_us"]) for sample in samples if sample["limiter"] == "compute"
    )
    memory_limited_duration = total_duration - compute_limited_duration
    near_ceiling_duration = sum(
        float(sample["duration_us"])
        for sample in samples
        if float(sample["limiter_pct"]) >= near_ceiling_pct
    )
    top_launches = sorted(
        samples,
        key=lambda sample: float(sample["duration_us"]),
        reverse=True,
    )[:top]
    return {
        "sampled_launches": len(samples),
        "sampled_duration_us": total_duration,
        "duration_weighted_compute_pct": weighted("compute_pct"),
        "duration_weighted_memory_pct": weighted("memory_pct"),
        "duration_weighted_limiter_pct": weighted("limiter_pct"),
        "near_ceiling_threshold_pct": near_ceiling_pct,
        "near_ceiling_duration_us": near_ceiling_duration,
        "near_ceiling_duration_fraction": near_ceiling_duration / total_duration,
        "compute_limited_launches": sum(sample["limiter"] == "compute" for sample in samples),
        "memory_limited_launches": sum(sample["limiter"] == "memory" for sample in samples),
        "compute_limited_duration_fraction": compute_limited_duration / total_duration,
        "memory_limited_duration_fraction": memory_limited_duration / total_duration,
        "diagnostics": {
            "max_compute_pct": max(float(sample["compute_pct"]) for sample in samples),
            "max_memory_pct": max(float(sample["memory_pct"]) for sample in samples),
            "max_limiter_pct": max(float(sample["limiter_pct"]) for sample in samples),
            "top_duration_launches": top_launches,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--benchmark-result", required=True, type=Path)
    parser.add_argument("--ncu-report", required=True, type=Path)
    parser.add_argument("--ncu-binary", default=_TRUSTED_NCU_IMPORTER_PATH)
    parser.add_argument(
        "--json-output",
        "--output-json",
        "--output",
        dest="json_output",
        type=Path,
        help="optional path for the machine-readable summary",
    )
    args = parser.parse_args()

    ncu_report = args.ncu_report.resolve()
    if not ncu_report.is_file():
        raise FileNotFoundError(f"Nsight Compute report does not exist: {ncu_report}")
    with ncu_report.open("rb") as handle:
        if handle.read(4) != b"NVR\0":
            raise ValueError(f"invalid Nsight Compute report header: {ncu_report}")
    temporary = tempfile.TemporaryDirectory(prefix="trtmc-ffs-ncu-")
    temporary_root = Path(temporary.name)
    details_export = temporary_root / "details.csv"
    session_export = temporary_root / "session.csv"
    details_command = _export_ncu_page(
        args.ncu_binary,
        ncu_report,
        page="details",
        output=details_export,
    )
    session_command = _export_ncu_page(
        args.ncu_binary,
        ncu_report,
        page="session",
        output=session_export,
    )
    samples = _load_samples([details_export])
    for sample in samples:
        sample["source"] = str(ncu_report)
    coverage = _validate_exhaustive_coverage(samples)
    profile_manifest = _load_profile_manifest(args.manifest)
    benchmark_receipt = _load_benchmark_receipt(
        args.benchmark_result,
        profile_manifest=profile_manifest,
    )
    ncu_session = _load_ncu_session(
        session_export,
        report=ncu_report,
        profile_manifest=profile_manifest,
    )
    summary = summarize_samples(
        samples,
        near_ceiling_pct=_ROOFLINE_THRESHOLD_PCT,
        top=args.top,
    )
    roofline_passed = (
        summary["duration_weighted_limiter_pct"] >= _ROOFLINE_THRESHOLD_PCT
        and summary["near_ceiling_duration_fraction"] >= _ROOFLINE_MINIMUM_DURATION_FRACTION
    )
    latency_passed = bool(benchmark_receipt["beats_recorded_baseline"])
    summary["qualification_gate"] = {
        "minimum_duration_weighted_limiter_pct": _ROOFLINE_THRESHOLD_PCT,
        "minimum_near_ceiling_duration_fraction": _ROOFLINE_MINIMUM_DURATION_FRACTION,
        "recorded_baseline_infer_5_pairs_mean_ms": (_RECORDED_BASELINE_INFER_5_PAIRS_MEAN_MS),
        "measured_infer_5_pairs_mean_ms": benchmark_receipt["infer_5_pairs"]["mean_ms"],
        "beats_recorded_baseline": latency_passed,
        "roofline_passed": roofline_passed,
        "passed": roofline_passed and latency_passed,
    }
    summary["ncu_exports"] = {
        "details": {
            "sha256": _sha256(details_export),
            "bytes": details_export.stat().st_size,
            "command": details_command,
        },
        "session": {
            "sha256": _sha256(session_export),
            "bytes": session_export.stat().st_size,
            "command": session_command,
        },
    }
    summary["coverage"] = coverage
    summary["profile_manifest"] = profile_manifest
    summary["benchmark_receipt"] = benchmark_receipt
    summary["ncu_report"] = {
        "path": str(ncu_report),
        "sha256": _sha256(ncu_report),
        "bytes": ncu_report.stat().st_size,
    }
    summary["ncu_session"] = ncu_session
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not summary["qualification_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
