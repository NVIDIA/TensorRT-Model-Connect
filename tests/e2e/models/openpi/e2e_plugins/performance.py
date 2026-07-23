# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, model-owned OpenPI performance evidence validation."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from tests.e2e.models.openpi import qualification

from . import openpi_proof_path

TORCH_EAGER_SHA256 = "3cd217412ddbe931e179038f39c74ffc4b7b35675132eb739370dc42dc53a8d8"
BENCHMARK_ITERATIONS = 1000
BENCHMARK_WARMUPS = 100
_PROFILE = "pi05_droid"
_UPSTREAM_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
_EXPECTED_WORKLOAD = {
    "action_horizon": 15,
    "batch_size": 1,
    "camera_slots": 3,
    "capture_manifest_sha256": "837695cfb46e9d550edd8c4d1126f2f052a256ceb597808cdde9032d3633a78b",
    "case_id": "droid_iris_20231204154425_f000005",
    "case_identity_sha256": "0800003c9dd4ba0d3cc63219bf73df361c1524c33e7a3533637236b522c8a866",
    "fixed_external_noise": True,
    "flow_steps": 10,
    "input_payload_sha256": {
        "base_0_rgb": "b2d1abbcc6987fd01440e590e741a908cbe2730f4e5e3a547d4ebcc4cd71962c",
        "initial_noise": "99a09944483723ee6221367ee8276ab8810f1cef7b95fc9306bcb673bc593e69",
        "left_wrist_0_rgb": "b137782b5e544c666fcb63721ccd198db717323c7ca891b251d92700b4efc196",
        "state": "1232e74cfa6c1f4da00392c35619d80e0cdec678392e08ad7e57755e626c035b",
    },
    "internal_action_dim": 32,
}
_BASELINE_FIELDS = {
    "artifact_type",
    "backend",
    "determinism",
    "first_request_ms",
    "hardware",
    "iterations",
    "latency_samples_ms",
    "profile_name",
    "provenance",
    "schema_version",
    "warmups",
    "workload",
}
_DECIMAL = r"(?:0|[1-9][0-9]*)\.[0-9]{6}"
_BENCHMARK_PREFIX = "[trtmc.openpi.benchmark]"
_BENCHMARK_LINE = re.compile(
    rf"^\[trtmc\.openpi\.benchmark\] action_ms=(?P<mean>{_DECIMAL}) "
    rf"p50_ms=(?P<p50>{_DECIMAL}) p95_ms=(?P<p95>{_DECIMAL}) "
    r"iterations=(?P<iterations>[1-9][0-9]*) warmup=(?P<warmups>0|[1-9][0-9]*)$"
)


def _positive_finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("OpenPI performance samples cannot be empty")
    ordered = sorted(values)
    return ordered[math.ceil(probability * len(ordered)) - 1]


def _latency_summary(samples: Sequence[Any], expected_count: int) -> dict[str, float]:
    if len(samples) != expected_count:
        raise ValueError(
            f"OpenPI performance sample count mismatch: expected {expected_count}, found {len(samples)}"
        )
    values = [
        _positive_finite(value, f"OpenPI performance sample {index}")
        for index, value in enumerate(samples)
    ]
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": _nearest_rank(values, 0.50),
        "p95_ms": _nearest_rank(values, 0.95),
    }


def validate_torch_eager_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and summarize the immutable Torch Eager qualification receipt."""

    if set(document) != _BASELINE_FIELDS:
        raise ValueError("Torch Eager baseline fields do not match the OpenPI contract")
    expected_identity = {
        "schema_version": 1,
        "artifact_type": "openpi_baseline_performance_measurement",
        "backend": "pytorch_eager",
        "profile_name": _PROFILE,
        "iterations": BENCHMARK_ITERATIONS,
        "warmups": BENCHMARK_WARMUPS,
    }
    for field, expected in expected_identity.items():
        if document.get(field) != expected or type(document.get(field)) is not type(expected):
            raise ValueError(f"Torch Eager baseline {field} is stale or mismatched")
    _positive_finite(document.get("first_request_ms"), "Torch Eager first request latency")

    hardware = document.get("hardware")
    if not isinstance(hardware, Mapping) or (
        hardware.get("gpu_name"),
        hardware.get("tensorrt_version"),
    ) != ("NVIDIA GB300", "11.2.0.113"):
        raise ValueError("Torch Eager baseline hardware does not match the qualified platform")
    if document.get("workload") != _EXPECTED_WORKLOAD:
        raise ValueError("Torch Eager baseline workload is stale or mismatched")

    provenance = document.get("provenance")
    if not isinstance(provenance, Mapping) or (
        provenance.get("upstream_repository"),
        provenance.get("upstream_commit"),
    ) != ("https://github.com/Physical-Intelligence/openpi.git", _UPSTREAM_COMMIT):
        raise ValueError("Torch Eager baseline upstream provenance is stale or mismatched")
    environment = provenance.get("environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("effective_compile_mode") is not None
    ):
        raise ValueError("Torch Eager baseline must have effective_compile_mode=null")
    if environment.get("forbidden_compiler_environment_names") != []:
        raise ValueError("Torch Eager baseline has forbidden compiler environment settings")
    guard = environment.get("torch_compile_guard")
    expected_guard = {
        "enforcement": "raise_on_invocation",
        "invocation_count": 0,
        "scope": "policy_construction_and_measurement",
    }
    if (
        guard != expected_guard
        or not isinstance(guard, Mapping)
        or type(guard.get("invocation_count")) is not int
    ):
        raise ValueError("Torch Eager baseline torch.compile guard is invalid")

    samples = document.get("latency_samples_ms")
    if not isinstance(samples, list):
        raise ValueError("Torch Eager baseline latency_samples_ms must be an array")
    latency = _latency_summary(samples, BENCHMARK_ITERATIONS)
    return {
        "artifact_sha256": TORCH_EAGER_SHA256,
        "backend": "pytorch_eager",
        "profile_name": _PROFILE,
        "iterations": BENCHMARK_ITERATIONS,
        "warmups": BENCHMARK_WARMUPS,
        "latency_ms": latency,
        "effective_compile_mode": None,
        "torch_compile_guard_invocation_count": 0,
        "upstream_commit": _UPSTREAM_COMMIT,
        "hardware": {
            "gpu_name": hardware["gpu_name"],
            "tensorrt_version": hardware["tensorrt_version"],
        },
        "workload": dict(_EXPECTED_WORKLOAD),
    }


def load_torch_eager_baseline() -> dict[str, Any]:
    """Load the Torch Eager receipt from the pinned offline snapshot."""

    path = openpi_proof_path("performance", "pytorch-eager.json")
    if not path.is_file():
        raise ValueError("Pinned OpenPI Torch Eager baseline is unavailable")
    digest = qualification.sha256_file(path)
    if digest != TORCH_EAGER_SHA256:
        raise ValueError(
            "Pinned OpenPI Torch Eager baseline SHA-256 mismatch: "
            f"expected {TORCH_EAGER_SHA256}, found {digest}"
        )
    return validate_torch_eager_document(qualification.strict_json_load(path))


def validate_case(case: Any, baseline: Mapping[str, Any]) -> None:
    """Reject a case whose declared workload differs from the pinned receipt."""

    inputs = case.inputs
    expected_inputs = {
        "profile": baseline["profile_name"],
        "reference_case_id": baseline["workload"]["case_id"],
        "action_horizon": baseline["workload"]["action_horizon"],
        "internal_action_dim": baseline["workload"]["internal_action_dim"],
        "flow_steps": baseline["workload"]["flow_steps"],
        "fixed_external_noise": baseline["workload"]["fixed_external_noise"],
    }
    for field, expected in expected_inputs.items():
        if (
            field not in inputs
            or inputs[field] != expected
            or type(inputs[field]) is not type(expected)
        ):
            raise ValueError(f"OpenPI performance case {field} differs from the pinned workload")

    expected_declaration = {
        "native_backend": "tensorrt_cpp",
        "baseline_backend": "pytorch_eager",
        "baseline_artifact": "performance/pytorch-eager.json",
        "baseline_sha256": TORCH_EAGER_SHA256,
        "iterations": BENCHMARK_ITERATIONS,
        "warmups": BENCHMARK_WARMUPS,
    }
    declared = case.metadata.get("performance_qualification")
    if (
        declared != expected_declaration
        or not isinstance(declared, Mapping)
        or any(
            type(declared[key]) is not type(value) for key, value in expected_declaration.items()
        )
    ):
        raise ValueError("OpenPI performance qualification declaration is stale or mismatched")


def parse_native_benchmark(stderr: str) -> dict[str, Any]:
    """Parse the single anchored benchmark receipt emitted by ``trtmc-openpi``."""

    prefix_lines = [line for line in stderr.splitlines() if _BENCHMARK_PREFIX in line]
    if len(prefix_lines) != 1:
        raise ValueError("trtmc-openpi must emit exactly one valid benchmark receipt line")
    match = _BENCHMARK_LINE.fullmatch(prefix_lines[0])
    if match is None:
        raise ValueError("trtmc-openpi benchmark receipt line is malformed")
    iterations = int(match.group("iterations"))
    warmups = int(match.group("warmups"))
    if (iterations, warmups) != (BENCHMARK_ITERATIONS, BENCHMARK_WARMUPS):
        raise ValueError("trtmc-openpi benchmark counts do not match the qualification contract")
    latency = {
        "mean_ms": _positive_finite(float(match.group("mean")), "native mean latency"),
        "p50_ms": _positive_finite(float(match.group("p50")), "native p50 latency"),
        "p95_ms": _positive_finite(float(match.group("p95")), "native p95 latency"),
    }
    if latency["p50_ms"] > latency["p95_ms"]:
        raise ValueError("trtmc-openpi benchmark p50 latency exceeds p95 latency")
    return {
        "backend": "tensorrt_cpp",
        "runtime": "trtmc-openpi",
        "runtime_contract": "native_cpp_tensorrt",
        "iterations": iterations,
        "warmups": warmups,
        "latency_ms": latency,
    }


def build_receipt(stderr: str, case: Any) -> dict[str, Any]:
    baseline = load_torch_eager_baseline()
    validate_case(case, baseline)
    return {
        "schema_version": 1,
        "artifact_type": "openpi_performance_receipt",
        "native": parse_native_benchmark(stderr),
        "torch_eager": baseline,
        "native_stderr": stderr,
    }


def validate_receipt(receipt: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate persisted evidence before applying performance gates."""

    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema_version",
        "artifact_type",
        "native",
        "torch_eager",
        "native_stderr",
    }:
        raise ValueError("OpenPI performance receipt fields do not match the contract")
    if (
        receipt["schema_version"] != 1
        or receipt["artifact_type"] != "openpi_performance_receipt"
        or not isinstance(receipt["native_stderr"], str)
    ):
        raise ValueError("OpenPI performance receipt identity is invalid")
    native = parse_native_benchmark(receipt["native_stderr"])
    baseline = load_torch_eager_baseline()
    if receipt["native"] != native:
        raise ValueError("OpenPI native benchmark summary differs from its stderr receipt")
    if receipt["torch_eager"] != baseline:
        raise ValueError("OpenPI Torch Eager summary differs from its pinned artifact")
    return native, baseline
