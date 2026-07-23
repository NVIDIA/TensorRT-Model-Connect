# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native model-owned OpenPI action runner."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any

from ... import qualification
from .. import openpi_proof_path, performance, resolve_model_asset
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

_ACTION_STAGES = frozenset({"actions", "act", "end_to_end"})
_OUTPUT_KEYS = frozenset({"actions", "horizon", "action_dim", "timings"})
_STAGE_TENSORS = {
    "preprocess": (
        "initial_noise",
        "token_ids",
        "token_mask",
        "preprocessed_images",
        "image_mask",
        "normalized_state",
    ),
    "vision": ("vision_tokens",),
    "prefix": ("prefix_kv_cache",),
    "flow": tuple(
        [f"velocity_{step:02d}" for step in range(10)]
        + [f"flow_state_{step:02d}" for step in range(11)]
    ),
}


def _resolve_bundle_path(case: E2ECase, ctx: RunContext) -> Path:
    bundle = Path(case.bundle)
    return bundle if bundle.is_absolute() else Path(ctx.engine_dir) / bundle


def _resolve_request_path(case: E2ECase) -> Path:
    value = str(case.inputs.get("request_json") or os.environ.get("TRTMC_OPENPI_REQUEST_JSON", ""))
    if value:
        return resolve_model_asset(value, str(case.metadata.get("model_test_dir", "")))

    snapshot_request = openpi_proof_path("request", "request.json")
    if not snapshot_request.is_file():
        raise FileNotFoundError(f"Pinned OpenPI request is missing: {snapshot_request}")
    return snapshot_request


def _diagnostic_cache_key(
    case: E2ECase, ctx: RunContext, profile_name: str
) -> tuple[str, str, str, str]:
    return (
        str(_resolve_bundle_path(case, ctx)),
        str(_resolve_request_path(case)),
        str(ctx.binary_path),
        profile_name,
    )


def build_openpi_act_command(
    case: E2ECase,
    ctx: RunContext,
    *,
    output_json: str | Path,
    qualification_diagnostics: str | Path | None = None,
    benchmark: bool = False,
) -> list[str]:
    """Build the pure-native action command used by the E2E harness."""

    if not ctx.binary_path:
        raise ValueError("OpenPI E2E requires a trtmc binary path")
    shared_binary = Path(ctx.binary_path)
    binary = (
        shared_binary
        if shared_binary.name == "trtmc-openpi"
        else shared_binary.with_name("trtmc-openpi")
    )
    command = [
        str(binary),
        str(_resolve_bundle_path(case, ctx)),
        "--request-json",
        str(_resolve_request_path(case)),
        "--output-json",
        str(output_json),
    ]
    if qualification_diagnostics is not None:
        command.extend(["--qualification-diagnostics", str(qualification_diagnostics)])
    if benchmark:
        command.extend(
            [
                "--benchmark",
                str(performance.BENCHMARK_ITERATIONS),
                "--warmup",
                str(performance.BENCHMARK_WARMUPS),
            ]
        )
    return command


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r}")


def _load_action_output(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read OpenPI action output {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("OpenPI action output must be a JSON object")
    unexpected = sorted(set(payload) - _OUTPUT_KEYS)
    if unexpected:
        raise ValueError(f"OpenPI action output contains unexpected fields: {unexpected}")
    missing = sorted({"actions", "horizon", "action_dim", "timings"} - set(payload))
    if missing:
        raise ValueError(f"OpenPI action output is missing fields: {missing}")
    horizon = payload["horizon"]
    action_dim = payload["action_dim"]
    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or horizon <= 0
        or not isinstance(action_dim, int)
        or isinstance(action_dim, bool)
        or action_dim <= 0
    ):
        raise ValueError("OpenPI action output horizon and action_dim must be positive integers")
    actions = payload["actions"]
    if not isinstance(actions, list):
        raise ValueError("OpenPI action output actions must be an array")
    if actions and isinstance(actions[0], list):
        if len(actions) != horizon or any(
            not isinstance(row, list) or len(row) != action_dim for row in actions
        ):
            raise ValueError("OpenPI nested action output does not match horizon x action_dim")
        values = [value for row in actions for value in row]
    else:
        values = actions
        if len(values) != horizon * action_dim:
            raise ValueError("OpenPI flat action output does not match horizon x action_dim")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("OpenPI action output contains a non-finite or non-numeric value")
    if not isinstance(payload["timings"], dict):
        raise ValueError("OpenPI action output timings must be an object")
    return payload


def _load_native_diagnostic_manifest(
    manifest_path: Path, profile_name: str
) -> dict[str, dict[str, Any]]:
    """Validate a native capture against the same tensor contract as the oracle."""

    manifest = qualification.strict_json_load(manifest_path)
    required_top = {
        "schema_version",
        "artifact_type",
        "runtime_contract",
        "model_id",
        "tensors",
    }
    if set(manifest) != required_top:
        raise ValueError("OpenPI native diagnostic manifest fields do not match the contract")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_type"] != "trtmc_action_qualification_diagnostics"
        or manifest["runtime_contract"] != "native_cpp_tensorrt"
    ):
        raise ValueError("OpenPI native diagnostic manifest identity is invalid")
    tensors = manifest["tensors"]
    if not isinstance(tensors, dict):
        raise ValueError("OpenPI native diagnostic tensors must be an object")
    expected = qualification._expected_tensor_contract(qualification.load_contract(profile_name))
    if set(tensors) != set(expected):
        raise ValueError(
            "OpenPI native diagnostic tensor set mismatch: "
            f"missing={sorted(set(expected) - set(tensors))}, "
            f"unexpected={sorted(set(tensors) - set(expected))}"
        )

    root = manifest_path.parent.resolve(strict=True)
    descriptors: dict[str, dict[str, Any]] = {}
    for name, contract in expected.items():
        descriptor = tensors[name]
        required_descriptor = {
            "path",
            "stage",
            "role",
            "dtype",
            "shape",
            "byte_length",
            "sha256",
        }
        if not isinstance(descriptor, dict) or set(descriptor) != required_descriptor:
            raise ValueError(f"OpenPI native diagnostic descriptor {name!r} is malformed")
        for field in ("stage", "role", "dtype", "shape"):
            if descriptor[field] != contract[field]:
                raise ValueError(
                    f"OpenPI native diagnostic {name!r}.{field} differs from the contract"
                )
        relative = PurePosixPath(str(descriptor["path"]))
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("tensors",):
            raise ValueError(f"OpenPI native diagnostic {name!r} has an unsafe path")
        payload = root.joinpath(*relative.parts).resolve(strict=True)
        if root not in payload.parents or not payload.is_file():
            raise ValueError(f"OpenPI native diagnostic {name!r} escapes its capture directory")
        expected_bytes = qualification._tensor_byte_length(
            str(contract["dtype"]), contract["shape"]
        )
        if descriptor["byte_length"] != expected_bytes or payload.stat().st_size != expected_bytes:
            raise ValueError(f"OpenPI native diagnostic {name!r} has an invalid byte count")
        if qualification.sha256_file(payload) != descriptor["sha256"]:
            raise ValueError(f"OpenPI native diagnostic {name!r} SHA-256 mismatch")
        descriptors[name] = {
            "path": str(payload),
            "stage": descriptor["stage"],
            "role": descriptor["role"],
            "dtype": descriptor["dtype"],
            "shape": descriptor["shape"],
            "byte_length": descriptor["byte_length"],
            "sha256": descriptor["sha256"],
        }
    return descriptors


class RobotActionGenerationRunner:
    """Execute OpenPI through the C++/TensorRT-only ``act`` runtime path."""

    @property
    def strategy_name(self) -> str:
        return "robot_action_generation"

    def __init__(self) -> None:
        self._diagnostic_cache: dict[
            tuple[str, str, str, str],
            tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]],
        ] = {}
        self._diagnostic_lock = threading.Lock()
        self._benchmark_cache: dict[tuple[int, str, str, str, str, str], dict[str, Any]] = {}
        self._benchmark_contexts: dict[int, RunContext] = {}
        self._benchmark_lock = threading.Lock()

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name not in _ACTION_STAGES:
            if stage.name not in _STAGE_TENSORS:
                return StageOutput(
                    stage_name=stage.name,
                    data={"error": f"Unsupported OpenPI stage {stage.name!r}", "returncode": -1},
                    metadata={"status": "unsupported_stage"},
                )
            return self._run_diagnostic_stage(case, stage, ctx)

        if ctx.artifacts_dir:
            output_path = Path(ctx.artifacts_dir) / case.name / "openpi_actions.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return self._run_actions(case, stage, ctx, output_path)

        with tempfile.TemporaryDirectory(prefix="trtmc_openpi_e2e_") as temporary_dir:
            return self._run_actions(case, stage, ctx, Path(temporary_dir) / "openpi_actions.json")

    def _run_diagnostic_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        profile_name = str(case.inputs.get("profile", ""))
        if not profile_name:
            raise ValueError("OpenPI diagnostic stage requires inputs.profile")
        cache_key = _diagnostic_cache_key(case, ctx, profile_name)
        with self._diagnostic_lock:
            cached = self._diagnostic_cache.get(cache_key)
            if cached is None:
                parent_root = Path(ctx.artifacts_dir) / case.name if ctx.artifacts_dir else None
                if parent_root is not None:
                    parent_root.mkdir(parents=True, exist_ok=True)
                capture_parent = Path(
                    tempfile.mkdtemp(prefix="openpi_qualification_", dir=parent_root)
                )
                diagnostics_dir = capture_parent / "capture"
                output_path = capture_parent / "openpi_actions.json"
                command = build_openpi_act_command(
                    case,
                    ctx,
                    output_json=output_path,
                    qualification_diagnostics=diagnostics_dir,
                )
                environment = os.environ.copy()
                if ctx.ld_library_path:
                    environment["LD_LIBRARY_PATH"] = ctx.ld_library_path
                timeout_s = int(case.metadata.get("runtime_timeout_s", 1800))
                started = time.monotonic()
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=environment,
                    timeout=timeout_s,
                )
                elapsed = time.monotonic() - started
                if result.returncode != 0:
                    return StageOutput(
                        stage_name=stage.name,
                        data={
                            "returncode": result.returncode,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                        },
                        timing_s=elapsed,
                        metadata={"command": command, "returncode": result.returncode},
                    )
                action_payload = _load_action_output(output_path)
                descriptors = _load_native_diagnostic_manifest(
                    diagnostics_dir / "manifest.json", profile_name
                )
                metadata = {
                    "command": command,
                    "returncode": result.returncode,
                    "diagnostic_manifest": str(diagnostics_dir / "manifest.json"),
                    "runtime_contract": "native_cpp_tensorrt",
                    "timing_s": elapsed,
                }
                cached = (action_payload, descriptors, metadata)
                self._diagnostic_cache[cache_key] = cached

        _action_payload, descriptors, metadata = cached
        selected = {name: descriptors[name] for name in _STAGE_TENSORS[stage.name]}
        return StageOutput(
            stage_name=stage.name,
            data={
                "tensor_files": selected,
                "returncode": 0,
                "profile_name": profile_name,
            },
            timing_s=float(metadata["timing_s"]),
            metadata=dict(metadata),
        )

    def _run_actions(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
        output_path: Path,
    ) -> StageOutput:
        context_identity = id(ctx)
        benchmark_key = (
            context_identity,
            case.name,
            str(_resolve_bundle_path(case, ctx)),
            str(_resolve_request_path(case)),
            str(ctx.binary_path),
            str(case.inputs.get("profile", "")),
        )
        with self._benchmark_lock:
            # Keep each context alive so Python cannot recycle its identity
            # while this runner retains the corresponding benchmark receipt.
            self._benchmark_contexts[context_identity] = ctx
            receipt = self._benchmark_cache.get(benchmark_key)
            if receipt is None:
                command = build_openpi_act_command(
                    case,
                    ctx,
                    output_json=output_path,
                    benchmark=True,
                )
                output = self._invoke_actions(case, stage, ctx, output_path, command)
                if int(output.data.get("returncode", 0)) != 0:
                    return output
                receipt = performance.build_receipt(str(output.metadata["runtime_stderr"]), case)
                self._benchmark_cache[benchmark_key] = receipt
                output.metadata["performance"] = receipt
                self._attach_normalized_actions(output, case, ctx)
                return output

        command = build_openpi_act_command(case, ctx, output_json=output_path)
        output = self._invoke_actions(case, stage, ctx, output_path, command)
        if int(output.data.get("returncode", 0)) == 0:
            output.metadata["performance"] = receipt
            self._attach_normalized_actions(output, case, ctx)
        return output

    def _attach_normalized_actions(
        self, output: StageOutput, case: E2ECase, ctx: RunContext
    ) -> None:
        profile_name = str(case.inputs.get("profile", ""))
        if not profile_name:
            return
        with self._diagnostic_lock:
            cached = self._diagnostic_cache.get(_diagnostic_cache_key(case, ctx, profile_name))
        if cached is not None:
            output.data["tensor_files"] = {"normalized_actions": cached[1]["normalized_actions"]}

    @staticmethod
    def _invoke_actions(
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
        output_path: Path,
        command: list[str],
    ) -> StageOutput:
        environment = os.environ.copy()
        if ctx.ld_library_path:
            environment["LD_LIBRARY_PATH"] = ctx.ld_library_path
        timeout_s = int(case.metadata.get("runtime_timeout_s", 1800))
        started = time.monotonic()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=timeout_s,
        )
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            return StageOutput(
                stage_name=stage.name,
                data={
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                timing_s=elapsed,
                metadata={"command": command, "returncode": result.returncode},
            )
        payload = _load_action_output(output_path)
        runtime_timings = payload.pop("timings")
        payload["output_field"] = payload["actions"]
        payload["returncode"] = result.returncode
        metadata = {
            "command": command,
            "returncode": result.returncode,
            "output_json": str(output_path),
            "runtime_contract": "native_cpp_tensorrt",
            "runtime_timings": runtime_timings,
            "runtime_stdout": result.stdout,
            "runtime_stderr": result.stderr,
        }
        return StageOutput(
            stage_name=stage.name,
            data=payload,
            timing_s=elapsed,
            metadata=metadata,
        )


plugin = RobotActionGenerationRunner()
