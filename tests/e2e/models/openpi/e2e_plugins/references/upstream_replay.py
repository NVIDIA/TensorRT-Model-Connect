# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned, content-addressed OpenPI upstream replay backend."""

from __future__ import annotations

import functools
import json
import os
import struct
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from tests.e2e.models.openpi import qualification

from .. import openpi_proof_path, openpi_snapshot_path, resolve_model_asset
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

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
    "actions": ("normalized_actions", "physical_actions"),
    "act": ("normalized_actions", "physical_actions"),
    "end_to_end": ("normalized_actions", "physical_actions"),
}


def _replay_path(case: E2ECase) -> Path:
    value = str(
        case.inputs.get("upstream_replay_artifact")
        or case.metadata.get("upstream_replay_artifact")
        or os.environ.get("TRTMC_OPENPI_REFERENCE_ARTIFACT", "")
    )
    if value:
        return resolve_model_asset(value, str(case.metadata.get("model_test_dir", "")))

    return openpi_proof_path("reference", "reference-set.json")


@functools.lru_cache(maxsize=8)
def _load_validated_document(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    document = qualification.strict_json_load(path)
    artifact_type = document.get("artifact_type")
    if artifact_type == "openpi_pinned_reference_set":
        return qualification.validate_reference_set(path, minimum_cases=512)
    raise ValueError(
        "OpenPI E2E requires a validated 512-case openpi_pinned_reference_set; "
        f"got {artifact_type!r}"
    )


@functools.lru_cache(maxsize=32)
def _load_validated_case(path_text: str) -> dict[str, Any]:
    return qualification.validate_reference_artifact(Path(path_text))


def _resolve_case_artifact(case: E2ECase) -> tuple[Path, dict[str, Any]]:
    source = _replay_path(case)
    if not source.is_file():
        raise FileNotFoundError(f"OpenPI reference set is missing: {source}")
    document = _load_validated_document(str(source))
    requested_id = str(
        case.inputs.get("reference_case_id")
        or case.metadata.get("reference_case_id")
        or os.environ.get("TRTMC_OPENPI_REFERENCE_CASE_ID", "")
    )
    entries = document["cases"]
    if not requested_id and len(entries) == 1:
        requested_id = str(entries[0]["id"])
    if not requested_id and any(entry["id"] == case.name for entry in entries):
        requested_id = case.name
    if not requested_id:
        raise ValueError(
            "A multi-case OpenPI reference set requires inputs.reference_case_id or "
            "TRTMC_OPENPI_REFERENCE_CASE_ID"
        )
    matches = [entry for entry in entries if entry["id"] == requested_id]
    if len(matches) != 1:
        raise ValueError(f"OpenPI reference case {requested_id!r} is not present exactly once")
    relative = PurePosixPath(str(matches[0]["path"]))
    artifact_path = source.parent.joinpath(*relative.parts)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"OpenPI reference case is missing: {artifact_path}")
    return artifact_path, _load_validated_case(str(artifact_path))


def _iter_tensor_values(path: Path, dtype: str) -> Iterable[float | int | bool]:
    payload = path.read_bytes()
    if dtype == "bool":
        return (value != 0 for value in payload)
    if dtype == "uint8":
        return iter(payload)
    if dtype == "int32":
        return (item[0] for item in struct.iter_unpack("<i", payload))
    if dtype == "int64":
        return (item[0] for item in struct.iter_unpack("<q", payload))
    if dtype == "float16":
        return (float(item[0]) for item in struct.iter_unpack("<e", payload))
    if dtype == "float32":
        return (float(item[0]) for item in struct.iter_unpack("<f", payload))
    if dtype == "bfloat16":
        return (
            float(struct.unpack("<f", struct.pack("<I", item[0] << 16))[0])
            for item in struct.iter_unpack("<H", payload)
        )
    raise ValueError(f"Unsupported OpenPI replay tensor dtype {dtype!r}")


def _tensor_file_descriptor(artifact_path: Path, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    relative = PurePosixPath(str(descriptor["path"]))
    return {
        "path": str(artifact_path.parent.joinpath(*relative.parts)),
        "dtype": str(descriptor["dtype"]),
        "shape": list(descriptor["shape"]),
        "sha256": str(descriptor["sha256"]),
    }


def _reshape_actions(values: list[float], shape: list[int]) -> list[list[float]]:
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError(f"OpenPI action tensor must have shape [1,H,D], got {shape}")
    horizon, action_dim = shape[1], shape[2]
    if len(values) != horizon * action_dim:
        raise ValueError("OpenPI action tensor payload length does not match its shape")
    return [values[row * action_dim : (row + 1) * action_dim] for row in range(horizon)]


def _snapshot_normalization_path(case: E2ECase) -> Path:
    profile = str(case.inputs.get("profile") or case.metadata.get("profile") or "")
    config_path = openpi_snapshot_path("openpi_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if profile and config.get("profile") != profile:
        raise ValueError(
            f"OpenPI snapshot profile {config.get('profile')!r} does not match {profile!r}"
        )
    normalization = openpi_snapshot_path("preprocessor_config.json")
    if not normalization.is_file():
        raise FileNotFoundError(f"OpenPI snapshot normalization asset is missing: {normalization}")
    return normalization


def _load_action_spans(case: E2ECase, action_dim: int) -> list[float] | None:
    direct = case.inputs.get("action_spans") or case.metadata.get("action_spans")
    if direct is not None:
        spans = [float(value) for value in direct]
    else:
        stats_value = str(
            case.inputs.get("normalization_stats")
            or case.metadata.get("normalization_stats")
            or os.environ.get("TRTMC_OPENPI_NORM_STATS", "")
        )
        if stats_value:
            stats_path = resolve_model_asset(
                stats_value, str(case.metadata.get("model_test_dir", ""))
            )
        else:
            stats_path = _snapshot_normalization_path(case)
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        root = payload.get("norm_stats", payload)
        actions = root.get("actions") if isinstance(root, dict) else None
        if not isinstance(actions, dict):
            raise ValueError("OpenPI normalization stats do not contain actions")
        q01, q99 = actions.get("q01"), actions.get("q99")
        if not isinstance(q01, list) or not isinstance(q99, list):
            raise ValueError("OpenPI action normalization must contain q01/q99 arrays")
        spans = [float(high) - float(low) for low, high in zip(q01, q99, strict=True)]
    if len(spans) < action_dim or any(span <= 0.0 for span in spans[:action_dim]):
        raise ValueError("OpenPI action spans must be positive and cover every output dimension")
    return spans[:action_dim]


class UpstreamReplayReference:
    """Read tensors captured from the audited OpenPI commit and checkpoint."""

    @property
    def backend_name(self) -> str:
        return "upstream_replay"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        del ctx
        artifact_path, artifact = _resolve_case_artifact(case)
        expected_profile = str(case.inputs.get("profile", ""))
        if expected_profile and artifact["profile_name"] != expected_profile:
            raise ValueError(
                f"OpenPI reference profile {artifact['profile_name']!r} does not match "
                f"case profile {expected_profile!r}"
            )
        tensor_names = _STAGE_TENSORS.get(stage.name)
        if stage.name == "flow":
            tensor_names = tuple(
                [f"velocity_{step:02d}" for step in range(10)]
                + [f"flow_state_{step:02d}" for step in range(11)]
            )
        if tensor_names is None:
            raise ValueError(f"Unsupported OpenPI replay stage {stage.name!r}")

        descriptors = {
            name: _tensor_file_descriptor(artifact_path, artifact["tensors"][name])
            for name in tensor_names
        }
        data: dict[str, Any] = {
            "tensor_files": descriptors,
            "profile_name": artifact["profile_name"],
            "reference_case_id": artifact["case"]["id"],
        }
        if stage.name in {"actions", "act", "end_to_end"}:
            for key in ("normalized_actions", "physical_actions"):
                descriptor = descriptors[key]
                values = [
                    float(value)
                    for value in _iter_tensor_values(
                        Path(descriptor["path"]), str(descriptor["dtype"])
                    )
                ]
                data[key] = _reshape_actions(values, list(descriptor["shape"]))
            data["actions"] = data["physical_actions"]
            data["output_field"] = data["physical_actions"]
            data["horizon"] = len(data["physical_actions"])
            data["action_dim"] = len(data["physical_actions"][0])
            spans = _load_action_spans(case, int(data["action_dim"]))
            if spans is not None:
                data["action_spans"] = spans

        return StageOutput(
            stage_name=stage.name,
            data=data,
            metadata={
                "source": "upstream_replay",
                "artifact_path": str(artifact_path),
                "upstream_commit": qualification.UPSTREAM_COMMIT,
                "checkpoint_sha256": artifact["upstream"]["checkpoint"]["sha256"],
            },
        )


plugin = UpstreamReplayReference()
