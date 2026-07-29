# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for model-owned reference-consistency workloads.

The validation driver and the independent reference process both use this
module to select the same manifest testcase, apply the same dataset inputs,
and persist ``StageOutput`` values without coupling either side to the E2E
orchestrator.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping

from tests.e2e_harness.contracts import StageOutput, StageSpec
from tests.e2e_harness.manifest_loader import load_model_manifest


_ARRAY_MARKER = "__trtmc_validation_array__"
_BYTES_MARKER = "__trtmc_validation_bytes__"
_COMMON_ARTIFACT_FIELDS = (
    "audio",
    "audio_output_path",
    "condition_image",
    "frames_dir",
    "image",
    "image_path",
    "output_video",
    "video",
    "video_path",
    "wav_path",
)


def safe_sample_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "sample"


def select_case(
    manifest_path: Path,
    request: Mapping[str, Any],
    *,
    source_index: int,
) -> tuple[Any, StageSpec]:
    """Select and specialize one model-owned testcase for a dataset row."""
    model = load_model_manifest(manifest_path)
    requested_name = str(request.get("testcase", "") or "")
    if requested_name:
        matches = [case for case in model.testcases if case.name == requested_name]
        if not matches:
            available = ", ".join(case.name for case in model.testcases)
            raise ValueError(
                f"{manifest_path}: testcase {requested_name!r} is not present; "
                f"available: {available}"
            )
        template = matches[0]
    else:
        template = model.build_case

    case = copy.deepcopy(template)
    inputs = request.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise ValueError(f"model-plugin request {source_index} inputs must be an object")
    case.inputs.update(copy.deepcopy(dict(inputs)))
    sample_id = str(request.get("sample_id", f"model_plugin_{source_index:06d}"))
    case.metadata["validation_sample_id"] = sample_id
    case.metadata["validation_manifest_case_name"] = template.name

    stage_name = str(request.get("stage", "") or "")
    if not stage_name:
        if len(case.stages) != 1:
            available = ", ".join(stage.name for stage in case.stages)
            raise ValueError(
                f"model-plugin request {sample_id!r} must select one stage; available: {available}"
            )
        return case, case.stages[0]
    for stage in case.stages:
        if stage.name == stage_name:
            return case, stage
    available = ", ".join(stage.name for stage in case.stages)
    raise ValueError(
        f"model-plugin request {sample_id!r} stage {stage_name!r} is not "
        f"present; available: {available}"
    )


def _array_value(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - runtime dependency
        np = None
    if np is not None and isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        tensor = detach()
        cpu = getattr(tensor, "cpu", None)
        numpy = getattr(cpu() if callable(cpu) else tensor, "numpy", None)
        if callable(numpy):
            return numpy()
    return None


def _serialize_value(value: Any, *, artifact_dir: Path, stem: str) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - runtime dependency
        np = None
    if np is not None and isinstance(value, np.generic):
        return value.item()
    array = _array_value(value)
    if array is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{safe_sample_name(stem)}.npy"
        np.save(path, array, allow_pickle=False)
        return {_ARRAY_MARKER: str(path.resolve())}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{safe_sample_name(stem)}.bin"
        path.write_bytes(value)
        return {_BYTES_MARKER: str(path.resolve())}
    if isinstance(value, Mapping):
        return {
            str(key): _serialize_value(
                item,
                artifact_dir=artifact_dir,
                stem=f"{stem}-{key}",
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _serialize_value(
                item,
                artifact_dir=artifact_dir,
                stem=f"{stem}-{index}",
            )
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def serialize_stage_output(
    output: StageOutput,
    *,
    artifact_dir: Path,
    sample_id: str,
) -> dict[str, Any]:
    prefix = safe_sample_name(sample_id)
    return {
        "stage_name": output.stage_name,
        "data": _serialize_value(
            output.data,
            artifact_dir=artifact_dir,
            stem=f"{prefix}-data",
        ),
        "text": output.text,
        "logits": _serialize_value(
            output.logits,
            artifact_dir=artifact_dir,
            stem=f"{prefix}-logits",
        ),
        "timing_s": float(output.timing_s),
        "metadata": _serialize_value(
            output.metadata,
            artifact_dir=artifact_dir,
            stem=f"{prefix}-metadata",
        ),
    }


def _deserialize_value(value: Any) -> Any:
    if isinstance(value, Mapping) and set(value) == {_ARRAY_MARKER}:
        import numpy as np

        return np.load(str(value[_ARRAY_MARKER]), allow_pickle=False)
    if isinstance(value, Mapping) and set(value) == {_BYTES_MARKER}:
        return Path(str(value[_BYTES_MARKER])).read_bytes()
    if isinstance(value, Mapping):
        return {str(key): _deserialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    return value


def deserialize_stage_output(payload: Mapping[str, Any]) -> StageOutput:
    return StageOutput(
        stage_name=str(payload.get("stage_name", "")),
        data=_deserialize_value(payload.get("data", {})),
        text=(str(payload["text"]) if payload.get("text") is not None else None),
        logits=_deserialize_value(payload.get("logits")),
        timing_s=float(payload.get("timing_s", 0.0) or 0.0),
        metadata=_deserialize_value(payload.get("metadata", {})),
    )


def response_from_output(
    *,
    sample_id: str,
    source: str,
    testcase: str,
    output: StageOutput,
    serialized_output: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a durable prediction row and expose common report artifacts."""
    response: dict[str, Any] = {
        "sample_id": sample_id,
        "source": source,
        "testcase": testcase,
        "stage": output.stage_name,
        "output_text": output.text or "",
        "stage_output": dict(serialized_output),
        "wall_ms": float(output.timing_s) * 1000.0,
    }
    sources = (
        output.data if isinstance(output.data, Mapping) else {},
        output.metadata if isinstance(output.metadata, Mapping) else {},
    )
    for field in _COMMON_ARTIFACT_FIELDS:
        value = next((mapping.get(field) for mapping in sources if mapping.get(field)), None)
        if isinstance(value, (str, Path)):
            response[field] = str(value)
    token_ids = next(
        (
            mapping.get(field)
            for mapping in sources
            for field in ("generated_token_ids", "token_ids")
            if isinstance(mapping.get(field), list)
        ),
        None,
    )
    if token_ids is not None:
        response["generated_token_ids"] = [int(value) for value in token_ids]
    return response


def manifest_path_from_work_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> Path:
    task_config = manifest.get("task_eval", {})
    task_config = task_config if isinstance(task_config, Mapping) else {}
    manifest_ref = str(task_config.get("model_manifest", "") or "")
    if not manifest_ref:
        raise ValueError("model-plugin validation requires task_eval.model_manifest")
    path = Path(manifest_ref)
    return path if path.is_absolute() else repo_root / path
