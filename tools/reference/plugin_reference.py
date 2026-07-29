#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run model-owned reference plugins without validation orchestration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.e2e_harness.contracts import RunContext, StageSpec  # noqa: E402
from tests.e2e_harness.manifest_loader import load_manifest  # noqa: E402
from tests.e2e_harness.registry import (  # noqa: E402
    activate_model_plugins,
    get_reference,
)


SCHEMA_VERSION = "trtmc.native-reference-reproduction/v1"
_VISION_DATASET_KINDS = {
    "image_classification_json",
    "prompted_segmentation_json",
    "semantic_segmentation_json",
}
_DIFFUSION_SAMPLE_INPUT_FIELDS = frozenset(
    {
        "action",
        "camera_intrinsics",
        "camera_intrinsics_file",
        "image",
        "image_path",
        "prompt_file",
        "rotation_speed_deg",
        "translation_speed",
    }
)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def _selected_rows(
    rows: Sequence[Mapping[str, Any]],
    sample_id: str,
) -> list[tuple[int, dict[str, Any]]]:
    selected = [
        (index, dict(row))
        for index, row in enumerate(rows)
        if not sample_id or str(row.get("sample_id", "")) == sample_id
    ]
    if sample_id and not selected:
        raise ValueError(f"sample_id {sample_id!r} is not present in the prepared prompts")
    return selected


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entrypoint_sha256() -> str:
    return _sha256_file(Path(__file__))


def _write_reproduction_metadata(arguments: argparse.Namespace) -> None:
    if arguments.repro_metadata is None:
        return
    arguments.repro_metadata.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "backend": "model_reference_plugin",
                "entrypoint": str(Path(__file__).resolve()),
                "entrypoint_sha256": _entrypoint_sha256(),
                "command_source": "hf_native_commands.jsonl",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _command_tokens(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)) and value and all(
        isinstance(token, (str, int, float, Path)) for token in value
    ):
        return [str(token) for token in value]
    if isinstance(value, str):
        tokens = shlex.split(value)
        if len(tokens) > 1:
            return tokens
    return []


def _native_command_from_metadata(metadata: Any) -> list[str]:
    if not isinstance(metadata, Mapping):
        return []
    command = _command_tokens(metadata.get("command"))
    if command:
        return command
    for value in metadata.values():
        nested = _native_command_from_metadata(value)
        if nested:
            return nested
    return []


def _record_native_command(path: Path, sample_id: str, output: Any) -> None:
    metadata = getattr(output, "metadata", {})
    command = _native_command_from_metadata(metadata)
    if not command:
        return
    with path.open("a", encoding="utf-8") as command_file:
        command_file.write(
            json.dumps(
                {"sample_id": sample_id, "command": command},
                ensure_ascii=False,
            )
            + "\n"
        )


def _run_reference_stage(
    reference: Any,
    case: Any,
    stage: Any,
    context: RunContext,
) -> Any:
    """Run a model reference while preserving its direct subprocess command."""
    run_stage = reference.run_stage
    function = getattr(run_stage, "__func__", run_stage)
    namespace = getattr(function, "__globals__", {})
    helper = namespace.get("run_reference_subprocess")
    subprocess_module = namespace.get("subprocess")
    direct_run = getattr(subprocess_module, "run", None)
    if not callable(helper) and not callable(direct_run):
        return run_stage(case, stage, context)
    captured_commands: list[list[str]] = []

    def run_and_record(*args: Any, **kwargs: Any) -> Any:
        output = helper(*args, **kwargs)
        command = _command_tokens(kwargs.get("command"))
        metadata = getattr(output, "metadata", {})
        if command and not _native_command_from_metadata(metadata):
            output.metadata = {
                **(metadata if isinstance(metadata, dict) else {}),
                "command": command,
            }
        return output

    def capture_direct_run(*args: Any, **kwargs: Any) -> Any:
        command_value = args[0] if args else kwargs.get("args")
        command = _command_tokens(command_value)
        if command:
            captured_commands.append(command)
        return direct_run(*args, **kwargs)

    if callable(helper):
        namespace["run_reference_subprocess"] = run_and_record
    if callable(direct_run):
        subprocess_module.run = capture_direct_run
    try:
        output = run_stage(case, stage, context)
    finally:
        if callable(helper):
            namespace["run_reference_subprocess"] = helper
        if callable(direct_run):
            subprocess_module.run = direct_run
    metadata = getattr(output, "metadata", {})
    if captured_commands and not _native_command_from_metadata(metadata):
        output.metadata = {
            **(metadata if isinstance(metadata, dict) else {}),
            "command": captured_commands[-1],
        }
    return output


def _model_manifest_path(manifest: Mapping[str, Any]) -> Path:
    task_config = manifest.get("task_eval", {})
    task_config = task_config if isinstance(task_config, dict) else {}
    manifest_ref = str(task_config.get("model_manifest", "") or "")
    if not manifest_ref:
        raise ValueError("reference plugin requires task_eval.model_manifest")
    manifest_path = Path(manifest_ref)
    return manifest_path if manifest_path.is_absolute() else REPO_ROOT / manifest_path


def _load_reference_plugin(manifest: Mapping[str, Any]) -> tuple[Any, Any]:
    case = load_manifest(_model_manifest_path(manifest))
    task_config = manifest.get("task_eval", {})
    if isinstance(task_config, Mapping):
        reference_precision = str(
            task_config.get("reference_precision", "") or ""
        )
        if reference_precision:
            case.metadata["reference_precision"] = reference_precision
    activate_model_plugins(str(case.metadata.get("model_test_dir", "") or ""))
    reference = get_reference(case.reference_backend)
    if reference is None:
        raise RuntimeError(
            f"No reference plugin {case.reference_backend!r} for {case.family}"
        )
    return case, reference


def _stage(case: Any, name: str) -> Any:
    for stage in case.stages:
        if stage.name == name:
            return stage
    return StageSpec(name=name, required=True)


def _context(case: Any, artifacts_dir: Path) -> RunContext:
    return RunContext(
        case=case,
        artifacts_dir=str(artifacts_dir),
        hf_python=sys.executable,
        reference_python=sys.executable,
    )


def _time_series_case(
    template: Any,
    prompt_row: Mapping[str, Any],
    index: int,
) -> Any:
    case = copy.deepcopy(template)
    case.name = str(prompt_row.get("sample_id", f"time_series_{index:06d}"))
    inputs = prompt_row.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError(f"Time-series sample {case.name!r} has no numeric inputs")
    case.inputs = copy.deepcopy(inputs)
    return case


def _time_series_values(data: Mapping[str, Any], sample_id: str) -> list[float]:
    error = str(data.get("error", "") or "")
    values = data.get("output_field", data.get("field"))
    if error:
        raise RuntimeError(f"HF time-series inference failed for {sample_id}: {error}")
    if not isinstance(values, list) or not values:
        raise RuntimeError(
            f"HF time-series inference produced no output tensor for {sample_id}"
        )
    return [float(value) for value in values]


def _time_series_shape(
    data: Mapping[str, Any],
    output_values: Sequence[float],
) -> list[int]:
    output_shape = data.get("output_shape")
    if not isinstance(output_shape, list):
        output_shape = [int(data.get("output_dim", len(output_values)))]
    return [int(dim) for dim in output_shape]


def _time_series_response(case: Any, output: Any) -> dict[str, Any]:
    data = output.data if isinstance(output.data, dict) else {}
    metadata = output.metadata if isinstance(output.metadata, dict) else {}
    output_values = _time_series_values(data, case.name)
    return {
        "sample_id": case.name,
        "source": "hf",
        "output_values": output_values,
        "output_shape": _time_series_shape(data, output_values),
        "output_name": str(data.get("reference_output_name", "") or ""),
        "returncode": int(metadata.get("returncode", 0) or 0),
        "wall_ms": float(output.timing_s) * 1000.0,
    }


def _run_time_series(
    template: Any,
    reference: Any,
    rows: Sequence[tuple[int, dict[str, Any]]],
    artifacts_dir: Path,
    command_path: Path,
) -> list[dict[str, Any]]:
    responses = []
    for run_index, (source_index, prompt_row) in enumerate(rows):
        case = _time_series_case(template, prompt_row, source_index)
        output = _run_reference_stage(
            reference,
            case,
            _stage(case, "full_inference"),
            _context(case, artifacts_dir),
        )
        _record_native_command(command_path, case.name, output)
        responses.append(_time_series_response(case, output))
        print(
            f"[reference.plugin.time_series] sample={run_index + 1}/{len(rows)}",
            file=sys.stderr,
        )
    return responses


def _vision_case(
    template: Any,
    prompt_row: Mapping[str, Any],
    task_config: Mapping[str, Any],
    index: int,
) -> Any:
    case = copy.deepcopy(template)
    case.name = str(prompt_row.get("sample_id", f"vision_{index:06d}"))
    case.inputs["image"] = str(prompt_row["image"])
    prompt_mode = str(task_config.get("prompt_mode", "") or "")
    if prompt_mode == "point":
        case.inputs["point_x"] = float(prompt_row["point_x"])
        case.inputs["point_y"] = float(prompt_row["point_y"])
        case.inputs["is_foreground"] = True
    elif prompt_mode == "text":
        text_prompt = str(prompt_row["text_prompt"])
        case.inputs["prompt"] = text_prompt
        case.inputs["text_prompt"] = text_prompt
    return case


def _persist_numpy_output(value: Any, path: Path) -> str:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value))
    return str(path)


def _classification_fields(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "top_class": int(data["top_class"]),
        "top_score": float(data.get("top_score", 0.0)),
        "num_classes": int(data.get("num_classes", 0)),
    }


def _semantic_class_map_path(
    data: Mapping[str, Any],
    artifact_dir: Path,
    sample_id: str,
) -> str:
    class_map_path = str(
        data.get("class_map_path")
        or data.get("segmentation_map_path")
        or data.get("output_path")
        or ""
    )
    if not class_map_path and data.get("class_map") is not None:
        class_map_path = _persist_numpy_output(
            data["class_map"],
            artifact_dir / sample_id / "hf_class_map.npy",
        )
    if not class_map_path or not Path(class_map_path).is_file():
        raise RuntimeError("HF semantic segmentation produced no class map")
    return class_map_path


def _semantic_raw_class_map(data: Mapping[str, Any]) -> str:
    raw_class_map_path = str(data.get("raw_class_map_path") or "")
    if raw_class_map_path and not Path(raw_class_map_path).is_file():
        raise RuntimeError(
            "HF semantic segmentation raw class map does not exist: "
            f"{raw_class_map_path}"
        )
    return raw_class_map_path


def _semantic_segmentation_fields(
    data: Mapping[str, Any],
    artifact_dir: Path,
    sample_id: str,
) -> dict[str, Any]:
    fields = {
        "class_map_path": _semantic_class_map_path(
            data,
            artifact_dir,
            sample_id,
        ),
        "visualization_path": str(
            data.get("viz_path") or data.get("output_path") or ""
        ),
    }
    raw_class_map_path = _semantic_raw_class_map(data)
    if raw_class_map_path:
        fields["raw_class_map_path"] = raw_class_map_path
    return fields


def _prompted_masks_path(
    data: Mapping[str, Any],
    artifact_dir: Path,
    sample_id: str,
) -> str:
    masks_path = str(data.get("masks_path", "") or "")
    if not masks_path and data.get("masks") is not None:
        masks_path = _persist_numpy_output(
            data["masks"],
            artifact_dir / sample_id / "hf_masks.npy",
        )
    if not masks_path or not Path(masks_path).is_file():
        raise RuntimeError("HF prompted segmentation produced no masks")
    return masks_path


def _prompted_segmentation_fields(
    data: Mapping[str, Any],
    artifact_dir: Path,
    prompt_row: Mapping[str, Any],
    sample_id: str,
) -> dict[str, Any]:
    scores = (
        data.get("mask_scores")
        or data.get("iou_scores")
        or data.get("scores")
        or []
    )
    return {
        "masks_path": _prompted_masks_path(data, artifact_dir, sample_id),
        "mask_scores": [float(value) for value in scores],
        "num_masks": int(data.get("num_masks", 0)),
        "point_x": prompt_row.get("point_x"),
        "point_y": prompt_row.get("point_y"),
        "text_prompt": str(prompt_row.get("text_prompt", "")),
        "segmented_image_path": str(data.get("segmented_image_path", "") or ""),
    }


def _vision_response(
    case: Any,
    output: Any,
    dataset_kind: str,
    prompt_row: Mapping[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    data = output.data if isinstance(output.data, dict) else {}
    metadata = output.metadata if isinstance(output.metadata, dict) else {}
    response = {
        "sample_id": case.name,
        "source": "hf",
        "returncode": int(data.get("returncode", metadata.get("returncode", 0))),
        "image": str(prompt_row["image"]),
        "wall_ms": float(output.timing_s) * 1000.0,
    }
    if dataset_kind == "image_classification_json":
        response.update(_classification_fields(data))
    elif dataset_kind == "semantic_segmentation_json":
        response.update(
            _semantic_segmentation_fields(data, artifact_dir, case.name)
        )
    elif dataset_kind == "prompted_segmentation_json":
        response.update(
            _prompted_segmentation_fields(
                data,
                artifact_dir,
                prompt_row,
                case.name,
            )
        )
    else:
        raise ValueError(f"Unsupported vision dataset kind {dataset_kind!r}")
    return response


def _run_vision(
    template: Any,
    reference: Any,
    rows: Sequence[tuple[int, dict[str, Any]]],
    artifacts_dir: Path,
    manifest: Mapping[str, Any],
    command_path: Path,
) -> list[dict[str, Any]]:
    task_config = manifest.get("task_eval", {})
    task_config = task_config if isinstance(task_config, dict) else {}
    dataset_kind = str(manifest.get("dataset_kind", "") or "")
    responses = []
    for run_index, (source_index, prompt_row) in enumerate(rows):
        case = _vision_case(template, prompt_row, task_config, source_index)
        output = _run_reference_stage(
            reference,
            case,
            _stage(case, "full_inference"),
            _context(case, artifacts_dir),
        )
        _record_native_command(command_path, case.name, output)
        responses.append(
            _vision_response(
                case,
                output,
                dataset_kind,
                prompt_row,
                artifacts_dir,
            )
        )
        print(
            f"[reference.plugin.vision] sample={run_index + 1}/{len(rows)}",
            file=sys.stderr,
        )
    return responses


def _reranking_case(
    template: Any,
    prompt_row: Mapping[str, Any],
    index: int,
) -> Any:
    case = copy.deepcopy(template)
    case.name = str(prompt_row.get("sample_id", f"reranking_{index:06d}"))
    case.inputs["prompt"] = str(prompt_row["query"])
    case.inputs["documents"] = [
        str(document) for document in prompt_row["documents"]
    ]
    return case


def _reranking_response(
    case: Any,
    output: Any,
    prompt_row: Mapping[str, Any],
) -> dict[str, Any]:
    data = output.data if isinstance(output.data, dict) else {}
    scores = data.get("scores")
    documents = prompt_row["documents"]
    if not isinstance(scores, list) or len(scores) != len(documents):
        raise RuntimeError(
            f"HF reranking produced "
            f"{len(scores) if isinstance(scores, list) else 0} "
            f"scores for {len(documents)} documents"
        )
    return {
        "sample_id": case.name,
        "source": "hf",
        "query": str(prompt_row["query"]),
        "documents": [str(document) for document in documents],
        "scores": [float(score) for score in scores],
        "wall_ms": float(output.timing_s) * 1000.0,
    }


def _run_reranking(
    template: Any,
    reference: Any,
    rows: Sequence[tuple[int, dict[str, Any]]],
    artifacts_dir: Path,
    command_path: Path,
) -> list[dict[str, Any]]:
    responses = []
    for run_index, (source_index, prompt_row) in enumerate(rows):
        case = _reranking_case(template, prompt_row, source_index)
        output = _run_reference_stage(
            reference,
            case,
            _stage(case, "full_inference"),
            _context(case, artifacts_dir),
        )
        _record_native_command(command_path, case.name, output)
        responses.append(_reranking_response(case, output, prompt_row))
        print(
            f"[reference.plugin.reranking] sample={run_index + 1}/{len(rows)}",
            file=sys.stderr,
        )
    return responses


def _diffusion_case(
    template: Any,
    prompt_row: Mapping[str, Any],
    generation: Mapping[str, Any],
    index: int,
) -> Any:
    case = copy.deepcopy(template)
    case.name = str(prompt_row.get("sample_id", f"diffusion_{index:06d}"))
    case.inputs.update(generation)
    case.inputs["prompt"] = str(prompt_row["prompt"])
    for field in _DIFFUSION_SAMPLE_INPUT_FIELDS:
        if field in prompt_row:
            case.inputs[field] = copy.deepcopy(prompt_row[field])
    seed = int(generation.get("seed", case.determinism.get("seed", 42)))
    case.inputs["seed"] = seed + index
    return case


def _diffusion_response(case: Any, output: Any, prompt: str) -> dict[str, Any]:
    data = output.data if isinstance(output.data, dict) else {}
    image_value = str(
        case.inputs.get("image") or case.inputs.get("image_path") or ""
    )
    response = {
        "sample_id": case.name,
        "source": "hf",
        "returncode": int(data.get("returncode", 1)),
        "num_frames": int(data.get("num_frames", 0)),
        "frames_dir": str(data.get("frames_dir", "")),
        "frame_stats": data.get("frame_stats", {}),
        "prompt": prompt,
        "initial_latents_sha256": str(data.get("initial_latents_sha256", "")),
        "wall_ms": float(output.timing_s) * 1000.0,
        "seed": int(case.inputs.get("seed", case.determinism.get("seed", 42))),
        "action": str(case.inputs.get("action", "")),
        "condition_image": image_value,
    }
    image_path = Path(image_value) if image_value else None
    if image_path is not None and image_path.is_file():
        response["condition_image_sha256"] = _sha256_file(image_path)
    if response["returncode"] != 0 or response["num_frames"] < 1:
        raise RuntimeError(
            f"HF diffusion reference failed for {case.name}: "
            f"returncode={response['returncode']} frames={response['num_frames']}"
        )
    return response


def _run_diffusion(
    template: Any,
    reference: Any,
    rows: Sequence[tuple[int, dict[str, Any]]],
    artifacts_dir: Path,
    manifest: Mapping[str, Any],
    command_path: Path,
) -> list[dict[str, Any]]:
    generation = manifest.get("generation", {})
    generation = generation if isinstance(generation, dict) else {}
    responses = []
    for run_index, (source_index, prompt_row) in enumerate(rows):
        case = _diffusion_case(template, prompt_row, generation, source_index)
        output = _run_reference_stage(
            reference,
            case,
            _stage(case, "end_to_end"),
            _context(case, artifacts_dir),
        )
        _record_native_command(command_path, case.name, output)
        responses.append(
            _diffusion_response(case, output, str(prompt_row["prompt"]))
        )
        print(
            f"[reference.plugin.diffusion] sample={run_index + 1}/{len(rows)}",
            file=sys.stderr,
        )
    return responses


def _run_dataset_kind(
    *,
    manifest: Mapping[str, Any],
    template: Any,
    reference: Any,
    rows: Sequence[tuple[int, dict[str, Any]]],
    artifacts_dir: Path,
    command_path: Path,
) -> list[dict[str, Any]]:
    dataset_kind = str(manifest.get("dataset_kind", "") or "")
    if dataset_kind == "time_series_csv":
        return _run_time_series(
            template,
            reference,
            rows,
            artifacts_dir,
            command_path,
        )
    if dataset_kind in _VISION_DATASET_KINDS:
        return _run_vision(
            template,
            reference,
            rows,
            artifacts_dir,
            manifest,
            command_path,
        )
    if dataset_kind == "reranking_json":
        return _run_reranking(
            template,
            reference,
            rows,
            artifacts_dir,
            command_path,
        )
    if dataset_kind == "diffusion_prompt_json":
        return _run_diffusion(
            template,
            reference,
            rows,
            artifacts_dir,
            manifest,
            command_path,
        )
    raise ValueError(f"Unsupported reference plugin dataset kind {dataset_kind!r}")


def run(arguments: argparse.Namespace) -> None:
    manifest = _load_json(arguments.manifest)
    rows = _selected_rows(
        _load_jsonl(arguments.prompts),
        arguments.sample_id,
    )
    template, reference = _load_reference_plugin(manifest)
    artifacts_dir = arguments.predictions.parent / "hf_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    command_path = arguments.predictions.parent / "hf_native_commands.jsonl"
    command_path.write_text("", encoding="utf-8")
    responses = _run_dataset_kind(
        manifest=manifest,
        template=template,
        reference=reference,
        rows=rows,
        artifacts_dir=artifacts_dir,
        command_path=command_path,
    )
    arguments.raw_output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.raw_output.open("w", encoding="utf-8") as raw_file:
        for response in responses:
            raw_file.write(json.dumps(response, ensure_ascii=False) + "\n")
    arguments.predictions.write_text(
        json.dumps({"responses": responses}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_reproduction_metadata(arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a model-owned reference plugin directly."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-family", default="")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--repro-metadata", type=Path)
    parser.add_argument("--sample-id", default="")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--attn-impl", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
