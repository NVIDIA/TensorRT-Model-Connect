#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score MiniMax-H3 candidate videos with a pinned SigLIP quality proxy."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SIGLIP_MODEL = "google/siglip-base-patch16-224"
SIGLIP_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
SIGLIP_LICENSE = "Apache-2.0"
SIGLIP_USE_FAST_PROCESSOR = False
SIGLIP_FILE_SHA256 = {
    "README.md": "86c231c4a7bf0ee2435295413ad5c7cf567c9426f00b79711ce8eda884b7a8d3",
    "config.json": "cd85b3d28829722820bcb89a2cfbb4160e55fd359249a3044da724166a8d9688",
    "model.safetensors": "2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8",
    "preprocessor_config.json": (
        "d11ccb80f15d358a11bdb070e92e2d889005874b7db15823d5f10d9b2533b14a"
    ),
    "special_tokens_map.json": ("2b6a1ff67a27e0df9ac0c7d93250fc0d87431c7b366b3d5669217104f9088a26"),
    "spiece.model": "1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec",
    "tokenizer.json": "c6e405cb7c670d56636a9402c81023a55bc6c3c53d89cf02b92f5c5005bfe920",
    "tokenizer_config.json": ("d6423dae508cc3a129d22ea443841c111832a1a73125b8f25ea8736951698bcb"),
}
EXPECTED_SHAPE = [124, 768, 1344, 3]
EXPECTED_RETAINED_FRAME_INDICES = [0, 18, 35, 53, 70, 88, 105, 123]
EXPECTED_FRAME_SIZE = (1344, 768)
METRIC_RANGES = {
    "siglip_alignment": (-1.0, 1.0),
    "temporal_consistency": (-1.0, 1.0),
    "motion_l1": (0.0, 1.0),
}
QUALITY_GATE_METRICS = {
    "min_siglip_alignment_mean": ("siglip_alignment", "minimum"),
    "min_temporal_consistency_mean": ("temporal_consistency", "minimum"),
    "min_motion_l1_mean": ("motion_l1", "minimum"),
    "max_motion_l1_mean": ("motion_l1", "maximum"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_snapshot(snapshot: Path) -> Path:
    """Verify every runtime and license file in the pinned SigLIP snapshot."""

    snapshot = snapshot.resolve(strict=True)
    if snapshot.name != SIGLIP_REVISION:
        raise ValueError(f"SigLIP snapshot must resolve to revision {SIGLIP_REVISION}")
    for name, expected_sha256 in SIGLIP_FILE_SHA256.items():
        path = snapshot / name
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise ValueError(f"pinned SigLIP file mismatch: {path}")
    return snapshot


def _stage_data(response: Mapping[str, Any]) -> Mapping[str, Any]:
    stage_output = response.get("stage_output")
    if not isinstance(stage_output, Mapping):
        raise ValueError("prediction has no serialized stage_output")
    data = stage_output.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("prediction stage_output has no data object")
    return data


def _candidate_frames(response: Mapping[str, Any]) -> list[Image.Image]:
    data = _stage_data(response)
    if int(data.get("returncode", 1)) != 0:
        raise ValueError(f"candidate returned {data.get('returncode')}")
    receipt = data.get("receipt")
    if not isinstance(receipt, Mapping) or receipt.get("status") != "passed":
        raise ValueError("candidate has no passed native receipt")
    if receipt.get("shape") != EXPECTED_SHAPE:
        raise ValueError(f"candidate shape is not {EXPECTED_SHAPE}")
    if receipt.get("retained_frame_indices") != EXPECTED_RETAINED_FRAME_INDICES:
        raise ValueError("candidate did not retain the required eight-frame subset")
    paths = data.get("frame_paths")
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
        raise ValueError("candidate frame_paths is not a sequence")
    if len(paths) != len(EXPECTED_RETAINED_FRAME_INDICES):
        raise ValueError("candidate does not contain exactly eight retained frames")

    frames = []
    for expected_index, value in zip(EXPECTED_RETAINED_FRAME_INDICES, paths, strict=True):
        path = Path(str(value))
        if path.name != f"frame_{expected_index:04d}.png":
            raise ValueError(f"candidate retained frame path is out of order: {path}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"candidate retained frame is missing or a symlink: {path}")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != EXPECTED_FRAME_SIZE:
                raise ValueError(
                    f"candidate retained frame has mode/size {image.mode}/{image.size}"
                )
            frames.append(image.copy())
    return frames


def _request_prompt(request: Mapping[str, Any]) -> str:
    prompt = request.get("prompt")
    inputs = request.get("inputs")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("VBench request has no prompt")
    if not isinstance(inputs, Mapping) or inputs.get("validation_mode") != "vbench_siglip":
        raise ValueError("VBench request does not select vbench_siglip validation")
    prompt_file = Path(str(inputs.get("prompt_file", "")))
    if prompt_file.is_symlink() or not prompt_file.is_file():
        raise ValueError(f"VBench prompt file is missing or a symlink: {prompt_file}")
    prompt_payload = json.loads(prompt_file.read_text(encoding="utf-8"))
    if not isinstance(prompt_payload, Mapping) or prompt_payload.get("prompt") != prompt:
        raise ValueError("VBench request prompt does not match its prompt file")
    return prompt


def _validate_metric_values(values: Mapping[str, Any]) -> dict[str, float]:
    if set(values) != set(METRIC_RANGES):
        raise ValueError(
            f"SigLIP scorer returned metrics {sorted(values)}; expected {sorted(METRIC_RANGES)}"
        )
    result = {}
    for name, (lower, upper) in METRIC_RANGES.items():
        value = float(values[name])
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"SigLIP scorer returned invalid {name} {value!r}")
        result[name] = value
    return result


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def _pooled_feature_tensor(output: Any) -> Any:
    """Accept Tensor or Transformers 5 model-output feature APIs."""

    features = getattr(output, "pooler_output", output)
    if not callable(getattr(features, "float", None)):
        raise TypeError("SigLIP feature output has no tensor pooler output")
    return features


def score_vbench_siglip_predictions(
    predictions: Mapping[str, Any],
    answers: Mapping[str, Any],
    *,
    scorer: Callable[[str, list[Image.Image]], Mapping[str, float]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate rows, report candidate metrics, and apply configured gates."""

    responses = predictions.get("responses")
    requests = answers.get("requests")
    if not isinstance(responses, list) or not isinstance(requests, list):
        raise ValueError("predictions and answers must contain lists")
    if len(responses) != len(requests):
        raise ValueError(f"prediction/request length mismatch: {len(responses)} != {len(requests)}")

    samples = []
    metric_values: dict[str, list[float]] = {name: [] for name in METRIC_RANGES}
    for index, (response, request) in enumerate(zip(responses, requests, strict=True)):
        if not isinstance(response, Mapping) or not isinstance(request, Mapping):
            raise ValueError(f"VBench/SigLIP row {index} must contain objects")
        expected_id = str(request.get("sample_id", ""))
        actual_id = str(response.get("sample_id", ""))
        if not expected_id or actual_id != expected_id:
            raise ValueError(
                f"VBench/SigLIP sample id mismatch at {index}: {expected_id!r} != {actual_id!r}"
            )
        sample = {
            "sample_id": expected_id,
            "selection_dimension": request.get("selection_dimension", ""),
            "source_index": request.get("source_index", index),
        }
        try:
            prompt = _request_prompt(request)
            frames = _candidate_frames(response)
            values = _validate_metric_values(scorer(prompt, frames))
            for name, value in values.items():
                metric_values[name].append(value)
            sample.update({"status": "passed", **values})
        except Exception as error:
            sample.update(
                {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        samples.append(sample)

    sample_count = len(samples)
    valid_count = len(metric_values["siglip_alignment"])
    structural_pass_rate = valid_count / sample_count if sample_count else 0.0
    metrics = {name: _metric_summary(values) for name, values in metric_values.items()}
    min_structural_pass_rate = float(gates.get("min_structural_pass_rate", 1.0))
    applied_gates: dict[str, float] = {
        "min_structural_pass_rate": min_structural_pass_rate,
    }
    gate_failures = []
    if structural_pass_rate < min_structural_pass_rate:
        gate_failures.append(
            {
                "gate": "min_structural_pass_rate",
                "actual": structural_pass_rate,
                "required": min_structural_pass_rate,
            }
        )
    for gate_name, (metric_name, direction) in QUALITY_GATE_METRICS.items():
        if gate_name not in gates:
            continue
        required = float(gates[gate_name])
        actual = metrics[metric_name]["mean"]
        applied_gates[gate_name] = required
        failed = actual < required if direction == "minimum" else actual > required
        if failed:
            gate_failures.append({"gate": gate_name, "actual": actual, "required": required})

    quality_gates = [name for name in QUALITY_GATE_METRICS if name in applied_gates]
    return {
        "status": "passed" if not gate_failures else "failed",
        "sample_count": sample_count,
        "valid_count": valid_count,
        "passed_count": valid_count,
        "structural_pass_rate": structural_pass_rate,
        "metrics": metrics,
        "primary_metric_name": "siglip_alignment",
        "calibration_status": ("quality_gated" if quality_gates else "pending_reference_baseline"),
        "quality_gate_status": "configured" if quality_gates else "report_only",
        "gates": applied_gates,
        "gate_failures": gate_failures,
        "samples": samples,
    }


def _load_pinned_scorer(
    *, device: str, local_files_only: bool
) -> tuple[Callable[[str, list[Image.Image]], Mapping[str, float]], dict[str, Any]]:
    from huggingface_hub import snapshot_download
    import torch
    from transformers import AutoModel, AutoProcessor

    snapshot = validate_model_snapshot(
        Path(
            snapshot_download(
                SIGLIP_MODEL,
                revision=SIGLIP_REVISION,
                local_files_only=local_files_only,
                allow_patterns=sorted(SIGLIP_FILE_SHA256),
            )
        )
    )
    processor = AutoProcessor.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=SIGLIP_USE_FAST_PROCESSOR,
    )
    model = AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    ).to(device)
    model.eval()

    def score(prompt: str, frames: list[Image.Image]) -> Mapping[str, float]:
        text_inputs = processor(
            text=[prompt],
            padding="max_length",
            return_tensors="pt",
        )
        image_inputs = processor(images=frames, return_tensors="pt")
        text_inputs = {name: value.to(device) for name, value in text_inputs.items()}
        image_inputs = {name: value.to(device) for name, value in image_inputs.items()}
        with torch.inference_mode():
            text_features = _pooled_feature_tensor(model.get_text_features(**text_inputs))
            image_features = _pooled_feature_tensor(model.get_image_features(**image_inputs))
            text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
            image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
            alignment = (image_features @ text_features.T).mean().item()
            temporal = (image_features[:-1] * image_features[1:]).sum(dim=-1).mean().item()

        motion_values = []
        previous = np.asarray(frames[0], dtype=np.float32)
        for frame in frames[1:]:
            current = np.asarray(frame, dtype=np.float32)
            motion_values.append(float(np.mean(np.abs(current - previous)) / 255.0))
            previous = current
        return {
            "siglip_alignment": float(alignment),
            "temporal_consistency": float(temporal),
            "motion_l1": sum(motion_values) / len(motion_values),
        }

    return score, {
        "prompt_repository": "https://github.com/Vchitect/VBench.git",
        "prompt_revision": "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490",
        "prompt_license": "Apache-2.0",
        "evaluator_model": SIGLIP_MODEL,
        "evaluator_revision": SIGLIP_REVISION,
        "evaluator_license": SIGLIP_LICENSE,
        "fast_image_processor": SIGLIP_USE_FAST_PROCESSOR,
        "frame_sampling": "8 evenly spaced frames: [0,18,35,53,70,88,105,123]",
        "metric_scope": (
            "TRTMC candidate-only semantic/temporal proxy; not an official VBench score"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--options-json", default="{}")
    parser.add_argument("--gates-json", default="{}")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _json_object(raw: str, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must decode to an object")
    return dict(value)


def main() -> int:
    args = _parse_args()
    options = _json_object(args.options_json, "--options-json")
    gates = _json_object(args.gates_json, "--gates-json")
    scorer, provenance = _load_pinned_scorer(
        device=str(options.get("device", "cuda:0")),
        local_files_only=args.local_files_only,
    )
    summary = score_vbench_siglip_predictions(
        json.loads(args.predictions.read_text(encoding="utf-8")),
        json.loads(args.answers.read_text(encoding="utf-8")),
        scorer=scorer,
        gates=gates,
    )
    summary["benchmark_provenance"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
