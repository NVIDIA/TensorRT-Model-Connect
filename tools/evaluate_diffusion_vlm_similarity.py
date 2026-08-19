#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare TRT and HF diffusion outputs with a paired vision-language judge.

This is an optional artifact-review tool. It feeds both images to the same
VLM prompt so the model can compare them directly, instead of captioning each
image independently.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from tools.count_diffusion_frame_pairs import discover_diffusion_frame_pairs
except ModuleNotFoundError:
    from count_diffusion_frame_pairs import discover_diffusion_frame_pairs


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_SCORED_NUMERIC_KEYS = (
    "semantic_similarity_0_to_5",
    "trt_prompt_alignment_0_to_5",
    "hf_prompt_alignment_0_to_5",
    "trt_visual_quality_0_to_5",
    "hf_visual_quality_0_to_5",
    "hf_visual_quality_5_to_5",
)
_SCORED_STRING_KEYS = (
    "trt_description",
    "hf_description",
    "trt_relative_to_hf",
    "reason",
)
_GATE_RULE = (
    "fail if semantic_similarity_0_to_5 < 3.0, "
    "trt_prompt_alignment_0_to_5 < 3.0, trt_visual_quality_0_to_5 < 2.5, "
    "hf_prompt_alignment_0_to_5 < 3.0, hf_visual_quality_0_to_5 < 3.0, "
    "is_regression is true, or the HF reference is invalid for a photo prompt; "
    "VLM gate failures always fail CI"
)

_PHOTO_PROMPT_TERMS = ("photo", "photograph", "photorealistic", "realistic")
_NON_PHOTO_DESCRIPTION_TERMS = (
    "cartoon", "drawing", "illustration", "painting", "sketch", "stylized",
    "vector",
)
_INVALID_REFERENCE_SCORE_CAP = 2.0


class _LoadingWeightsProgressFilter:
    """Drop tqdm weight-loading progress while preserving other output."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def write(self, text: str) -> int:
        if "Loading weights:" not in text:
            self._wrapped.write(text)
        return len(text)

    def flush(self) -> None:
        self._wrapped.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._wrapped, "isatty", lambda: False)())


@contextlib.contextmanager
def _suppress_loading_weights_progress():
    filtered_stdout = _LoadingWeightsProgressFilter(sys.stdout)
    filtered_stderr = _LoadingWeightsProgressFilter(sys.stderr)
    with contextlib.redirect_stdout(filtered_stdout), contextlib.redirect_stderr(filtered_stderr):
        yield


def _default_assessment_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    configs = sorted(
        (repo_root / "python" / "tensorrt_model_connect" / "models").glob(
            "*/tests/diffusion_vlm_assessment.json"
        )
    )
    defaults = []
    for path in configs:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("default") is True:
            defaults.append(path)
    if len(defaults) != 1:
        listed = ", ".join(str(path) for path in defaults) or "none"
        raise SystemExit(
            "Expected exactly one default diffusion VLM assessment config under "
            f"python/tensorrt_model_connect/models/*/diffusion_vlm_assessment.json; found {listed}"
        )
    return defaults[0]


def _load_assessment_config(path: Path | None) -> dict[str, Any]:
    config_path = path or _default_assessment_config_path()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{config_path} must contain a JSON object")
    return data


def _load_image(path: Path, max_side: int) -> Any:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def _discover_pairs(artifacts_dir: Path) -> list[dict[str, Any]]:
    return discover_diffusion_frame_pairs(artifacts_dir)


def _expand_pair_samples(pair: dict[str, Any]) -> list[dict[str, Any]]:
    trt_paths = pair.get("trt_images")
    hf_paths = pair.get("hf_images")
    frame_indices = pair.get("frame_indices")
    if not isinstance(trt_paths, list) or not trt_paths:
        trt_paths = [pair.get("trt_image")]
    if not isinstance(hf_paths, list) or not hf_paths:
        hf_paths = [pair.get("hf_image")]
    if not isinstance(frame_indices, list) or not frame_indices:
        frame_indices = list(range(len(trt_paths)))
    if (
        len(trt_paths) != len(hf_paths)
        or len(trt_paths) != len(frame_indices)
        or any(not isinstance(path, str) or not path for path in [*trt_paths, *hf_paths])
    ):
        raise ValueError("diffusion VLM frame sample lists are incomplete or mismatched")

    shared = {
        key: value
        for key, value in pair.items()
        if key not in {"trt_image", "hf_image", "trt_images", "hf_images", "frame_indices"}
    }
    return [
        {
            **shared,
            "trt_image": trt_path,
            "hf_image": hf_path,
            "frame_index": frame_index,
        }
        for trt_path, hf_path, frame_index in zip(trt_paths, hf_paths, frame_indices)
    ]


def _parse_json(text: str) -> dict[str, Any]:
    match = _JSON_RE.search(text)
    if not match:
        return _parse_scored_fields(text)
    try:
        parsed = json.loads(match.group(0))
        if (
            "hf_visual_quality_0_to_5" not in parsed
            and "hf_visual_quality_5_to_5" in parsed
        ):
            parsed["hf_visual_quality_0_to_5"] = parsed["hf_visual_quality_5_to_5"]
        parsed["raw"] = text
        return parsed
    except json.JSONDecodeError:
        return _parse_scored_fields(text)


def _parse_scored_fields(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"raw": text}
    for key in _SCORED_NUMERIC_KEYS:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if match:
            parsed[key] = float(match.group(1))
    for key in _SCORED_STRING_KEYS:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', text)
        if match:
            parsed[key] = match.group(1)
    match = re.search(r'"is_regression"\s*:\s*(true|false)', text, re.IGNORECASE)
    if match:
        parsed["is_regression"] = match.group(1).lower() == "true"
    if (
        "hf_visual_quality_0_to_5" not in parsed
        and "hf_visual_quality_5_to_5" in parsed
    ):
        parsed["hf_visual_quality_0_to_5"] = parsed["hf_visual_quality_5_to_5"]
    return parsed


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _mentions_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _photo_reference_invalid(judgment: dict[str, Any], prompt: str) -> bool:
    hf_description = str(judgment.get("hf_description") or "")
    return bool(
        prompt
        and _mentions_any(prompt, _PHOTO_PROMPT_TERMS)
        and _mentions_any(hf_description, _NON_PHOTO_DESCRIPTION_TERMS)
    )


def _normalize_judgment_consistency(
    judgment: dict[str, Any], prompt: str = "",
) -> dict[str, Any]:
    """Make structured scores consistent with the VLM's own descriptions."""
    if not _photo_reference_invalid(judgment, prompt):
        return judgment

    normalized = dict(judgment)
    normalized.setdefault(
        "judgment_consistency_note",
        "HF reference description is non-photo/stylized for a photo prompt.",
    )
    for key in ("hf_prompt_alignment_0_to_5", "hf_visual_quality_0_to_5"):
        value = _as_float(normalized.get(key))
        if value is None or value > _INVALID_REFERENCE_SCORE_CAP:
            if key in normalized:
                normalized.setdefault(f"{key}_original", normalized[key])
            normalized[key] = _INVALID_REFERENCE_SCORE_CAP

    trt_alignment = _as_float(normalized.get("trt_prompt_alignment_0_to_5"))
    trt_quality = _as_float(normalized.get("trt_visual_quality_0_to_5"))
    trt_description = str(normalized.get("trt_description") or "")
    trt_is_acceptable_photo = (
        trt_alignment is not None
        and trt_alignment >= 3.0
        and trt_quality is not None
        and trt_quality >= 2.5
        and not _mentions_any(trt_description, _NON_PHOTO_DESCRIPTION_TERMS)
    )
    if trt_is_acceptable_photo:
        relative = str(normalized.get("trt_relative_to_hf") or "").strip().lower()
        if relative != "better":
            if "trt_relative_to_hf" in normalized:
                normalized.setdefault(
                    "trt_relative_to_hf_original",
                    normalized["trt_relative_to_hf"],
                )
            normalized["trt_relative_to_hf"] = "better"

    return normalized


def _apply_gate(judgment: dict[str, Any], prompt: str = "") -> dict[str, Any]:
    reasons = []
    similarity = _as_float(judgment.get("semantic_similarity_0_to_5"))
    alignment = _as_float(judgment.get("trt_prompt_alignment_0_to_5"))
    quality = _as_float(judgment.get("trt_visual_quality_0_to_5"))
    hf_alignment = _as_float(judgment.get("hf_prompt_alignment_0_to_5"))
    hf_quality = _as_float(judgment.get("hf_visual_quality_0_to_5"))
    is_regression = _as_bool(judgment.get("is_regression"))

    if similarity is None:
        reasons.append("missing semantic_similarity_0_to_5")
    elif similarity < 3.0:
        reasons.append(f"semantic_similarity_0_to_5={similarity:.2f} < 3.0")

    if alignment is None:
        reasons.append("missing trt_prompt_alignment_0_to_5")
    elif alignment < 3.0:
        reasons.append(f"trt_prompt_alignment_0_to_5={alignment:.2f} < 3.0")

    if quality is None:
        reasons.append("missing trt_visual_quality_0_to_5")
    elif quality < 2.5:
        reasons.append(f"trt_visual_quality_0_to_5={quality:.2f} < 2.5")

    if hf_alignment is None:
        reasons.append("missing hf_prompt_alignment_0_to_5")
    elif hf_alignment < 3.0:
        reasons.append(f"hf_prompt_alignment_0_to_5={hf_alignment:.2f} < 3.0")

    if hf_quality is None:
        reasons.append("missing hf_visual_quality_0_to_5")
    elif hf_quality < 3.0:
        reasons.append(f"hf_visual_quality_0_to_5={hf_quality:.2f} < 3.0")

    if is_regression is None:
        reasons.append("missing is_regression")
    elif bool(is_regression):
        reasons.append("is_regression is true")

    if _photo_reference_invalid(judgment, prompt):
        reasons.append(
            "HF reference description suggests non-photo/stylized output for a photo prompt")

    return {
        "failed": bool(reasons),
        "rule": _GATE_RULE,
        "reasons": reasons,
    }


def _judge_pair(
    model: Any,
    processor: Any,
    device: str,
    pair: dict[str, Any],
    *,
    max_side: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    trt_image = _load_image(Path(pair["trt_image"]), max_side)
    hf_image = _load_image(Path(pair["hf_image"]), max_side)
    prompt = pair.get("prompt") or ""
    frame_index = pair.get("frame_index")
    frame_note = f"\nThis is sampled video frame {frame_index}." if frame_index is not None else ""

    question = f"""You are comparing two diffusion pipeline outputs for the same prompt.
Image 1 is TensorRT/TRT output. Image 2 is the configured reference output.
Prompt: {prompt}{frame_note}

Compare Image 1 to Image 2. Focus on semantic content, scene layout, prompt alignment,
major missing objects, visible artifacts, reference validity, and whether Image 1 is a
user-visible regression.
Do not require pixel-level identity.
Set "is_regression" to true only for gate-level failures such as garbage output,
severe artifacts, missing or incorrect primary subjects, poor prompt alignment, or Image 1
being materially worse than a valid Image 2 reference.
Do not mark lower detail than the HF reference alone as a regression.
Score Image 2 independently as a reference: wrong primary subject count, missing primary
subjects, obvious artifacts, or non-photo/stylized output (such as a cartoon, drawing,
illustration, painting, sketch, or vector art) for a photo prompt must reduce its
prompt-alignment and visual-quality scores. A photographic backlit subject may be
described as a silhouette; do not treat the word "silhouette" alone as evidence that
an image is non-photographic.
Set "trt_relative_to_hf" from prompt alignment plus visual quality, not just shared
objects. If Image 2 is non-photo/stylized for a photo prompt and Image 1 is a normal
photo-like rendering of the prompt, set "trt_relative_to_hf" to "better".

Return only JSON with these keys:
{{
  "trt_description": string,
  "hf_description": string,
  "semantic_similarity_0_to_5": number,
  "trt_prompt_alignment_0_to_5": number,
  "hf_prompt_alignment_0_to_5": number,
  "trt_visual_quality_0_to_5": number,
  "hf_visual_quality_0_to_5": number,
  "trt_relative_to_hf": "better" | "similar" | "worse",
  "is_regression": boolean,
  "reason": string
}}
Keep "reason" under 30 words."""

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": trt_image},
            {"type": "image", "image": hf_image},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[trt_image, hf_image], padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = generated[:, inputs.input_ids.shape[1]:]
    answer = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    judged = _normalize_judgment_consistency(_parse_json(answer.strip()), prompt=prompt)
    judged["vlm_gate"] = _apply_gate(judged, prompt=prompt)
    return {**pair, "vlm_judgment": judged}


def _aggregate_sample_results(
    pair: dict[str, Any],
    sample_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not sample_results:
        raise ValueError("diffusion VLM produced no sampled-frame judgments")

    def score(result: dict[str, Any], key: str) -> float:
        value = _as_float(result["vlm_judgment"].get(key))
        return value if value is not None else float("-inf")

    failed_samples = [
        result
        for result in sample_results
        if result["vlm_judgment"].get("vlm_gate", {}).get("failed") is not False
    ]
    worst = min(
        failed_samples or sample_results,
        key=lambda result: (
            score(result, "semantic_similarity_0_to_5"),
            score(result, "trt_prompt_alignment_0_to_5"),
            score(result, "trt_visual_quality_0_to_5"),
        ),
    )
    aggregate = dict(worst["vlm_judgment"])
    for key in (
        "semantic_similarity_0_to_5",
        "trt_prompt_alignment_0_to_5",
        "hf_prompt_alignment_0_to_5",
        "trt_visual_quality_0_to_5",
        "hf_visual_quality_0_to_5",
    ):
        values = [
            value
            for result in sample_results
            if (value := _as_float(result["vlm_judgment"].get(key))) is not None
        ]
        if values:
            aggregate[key] = min(values)
    aggregate["is_regression"] = any(
        _as_bool(result["vlm_judgment"].get("is_regression")) is True for result in sample_results
    )
    reasons = []
    for result in failed_samples:
        frame_index = result.get("frame_index")
        for reason in result["vlm_judgment"].get("vlm_gate", {}).get("reasons", []):
            reasons.append(f"frame {frame_index}: {reason}")
    aggregate["vlm_gate"] = {
        "failed": bool(failed_samples),
        "rule": _GATE_RULE,
        "reasons": reasons,
    }
    aggregate["reason"] = (
        f"{len(sample_results)} sampled frames assessed; "
        f"worst frame {worst.get('frame_index')}: "
        f"{worst['vlm_judgment'].get('reason', '')}"
    )
    return {
        "case_name": pair.get("case_name"),
        "prompt": pair.get("prompt", ""),
        "sampled_frame_indices": [result.get("frame_index") for result in sample_results],
        "sample_assessments": sample_results,
        "vlm_judgment": aggregate,
    }


def _judge_pair_samples(
    model: Any,
    processor: Any,
    device: str,
    pair: dict[str, Any],
    *,
    max_side: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    trt_paths = pair.get("trt_images")
    if not isinstance(trt_paths, list) or len(trt_paths) <= 1:
        return _judge_pair(
            model, processor, device, pair,
            max_side=max_side, max_new_tokens=max_new_tokens)
    samples = _expand_pair_samples(pair)
    sample_results = [
        _judge_pair(
            model,
            processor,
            device,
            sample,
            max_side=max_side,
            max_new_tokens=max_new_tokens,
        )
        for sample in samples
    ]
    return _aggregate_sample_results(pair, sample_results)


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path,
                        help="Model-owned diffusion VLM assessment config.")
    parser.add_argument("--model-id")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-side", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--case", action="append", default=[],
                        help="Optional case name filter. May be passed more than once.")
    parser.add_argument("--waives", type=Path,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.waives:
        print("Ignoring deprecated --waives; diffusion VLM gate failures fail CI.")

    assessment_config = _load_assessment_config(args.config)
    model_id = args.model_id or assessment_config.get("model_id")
    if not model_id:
        raise SystemExit(
            "Diffusion VLM assessment requires --model-id or a config model_id"
        )
    max_side = int(args.max_side or assessment_config.get("max_side", 512))
    max_new_tokens = int(
        args.max_new_tokens or assessment_config.get("max_new_tokens", 384)
    )

    pairs = _discover_pairs(args.artifacts_dir)
    if args.case:
        wanted = set(args.case)
        pairs = [pair for pair in pairs if pair["case_name"] in wanted]
    if not pairs:
        raise SystemExit(f"No TRT/HF diffusion frame pairs found in {args.artifacts_dir}")

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    with _suppress_loading_weights_progress():
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            local_files_only=args.local_files_only,
            dtype=dtype,
            trust_remote_code=True,
        ).to(args.device)
        processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=args.local_files_only,
            trust_remote_code=True,
            min_pixels=224 * 224,
            max_pixels=max_side * max_side,
        )

    results = []
    any_gate_failed = False
    for pair in pairs:
        result = _judge_pair_samples(
            model,
            processor,
            args.device,
            pair,
            max_side=max_side,
            max_new_tokens=max_new_tokens,
        )
        results.append(result)
        judgment = result["vlm_judgment"]
        gate = judgment.get("vlm_gate", {})
        gate_failed = bool(gate.get("failed"))
        any_gate_failed = any_gate_failed or gate_failed
        print(
            f"{result['case_name']}: similarity="
            f"{judgment.get('semantic_similarity_0_to_5')} "
            f"trt_quality={judgment.get('trt_visual_quality_0_to_5')} "
            f"hf_quality={judgment.get('hf_visual_quality_0_to_5')} "
            f"regression={judgment.get('is_regression')} "
            f"gate_failed={gate.get('failed')}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model_id": model_id,
        "artifacts_dir": str(args.artifacts_dir),
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 1 if any_gate_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
