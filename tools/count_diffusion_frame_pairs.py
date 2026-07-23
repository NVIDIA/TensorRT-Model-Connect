#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Count diffusion TRT/reference frame pairs in E2E artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _frames_in(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    frames = sorted(path.glob("frame_*.png"))
    if frames:
        return frames
    return sorted(path.glob("*.png"))


def _select_frame(frames: list[Path]) -> Path | None:
    if not frames:
        return None
    return frames[(len(frames) - 1) // 2]


def _sample_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if sample_count > frame_count:
        raise ValueError(f"cannot select {sample_count} VLM samples from {frame_count} frames")
    if sample_count <= 1:
        return [(frame_count - 1) // 2]
    return [round(index * (frame_count - 1) / (sample_count - 1)) for index in range(sample_count)]


def _vlm_frame_contract(result: dict[str, Any]) -> tuple[int, int | None]:
    case = result.get("case_config")
    if not isinstance(case, dict):
        return 1, None
    metadata = case.get("metadata")
    if not isinstance(metadata, dict):
        return 1, None
    policy = metadata.get("native_acceptance")
    if policy is None:
        return 1, None
    if not isinstance(policy, dict):
        raise ValueError("native_acceptance must be an object")
    requested = policy.get("vlm_frame_samples")
    if not isinstance(requested, int) or isinstance(requested, bool) or not 1 <= requested <= 6:
        raise ValueError("native_acceptance.vlm_frame_samples must be an integer from 1 to 6")
    inputs = case.get("inputs")
    expected_frames = inputs.get("video_num_frames") if isinstance(inputs, dict) else None
    if (
        not isinstance(expected_frames, int)
        or isinstance(expected_frames, bool)
        or expected_frames < requested
    ):
        raise ValueError("native_acceptance requires a valid case_config.inputs.video_num_frames")
    return requested, expected_frames


def _load_result(path: Path) -> dict[str, Any] | None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return result if isinstance(result, dict) else None


def discover_diffusion_frame_pairs(artifacts_dir: Path) -> list[dict[str, Any]]:
    """Return paired TRT/reference diffusion frames discovered under artifacts."""
    pairs: list[dict[str, Any]] = []
    for result_path in sorted(artifacts_dir.glob("*/result.json")):
        result = _load_result(result_path)
        if result is None:
            continue

        case = result.get("case_config", {})
        if not isinstance(case, dict):
            continue
        if case.get("task_strategy") != "diffusion_media_generation":
            continue

        model_dir = result_path.parent
        trt_frames = _frames_in(model_dir / "frames")
        hf_frames = _frames_in(model_dir / "hf_frames")
        if not hf_frames:
            hf_frames = _frames_in(model_dir / "ref_frames")
        if not trt_frames or len(trt_frames) != len(hf_frames):
            continue
        sample_count, expected_frame_count = _vlm_frame_contract(result)
        if expected_frame_count is not None:
            expected_names = [f"frame_{index:04d}.png" for index in range(expected_frame_count)]
            trt_names = [path.name for path in trt_frames]
            hf_names = [path.name for path in hf_frames]
            if (
                len(trt_frames) != expected_frame_count
                or trt_names != expected_names
                or hf_names != expected_names
            ):
                raise ValueError(
                    "native_acceptance requires complete contiguous TRT/reference "
                    f"frame sequences: expected={expected_frame_count}, "
                    f"TRT={len(trt_frames)}, reference={len(hf_frames)}"
                )
        sample_indices = _sample_indices(
            len(trt_frames),
            sample_count,
        )
        trt_samples = [trt_frames[index] for index in sample_indices]
        hf_samples = [hf_frames[index] for index in sample_indices]
        trt_frame = _select_frame(trt_frames)
        hf_frame = _select_frame(hf_frames)
        if trt_frame is None or hf_frame is None:
            continue

        inputs = case.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}
        pair = {
            "case_name": result.get("case_name") or case.get("name") or model_dir.name,
            "prompt": inputs.get("prompt", ""),
            "trt_image": str(trt_frame),
            "hf_image": str(hf_frame),
        }
        if sample_count > 1:
            pair.update({
                "trt_images": [str(path) for path in trt_samples],
                "hf_images": [str(path) for path in hf_samples],
                "frame_indices": sample_indices,
            })
        pairs.append(pair)
    return pairs


def count_diffusion_frame_pairs(artifacts_dir: Path) -> int:
    """Return the number of discoverable diffusion frame pairs."""
    return len(discover_diffusion_frame_pairs(artifacts_dir))


def validate_complete_diffusion_frame_pairs(
    artifacts_dir: Path,
) -> list[dict[str, Any]]:
    """Require one unique frame pair for every diffusion E2E result."""
    expected: list[str] = []
    for result_path in sorted(artifacts_dir.glob("*/result.json")):
        result = _load_result(result_path)
        if result is None:
            raise ValueError(f"unreadable E2E result: {result_path}")
        case = result.get("case_config")
        if not isinstance(case, dict):
            raise ValueError(f"E2E result has no case_config object: {result_path}")
        if case.get("task_strategy") != "diffusion_media_generation":
            continue
        case_name = result.get("case_name") or case.get("name") or result_path.parent.name
        if not isinstance(case_name, str) or not case_name:
            raise ValueError(f"diffusion E2E result has no case name: {result_path}")
        expected.append(case_name)

    if not expected:
        raise ValueError("no diffusion E2E results were produced")
    duplicate_expected = sorted({name for name in expected if expected.count(name) > 1})
    if duplicate_expected:
        raise ValueError(f"duplicate diffusion E2E results: {duplicate_expected}")

    pairs = discover_diffusion_frame_pairs(artifacts_dir)
    paired = [str(pair.get("case_name") or "") for pair in pairs]
    duplicate_pairs = sorted({name for name in paired if paired.count(name) > 1})
    missing = sorted(set(expected) - set(paired))
    unexpected = sorted(set(paired) - set(expected))
    if duplicate_pairs or missing or unexpected:
        raise ValueError(
            "diffusion frame-pair coverage is incomplete: "
            f"missing={missing}, unexpected={unexpected}, duplicates={duplicate_pairs}"
        )
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count diffusion TRT/reference frame pairs in E2E artifacts."
    )
    parser.add_argument("artifacts_dir", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="print JSON details instead of only the pair count",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless every diffusion E2E result has one unique TRT/reference pair",
    )
    args = parser.parse_args(argv)

    try:
        pairs = (
            validate_complete_diffusion_frame_pairs(args.artifacts_dir)
            if args.require_complete
            else discover_diffusion_frame_pairs(args.artifacts_dir)
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.json_output:
        print(json.dumps({"count": len(pairs), "pairs": pairs}, indent=2))
    else:
        print(len(pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
