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
        trt_frame = _select_frame(_frames_in(model_dir / "frames"))
        hf_frame = _select_frame(_frames_in(model_dir / "hf_frames"))
        if hf_frame is None:
            hf_frame = _select_frame(_frames_in(model_dir / "ref_frames"))
        if trt_frame is None or hf_frame is None:
            continue

        inputs = case.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}
        pairs.append({
            "case_name": result.get("case_name") or case.get("name") or model_dir.name,
            "prompt": inputs.get("prompt", ""),
            "trt_image": str(trt_frame),
            "hf_image": str(hf_frame),
        })
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
