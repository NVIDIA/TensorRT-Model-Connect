# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object detection strategy runner — TRT inference for detection models.

Handles detection models that output bounding boxes, scores, and class IDs.

Stages:
  - "full_inference": Run C++ binary detection and parse output.

Auto-discovered by the registry via the module-level ``plugin`` attribute.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from .. import save_full_stderr, _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[6]


class ObjectDetectionRunner:
    """TRT inference runner for object detection models."""

    @property
    def strategy_name(self) -> str:
        return "object_detection"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name == "full_inference":
            return self._run_full_inference(case, ctx)
        else:
            return StageOutput(
                stage_name=stage.name,
                metadata={"error": f"Unknown stage: {stage.name}"},
            )

    def _resolve_bundle_path(self, case: E2ECase, ctx: RunContext) -> str:
        bundle = case.bundle or f"{case.name}.trtfb"
        if os.path.isabs(bundle):
            return bundle
        return os.path.join(ctx.engine_dir, bundle)

    def _resolve_image_path(self, case: E2ECase, ctx: RunContext) -> str | None:
        image = (case.inputs.get("image") or case.inputs.get("test_image")
                 or case.inputs.get("image_path"))
        if not image:
            return None
        p = Path(image)
        if p.is_absolute():
            return str(p)
        for base in [ctx.engine_dir, str(PROJECT_DIR), str(PROJECT_DIR / "tests" / "e2e")]:
            candidate = os.path.join(base, image)
            if os.path.isfile(candidate):
                return candidate
        return str(p)

    def _run_full_inference(
        self, case: E2ECase, ctx: RunContext
    ) -> StageOutput:
        """Run detection via C++ binary: trtmc detect --image <path>."""
        bundle_path = self._resolve_bundle_path(case, ctx)
        image_path = self._resolve_image_path(case, ctx)

        if not image_path or not os.path.isfile(image_path):
            return StageOutput(
                stage_name="full_inference",
                metadata={"error": f"Image not found: {image_path}",
                          "skipped": True},
            )

        if not ctx.binary_path or not os.path.isfile(ctx.binary_path):
            return StageOutput(
                stage_name="full_inference",
                metadata={"error": f"Binary not found: {ctx.binary_path}",
                          "skipped": True},
            )

        _model_dir = _case_artifact_dir(ctx.artifacts_dir or "/tmp/claude", case.name)
        json_output_path = os.path.join(_model_dir, "detections.json")

        # Detection confidence threshold from case inputs
        score_threshold = case.inputs.get("score_threshold", 0.3)

        cmd = [
            str(ctx.binary_path), "detect", str(bundle_path),
            "--image", str(image_path),
            "--output-json", str(json_output_path),
            "--score-threshold", str(score_threshold),
        ]
        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            cmd.extend(["--hf-python", str(runtime_cli_python)])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, env=env)
        except subprocess.TimeoutExpired:
            return StageOutput(
                stage_name="full_inference",
                timing_s=time.monotonic() - t0,
                metadata={"error": "Detection subprocess timed out",
                          "command": cmd},
            )
        elapsed = time.monotonic() - t0

        # Parse detection results
        boxes, scores, class_ids = _parse_detection_output(
            json_output_path, result.stdout)

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "object_detection", case.name)
        det_meta: dict = {
            "command": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": stderr_truncated,
        }
        if stderr_log:
            det_meta["stderr_log"] = stderr_log

        return StageOutput(
            stage_name="full_inference",
            data={
                "boxes": boxes,
                "scores": scores,
                "class_ids": class_ids,
                "num_detections": len(boxes),
                "image_path": str(image_path),
                "output_json": json_output_path,
            },
            timing_s=elapsed,
            metadata=det_meta,
        )


def _parse_detection_output(
    json_path: str, stdout_text: str
) -> tuple[list, list[float], list[int]]:
    """Parse detection outputs from JSON file or stdout.

    Returns (boxes, scores, class_ids) where:
      - boxes: list of [x1, y1, x2, y2] normalized coordinates
      - scores: list of confidence scores
      - class_ids: list of predicted class indices
    """
    boxes: list = []
    scores: list[float] = []
    class_ids: list[int] = []

    # Try JSON output file first
    if os.path.isfile(json_path):
        try:
            with open(json_path) as f:
                data = json.load(f)
            detections = data if isinstance(data, list) else data.get("detections", [])
            for det in detections:
                box = det.get("box") or det.get("bbox")
                if box:
                    boxes.append(box)
                score = det.get("score") or det.get("confidence")
                if score is not None:
                    scores.append(float(score))
                cls = det.get("class_id") or det.get("label") or det.get("category_id")
                if cls is not None:
                    class_ids.append(int(cls))
            return boxes, scores, class_ids
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to parse detection JSON: %s", e)

    # Fall back to parsing stdout
    if stdout_text:
        for line in stdout_text.splitlines():
            line = line.strip()
            # Try to parse lines like: "class=5 score=0.95 box=[0.1,0.2,0.3,0.4]"
            if "score=" in line and "box=" in line:
                try:
                    parts = {}
                    for token in line.split():
                        if "=" in token:
                            k, v = token.split("=", 1)
                            parts[k] = v

                    if "box" in parts:
                        box_str = parts["box"].strip("[]")
                        box = [float(x) for x in box_str.split(",")]
                        boxes.append(box)
                    if "score" in parts:
                        scores.append(float(parts["score"]))
                    if "class" in parts:
                        class_ids.append(int(parts["class"]))
                except (ValueError, IndexError):
                    continue

    return boxes, scores, class_ids


plugin = ObjectDetectionRunner()
