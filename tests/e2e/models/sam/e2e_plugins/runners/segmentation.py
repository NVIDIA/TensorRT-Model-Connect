# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM prompted segmentation strategy runner."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from pathlib import Path

from .. import save_full_stderr, _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[6]


def _distributed_runtime_config(case: E2ECase) -> dict:
    config = case.metadata.get("distributed_runtime", {})
    return config if isinstance(config, dict) and config.get("enabled") else {}


def _wrap_distributed_command(cmd: list[str], case: E2ECase) -> list[str]:
    config = _distributed_runtime_config(case)
    if not config:
        return cmd
    launcher = str(config.get("launcher", "mpirun") or "mpirun")
    world_size = int(config.get("world_size", config.get("tp_size", 2)) or 2)
    launcher_args = config.get("launcher_args")
    if isinstance(launcher_args, list):
        return [launcher] + [str(arg) for arg in launcher_args] + cmd
    return [launcher, "--tag-output", "-np", str(world_size)] + cmd


def _strip_mpirun_tags(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lines.append(re.sub(r"^\[[^\]]+\]<std(?:out|err)>:\s?", "", line))
    return "\n".join(lines)


class PromptedSegmentationRunner:
    """TRT inference runner for prompted segmentation models (e.g. SAM)."""

    @property
    def strategy_name(self) -> str:
        return "prompted_segmentation"

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
        bundle = case.bundle or f"{case.name}.bundle"
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
        """Run prompted segmentation via the generic prompted-segmentation CLI."""
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

        # Extract point prompts from case inputs
        point_x = case.inputs.get("point_x", 0.5)
        point_y = case.inputs.get("point_y", 0.5)
        is_foreground = case.inputs.get("is_foreground", True)
        num_expected_masks = case.inputs.get("num_expected_masks", 4)

        _model_dir = _case_artifact_dir(ctx.artifacts_dir or "/tmp/claude", case.name)
        distributed_runtime = _distributed_runtime_config(case)
        output_root = os.path.join(_model_dir, "rank_outputs")
        output_dir = (
            os.path.join(output_root, "rank_0", "masks")
            if distributed_runtime else os.path.join(_model_dir, "masks")
        )

        cmd = [
            str(ctx.binary_path), "segment-prompted", str(bundle_path),
            "--image", str(image_path),
            "--point-x", str(point_x),
            "--point-y", str(point_y),
        ]
        if not is_foreground:
            cmd.append("--background")
        if distributed_runtime:
            wrapper = (
                'rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-${RANK:-0}}}}"; '
                'out="$1/rank_${rank}/masks"; mkdir -p "$out"; shift; '
                'exec "$@" --output "$out"'
            )
            cmd = ["bash", "-lc", wrapper, "trtmc_rank_sam", output_root] + cmd
        else:
            cmd.extend(["--output", str(output_dir)])
        if ctx.model_plugin_dir:
            cmd.extend(["--model-plugin-dir", ctx.model_plugin_dir])
        cmd = _wrap_distributed_command(cmd, case)

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
                metadata={"error": "Prompted segmentation timed out",
                          "command": cmd},
            )
        elapsed = time.monotonic() - t0

        # Parse mask outputs from the output directory
        masks = []
        mask_scores = []
        segmented_image_path = None
        if result.returncode == 0:
            masks, mask_scores = _load_mask_outputs(output_dir, result.stdout)
            segmented_image_path = str(Path(output_dir) / "segmented.png")
            if not Path(segmented_image_path).is_file():
                segmented_image_path = _write_segmented_overlay(
                    image_path, masks, mask_scores, output_dir)

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "prompted_segmentation", case.name)
        pseg_meta: dict = {
            "command": cmd,
            "returncode": result.returncode,
            "stdout": _strip_mpirun_tags(result.stdout)[-2000:] if result.stdout else "",
            "stderr": _strip_mpirun_tags(stderr_truncated),
        }
        if stderr_log:
            pseg_meta["stderr_log"] = stderr_log

        return StageOutput(
            stage_name="full_inference",
            data={
                "masks": masks,
                "mask_scores": mask_scores,
                "num_masks": len(masks),
                "num_expected_masks": num_expected_masks,
                "point_prompt": {"x": point_x, "y": point_y},
                "output_dir": output_dir,
                "image_path": str(image_path),
                "segmented_image_path": segmented_image_path,
            },
            timing_s=elapsed,
            metadata=pseg_meta,
        )


def _load_mask_outputs(
    output_dir: str, stdout_text: str
) -> tuple[list, list[float]]:
    """Load mask arrays and scores from the output directory or stdout.

    Returns (masks_list, scores_list) where masks are numpy arrays.
    """
    masks = []
    scores = []

    # Try loading .npy mask files from output directory
    try:
        import numpy as np
        mask_dir = Path(output_dir)
        if mask_dir.is_dir():
            for mask_file in sorted(mask_dir.glob("mask_*.npy")):
                mask = np.load(str(mask_file))
                masks.append(mask)
            for score_file in sorted(mask_dir.glob("score_*.txt")):
                try:
                    score = float(score_file.read_text().strip())
                    scores.append(score)
                except (ValueError, OSError):
                    pass
    except ImportError:
        logger.warning("numpy not available for loading mask outputs")

    # Fall back to parsing scores from stdout
    if not scores and stdout_text:
        for line in stdout_text.splitlines():
            if "score=" in line or "iou_prediction=" in line:
                try:
                    val = line.split("=")[-1].strip().rstrip(",)")
                    scores.append(float(val))
                except (ValueError, IndexError):
                    pass

    # Try loading PNG masks if no .npy files found
    if not masks:
        try:
            import numpy as np
            from PIL import Image
            mask_dir = Path(output_dir)
            if mask_dir.is_dir():
                for mask_file in sorted(mask_dir.glob("mask_*.png")):
                    mask_img = np.array(Image.open(str(mask_file)).convert("L"))
                    masks.append((mask_img > 127).astype(np.uint8))
        except (ImportError, Exception) as e:
            logger.warning("Failed to load PNG masks: %s", e)

    return masks, scores


def _write_segmented_overlay(
    image_path: str,
    masks: list,
    mask_scores: list[float],
    output_dir: str,
) -> str | None:
    """Write an input-image overlay for the highest-scoring SAM mask."""
    if not masks:
        return None

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    mask_index = 0
    if mask_scores and len(mask_scores) >= len(masks):
        mask_index = max(range(len(masks)), key=lambda i: mask_scores[i])

    mask_arr = np.asarray(masks[mask_index], dtype=np.float32)
    if mask_arr.ndim > 2:
        mask_arr = np.squeeze(mask_arr)
    if mask_arr.ndim != 2:
        return None

    mask_img = Image.fromarray((mask_arr > 0).astype(np.uint8) * 255)
    if mask_img.size != image.size:
        mask_img = mask_img.resize(image.size, Image.NEAREST)

    image_arr = np.asarray(image, dtype=np.float32)
    mask_bool = np.asarray(mask_img, dtype=np.uint8) > 0
    overlay = np.zeros_like(image_arr)
    overlay[..., 1] = 220.0
    overlay[..., 2] = 64.0
    alpha = 0.55
    image_arr[mask_bool] = image_arr[mask_bool] * (1.0 - alpha) + overlay[mask_bool] * alpha

    out_path = Path(output_dir) / "segmented.png"
    try:
        Image.fromarray(np.clip(image_arr, 0, 255).astype(np.uint8)).save(out_path)
    except Exception:
        return None
    return str(out_path)


plugin = None
