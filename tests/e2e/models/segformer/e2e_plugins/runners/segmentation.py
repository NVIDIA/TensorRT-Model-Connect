"""Semantic segmentation strategy runner.

Uses subprocess isolation for GPU operations.

Auto-discovered by the registry via the module-level ``plugin`` attribute.
"""

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


class SegmentationRunner:
    """TRT inference runner for semantic segmentation models."""

    @property
    def strategy_name(self) -> str:
        return "segmentation"

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
        """Run segmentation via C++ binary: trtmc segment --image <path>."""
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
        distributed_runtime = _distributed_runtime_config(case)
        output_root = os.path.join(_model_dir, "rank_outputs")
        output_path = (
            os.path.join(output_root, "rank_0", "seg_output.png")
            if distributed_runtime else os.path.join(_model_dir, "seg_output.png")
        )

        cmd = [
            str(ctx.binary_path), "segment", str(bundle_path),
            "--image", str(image_path),
        ]
        if distributed_runtime:
            wrapper = (
                'rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-${RANK:-0}}}}"; '
                'out="$1/rank_${rank}"; mkdir -p "$out"; shift; '
                'exec "$@" --output "$out/seg_output.png"'
            )
            cmd = ["bash", "-lc", wrapper, "trtmc_rank_segment", output_root] + cmd
        else:
            cmd.extend(["--output", str(output_path)])
        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            cmd.extend(["--hf-python", str(runtime_cli_python)])
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
                metadata={"error": "Segmentation subprocess timed out",
                          "command": cmd},
            )
        elapsed = time.monotonic() - t0

        # Parse class map from output image
        class_map = None
        if result.returncode == 0 and os.path.isfile(output_path):
            try:
                from PIL import Image
                import numpy as np
                class_img = np.array(Image.open(output_path).convert("L"))
                class_map = class_img.astype(np.int32)
            except Exception as e:
                logger.warning("Failed to load segmentation output: %s", e)

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "segmentation", case.name)
        seg_meta: dict = {
            "command": cmd,
            "returncode": result.returncode,
            "stdout": _strip_mpirun_tags(result.stdout),
            "stderr": _strip_mpirun_tags(stderr_truncated),
        }
        if stderr_log:
            seg_meta["stderr_log"] = stderr_log

        return StageOutput(
            stage_name="full_inference",
            data={
                "class_map": class_map,
                "output_path": output_path,
                "segmentation_map_path": output_path,
                "image_path": str(image_path),
            },
            timing_s=elapsed,
            metadata=seg_meta,
        )


# Primary plugin for auto-discovery.
plugin = SegmentationRunner()
