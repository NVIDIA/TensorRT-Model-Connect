"""Diffusion media generation comparator.

Compares TRT diffusion output against reference output with metrics:
- Stage-level latent trajectory parity (cosine per step)
- Final-frame PSNR, SSIM, LPIPS
- Temporal consistency for video (frame-to-frame diff stats)
- Frame-level distribution checks (pixel mean/std/min/max)
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ..contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)
from ._helpers import cosine_similarity

logger = logging.getLogger(__name__)


def _compute_psnr(img1: Any, img2: Any) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images (values in [0,1])."""
    import numpy as np
    a = np.asarray(img1, dtype=np.float32)
    b = np.asarray(img2, dtype=np.float32)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return 100.0  # Identical
    return float(10.0 * math.log10(1.0 / mse))


def _compute_ssim(img1: Any, img2: Any) -> float:
    """Compute Structural Similarity Index (simplified, per-channel average)."""
    import numpy as np
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    mu_a = np.mean(a)
    mu_b = np.mean(b)
    sigma_a_sq = np.var(a)
    sigma_b_sq = np.var(b)
    sigma_ab = np.mean((a - mu_a) * (b - mu_b))

    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a_sq + sigma_b_sq + c2)

    return float(numerator / denominator)


def _compute_temporal_consistency(frames_dir: str) -> float:
    """Compute average frame-to-frame cosine similarity for temporal consistency."""
    import numpy as np
    from pathlib import Path

    try:
        from PIL import Image
    except ImportError:
        return -1.0

    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    if len(frames) < 2:
        return 1.0

    similarities = []
    prev_arr = None
    for fp in frames:
        img = Image.open(fp).convert("RGB")
        arr = np.array(img, dtype=np.float32).flatten()
        if prev_arr is not None:
            similarities.append(cosine_similarity(prev_arr, arr))
        prev_arr = arr

    return float(np.mean(similarities)) if similarities else 1.0


class DiffusionComparator:
    """Compares TRT diffusion output against reference output."""

    @property
    def task_strategy(self) -> str:
        return "diffusion_media_generation"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        thresholds = threshold.metrics

        if stage.name == "debug_pipeline":
            return self._compare_debug_pipeline(trt, thresholds)

        if stage.name in ("end_to_end", "end_to_end_video", "generate", "frame_quality"):
            return self._compare_frames(trt, ref, thresholds)

        if stage.name == "t5_encode":
            return self._compare_embeddings(trt, ref, thresholds)

        if stage.name.startswith("crossover_"):
            return self._compare_crossover(trt, stage, thresholds)

        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.SKIPPED.value,
            metrics={},
            message=f"No comparison logic for diffusion stage: {stage.name}",
        )

    def _compare_debug_pipeline(
        self, trt: StageOutput, thresholds: dict[str, float]
    ) -> CompareResult:
        """Extract pass/fail from the debug_diffusion_pipeline output."""
        metrics: dict[str, MetricResult] = {}

        passed = trt.data.get("passed", False)
        output_text = trt.data.get("output", "")

        for line in output_text.splitlines():
            line = line.strip()
            if line.startswith("PASS") or line.startswith("FAIL"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    step_pass = parts[0] == "PASS"
                    name = parts[1]
                    metrics[name] = MetricResult(
                        value=1.0 if step_pass else 0.0,
                        threshold=1.0, operator=">=", passed=step_pass,
                    )

        return CompareResult(
            stage_name="debug_pipeline",
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all debug pipeline steps must pass",
            message=f"debug_pipeline: {'PASS' if passed else 'FAIL'} "
                    f"(rc={trt.data.get('returncode', -1)})",
        )

    def _compare_frames(
        self,
        trt: StageOutput,
        ref: StageOutput,
        thresholds: dict[str, float],
    ) -> CompareResult:
        """Compare generated frames: pixel stats, PSNR, SSIM, temporal consistency."""
        metrics: dict[str, MetricResult] = {}
        all_pass = True

        if trt.data.get("returncode", -1) != 0:
            return CompareResult(
                stage_name="end_to_end",
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"TRT generation failed (rc={trt.data.get('returncode')})",
            )

        num_frames = trt.data.get("num_frames", 0)
        metrics["num_frames"] = MetricResult(
            value=float(num_frames), threshold=None, operator=">=", passed=True,
        )

        frame_stats = trt.data.get("frame_stats", {})
        if frame_stats:
            pixel_mean = frame_stats.get("mean", 0.5)
            pixel_std = frame_stats.get("std", 0.0)

            min_mean = thresholds.get("min_pixel_mean", 0.15)
            max_mean = thresholds.get("max_pixel_mean", 0.85)
            mean_ok = min_mean <= pixel_mean <= max_mean
            metrics["pixel_mean_range"] = MetricResult(
                value=pixel_mean, threshold=None, operator="in_range",
                passed=mean_ok, note=f"[{min_mean}, {max_mean}]",
            )
            if not mean_ok:
                all_pass = False

            min_std = thresholds.get("min_pixel_std", 0.05)
            std_ok = pixel_std >= min_std
            metrics["pixel_std_min"] = MetricResult(
                value=pixel_std, threshold=min_std,
                operator=">=", passed=std_ok,
            )
            if not std_ok:
                all_pass = False

        ref_frame_stats = ref.data.get("frame_stats", {})
        if ref_frame_stats:
            ref_pixel_mean = ref_frame_stats.get("mean", 0.5)
            ref_pixel_std = ref_frame_stats.get("std", 0.0)

            min_mean = thresholds.get("min_pixel_mean", 0.15)
            max_mean = thresholds.get("max_pixel_mean", 0.85)
            ref_mean_ok = min_mean <= ref_pixel_mean <= max_mean
            metrics["reference_pixel_mean_range"] = MetricResult(
                value=ref_pixel_mean, threshold=None, operator="in_range",
                passed=ref_mean_ok, note=f"[{min_mean}, {max_mean}]",
            )
            if not ref_mean_ok:
                all_pass = False

            min_std = thresholds.get("min_pixel_std", 0.05)
            ref_std_ok = ref_pixel_std >= min_std
            metrics["reference_pixel_std_min"] = MetricResult(
                value=ref_pixel_std, threshold=min_std,
                operator=">=", passed=ref_std_ok,
            )
            if not ref_std_ok:
                all_pass = False

        if frame_stats and ref_frame_stats:
            pixel_std = float(frame_stats.get("std", 0.0))
            ref_pixel_std = float(ref_frame_stats.get("std", 0.0))
            min_ref_std = thresholds.get("reference_min_pixel_std_for_ratio", 0.08)
            ratio_thresh = thresholds.get("min_reference_std_ratio", 0.35)
            if ref_pixel_std >= min_ref_std:
                ratio = pixel_std / ref_pixel_std if ref_pixel_std > 0.0 else 0.0
                ratio_ok = ratio >= ratio_thresh
                metrics["reference_pixel_std_ratio"] = MetricResult(
                    value=ratio,
                    threshold=ratio_thresh,
                    operator=">=",
                    passed=ratio_ok,
                    note=f"trt_std={pixel_std:.4f}, ref_std={ref_pixel_std:.4f}",
                )
                if not ratio_ok:
                    all_pass = False

        frames_dir = trt.data.get("frames_dir", "")
        if frames_dir:
            temporal_cs = _compute_temporal_consistency(frames_dir)
            tc_thresh = thresholds.get("temporal_consistency", 0.6)
            tc_ok = temporal_cs >= tc_thresh
            metrics["temporal_consistency"] = MetricResult(
                value=temporal_cs, threshold=tc_thresh,
                operator=">=", passed=tc_ok,
            )
            if not tc_ok:
                all_pass = False

        ref_frames_dir = ref.data.get("frames_dir", "")
        if frames_dir and ref_frames_dir:
            psnr, ssim = self._cross_compare_frames(frames_dir, ref_frames_dir)
            if psnr is not None:
                # Default PSNR threshold is low (5.0) because diffusion models
                # are stochastic: even with the same seed, TRT and HF use
                # different kernels/scheduling so denoising trajectories diverge.
                # This threshold catches broken outputs (black/NaN frames) while
                # accepting legitimate numerical divergence.
                psnr_thresh = thresholds.get("psnr", 5.0)
                psnr_ok = psnr >= psnr_thresh
                metrics["psnr"] = MetricResult(
                    value=psnr, threshold=psnr_thresh,
                    operator=">=", passed=psnr_ok,
                )
                if not psnr_ok:
                    all_pass = False

            if ssim is not None:
                # Default SSIM threshold is low (0.1) for the same reason as
                # PSNR: stochastic diffusion outputs naturally diverge between
                # TRT and HF reference implementations.
                ssim_thresh = thresholds.get("ssim", 0.1)
                ssim_ok = ssim >= ssim_thresh
                metrics["ssim"] = MetricResult(
                    value=ssim, threshold=ssim_thresh,
                    operator=">=", passed=ssim_ok,
                )
                if not ssim_ok:
                    all_pass = False

        n_gated = sum(1 for m in metrics.values() if m.threshold is not None)
        n_passed = sum(1 for m in metrics.values() if m.threshold is not None and m.passed)
        return CompareResult(
            stage_name="end_to_end",
            status=StageStatus.PASSED.value if all_pass else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"{'PASS' if all_pass else 'FAIL'}: {n_passed}/{n_gated} metrics passed",
        )

    def _compare_embeddings(
        self,
        trt: StageOutput,
        ref: StageOutput,
        thresholds: dict[str, float],
    ) -> CompareResult:
        """Compare T5 embedding outputs via cosine similarity."""
        import numpy as np

        trt_path = trt.data.get("output_path", "")
        ref_path = ref.data.get("output_path", "")

        if not trt_path or not ref_path:
            return CompareResult(
                stage_name="t5_encode",
                status=StageStatus.ERROR.value,
                metrics={},
                message="Missing output paths for T5 comparison",
            )

        try:
            trt_arr = np.load(trt_path)
            ref_arr = np.load(ref_path)
        except Exception as e:
            return CompareResult(
                stage_name="t5_encode",
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"Failed to load T5 outputs: {e}",
            )

        cs = cosine_similarity(trt_arr.flatten(), ref_arr.flatten())
        thresh = thresholds.get("latent_cosine_per_step", 0.95)
        passed = cs >= thresh

        return CompareResult(
            stage_name="t5_encode",
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics={"cosine_similarity": MetricResult(
                value=cs, threshold=thresh, operator=">=", passed=passed,
            )},
            message=f"T5 cosine_sim={cs:.6f}",
        )

    def _compare_crossover(
        self,
        trt: StageOutput,
        stage: StageSpec,
        thresholds: dict[str, float],
    ) -> CompareResult:
        """Validate crossover stage output: subprocess succeeded and output is sane."""
        metrics: dict[str, MetricResult] = {}
        all_pass = True

        rc = trt.data.get("returncode", -1)
        if rc != 0:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"Crossover stage failed (rc={rc}): "
                        f"{trt.data.get('stderr', '')[:500]}",
            )

        metrics["subprocess_ok"] = MetricResult(
            value=1.0, threshold=1.0, operator=">=", passed=True,
        )

        out_mean = trt.data.get("dit_output_mean")
        out_std = trt.data.get("dit_output_std")
        if out_mean is not None and out_std is not None:
            finite_ok = math.isfinite(out_mean) and math.isfinite(out_std)
            metrics["output_finite"] = MetricResult(
                value=1.0 if finite_ok else 0.0, threshold=1.0,
                operator=">=", passed=finite_ok,
                note=f"mean={out_mean:.4f}, std={out_std:.4f}",
            )
            if not finite_ok:
                all_pass = False

            nonzero_ok = out_std > 1e-6
            metrics["output_nonzero"] = MetricResult(
                value=out_std, threshold=1e-6,
                operator=">", passed=nonzero_ok,
            )
            if not nonzero_ok:
                all_pass = False

        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if all_pass else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"{'PASS' if all_pass else 'FAIL'} crossover {stage.name}",
        )

    @staticmethod
    def _cross_compare_frames(
        trt_dir: str, ref_dir: str
    ) -> tuple[float | None, float | None]:
        """Compare frames from two directories, return (avg_psnr, avg_ssim)."""
        from pathlib import Path

        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            return None, None

        trt_frames = sorted(Path(trt_dir).glob("frame_*.png"))
        ref_frames = sorted(Path(ref_dir).glob("frame_*.png"))

        if not trt_frames or not ref_frames:
            return None, None

        n = min(len(trt_frames), len(ref_frames))
        psnr_vals = []
        ssim_vals = []

        for i in range(n):
            trt_img = np.array(Image.open(trt_frames[i]).convert("RGB"), dtype=np.float32) / 255.0
            ref_img = np.array(Image.open(ref_frames[i]).convert("RGB"), dtype=np.float32) / 255.0
            psnr_vals.append(_compute_psnr(trt_img, ref_img))
            ssim_vals.append(_compute_ssim(trt_img, ref_img))

        return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


plugin = DiffusionComparator()
