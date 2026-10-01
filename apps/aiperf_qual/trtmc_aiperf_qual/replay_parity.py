# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pixel parity of generated images and videos under latent replay (``replay-parity``).

With the same initial noise (``latent_seed``, see trtmc_perf_serving.latents) TRTMC and the native
model denoise the same starting point, so their outputs are comparable pixel by pixel. The yardstick
is full precision: the first samples are also rendered by the native model at fp32 (else the other
half precision), and the check compares how far TRTMC and the native half-precision run each are from
it. It fails when TRTMC is on average more than ``max_psnr_gap_db`` (PSNR) or ``max_ssim_gap`` (SSIM)
further away and that gap is significant (one-sided 95%). Without a usable full-precision render
(it fails, or renders noise such as an fp16 overflow), each sample needs ``fallback_min`` PSNR / SSIM
against the native output instead.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import Environment
from .services import _serve_env

# Full-resolution PSNR (RGB) and SSIM (luminance, 7x7 box windows) over evenly sampled frames.
PIXELS = r"""
import json, sys
from pathlib import Path
import numpy as np
from trtmc_perf_serving.digests import media_frames
items, max_frames = json.loads(sys.argv[1]), int(sys.argv[2])

def gray(frame):
    return frame.astype(np.float64) @ np.array([0.299, 0.587, 0.114])

def box(a, k=7):
    c = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)

def ssim(x, y):
    x, y = gray(x), gray(y)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = box(x), box(y)
    sx, sy, sxy = box(x * x) - mx * mx, box(y * y) - my * my, box(x * y) - mx * my
    return float((((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sx + sy + c2))).mean())

def psnr(x, y):
    mse = float(((x.astype(np.float64) - y.astype(np.float64)) ** 2).mean())
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))

def compare(a, b):
    if not a or not b or len(a) != len(b) or a[0].shape != b[0].shape:
        return None
    picks = sorted({int(i) for i in np.linspace(0, len(a) - 1, min(max_frames, len(a))).round()})
    return {"psnr": float(np.mean([psnr(a[i], b[i]) for i in picks])),
            "ssim": float(np.mean([ssim(a[i], b[i]) for i in picks]))}

rows = []
for candidate, native, reference in items:
    c, n = media_frames(Path(candidate)), media_frames(Path(native))
    r = media_frames(Path(reference)) if reference else []
    rows.append({"candidate": compare(c, n), "candidate_ref": compare(c, r) if r else None,
                 "native_ref": compare(n, r) if r else None,
                 "frames": [len(c), len(n)], "shapes": [list(c[0].shape) if c else None, list(n[0].shape) if n else None]})
print(json.dumps(rows))
"""


MIN_FLOOR_PSNR_DB, MIN_FLOOR_SSIM = 12.0, 0.3


def _valid_floor(floor: Mapping[str, Any]) -> bool:
    return floor["psnr"] >= MIN_FLOOR_PSNR_DB and floor["ssim"] >= MIN_FLOOR_SSIM


def measure(environment: Environment, items: Sequence[tuple[str, str, str | None]],
            max_frames: int) -> list[dict[str, Any]]:
    completed = subprocess.run([str(environment["serve_python"]), "-c", PIXELS, json.dumps(list(items)),
                                str(max_frames)], capture_output=True, text=True, timeout=7200,
                               env=_serve_env(environment))
    if completed.returncode:
        raise RuntimeError(f"pixel comparison failed: {completed.stderr[-400:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def judge(rows: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]],
          check: Mapping[str, Any]) -> dict[str, Any]:
    from .alignment import drop

    psnr_gap, ssim_gap = float(check.get("max_psnr_gap_db", 3.0)), float(check.get("max_ssim_gap", 0.05))
    referenced = [(sample, row) for sample, row in zip(samples, rows) if row.get("candidate_ref") and row.get("native_ref")]
    valid = [(sample, row) for sample, row in referenced if _valid_floor(row["native_ref"])]
    missing = [{"sample_id": sample["sample_id"],
                "explanation": f"geometry differs (frames {row.get('frames')}, shapes {row.get('shapes')})"}
               for sample, row in zip(samples, rows) if row.get("candidate") is None]
    candidates = [row["candidate"] for row in rows if row.get("candidate")]
    metrics: dict[str, Any] = {"candidate_vs_native_psnr_db": _mean([c["psnr"] for c in candidates]),
                               "candidate_vs_native_ssim": _mean([c["ssim"] for c in candidates])}
    per_sample = [{"sample_id": sample["sample_id"], **{key: row.get(key) for key in
                                                        ("candidate", "candidate_ref", "native_ref")}}
                  for sample, row in zip(samples, rows)]
    # A full-precision render that broke for most samples (fp16 overflow renders noise) is no yardstick.
    if valid and len(valid) * 2 >= len(referenced):
        gaps = [(row["native_ref"]["psnr"] - row["candidate_ref"]["psnr"],
                 row["native_ref"]["ssim"] - row["candidate_ref"]["ssim"]) for _, row in valid]
        psnr_mean, psnr_lower, psnr_failed = drop([gap[0] for gap in gaps], psnr_gap)
        ssim_mean, ssim_lower, ssim_failed = drop([gap[1] for gap in gaps], ssim_gap)
        failures = missing + [
            {"sample_id": sample["sample_id"],
             "explanation": f"vs full precision: TRTMC {row['candidate_ref']['psnr']:.2f} dB / "
                            f"{row['candidate_ref']['ssim']:.3f}, native {row['native_ref']['psnr']:.2f} dB / "
                            f"{row['native_ref']['ssim']:.3f}"}
            for (sample, row), gap in zip(valid, gaps) if gap[0] > psnr_gap or gap[1] > ssim_gap]
        metrics.update(
            candidate_vs_reference_psnr_db=_mean([row["candidate_ref"]["psnr"] for _, row in valid]),
            native_vs_reference_psnr_db=_mean([row["native_ref"]["psnr"] for _, row in valid]),
            candidate_vs_reference_ssim=_mean([row["candidate_ref"]["ssim"] for _, row in valid]),
            native_vs_reference_ssim=_mean([row["native_ref"]["ssim"] for _, row in valid]),
            psnr_gap_db=psnr_mean, psnr_gap_lower_bound_db=psnr_lower, ssim_gap=ssim_mean,
            ssim_gap_lower_bound=ssim_lower, reference_samples=len(valid), yardstick="full precision",
            per_sample=per_sample)
        return {"status": "fail" if missing or psnr_failed or ssim_failed else "pass",
                "passed": len(valid) - len(failures) + len(missing), "samples": len(valid), "failures": failures,
                "metrics": metrics, "gate": {"max_psnr_gap_db": psnr_gap, "max_ssim_gap": ssim_gap}}
    floor = check.get("fallback_min", {"psnr_db": 19.0, "ssim": 0.8})
    min_psnr, min_ssim = float(floor["psnr_db"]), float(floor["ssim"])
    min_rate = float(check.get("min_pass_rate", 0.9))
    failures = missing + [
        {"sample_id": sample["sample_id"],
         "explanation": f"PSNR {row['candidate']['psnr']:.2f} dB / SSIM {row['candidate']['ssim']:.3f} vs native"}
        for sample, row in zip(samples, rows)
        if row.get("candidate") and (row["candidate"]["psnr"] < min_psnr or row["candidate"]["ssim"] < min_ssim)]
    passed = len(rows) - len(failures)
    metrics.update(yardstick="fallback", reference_samples=0, per_sample=per_sample)
    return {"status": "pass" if rows and passed >= min_rate * len(rows) else "fail", "passed": passed,
            "samples": len(rows), "failures": failures, "metrics": metrics,
            "gate": {"min_psnr_db": min_psnr, "min_ssim": min_ssim, "min_pass_rate": min_rate}}


def not_replayed(outputs: Mapping[str, Sequence[tuple[Path, Mapping[str, Any]]]]) -> list[str]:
    """Sides whose server did not replay the noise for every sample."""
    return [side for side, rows in outputs.items()
            if not all((record.get("observation") or {}).get("latent_replay") is True for _, record in rows)]


def item(suite: Any, verdict: Mapping[str, Any], native: str, floor: str | None, outputs: Mapping[str, Any],
         reasons: Sequence[str] = ()) -> dict[str, Any]:
    failures = [dict(failure) for failure in verdict.get("failures", [])[:10]]
    for failure in failures:  # where to look at the two renderings
        index = next(i for i, sample in enumerate(suite.samples) if sample["sample_id"] == failure["sample_id"])
        failure.update(actual=str(outputs["candidate"][index][0]), expected=str(outputs["native"][index][0]))
    return {"suite": "replay-parity", "source": "task",
            "benchmark": f"pixel parity under latent replay ({suite.name}, native {native}, full precision {floor or 'none'})",
            "status": "not-comparable" if reasons else verdict["status"],
            "samples": verdict.get("samples", len(suite.samples)), "expected_samples": verdict.get("samples", len(suite.samples)),
            "passed": None if reasons else verdict["passed"],
            "required_passes": None, "gate": dict(verdict.get("gate", {})), "metrics": dict(verdict.get("metrics", {})),
            "failures": failures, **({"reasons": list(reasons)} if reasons else {})}
