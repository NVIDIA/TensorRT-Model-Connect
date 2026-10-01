# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pixel parity of generated images and videos under latent replay (``replay-parity``).

With the same initial noise (``latent_seed``, see trtmc_perf_serving.latents) TRTMC and the native
model denoise the same starting point, so their outputs are comparable pixel by pixel. The yardstick
is the native model against itself: the same prompts rendered at a second native precision give a
per-sample floor. TRTMC passes a sample when its full-resolution PSNR and SSIM against the native
output are at most ``max_psnr_gap_db`` and ``max_ssim_gap`` below that floor. Without a second native
precision (it fails or is the same), ``fallback_floor`` (25 dB / 0.9 SSIM) stands in for it.
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
for candidate, native, floor in items:
    c, n = media_frames(Path(candidate)), media_frames(Path(native))
    rows.append({"candidate": compare(c, n), "floor": compare(media_frames(Path(floor)), n) if floor else None,
                 "frames": [len(c), len(n)], "shapes": [list(c[0].shape) if c else None, list(n[0].shape) if n else None]})
print(json.dumps(rows))
"""


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
    """Per-sample comparison against the native precision floor (the mean floor where a sample has
    none); passes at ``min_pass_rate``."""
    psnr_gap, ssim_gap = float(check.get("max_psnr_gap_db", 6.0)), float(check.get("max_ssim_gap", 0.1))
    min_rate = float(check.get("min_pass_rate", 0.9))
    floors = [row["floor"] for row in rows if row.get("floor")]
    fallback = check.get("fallback_floor", {"psnr_db": 25.0, "ssim": 0.9})
    mean_floor = ({"psnr": _mean([f["psnr"] for f in floors]), "ssim": _mean([f["ssim"] for f in floors])}
                  if floors else {"psnr": float(fallback["psnr_db"]), "ssim": float(fallback["ssim"])})
    passed, failures = 0, []
    for sample, row in zip(samples, rows):
        candidate, floor = row.get("candidate"), row.get("floor") or mean_floor
        if candidate is None:
            failures.append({"sample_id": sample["sample_id"],
                             "explanation": f"geometry differs (frames {row.get('frames')}, shapes {row.get('shapes')})"})
            continue
        if candidate["psnr"] >= floor["psnr"] - psnr_gap and candidate["ssim"] >= floor["ssim"] - ssim_gap:
            passed += 1
            continue
        failures.append({"sample_id": sample["sample_id"],
                         "explanation": f"PSNR {candidate['psnr']:.2f} dB / SSIM {candidate['ssim']:.3f} vs native "
                                        f"precision floor {floor['psnr']:.2f} dB / {floor['ssim']:.3f}"})
    candidates = [row["candidate"] for row in rows if row.get("candidate")]
    metrics = {"candidate_psnr_db": _mean([c["psnr"] for c in candidates]),
               "candidate_ssim": _mean([c["ssim"] for c in candidates]),
               "floor_psnr_db": mean_floor["psnr"], "floor_ssim": mean_floor["ssim"], "floor_samples": len(floors),
               "floor_source": "native precision" if floors else "fallback",
               "per_sample": [{"sample_id": sample["sample_id"], "candidate": row.get("candidate"),
                               "floor": row.get("floor")} for sample, row in zip(samples, rows)]}
    status = "pass" if rows and passed >= min_rate * len(rows) else "fail"
    return {"status": status, "passed": passed, "failures": failures, "metrics": metrics,
            "gate": {"max_psnr_gap_db": psnr_gap, "max_ssim_gap": ssim_gap, "min_pass_rate": min_rate}}


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
            "benchmark": f"pixel parity under latent replay ({suite.name}, native {native}, floor {floor or 'none'})",
            "status": "not-comparable" if reasons else verdict["status"], "samples": len(suite.samples),
            "expected_samples": len(suite.samples), "passed": None if reasons else verdict["passed"],
            "required_passes": None, "gate": dict(verdict.get("gate", {})), "metrics": dict(verdict.get("metrics", {})),
            "failures": failures, **({"reasons": list(reasons)} if reasons else {})}
