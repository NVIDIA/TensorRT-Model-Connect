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
against the native output instead. Generated media drift between TRTMC and the native model at any
precision, so config/tasks.yaml marks this check ``informational``: reported, not part of the verdict.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import Environment
from .generation import generate, generate_native, is_video
from .services import _serve_env
from .suites import build_suite, limit_suite, with_latent_seeds

# Full-resolution PSNR (RGB) and SSIM (luminance, 7x7 box windows) over evenly sampled frames.
PIXEL_HELPERS = r"""
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

"""
PIXELS = PIXEL_HELPERS + r"""
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


def drop(differences: Sequence[float], limit: float) -> tuple[float, float | None, bool]:
    """Mean paired drop (native minus TRTMC), its one-sided 95% lower bound, and whether it fails:
    above ``limit`` and significant (a single pair can only be judged by the limit)."""
    import statistics

    mean = statistics.fmean(differences)
    if len(differences) < 2:
        return mean, None, mean > limit
    from .noninferiority import t_quantile

    lower = mean - t_quantile(0.95, len(differences) - 1) * statistics.stdev(differences) / len(differences) ** 0.5
    return mean, lower, mean > limit and lower > 0


def judge(rows: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]], check: Mapping[str, Any],
          planned: int | None = None) -> dict[str, Any]:
    """Judge the first ``planned`` samples (the full-precision budget) and nothing less: a native output
    missing there is an error, a TRTMC output missing or of another geometry is a failure, and unless
    every planned full-precision render is usable the whole budget is judged by ``fallback_min``.
    Samples beyond the budget only report TRTMC against the native output."""
    psnr_gap, ssim_gap = float(check.get("max_psnr_gap_db", 3.0)), float(check.get("max_ssim_gap", 0.05))
    planned = min(planned or len(rows), len(rows))
    budget = list(zip(samples[:planned], rows[:planned]))
    candidates = [row["candidate"] for row in rows if row.get("candidate")]
    metrics: dict[str, Any] = {
        "candidate_vs_native_psnr_db": _mean([c["psnr"] for c in candidates]),
        "candidate_vs_native_ssim": _mean([c["ssim"] for c in candidates]), "planned_samples": planned,
        "per_sample": [{"sample_id": sample["sample_id"], **{key: row.get(key) for key in
                                                             ("candidate", "candidate_ref", "native_ref")}}
                       for sample, row in zip(samples, rows)]}
    native_missing = sum((row.get("frames") or [1, 1])[1] == 0 for _, row in budget)
    if native_missing:
        return {"status": "error", "passed": None, "samples": planned, "failures": [], "metrics": metrics,
                "gate": {"max_psnr_gap_db": psnr_gap, "max_ssim_gap": ssim_gap},
                "error": f"the native model wrote {planned - native_missing} of {planned} planned outputs"}
    unmatched = [{"sample_id": sample["sample_id"],
                  "explanation": "no TRTMC output" if (row.get("frames") or [0])[0] == 0 else
                  f"geometry differs (frames {row.get('frames')}, shapes {row.get('shapes')})"}
                 for sample, row in budget if row.get("candidate") is None]
    usable = [row for _, row in budget if row.get("candidate_ref") and row.get("native_ref")
              and _valid_floor(row["native_ref"])]
    if not unmatched and len(usable) == planned:
        gaps = [(row["native_ref"]["psnr"] - row["candidate_ref"]["psnr"],
                 row["native_ref"]["ssim"] - row["candidate_ref"]["ssim"]) for row in usable]
        psnr_mean, psnr_lower, psnr_failed = drop([gap[0] for gap in gaps], psnr_gap)
        ssim_mean, ssim_lower, ssim_failed = drop([gap[1] for gap in gaps], ssim_gap)
        failures = [{"sample_id": sample["sample_id"],
                     "explanation": f"vs full precision: TRTMC {row['candidate_ref']['psnr']:.2f} dB / "
                                    f"{row['candidate_ref']['ssim']:.3f}, native {row['native_ref']['psnr']:.2f} dB / "
                                    f"{row['native_ref']['ssim']:.3f}"}
                    for (sample, row), gap in zip(budget, gaps) if gap[0] > psnr_gap or gap[1] > ssim_gap]
        metrics.update(
            candidate_vs_reference_psnr_db=_mean([row["candidate_ref"]["psnr"] for row in usable]),
            native_vs_reference_psnr_db=_mean([row["native_ref"]["psnr"] for row in usable]),
            candidate_vs_reference_ssim=_mean([row["candidate_ref"]["ssim"] for row in usable]),
            native_vs_reference_ssim=_mean([row["native_ref"]["ssim"] for row in usable]),
            psnr_gap_db=psnr_mean, psnr_gap_lower_bound_db=psnr_lower, ssim_gap=ssim_mean,
            ssim_gap_lower_bound=ssim_lower, reference_samples=planned, yardstick="full precision")
        return {"status": "fail" if psnr_failed or ssim_failed else "pass", "passed": planned - len(failures),
                "samples": planned, "failures": failures, "metrics": metrics,
                "gate": {"max_psnr_gap_db": psnr_gap, "max_ssim_gap": ssim_gap}}
    floor = check.get("fallback_min", {"psnr_db": 19.0, "ssim": 0.8})
    min_psnr, min_ssim = float(floor["psnr_db"]), float(floor["ssim"])
    min_rate = float(check.get("min_pass_rate", 0.9))
    failures = unmatched + [
        {"sample_id": sample["sample_id"],
         "explanation": f"PSNR {row['candidate']['psnr']:.2f} dB / SSIM {row['candidate']['ssim']:.3f} vs native"}
        for sample, row in budget
        if row.get("candidate") and (row["candidate"]["psnr"] < min_psnr or row["candidate"]["ssim"] < min_ssim)]
    passed = planned - len(failures)
    metrics.update(yardstick="fallback", reference_samples=len(usable),
                   fallback_reason=f"{len(usable)} of {planned} full-precision renders usable")
    return {"status": "pass" if planned and not unmatched and passed >= min_rate * planned else "fail",
            "passed": passed, "samples": planned, "failures": failures, "metrics": metrics,
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
            **({"error": verdict["error"]} if verdict.get("error") else {}),
            "samples": verdict.get("samples", len(suite.samples)), "expected_samples": verdict.get("samples", len(suite.samples)),
            "passed": None if reasons else verdict["passed"],
            "required_passes": None, "gate": dict(verdict.get("gate", {})), "metrics": dict(verdict.get("metrics", {})),
            "failures": failures, **({"reasons": list(reasons)} if reasons else {})}


def _floor(environment: Environment, model: dict[str, Any], suite: Any, python: str, out: Path, backend: str,
           precision: str, reuse: bool = False) -> tuple[list[tuple[Path, dict[str, Any]]], str] | tuple[None, None]:
    """The same backend at the next native precision (a script reference does not replay the noise)."""
    from .runner import timing_precisions

    # fp32 first (how far the native half-precision run is from full precision; it never overflows),
    # then the other half precision where fp32 does not fit.
    preferred = {"bf16": ["fp32", "fp16"], "fp16": ["fp32", "bf16"]}.get(precision, ["bf16", "fp16"])
    for other in dict.fromkeys([*preferred, *timing_precisions(model["reference"])]):
        if other == precision:
            continue
        try:
            return generate(environment, model, backend, out / f"replay-floor-{backend}-{other}", suite, python,
                            other, reuse), f"{backend} {other}"
        except Exception:  # noqa: BLE001 - try the next precision; no floor at all is reported
            continue
    return None, None


def _parity(environment: Environment, model: dict[str, Any], check: Mapping[str, Any], python: str,
                   out: Path, suite: Any, outputs: Mapping[str, Any], native: tuple[str, str],
                   max_frames: int) -> dict[str, Any]:
    label = " ".join(native)
    missing = not_replayed(outputs)
    if missing:
        return item(suite, {}, label, None, outputs,
                                  [f"the initial noise was not replayed on {', '.join(missing)}"])
    floor_suite = limit_suite(suite, int(check.get("parity_floor_samples", 10)))
    floor, floor_label = _floor(environment, model, floor_suite, python, out, *native,
                                reuse=bool(check.get("reuse_outputs")))
    if floor is not None and not_replayed({"floor": floor}):
        floor, floor_label = None, None
    items = [(str(outputs["candidate"][index][0]), str(outputs["native"][index][0]),
              str(floor[index][0]) if floor is not None and index < len(floor) else None)
             for index in range(len(suite.samples))]
    rows = measure(environment, items, max_frames)
    if model["candidate"].get("quantization"):  # quantized weights deviate more than a native precision change
        check = {**check, **check.get("quantized_gap", {})}
    return item(suite, judge(rows, suite.samples, check, len(floor_suite.samples)),
                              label, floor_label, outputs)


def run(environment: Environment, model: dict[str, Any], check: Mapping[str, Any], python: str,
        out: Path) -> list[dict[str, Any]]:
    """The ``replay-parity`` entry of a family that takes caller latents (nothing otherwise): both sides
    render the first ``parity_floor_samples`` prompts (``video_samples`` videos) with the same initial
    noise, the native model also at full precision."""
    from .models import _suite

    if model.get("family") not in check.get("latent_replay_families", ()):
        return []
    suite = build_suite(_suite(check["suite"], model["catalog_profile"]), environment)
    count = int(check.get("video_samples", 3) if is_video(suite.samples[0]["request"])
                else check.get("parity_floor_samples", 10))
    suite = with_latent_seeds(limit_suite(suite, count))
    reuse = bool(check.get("reuse_outputs"))
    native, native_backend, native_precision = generate_native(environment, model, suite, python, out, "replay",
                                                               reuse=reuse)
    outputs = {"candidate": generate(environment, model, "trtmc", out / "replay-candidate", suite, reuse=reuse),
               "native": native}
    return [_parity(environment, model, check, python, out, suite, outputs,
                    (native_backend, native_precision), int(check.get("max_frames", 8)))]
