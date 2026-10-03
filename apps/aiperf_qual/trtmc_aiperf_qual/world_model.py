# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""World-model video parity (``world_model_parity``): both sides render each input with the same seed; every
TRTMC video must have the native frame count, move (mean change between frames at least MIN_MOTION_RATIO of
the native video's), and agree coarsely with the native video (PSNR and SSIM over sampled frames at least the
check's floors). Action-conditional fidelity is not covered (DESIGN.md Section 6)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .config import Environment
from .generation import generate, generate_native
from .replay_parity import PIXEL_HELPERS
from .services import _serve_env
from .suites import build_suite

MIN_MOTION_RATIO = 0.25
SAMPLED_FRAMES = 5

SCRIPT = PIXEL_HELPERS + r"""
def motion(frames):
    picks = sorted({int(i) for i in np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).round()})
    if len(picks) < 2:
        return 0.0
    return float(np.mean([np.abs(frames[a].astype(np.float32) - frames[b].astype(np.float32)).mean()
                          for a, b in zip(picks, picks[1:])]))

rows = []
for candidate, native in items:
    c, n = media_frames(Path(candidate)), media_frames(Path(native))
    rows.append({"frames": [len(c), len(n)], "motion": [motion(c), motion(n)], "pixels": compare(c, n)})
print(json.dumps(rows))
"""


def run(environment: Environment, model: dict[str, Any], check: Mapping[str, Any], python: str,
        out: Path) -> dict[str, Any]:
    from .models import model_suite

    suite = build_suite(model_suite(check["suite"], model), environment)
    native, _, _ = generate_native(environment, model, suite, python, out, "world-model")
    candidate = generate(environment, model, "trtmc", out / "world-model-candidate", suite)
    items = [(str(mine[0]), str(theirs[0])) for mine, theirs in zip(candidate, native)]
    completed = subprocess.run([str(environment["serve_python"]), "-c", SCRIPT, json.dumps(items), str(SAMPLED_FRAMES)],
                               capture_output=True, text=True, timeout=7200, env=_serve_env(environment))
    if completed.returncode:
        raise RuntimeError(f"video comparison failed: {completed.stderr[-400:]}")
    rows = json.loads(completed.stdout.strip().splitlines()[-1])
    floor_psnr, floor_ssim = float(check.get("min_psnr_db", 5.0)), float(check.get("min_ssim", 0.1))
    failures = []
    for sample, row in zip(suite.samples, rows):
        mine, theirs = row["frames"]
        pixels = row["pixels"] or {}
        reason = (f"{mine} frames, native {theirs}" if mine != theirs or not mine else
                  f"frozen: motion {row['motion'][0]:.2f} vs native {row['motion'][1]:.2f}"
                  if row["motion"][1] > 0 and row["motion"][0] < MIN_MOTION_RATIO * row["motion"][1] else
                  f"PSNR {pixels.get('psnr', 0):.1f} dB / SSIM {pixels.get('ssim', 0):.3f} below the floor"
                  if pixels.get("psnr", 0) < floor_psnr or pixels.get("ssim", 0) < floor_ssim else None)
        if reason:
            failures.append({"sample_id": sample["sample_id"], "explanation": reason})
    count = len(suite.samples)
    return {"suite": "world-model-parity", "source": "task", "benchmark": "video parity (frames, motion, PSNR/SSIM)",
            "samples": count, "expected_samples": count, "passed": count - len(failures), "required_passes": count,
            "status": "pass" if not failures and len(rows) == count else "fail", "failures": failures[:5],
            "reasons": [f"{len(failures)} of {count} videos outside the floors"] if failures else [],
            "gate": {"min_psnr_db": floor_psnr, "min_ssim": floor_ssim, "min_motion_ratio": MIN_MOTION_RATIO}}
