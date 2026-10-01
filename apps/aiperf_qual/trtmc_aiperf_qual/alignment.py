# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Task-level check for generated images and videos: CLIP text alignment (``clip_alignment``).

Diffusion output drifts between TRTMC and the native model at any precision, so pixel parity only
catches gross failures (black or noise frames, wrong geometry). Both sides render the same prompts;
CLIP scores every image (evenly sampled frames of a video) against its prompt. The check fails when
TRTMC's mean score drops by more than ``max_mean_clip_drop`` below the native model's and the paired
drop is significant (one-sided 95%), so a few prompts that happen to render worse do not fail it.
Videos also compare temporal consistency (mean cosine of adjacent sampled frames' CLIP embeddings).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import Environment
from .generation import generate, generate_native
from .services import _serve_env
from .suites import build_suite, limit_suite

# CLIPScore (Hessel et al. 2021, as torchmetrics computes it): 100 * max(cosine(image, text), 0).
SCORE = r"""
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import CLIPModel, CLIPProcessor
from trtmc_perf_serving.digests import media_frames
model_id, items, max_frames = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_id).to(device).eval()
processor = CLIPProcessor.from_pretrained(model_id)
rows, firsts = [], []
with torch.no_grad():
    for workdir, prompt in items:
        frames = media_frames(Path(workdir))
        if not frames:
            rows.append(None)
            continue
        picks = sorted({int(i) for i in np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).round()})
        inputs = processor(text=[prompt], images=[frames[i] for i in picks], return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to(device)
        output = model(**inputs)  # image_embeds and text_embeds come out L2-normalized
        cosine = (output.image_embeds @ output.text_embeds.T).squeeze(-1)
        row = {"clip_score": float((100 * cosine.clamp(min=0)).mean()), "frames": len(frames)}
        if len(picks) > 1:
            row["temporal_consistency"] = float((output.image_embeds[1:] * output.image_embeds[:-1]).sum(-1).mean())
        rows.append(row)
        firsts.append(output.image_embeds[0])
# Mean cosine between the first frames of different prompts: near 1 when the output ignores the prompt.
cross = None
if len(firsts) > 1:
    stacked = torch.stack(firsts)
    similarity = stacked @ stacked.T
    count = len(firsts)
    cross = float((similarity.sum() - similarity.diagonal().sum()) / (count * (count - 1)))
print(json.dumps({"rows": rows, "cross_prompt_cosine": cross}))
"""

# Different prompts rendering this alike (CLIP image cosine) means the output ignores the prompt.
MAX_CROSS_PROMPT_COSINE = 0.95


def is_video(request: Mapping[str, Any]) -> bool:
    return request.get("media_type") == "video" or int(request.get("num_frames") or 1) > 1


def _score(environment: Environment, clip_model: str, items: Sequence[tuple[str, str]],
           max_frames: int) -> dict[str, Any]:
    completed = subprocess.run(
        [str(environment["serve_python"]), "-c", SCORE, clip_model, json.dumps(list(items)), str(max_frames)],
        capture_output=True, text=True, timeout=3600,
        env={key: value for key, value in _serve_env(environment).items() if key != "HF_HUB_OFFLINE"})
    if completed.returncode:
        raise RuntimeError(f"CLIP scoring failed: {completed.stderr[-400:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


# One-sided 95% Student-t critical values by degrees of freedom (larger df use the next lower entry).
_T95_ONE_SIDED = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833,
                  10: 1.812, 15: 1.753, 20: 1.725, 30: 1.697}


def drop(differences: Sequence[float], limit: float) -> tuple[float, float | None, bool]:
    """Mean paired drop (native minus TRTMC), its one-sided 95% lower bound, and whether it fails:
    above ``limit`` and significant (a single pair can only be judged by the limit)."""
    import statistics

    mean = statistics.fmean(differences)
    if len(differences) < 2:
        return mean, None, mean > limit
    degrees = len(differences) - 1
    critical = _T95_ONE_SIDED[max(key for key in _T95_ONE_SIDED if key <= degrees)] if degrees <= 30 else 1.645
    lower = mean - critical * statistics.stdev(differences) / len(differences) ** 0.5
    return mean, lower, mean > limit and lower > 0


def judge(candidate: Sequence[Mapping[str, Any] | None], native: Sequence[Mapping[str, Any] | None],
          samples: Sequence[Mapping[str, Any]], check: Mapping[str, Any], video: bool,
          cross: Mapping[str, float | None] | None = None) -> dict[str, Any]:
    """Status, gate, metrics, and failing samples from per-sample CLIP rows of both sides; ``cross``
    holds each side's cross-prompt cosine (prompt-independent output when near 1)."""
    clip_drop = float(check.get("max_mean_clip_drop", 1.0))
    consistency_drop = float(check.get("max_temporal_consistency_drop", 0.02))
    tolerance = float(check.get("sample_tolerance", 3.0))
    gate = {"max_mean_clip_drop": clip_drop, **({"max_temporal_consistency_drop": consistency_drop} if video else {})}
    # Means over the samples both sides rendered; a missing TRTMC image fails on its own.
    pairs = [(c, n) for c, n in zip(candidate, native) if c is not None and n is not None]
    metrics: dict[str, Any] = {"candidate_clip_score": _mean([c["clip_score"] for c, _ in pairs]),
                               "native_clip_score": _mean([n["clip_score"] for _, n in pairs])}
    reasons = []
    missing = [samples[index]["sample_id"] for index, row in enumerate(candidate) if row is None]
    if missing:
        reasons.append(f"TRTMC wrote no image for {len(missing)} sample(s)")
    if not any(row is not None for row in native):
        reasons.append("the native model wrote no image")
    if pairs:
        mean, lower, failed = drop([n["clip_score"] - c["clip_score"] for c, n in pairs], clip_drop)
        metrics.update(clip_score_drop=mean, clip_score_drop_lower_bound=lower)
        if failed:
            reasons.append(f"mean CLIP score {metrics['candidate_clip_score']:.2f} vs native "
                           f"{metrics['native_clip_score']:.2f} (drop {mean:.2f}, 95% lower bound {lower or mean:.2f})")
    frames = [(c["temporal_consistency"], n["temporal_consistency"]) for c, n in pairs
              if "temporal_consistency" in c and "temporal_consistency" in n]
    if video and frames:
        metrics.update(candidate_temporal_consistency=_mean([c for c, _ in frames]),
                       native_temporal_consistency=_mean([n for _, n in frames]))
        mean, lower, failed = drop([n - c for c, n in frames], consistency_drop)
        metrics.update(temporal_consistency_drop=mean, temporal_consistency_drop_lower_bound=lower)
        if failed:
            reasons.append(f"temporal consistency {metrics['candidate_temporal_consistency']:.3f} vs native "
                           f"{metrics['native_temporal_consistency']:.3f}")
    cross = dict(cross or {})
    metrics.update(candidate_cross_prompt_cosine=cross.get("candidate"), native_cross_prompt_cosine=cross.get("native"))
    if (cross.get("candidate") or 0) >= MAX_CROSS_PROMPT_COSINE:
        reasons.append(f"TRTMC renders near-identical images for different prompts (cosine {cross['candidate']:.3f})")
    status = "fail" if reasons else "pass"
    if (cross.get("native") or 0) >= MAX_CROSS_PROMPT_COSINE:  # no baseline to judge TRTMC against
        status = "not-comparable"
        reasons.append(f"the native model renders near-identical images for different prompts (cosine "
                       f"{cross['native']:.3f}): broken reference")
    passed, failures = 0, []
    for index, (c, n) in enumerate(zip(candidate, native)):
        if n is None:
            continue
        if c is not None and c["clip_score"] >= n["clip_score"] - tolerance:
            passed += 1
        else:
            failures.append({"sample_id": samples[index]["sample_id"], "prompt": samples[index]["request"]["prompt"],
                             "explanation": f"CLIP {c['clip_score']:.2f} vs native {n['clip_score']:.2f}"
                             if c is not None else "no TRTMC image"})
    return {"status": status, "passed": passed, "gate": gate, "metrics": metrics, "reasons": reasons,
            "failures": failures}


def run(environment: Environment, model: dict[str, Any], check: Mapping[str, Any], python: str,
        out: Path) -> dict[str, Any]:
    from .models import _suite

    suite = build_suite(_suite(check["suite"], model["catalog_profile"], environment.path("repo")), environment)
    video = is_video(suite.samples[0]["request"])
    if video:  # videos take minutes each
        suite = limit_suite(suite, int(check.get("video_samples", 5)))
    clip_model = str(check.get("clip_model", "openai/clip-vit-large-patch14"))
    max_frames = int(check.get("max_frames", 8))
    native, native_reference = generate_native(environment, model, suite, python, out, "clip")
    outputs = {"candidate": generate(environment, model, "trtmc", out / "clip-candidate", suite), "native": native}
    prompts = [str(sample["request"]["prompt"]) for sample in suite.samples]
    scored = {side: _score(environment, clip_model, [(str(workdir), prompt) for (workdir, _), prompt
                                                     in zip(rows, prompts)], max_frames)
              for side, rows in outputs.items()}
    scores = {side: result["rows"] for side, result in scored.items()}
    verdict = judge(scores["candidate"], scores["native"], suite.samples, check, video,
                    {side: result["cross_prompt_cosine"] for side, result in scored.items()})
    for failure in verdict["failures"]:  # where to look at the two renderings
        index = next(i for i, sample in enumerate(suite.samples) if sample["sample_id"] == failure["sample_id"])
        failure.update(actual=str(outputs["candidate"][index][0]), expected=str(outputs["native"][index][0]))
    return {"suite": "clip-alignment", "source": "task",
            "benchmark": f"CLIP text alignment ({suite.name}, {clip_model}, native {native_reference})",
            "status": verdict["status"], "samples": len(suite.samples), "expected_samples": len(suite.samples),
            "passed": verdict["passed"], "required_passes": None, "gate": verdict["gate"],
            "metrics": {**verdict["metrics"], "per_sample": [
                {"sample_id": sample["sample_id"], "candidate": c, "native": n}
                for sample, c, n in zip(suite.samples, scores["candidate"], scores["native"])]},
            "failures": verdict["failures"][:10],
            **({"reasons": verdict["reasons"]} if verdict["reasons"] else {})}
