# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image edits against human target images (``edit_similarity``, MagicBrush).

Both sides apply each instruction to its source image (with the same initial noise where the family
takes replayed latents); CLIP-I, the cosine x 100 of the CLIP image embeddings of the edit and of the
human-made target, scores each edit. TRTMC's mean CLIP-I must stay within ``max_delta_points`` of the
native model's (the paired bootstrap interval is a note).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from . import absolute
from .config import Environment
from .generation import generate, generate_native, media_source
from .services import _serve_env
from .suites import build_suite, with_latent_seeds

SCORE = r"""
import base64, io, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from trtmc_perf_serving.digests import media_frames

clip_name, clip_revision, items = sys.argv[1:4]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(clip_name, revision=clip_revision).to(device).eval()
processor = CLIPProcessor.from_pretrained(clip_name, revision=clip_revision)


def output(source):
    frames = media_frames(Path(source["dir"])) if source else []
    if frames:
        return Image.fromarray(np.asarray(frames[0])).convert("RGB")
    files = [path for path in (source or {}).get("files") or [] if Path(path).is_file()]
    return Image.open(files[0]).convert("RGB") if files else None


scores = []
with torch.no_grad():
    for source, target in json.load(open(items)):
        edit = output(source)
        if edit is None:
            scores.append(0.0)
            continue
        reference = Image.open(io.BytesIO(base64.b64decode(target))).convert("RGB")
        embeds = model.get_image_features(**processor(images=[edit, reference], return_tensors="pt").to(device))
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        scores.append(float(100 * (embeds[0] @ embeds[1])))
print(json.dumps(scores))
"""


def run(environment: Environment, model: dict[str, Any], check: Mapping[str, Any], python: str,
        out: Path) -> dict[str, Any]:
    from .models import _suite

    suite = build_suite(_suite(check["suite"], model["catalog_profile"], environment.path("repo")), environment)
    if model.get("family") in check.get("latent_replay_families", ()):
        suite = with_latent_seeds(suite)
    native, native_backend, native_precision = generate_native(environment, model, suite, python, out, "edits",
                                                               reuse=bool(check.get("reuse_outputs")))
    outputs = {"candidate": generate(environment, model, "trtmc", out / "edits-candidate", suite,
                                     reuse=bool(check.get("reuse_outputs"))), "native": native}
    targets = [sample["label"]["png_b64"] for sample in suite.samples]
    sides = {}
    for side, rows in outputs.items():
        items = out / f"edits-{side}.items.json"
        items.write_text(json.dumps([[media_source(workdir, record), target] for (workdir, record), target
                                     in zip(rows, targets)]))
        done = subprocess.run([str(environment["serve_python"]), "-c", SCORE, str(check["clip_model"]),
                               str(check["clip_revision"]), str(items)], capture_output=True, text=True,
                              timeout=4 * 3600,
                              env={k: v for k, v in _serve_env(environment).items() if k != "HF_HUB_OFFLINE"})
        if done.returncode:
            raise RuntimeError(f"CLIP-I scoring failed: {done.stderr[-600:]}")
        scores = json.loads(done.stdout.strip().splitlines()[-1])
        sides[side] = {"observations": {"greedy": {index: {"value": value} for index, value in enumerate(scores)}},
                       "exit": {"greedy": 0}, "timings": {"greedy": {}}}
    item = {"suite": check.get("entry", "edit-similarity"), "metric": "precomputed_mean",
            "gate": dict(check.get("gate") or {"max_delta_points": 1.0})}
    problems = [{"task": "edit", "gold": sample["request"].get("prompt")} for sample in suite.samples]
    entry = absolute.judge(item, problems, sides["candidate"], sides["native"])
    entry["benchmark"] = f"CLIP-I to the human target ({check['clip_model']})"
    entry["native"] = {"backend": native_backend, "precision": native_precision}
    return entry
