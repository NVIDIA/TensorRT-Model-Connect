# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image edits against human target images (``edit_similarity``, MagicBrush).

Both sides apply each instruction to its source image (with the same initial noise where the family
takes replayed latents); CLIP-I, the cosine x 100 of the CLIP image embeddings of the edit and of the
human-made target, scores each edit; TRTMC's mean CLIP-I is judged by the paired bootstrap against the
check's ``margin`` (``noninferiority``).
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
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
from trtmc_perf_serving.digests import media_frames

clip_name, clip_revision, dino_name, dino_revision, items = sys.argv[1:6]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(clip_name, revision=clip_revision).to(device).eval()
processor = CLIPProcessor.from_pretrained(clip_name, revision=clip_revision)
dino = AutoModel.from_pretrained(dino_name, revision=dino_revision).to(device).eval()
dino_processor = AutoImageProcessor.from_pretrained(dino_name, revision=dino_revision)


def similarity(a, b):
    # (CLIP-I, DINO) x 100: cosines of the two images' CLIP embeddings and DINO class tokens
    clip_embeds = model.get_image_features(**processor(images=[a, b], return_tensors="pt").to(device))
    clip_embeds = clip_embeds / clip_embeds.norm(dim=-1, keepdim=True)
    tokens = dino(**dino_processor(images=[a, b], return_tensors="pt").to(device)).last_hidden_state[:, 0]
    tokens = tokens / tokens.norm(dim=-1, keepdim=True)
    return float(100 * (clip_embeds[0] @ clip_embeds[1])), float(100 * (tokens[0] @ tokens[1]))


def output(source):
    frames = media_frames(Path(source["dir"])) if source else []
    if frames:
        return Image.fromarray(np.asarray(frames[0])).convert("RGB")
    files = [path for path in (source or {}).get("files") or [] if Path(path).is_file()]
    return Image.open(files[0]).convert("RGB") if files else None


rows = []
with torch.no_grad():
    for source, target, original in json.load(open(items)):
        edit = output(source)
        reference = Image.open(io.BytesIO(base64.b64decode(target))).convert("RGB")
        before = Image.open(original).convert("RGB")
        baseline = similarity(before, reference)  # the no-edit baseline: the source image against the target
        if edit is None:  # no output image: a missing answer, never a score
            rows.append({"missing": True, "baseline": baseline})
            continue
        same_size = edit.resize(before.size)
        unchanged = float(np.abs(np.asarray(same_size, np.float32) - np.asarray(before, np.float32)).mean()) < 1.0
        clip_i, dino_score = similarity(edit, reference)
        rows.append({"clip_i": clip_i, "dino": dino_score, "unchanged": unchanged, "baseline": baseline})
print(json.dumps(rows))
"""


def run(environment: Environment, model: dict[str, Any], check: Mapping[str, Any], python: str,
        out: Path) -> dict[str, Any]:
    from .models import _suite

    suite = build_suite(_suite(check["suite"], model["catalog_profile"]), environment)
    if model.get("family") in check.get("latent_replay_families", ()):
        suite = with_latent_seeds(suite)
    native, native_backend, native_precision = generate_native(environment, model, suite, python, out, "edits",
                                                               reuse=bool(check.get("reuse_outputs")))
    outputs = {"candidate": generate(environment, model, "trtmc", out / "edits-candidate", suite,
                                     reuse=bool(check.get("reuse_outputs"))), "native": native}
    targets = [sample["label"]["png_b64"] for sample in suite.samples]
    from trtmc_perf_serving.files import materialize_files

    # The source images the edits start from (inlined in the requests): files for the scorer's baseline.
    sources = [materialize_files(sample["request"], out / "edits-sources" / str(index)).get("image_path")
               for index, sample in enumerate(suite.samples)]
    scored = {}
    for side, rows in outputs.items():
        items = out / f"edits-{side}.items.json"
        items.write_text(json.dumps([[media_source(workdir, record), target, source] for (workdir, record), target, source
                                     in zip(rows, targets, sources)]))
        done = subprocess.run([str(environment["serve_python"]), "-c", SCORE, str(check["clip_model"]),
                               str(check["clip_revision"]), str(check["dino_model"]), str(check["dino_revision"]),
                               str(items)], capture_output=True, text=True, timeout=4 * 3600,
                              env={k: v for k, v in _serve_env(environment).items() if k != "HF_HUB_OFFLINE"})
        if done.returncode:
            raise RuntimeError(f"edit scoring failed: {done.stderr[-600:]}")
        scored[side] = json.loads(done.stdout.strip().splitlines()[-1])
    problems = [{"task": "edit", "gold": sample["request"].get("prompt"), "sample_id": sample["sample_id"]}
                for sample in suite.samples]
    name = check.get("entry", "edit-similarity")
    entries = []
    for metric, label in (("clip_i", f"CLIP-I to the human target ({check['clip_model']})"),
                          ("dino", f"DINO to the human target ({check['dino_model']})")):
        sides = {side: {"observations": {"greedy": {index: {"value": row[metric]} for index, row in enumerate(rows)
                                                    if not row.get("missing")}},
                        "exit": {"greedy": 0}, "timings": {"greedy": {}}} for side, rows in scored.items()}
        item = {"suite": f"{name}-{metric.replace('_', '-')}", "metric": "precomputed_mean",
                "gate": dict(check.get("gate") or {"margin": 1.0})}
        entry = absolute.judge(item, problems, sides["candidate"], sides["native"])
        baseline = [row["baseline"][0 if metric == "clip_i" else 1] for row in scored["native"]]
        entry.update(benchmark=label, native={"backend": native_backend, "precision": native_precision},
                     notes=[*entry.get("notes", []), f"no-edit baseline (source vs target): "
                                                     f"{sum(baseline) / max(len(baseline), 1):.2f}"])
        entries.append(entry)
    rows = scored["candidate"]
    missing = [problem["sample_id"] for problem, row in zip(problems, rows) if row.get("missing")]
    unchanged = [problem["sample_id"] for problem, row in zip(problems, rows) if not row.get("missing") and row["unchanged"]]
    count = len(problems)
    entries.append({"suite": f"{name}-changed", "source": "task", "benchmark": "the edit changes its source image",
                    "samples": count - len(missing), "expected_samples": count, "passed": count - len(missing) - len(unchanged),
                    "required_passes": count, "status": "error" if missing else "fail" if unchanged else "pass",
                    "failures": [{"sample_id": sample, "explanation": "output equals its source"} for sample in unchanged[:5]],
                    "reasons": ([f"{len(missing)} of {count} edits have no output image"] if missing else [])
                               + ([f"{len(unchanged)} of {count} outputs equal their source"] if unchanged else []),
                    "gate": {"max_unchanged": 0}})
    return entries
