# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenEval for text-to-image models: what each image must contain, verified on both sides (``geneval``).

Both sides render the suite's GenEval prompts (with the same initial noise where the family takes replayed
latents). Each image passes when every requirement of its prompt holds, by GenEval's rules: the
objects are detected (an open-vocabulary detector, OWLv2, at score >= 0.2 after per-class NMS) at
least as often as required and not as often as excluded, an object's CLIP-classified color is the
required one, and a required relative position (left of, right of, above, below) holds between the
objects' box centers along the dominant axis. The detector differs from GenEval's Mask2Former, so the
absolute scores are GenEval-style; both sides are scored alike, and TRTMC's pass rate is judged by the
paired score test against the check's ``margin`` (``noninferiority``).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import absolute
from .config import Environment
from .generation import generate, generate_native, media_source
from .services import _serve_env
from .suites import build_suite, with_latent_seeds

SCORE = r"""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, Owlv2ForObjectDetection, Owlv2Processor
from trtmc_perf_serving.digests import media_frames

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
owl_name, owl_revision, clip_name, clip_revision, threshold, items = sys.argv[1:7]
threshold = float(threshold)
device = "cuda" if torch.cuda.is_available() else "cpu"
owl_processor = Owlv2Processor.from_pretrained(owl_name, revision=owl_revision)
owl = Owlv2ForObjectDetection.from_pretrained(owl_name, revision=owl_revision).to(device).eval()
clip_processor = CLIPProcessor.from_pretrained(clip_name, revision=clip_revision)
clip = CLIPModel.from_pretrained(clip_name, revision=clip_revision).to(device).eval()


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detect(image, classes):
    inputs = owl_processor(text=[[f"a photo of a {name}" for name in classes]], images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = owl(**inputs)
    side = max(image.size)  # OWLv2 pads the image to a square
    post = (getattr(owl_processor, "post_process_grounded_object_detection", None)  # transformers >= 5
            or owl_processor.post_process_object_detection)
    result = post(outputs=outputs, threshold=threshold, target_sizes=torch.tensor([[side, side]], device=device))[0]
    found = {name: [] for name in classes}
    for box, score, label in sorted(zip(result["boxes"].tolist(), result["scores"].tolist(), result["labels"].tolist()),
                                    key=lambda item: -item[1]):
        kept = found[classes[label]]
        if all(iou(box, other) <= 0.5 for other in kept):  # per-class NMS
            kept.append(box)
    return found


def color(image, box, name):
    crop = image.crop(tuple(int(round(v)) for v in box))
    if crop.width < 2 or crop.height < 2:
        return None
    inputs = clip_processor(text=[f"a photo of a {c} {name}" for c in COLORS], images=crop, return_tensors="pt",
                            padding=True).to(device)
    with torch.no_grad():
        logits = clip(**inputs).logits_per_image[0]
    return COLORS[int(logits.argmax())]


def center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def holds(relation, a, b):
    (ax, ay), (bx, by) = center(a), center(b)
    dx, dy = ax - bx, ay - by
    return {"left of": dx < 0 and abs(dx) >= abs(dy), "right of": dx > 0 and abs(dx) >= abs(dy),
            "above": dy < 0 and abs(dy) >= abs(dx), "below": dy > 0 and abs(dy) >= abs(dx)}[relation]


def all_frames(source):
    frames = media_frames(Path(source["dir"])) if source else []
    if frames:
        return [np.asarray(frame) for frame in frames]
    files = [path for path in (source or {}).get("files") or [] if Path(path).is_file()]
    return [np.asarray(Image.open(path).convert("RGB")) for path in files]


def first_frame(source):
    # An image, or a video's middle frame.
    frames = all_frames(source)
    return Image.fromarray(frames[len(frames) // 2]).convert("RGB") if frames else None


def projected(output):
    return output if isinstance(output, torch.Tensor) else output.pooler_output


def clip_t(image, prompt):
    # CLIP-T: 100 x the cosine of the image and prompt embeddings.
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        # Transformers 5 returns a model output whose pooler_output is the projected embedding.
        image_embedding = projected(clip.get_image_features(pixel_values=inputs["pixel_values"]))
        text_embedding = projected(clip.get_text_features(input_ids=inputs["input_ids"],
                                                          attention_mask=inputs["attention_mask"]))
    return 100.0 * float(torch.nn.functional.cosine_similarity(image_embedding, text_embedding).item())


def motion(frames):
    # Mean absolute change between consecutive frames (0-255 scale); 0 for an image or a frozen video.
    if len(frames) < 2:
        return 0.0
    return float(np.mean([np.abs(a.astype(np.float32) - b.astype(np.float32)).mean() for a, b in zip(frames, frames[1:])]))


def evaluate(source, meta):
    image = first_frame(source)
    if image is None:
        return False, "no image"
    # Exact counts: GenEval's counting prompts also exclude count + 1 of the class (below).
    classes = sorted({req["class"] for req in meta["include"] + meta.get("exclude", [])})
    found = detect(image, classes)
    for req in meta["include"]:
        boxes = found[req["class"]]
        if len(boxes) < req["count"]:
            return False, f"{req['class']}: {len(boxes)} of {req['count']}"
        if "color" in req:
            seen = [color(image, box, req["class"]) for box in boxes[:req["count"]]]
            if any(value != req["color"] for value in seen):
                return False, f"{req['class']} is {seen}, not {req['color']}"
        if "position" in req:
            relation, target = req["position"]
            other = found[meta["include"][target]["class"]]
            if not other or not holds(relation, boxes[0], other[0]):
                return False, f"{req['class']} not {relation} {meta['include'][target]['class']}"
    for req in meta.get("exclude", []):
        if len(found[req["class"]]) >= req["count"]:
            return False, f"{req['class']}: {len(found[req['class']])} >= excluded {req['count']}"
    return True, "all requirements hold"


rows = []
for source, meta in json.load(open(items)):
    passed, reason = evaluate(source, meta)
    frames = all_frames(source)
    image = first_frame(source)
    if image is None:  # no output: a missing answer, never a wrong one or a zero score
        rows.append({"missing": True, "frames": 0, "motion": 0.0})
        continue
    rows.append({"passed": passed, "reason": reason, "frames": len(frames), "motion": motion(frames),
                 "clip_t": clip_t(image, meta.get("prompt", ""))})
print(json.dumps(rows))
"""
# A video counts as frozen when it changes less than this fraction of the native video between frames.
MIN_MOTION_RATIO = 0.25


def _score(environment: Environment, check: Mapping[str, Any], items: Sequence[tuple[Mapping, Mapping]],
           out: Path) -> list[dict[str, Any]]:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(list(items)))
    done = subprocess.run([str(environment["serve_python"]), "-c", SCORE, str(check["detector"]),
                           str(check["detector_revision"]), str(check["clip_model"]), str(check["clip_revision"]),
                           str(check.get("detector_threshold", 0.2)), str(out)],
                          capture_output=True, text=True, timeout=6 * 3600,
                          env={key: value for key, value in _serve_env(environment).items() if key != "HF_HUB_OFFLINE"})
    if done.returncode:
        raise RuntimeError(f"GenEval scoring failed: {done.stderr[-600:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def run(environment: Environment, model: dict[str, Any], check: Mapping[str, Any], python: str,
        out: Path) -> dict[str, Any]:
    from .models import model_suite

    suite = build_suite(model_suite(check["suite"], model), environment)
    if model.get("family") in check.get("latent_replay_families", ()):
        suite = with_latent_seeds(suite)
    native, native_backend, native_precision = generate_native(environment, model, suite, python, out, "geneval",
                                                               reuse=bool(check.get("reuse_outputs")))
    outputs = {"candidate": generate(environment, model, "trtmc", out / "geneval-candidate", suite,
                                     reuse=bool(check.get("reuse_outputs"))), "native": native}
    metadata = [{**sample["label"], "prompt": sample["request"].get("prompt", "")} for sample in suite.samples]
    scored = {}
    for side, rows in outputs.items():
        sources = [media_source(workdir, record) for workdir, record in rows]
        scored[side] = _score(environment, check, list(zip(sources, metadata)), out / f"geneval-{side}.items.json")
    problems = [{"task": meta.get("tag", "geneval"), "gold": meta["prompt"]} for meta in metadata]
    name = check.get("entry", "geneval")
    if check.get("metric") == "clip_t":  # without shared noise: a continuous prompt-paired score
        sides = {side: {"observations": {"greedy": {index: {"value": row["clip_t"]} for index, row in enumerate(rows)
                                                    if not row.get("missing")}},
                        "exit": {"greedy": 0}, "timings": {"greedy": {}}} for side, rows in scored.items()}
        item = {"suite": name, "metric": "precomputed_mean", "gate": dict(check.get("gate") or {"margin": 1.0})}
        entry = absolute.judge(item, problems, sides["candidate"], sides["native"])
        entry["benchmark"] = f"CLIP-T ({check['clip_model']})"
    else:  # the same initial noise on both sides: right/wrong per prompt, paired
        graded = {side: {"records": {"greedy": {index: {"passed": row["passed"], "unparsed": False, "actual": row["reason"]}
                                                for index, row in enumerate(rows) if not row.get("missing")}},
                         "exit": {"greedy": 0}, "timings": {"greedy": {}}} for side, rows in scored.items()}
        item = {"suite": name, "plugin": f"GenEval-style ({check['detector']} + CLIP colors)",
                "endpoint": "image_generation", "gate": dict(check.get("gate") or {"margin": 5.0})}
        entry = absolute.judge(item, problems, graded["candidate"], graded["native"])
    entry["native"] = {"backend": native_backend, "precision": native_precision}
    entries = [entry]
    if any(row["frames"] > 1 for row in scored["native"]):  # videos: every one valid (frames, not frozen)
        entries.append(video_validity(name, problems, scored["candidate"], scored["native"]))
    return entries


def video_validity(name: str, problems: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]],
                   native: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every TRTMC video has the native video's frame count and moves at least MIN_MOTION_RATIO as much."""
    failures = []
    for index, (mine, theirs) in enumerate(zip(candidate, native)):
        if mine["frames"] != theirs["frames"]:
            failures.append({"sample_id": str(index), "explanation": f"{mine['frames']} frames, native {theirs['frames']}"})
        elif theirs["motion"] > 0 and mine["motion"] < MIN_MOTION_RATIO * theirs["motion"]:
            failures.append({"sample_id": str(index),
                             "explanation": f"frozen: motion {mine['motion']:.2f} vs native {theirs['motion']:.2f}"})
    count = len(problems)
    return {"suite": f"{name}-video-validity", "source": "task", "benchmark": "video validity (frames, motion)",
            "samples": count, "expected_samples": count, "passed": count - len(failures), "required_passes": count,
            "status": "pass" if not failures and len(candidate) == count else "fail", "failures": failures[:5],
            "reasons": [f"{len(failures)} of {count} videos invalid"] if failures else [],
            "gate": {"same_frames": True, "min_motion_ratio": MIN_MOTION_RATIO}}
