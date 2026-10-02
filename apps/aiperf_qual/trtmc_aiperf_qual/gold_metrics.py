# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scores of task outputs against gold labels, for absolute-accuracy suites.

A binary metric grades each sample right or wrong (both sides are then compared with the paired
McNemar test). A corpus metric is a statistic over units (utterances, sentence pairs): each side's
units are scored, and a paired bootstrap over units gives the confidence interval of the difference.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any, Callable, Mapping, Sequence

BOOTSTRAP_SAMPLES = 1000
# Corpus statistics that cost seconds per evaluation resample less.
BOOTSTRAP_SAMPLES_BY_METRIC = {"chrf": 200, "miou": 200}


def _text(observation: Mapping[str, Any] | None) -> str:
    return str((observation or {}).get("text") or "")


# ---------------- binary metrics: (gold, observation, task) -> (correct, extracted answer) ----------------

_LETTER = re.compile(r"(?<![A-Za-z])\(?([A-D])\)?(?![A-Za-z])")


def choice(gold: Any, observation: Mapping[str, Any] | None, task: str | None = None) -> tuple[bool, str]:
    """A multiple-choice letter: the first standalone A-D (``B``, ``(B)``, ``B.``, ``Answer: B``)."""
    match = _LETTER.search(_text(observation))
    answer = match.group(1) if match else ""
    return bool(answer) and answer == str(gold).strip().upper(), answer


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


FORMULA_TASKS = {"Handwritten Mathematical Expression Recognition"}


def contains(gold: Any, observation: Mapping[str, Any] | None, task: str | None = None) -> tuple[bool, str]:
    """OCRBench scoring: some gold answer occurs in the prediction (case- and space-insensitive;
    formulas ignore spaces entirely, as the official script does)."""
    squash = (lambda text: re.sub(r"\s+", "", text.strip().lower())) if task in FORMULA_TASKS else _squash
    prediction = squash(_text(observation))
    answers = gold if isinstance(gold, list) else [gold]
    correct = any(squash(str(answer)) and squash(str(answer)) in prediction for answer in answers)
    return correct, prediction[:200]


CODE_TIMEOUT_S = 10.0
CODE_STOPS = ("\nclass ", "\ndef ", "\n#", "\nif ", "\nprint(", "\nassert ")


def _limits() -> None:  # in the child: bounded CPU time and memory, no core dumps
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (int(CODE_TIMEOUT_S), int(CODE_TIMEOUT_S)))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 2**30, 4 * 2**30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def code_pass(gold: Any, observation: Mapping[str, Any] | None, task: str | None = None) -> tuple[bool, str]:
    """HumanEval: the prompt plus the generated body passes the problem's unit tests. The program runs
    isolated (``python -I`` in an empty temporary directory, CPU, memory, and wall-time limits)."""
    import subprocess
    import sys
    import tempfile

    completion = _text(observation)
    cut = min((completion.find(stop) for stop in CODE_STOPS if completion.find(stop) >= 0), default=len(completion))
    completion = completion[:cut]  # the function body only (HumanEval's stop sequences)
    program = f"{gold['prompt']}{completion}\n\n{gold['test']}\n\ncheck({gold['entry_point']})\n"
    with tempfile.TemporaryDirectory(prefix="trtmc-code-") as directory:
        try:
            result = subprocess.run([sys.executable, "-I", "-c", program], cwd=directory, capture_output=True,
                                    timeout=CODE_TIMEOUT_S + 5, preexec_fn=_limits, env={"PATH": "/usr/bin:/bin"})
            passed = result.returncode == 0
        except subprocess.TimeoutExpired:
            passed = False
    return passed, completion[:200]


def top1(gold: Any, observation: Mapping[str, Any] | None, task: str | None = None) -> tuple[bool, str]:
    """Image classification: the predicted class (``top_class``, else the arg-max of the class scores)
    is the gold class index."""
    observation = observation or {}
    if observation.get("top_class") is not None:
        best = int(observation["top_class"])
    else:
        scores = observation.get("scores") or []
        if not scores:
            return False, ""
        best = max(range(len(scores)), key=lambda index: float(scores[index]))
    return best == int(gold), str(best)


_BOX = re.compile(r"<box>\s*<?(-?[\d.]+)>?\s*<?(-?[\d.]+)>?\s*<?(-?[\d.]+)>?\s*<?(-?[\d.]+)>?\s*</box>")


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_iou50(gold: Any, observation: Mapping[str, Any] | None, task: str | None = None) -> tuple[bool, str]:
    """Referring-expression grounding: the first ``<box>`` (x1, y1, x2, y2 on a 0-1000 grid) overlaps the
    gold box (pixel x, y, width, height of an image of ``image_size``) with IoU >= 0.5."""
    match = _BOX.search(_text(observation))
    if not match:
        return False, ""
    predicted = [float(value) for value in match.groups()]
    (x, y, w, h), (width, height) = gold["value"], gold["image_size"]
    target = [1000.0 * x / width, 1000.0 * y / height, 1000.0 * (x + w) / width, 1000.0 * (y + h) / height]
    return _iou(predicted, target) >= 0.5, match.group(0)


BINARY: dict[str, Callable[..., tuple[bool, str]]] = {"choice": choice, "contains": contains, "code_pass": code_pass,
                                                      "top1": top1, "box_iou50": box_iou50}


# ---------------- corpus metrics ----------------

def _words(text: str) -> list[str]:
    """Lower-cased words without punctuation (a light normalization applied to both sides alike)."""
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1]


def wer_units(problems: Sequence[Mapping[str, Any]], observations: Mapping[int, Any]) -> list[tuple[float, float]]:
    """Per utterance: (word errors, reference words)."""
    units = []
    for index, problem in enumerate(problems):
        reference = _words(str(problem["gold"]))
        units.append((float(_edit_distance(reference, _words(_text(observations.get(index))))), float(len(reference))))
    return units


def wer(units: Sequence[tuple[float, float]]) -> float:
    """Corpus word error rate in percent."""
    words = sum(count for _, count in units)
    return 100.0 * sum(errors for errors, _ in units) / words if words else 0.0


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def sts_units(problems: Sequence[Mapping[str, Any]], observations: Mapping[int, Any]) -> list[tuple[float, float]]:
    """Per sentence pair (two consecutive samples labelled with the pair's gold score): (cosine, gold)."""
    units = []
    for index in range(0, len(problems) - 1, 2):
        first, second = observations.get(index) or {}, observations.get(index + 1) or {}
        units.append((_cosine(first.get("values") or [], second.get("values") or []), float(problems[index]["gold"])))
    return units


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        for k in range(position, end + 1):  # ties share their average rank
            ranks[order[k]] = (position + end) / 2.0
        position = end + 1
    return ranks


def spearman(units: Sequence[tuple[float, float]]) -> float:
    """Spearman correlation x 100 of the predicted similarity with the gold score."""
    if len(units) < 3:
        return 0.0
    x, y = _ranks([unit[0] for unit in units]), _ranks([unit[1] for unit in units])
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    norm = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return 100.0 * cov / norm if norm else 0.0


def _ndcg(scores: Sequence[float], relevant: Sequence[int], k: int) -> float:
    order = sorted(range(len(scores)), key=lambda index: -float(scores[index]))[:k]
    gain = sum(1.0 / math.log2(rank + 2) for rank, index in enumerate(order) if index in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return gain / ideal if ideal else 0.0


def rerank_units(problems: Sequence[Mapping[str, Any]], observations: Mapping[int, Any]) -> list[float]:
    """Per query: nDCG over its candidate documents, ranked by the model's scores."""
    units = []
    for index, problem in enumerate(problems):
        scores = (observations.get(index) or {}).get("scores") or []
        units.append(_ndcg(scores, list(problem["gold"]), max(1, len(scores))))
    return units


def retrieval_units(problems: Sequence[Mapping[str, Any]], observations: Mapping[int, Any]) -> list[float]:
    """Per query (a ``query`` sample followed by its ``document`` samples): nDCG of the documents ranked
    by embedding cosine with the query."""
    units, index = [], 0
    while index < len(problems):
        if problems[index].get("task") != "query":
            index += 1
            continue
        end = index + 1
        while end < len(problems) and problems[end].get("task") == "document":
            end += 1
        query = (observations.get(index) or {}).get("values") or []
        scores = [_cosine(query, (observations.get(position) or {}).get("values") or []) for position in range(index + 1, end)]
        units.append(_ndcg(scores, list(problems[index]["gold"]), max(1, len(scores))))
        index = end
    return units


def mean_percent(units: Sequence[float]) -> float:
    return 100.0 * sum(units) / len(units) if units else 0.0


def point_forecast(observation: Mapping[str, Any] | None, length: int) -> list[float]:
    """The point forecast of an output with ``length`` values: the values themselves, or the median
    row of a quantile forecast ([1, quantiles, horizon], e.g. Chronos-Bolt's nine)."""
    values = [float(value) for value in (observation or {}).get("values") or []]
    if len(values) == length:
        return values
    shape = (observation or {}).get("shape") or []
    if len(shape) == 3 and len(values) == shape[1] * shape[2] and shape[2] == length:
        middle = shape[1] // 2
        return values[middle * length:(middle + 1) * length]
    return []


def forecast_units(problems: Sequence[Mapping[str, Any]], observations: Mapping[int, Any]) -> list[tuple[float, float]]:
    """Per window: (squared error summed over the horizon, values); a missing forecast counts its full
    squared target as error."""
    units = []
    for index, problem in enumerate(problems):
        gold = [float(value) for value in problem["gold"]]
        forecast = point_forecast(observations.get(index), len(gold)) or [0.0] * len(gold)
        units.append((sum((a - b) ** 2 for a, b in zip(forecast, gold)), float(len(gold))))
    return units


def mse(units: Sequence[tuple[float, float]]) -> float:
    count = sum(n for _, n in units)
    return sum(error for error, _ in units) / count if count else 0.0


def translation_units(problems: Sequence[Mapping[str, Any]], observations: Mapping[int, Any]) -> list[tuple[str, str]]:
    """Per sentence: (the model's translation, the reference)."""
    return [(_text(observations.get(index)).strip(), str(problem["gold"])) for index, problem in enumerate(problems)]


def chrf(units: Sequence[tuple[str, str]]) -> float:
    """Corpus chrF++ (sacreBLEU, word order 2)."""
    import sacrebleu

    return sacrebleu.corpus_chrf([unit[0] for unit in units], [[unit[1] for unit in units]], word_order=2).score


def mask_iou_units(problems: Sequence[Mapping[str, Any]], observations: Mapping[int, Any]) -> list[float]:
    """Per object: the IoU of the highest-scoring predicted mask (``masks``: num_masks x H x W logits or
    0/1, ``iou_scores``) with the gold object mask (a polygon on the image)."""
    import numpy as np

    from .suites import polygon_mask

    units = []
    for index, problem in enumerate(problems):
        gold = polygon_mask(problem["gold"]["polygon"], problem["gold"]["image_size"])
        observation = observations.get(index) or {}
        values = np.asarray(observation.get("masks") or [], dtype=np.float32)
        height, width = int(observation.get("height") or 0), int(observation.get("width") or 0)
        count = values.size // (height * width) if height and width else 0
        if not count:
            units.append(0.0)
            continue
        scores = list(observation.get("iou_scores") or [0.0] * count)[:count]
        best = int(np.argmax(scores)) if scores else 0
        predicted = values.reshape(count, height, width)[best] > 0
        if predicted.shape != gold.shape:
            rows = np.arange(gold.shape[0]) * height // gold.shape[0]
            columns = np.arange(gold.shape[1]) * width // gold.shape[1]
            predicted = predicted[rows][:, columns]
        union = np.logical_or(predicted, gold).sum()
        units.append(float(np.logical_and(predicted, gold).sum() / union) if union else 0.0)
    return units


ADE_CLASSES = 150


def miou_units(problems: Sequence[Mapping[str, Any]], observations: Mapping[int, Any]) -> list[tuple[Any, Any]]:
    """Per image: (intersection, union) per ADE20K class of the predicted class map (``mask``, class index
    0-149) against the annotation (1-150, 0 ignored). A prediction of another size is resized nearest."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    units = []
    for index, problem in enumerate(problems):
        gold = np.asarray(Image.open(io.BytesIO(base64.b64decode(problem["gold"]["png_b64"]))), dtype=np.int64)
        observation = observations.get(index) or {}
        mask = np.asarray(observation.get("mask") or [], dtype=np.int64)
        height, width = int(observation.get("height") or 0), int(observation.get("width") or 0)
        if mask.size != height * width or not mask.size:
            predicted = np.full(gold.shape, -1, dtype=np.int64)
        else:
            predicted = mask.reshape(height, width) + 1
            if predicted.shape != gold.shape:
                rows = (np.arange(gold.shape[0]) * height // gold.shape[0])
                columns = (np.arange(gold.shape[1]) * width // gold.shape[1])
                predicted = predicted[rows][:, columns]
        valid = gold > 0
        intersection = np.bincount(gold[valid & (predicted == gold)] - 1, minlength=ADE_CLASSES)[:ADE_CLASSES]
        area_gold = np.bincount(gold[valid] - 1, minlength=ADE_CLASSES)[:ADE_CLASSES]
        hits = predicted[valid]
        area_predicted = np.bincount(np.clip(hits - 1, 0, ADE_CLASSES - 1)[(hits >= 1) & (hits <= ADE_CLASSES)],
                                     minlength=ADE_CLASSES)[:ADE_CLASSES]
        units.append((intersection, area_gold + area_predicted - intersection))
    return units


def miou(units: Sequence[tuple[Any, Any]]) -> float:
    """Mean IoU x 100 over the classes present (union > 0) in the corpus."""
    import numpy as np

    if not units:
        return 0.0
    intersection = np.sum([unit[0] for unit in units], axis=0)
    union = np.sum([unit[1] for unit in units], axis=0)
    present = union > 0
    return float(100.0 * np.mean(intersection[present] / union[present])) if present.any() else 0.0


# COCO category ids of the 80 contiguous class indices.
COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
LABEL_FIELDS = {"coco-contiguous-80": "category_index", "coco-category-id": "category_id"}


def coco_units(problems: Sequence[Mapping[str, Any]], observations: Mapping[int, Any],
               params: Mapping[str, Any]) -> list[tuple[dict, dict, str]]:
    """Per image: its gold annotations (``objects`` of detection-datasets/coco: xyxy boxes, contiguous
    categories), the model's detections, and the label space they are compared in."""
    field = LABEL_FIELDS[params.get("label_space", "coco-contiguous-80")]
    units = []
    for index, problem in enumerate(problems):
        objects = problem["gold"]
        annotations = [{"box": [float(v) for v in box], "category_index": int(category),
                        "category_id": COCO_IDS[int(category)]}
                       for box, category in zip(objects["bbox"], objects["category"])]
        output = observations.get(index) or {"boxes": [], "scores": [], "class_ids": []}
        units.append(({"sample_id": problem.get("sample_id", str(index)), "annotations": annotations}, output, field))
    return units


def coco_map(units: Sequence[tuple[dict, dict, str]]) -> float:
    """COCO mAP@[.5:.95] x 100 (the family qualification's AP, all images, 100 detections each)."""
    from qualification_tests.benchmark_qualification.accuracy import _coco_detection_metrics

    if not units:
        return 0.0
    samples, outputs = [unit[0] for unit in units], [unit[1] for unit in units]
    return 100.0 * _coco_detection_metrics(samples, outputs, units[0][2])["map_50_95"]


def precomputed_units(problems: Sequence[Mapping[str, Any]], observations: Mapping[int, Any]) -> list[float]:
    """Per sample: a score a check computed already (``value``, in points), e.g. CLIP-I to a target."""
    return [float((observations.get(index) or {}).get("value") or 0.0) for index in range(len(problems))]


def mean(units: Sequence[float]) -> float:
    return sum(units) / len(units) if units else 0.0


# name -> (units of a side, statistic over units, higher is better); NO_BOOTSTRAP statistics are too
# costly to resample (the gate then rests on the difference alone).
NO_BOOTSTRAP = {"coco_map"}
PARAMETRIC = {"coco_map"}  # unit functions that take the benchmark's ``metric_params``
CORPUS: dict[str, tuple[Callable, Callable, bool]] = {"chrf": (translation_units, chrf, True),
                                                      "coco_map": (coco_units, coco_map, True),
                                                      "forecast_mse": (forecast_units, mse, False),
                                                      "miou": (miou_units, miou, True),
                                                      "mask_iou": (mask_iou_units, mean_percent, True),
                                                      "precomputed_mean": (precomputed_units, mean, True),
                                                      "wer": (wer_units, wer, False),
                                                      "sts_spearman": (sts_units, spearman, True),
                                                      "rerank_ndcg": (rerank_units, mean_percent, True),
                                                      "retrieval_ndcg": (retrieval_units, mean_percent, True)}


def compare_corpus(metric: str, problems: Sequence[Mapping[str, Any]], candidate: Mapping[int, Any],
                   native: Mapping[int, Any], params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Both sides' statistic, their difference (TRTMC - native, in the metric's points), and a paired
    bootstrap 95% interval of the difference; ``worse`` when the interval excludes zero on TRTMC's
    losing side."""
    units_of, statistic, higher = CORPUS[metric]
    if metric in PARAMETRIC:
        mine, theirs = units_of(problems, candidate, params or {}), units_of(problems, native, params or {})
    else:
        mine, theirs = units_of(problems, candidate), units_of(problems, native)
    mine_value, theirs_value = statistic(mine), statistic(theirs)
    delta = mine_value - theirs_value
    if metric in NO_BOOTSTRAP:
        return {"trtmc": round(mine_value, 3), "native": round(theirs_value, 3), "delta_points": round(delta, 3),
                "ci95": None, "significantly_worse": False, "higher_is_better": higher, "units": len(mine)}
    generator, deltas = random.Random(0), []
    for _ in range(BOOTSTRAP_SAMPLES_BY_METRIC.get(metric, BOOTSTRAP_SAMPLES)):
        picks = [generator.randrange(len(mine)) for _ in range(len(mine))]
        deltas.append(statistic([mine[i] for i in picks]) - statistic([theirs[i] for i in picks]))
    deltas.sort()
    low, high = deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas)) - 1]
    worse = high < 0 if higher else low > 0
    return {"trtmc": round(mine_value, 3), "native": round(theirs_value, 3), "delta_points": round(delta, 3),
            "ci95": [round(low, 3), round(high, 3)], "significantly_worse": worse, "higher_is_better": higher,
            "units": len(mine)}
