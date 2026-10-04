# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output comparators: does the TRTMC observation match the native one?

The Perf L1 output check compares the first output of each timed side with these (``COMPARATORS``,
named by ``output_grader`` in config/tasks.yaml). Each returns ``(match, reason, candidate summary,
reference summary)``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Sequence


def _first_divergence(left: list, right: list) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"length mismatch {len(left)} vs {len(right)}")
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


# Comparators return (match, reason, candidate_answer, reference_answer). They are shared by the
# AIPerf graders (Acc) and the orchestrator's Perf output check.
def compare_text(candidate: dict, reference: dict, **_: Any) -> tuple[bool, str, str, str]:
    left, right = " ".join(candidate["text"].split()), " ".join(reference["text"].split())
    return left == right, "text identical" if left == right else "text differs", left, right


def _words(text: str) -> list[str]:
    # Markup tokens such as language tags ("<en-US>") are not words.
    return re.sub(r"[^\w\s']", " ", re.sub(r"<[^<>\s]{1,32}>", " ", text).lower()).split()


def _edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def word_error_rate(hypothesis: str, reference: str) -> float:
    reference_words = _words(reference)
    distance = _edit_distance(_words(hypothesis), reference_words)
    return distance / len(reference_words) if reference_words else float(bool(_words(hypothesis)))


def compare_wer(candidate: dict, reference: dict, max_wer: float = 0.1, **_: Any) -> tuple[bool, str, str, str]:
    wer = word_error_rate(candidate["text"], reference["text"])
    return wer <= max_wer, f"WER to reference {wer:.4f} (max {max_wer})", candidate["text"], reference["text"]


def normalized_edit_distance(left: str, right: str) -> float:
    a, b = " ".join(left.lower().split()), " ".join(right.lower().split())
    return _edit_distance(a, b) / max(len(a), len(b), 1)


def compare_edit_distance(candidate: dict, reference: dict, max_distance: float = 0.15,
                          **_: Any) -> tuple[bool, str, str, str]:
    distance = normalized_edit_distance(candidate["text"], reference["text"])
    return (distance <= max_distance, f"normalized edit distance {distance:.4f} (max {max_distance})",
            candidate["text"], reference["text"])


def compare_answer_line(candidate: dict, reference: dict, stop: str = "\n", **_: Any) -> tuple[bool, str, str, str]:
    """Compare only the answer: text up to the first stop sequence (AIPerf's MMLU uses "\\n")."""
    left = candidate["text"].split(stop, 1)[0].strip()
    right = reference["text"].split(stop, 1)[0].strip()
    if left != right and candidate.get("token_ids") and candidate.get("token_ids") == reference.get("token_ids"):
        # Same generated tokens, different detokenization (for example special tokens kept in the text).
        return True, "token IDs identical; text rendering differs", left, right
    return left == right, "answer identical" if left == right else "answer differs", left, right


def compare_token_exact(candidate: dict, reference: dict, min_prefix: int | None = None,
                        accept_equal_text: bool = False, sampled: bool = False,
                        **_: Any) -> tuple[bool, str, str, str]:
    if "token_ids" not in candidate or "token_ids" not in reference:
        return compare_text(candidate, reference)
    if sampled:  # different random generators: only the generated length is comparable
        left, right = len(candidate["token_ids"]), len(reference["token_ids"])
        return left == right, f"sampled request: {left} vs {right} generated tokens", str(left), str(right)
    divergence = _first_divergence(candidate["token_ids"], reference["token_ids"])
    reason = ("token IDs identical" if divergence is None else
              f"first divergence at token {divergence} "
              f"({len(candidate['token_ids'])} vs {len(reference['token_ids'])} tokens)")
    # min_prefix (Perf output sanity checks only): agreeing on the first tokens is enough there.
    ok = divergence is None or (min_prefix is not None and divergence >= min_prefix)
    # accept_equal_text (Perf output checks and conversion parity): different token IDs for exactly the
    # same returned text (no whitespace normalization: whitespace is generated output too).
    if not ok and accept_equal_text and candidate.get("text") and str(candidate.get("text")) == str(reference.get("text")):
        ok, reason = True, reason + "; text identical"
    return ok, reason, candidate.get("text", ""), reference.get("text", "")


def _top_class(observation: dict) -> int:
    """Arg-max of the reported scores, or the reported top class when only that is returned."""
    scores = observation.get("scores")
    if isinstance(scores, list) and scores:
        return max(range(len(scores)), key=scores.__getitem__)
    return int(observation["top_class"])


def compare_top1(candidate: dict, reference: dict, tie_cosine: float | None = None,
                 **_: Any) -> tuple[bool, str, str, str]:
    """Top-1 class parity. ``tie_cosine`` (Perf output sanity checks only) accepts a different top
    class when the logits agree that closely, i.e. a near tie decided by precision."""
    top_left, top_right = _top_class(candidate), _top_class(reference)
    detail = f"top1 {top_left} vs {top_right}"
    ok = top_left == top_right
    if isinstance(candidate.get("scores"), list) and isinstance(reference.get("scores"), list):
        cosine = _cosine(candidate["scores"], reference["scores"])
        detail += f"; logit cosine {cosine:.5f}"
        if not ok and tie_cosine is not None and cosine >= tie_cosine:
            ok, detail = True, detail + " (near tie)"
    return ok, detail, str(top_left), str(top_right)


def _boxes(observation: dict) -> list[tuple[int, list[float]]]:
    """(class, xyxy) pairs from flat ``boxes`` or nested ``[[x1, y1, x2, y2], ...]``."""
    boxes, classes = observation["boxes"], observation["class_ids"]
    flat = [value for box in boxes for value in box] if boxes and isinstance(boxes[0], list) else boxes
    if len(flat) != 4 * len(classes):
        raise ValueError(f"{len(flat)} box values for {len(classes)} classes")
    return [(int(classes[i]), [float(v) for v in flat[4 * i: 4 * i + 4]]) for i in range(len(classes))]


def _iou(a: list[float], b: list[float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare_boxes(candidate: dict, reference: dict, min_iou: float = 0.5, min_precision: float = 0.9,
                  min_recall: float = 0.9, max_score_delta: float = 0.1, **_: Any) -> tuple[bool, str, str, str]:
    """Greedy same-class matching at IoU >= min_iou; both precision and recall must reach thresholds, and
    matched boxes must agree on confidence within ``max_score_delta`` when both sides report scores (a
    reordered ranking changes COCO mAP while boxes and classes still match)."""
    left, right = _boxes(candidate), _boxes(reference)
    if not left and not right:
        return True, "no detections on either side", "0", "0"
    unmatched = list(range(len(right)))
    pairs = []
    for index, (label, box) in enumerate(left):
        best = max(((j, _iou(box, right[j][1])) for j in unmatched if right[j][0] == label),
                   key=lambda item: item[1], default=(None, 0.0))
        if best[0] is not None and best[1] >= min_iou:
            unmatched.remove(best[0])
            pairs.append((index, best[0]))
    matched = len(pairs)
    precision = matched / len(left) if left else 0.0
    recall = matched / len(right) if right else 0.0
    ok = precision >= min_precision and recall >= min_recall
    detail = f"{matched} matched; precision {precision:.3f} recall {recall:.3f} at IoU {min_iou}"
    scores = [observation.get("scores") for observation in (candidate, reference)]
    if pairs and all(isinstance(values, list) and values for values in scores):
        delta = max(abs(float(scores[0][i]) - float(scores[1][j])) for i, j in pairs)
        ok = ok and delta <= max_score_delta
        detail += f"; max score difference {delta:.3f}"
    return ok, detail, str(len(left)), str(len(right))


def _binary_masks(observation: dict) -> list[list[bool]]:
    """Split flattened masks and binarize: logits at 0, probability/binary masks at 0.5."""
    count, area = int(observation["num_masks"]), int(observation["height"]) * int(observation["width"])
    values = observation["masks"]
    if len(values) != count * area:
        raise ValueError(f"{len(values)} mask values for {count} masks of {area} pixels")
    threshold = 0.0 if observation.get("mask_kind", "logits") == "logits" else 0.5
    return [[float(v) > threshold for v in values[i * area:(i + 1) * area]] for i in range(count)]


def _mask_iou(left: list[bool], right: list[bool]) -> float:
    inter = sum(1 for a, b in zip(left, right) if a and b)
    union = sum(1 for a, b in zip(left, right) if a or b)
    return inter / union if union else 1.0


def compare_mask(candidate: dict, reference: dict, min_pixel_accuracy: float = 0.99, min_mean_iou: float = 0.94,
                 min_mask_iou: float = 0.7, min_mask_match_rate: float = 1.0, **_: Any) -> tuple[bool, str, str, str]:
    """Semantic label map (``mask``): pixel accuracy and mean IoU over present classes.
    Prompted masks (``masks``, num_masks per image, ``mask_kind``): binarized, greedily matched by IoU."""
    if (candidate.get("height"), candidate.get("width")) != (reference.get("height"), reference.get("width")):
        return False, "mask resolution differs", str(candidate.get("height")), str(reference.get("height"))
    if "masks" in candidate or "masks" in reference:
        left, right = _binary_masks(candidate), _binary_masks(reference)
        taken: set[int] = set()
        ious = []
        for mask in left:  # greedy best-IoU matching, as in the Accuracy qualification
            choices = [(_mask_iou(mask, other), index) for index, other in enumerate(right) if index not in taken]
            if choices:
                iou, index = max(choices)
                taken.add(index)
                ious.append(iou)
        match_rate = len(ious) / max(len(left), len(right), 1)
        worst = min(ious, default=0.0)
        ok = match_rate >= min_mask_match_rate and worst >= min_mask_iou
        return (ok, f"mask match rate {match_rate:.2f}; IoU " + ", ".join(f"{v:.3f}" for v in ious),
                f"{worst:.4f}", str(min_mask_iou))
    left, right = candidate["mask"], reference["mask"]
    if len(left) != len(right):
        return False, f"{len(left)} vs {len(right)} pixels", "", ""
    pairs = Counter(zip(left, right))  # one pass: confusion counts
    predicted, expected = Counter(left), Counter(right)
    accuracy = sum(count for (a, b), count in pairs.items() if a == b) / len(right)
    ious = [pairs[(label, label)] / (predicted[label] + expected[label] - pairs[(label, label)])
            for label in set(predicted) | set(expected)]
    mean_iou = sum(ious) / len(ious)
    ok = accuracy >= min_pixel_accuracy and mean_iou >= min_mean_iou
    return ok, f"pixel accuracy {accuracy:.4f}, mean IoU {mean_iou:.4f}", f"{accuracy:.4f}", f"{mean_iou:.4f}"


VECTOR_COSINE_THRESHOLD = 0.99


def compare_vector(candidate: dict, reference: dict, min_cosine: float = VECTOR_COSINE_THRESHOLD,
                   **_: Any) -> tuple[bool, str, str, str]:
    cosine = _cosine(candidate["values"], reference["values"])
    return (cosine >= min_cosine, f"cosine {cosine:.5f} (threshold {min_cosine})",
            f"{cosine:.5f}", "1")


def compare_scores(candidate: dict, reference: dict, max_abs_diff: float | None = None,
                   **_: Any) -> tuple[bool, str, str, str]:
    """Reranking: identical document order; optionally bounded score differences."""
    left, right = [float(v) for v in candidate["scores"]], [float(v) for v in reference["scores"]]
    if len(left) != len(right):
        return False, f"{len(left)} vs {len(right)} scores", str(len(left)), str(len(right))
    order_left = sorted(range(len(left)), key=lambda i: (-left[i], i))
    order_right = sorted(range(len(right)), key=lambda i: (-right[i], i))
    worst = max((abs(a - b) for a, b in zip(left, right)), default=0.0)
    ok = order_left == order_right and (max_abs_diff is None or worst <= max_abs_diff)
    return ok, f"order {'identical' if order_left == order_right else 'differs'}; max |score diff| {worst:.4g}", \
        str(order_left), str(order_right)


_SKIPPED_NUMERIC = re.compile(r"(^|_)(ms|seconds|time|latency|elapsed|setup|inference|load|sha256|seed)(_|$)")


def _numeric_fields(value: Any, prefix: str = "") -> dict[str, list[float]]:
    """Flattened numeric fields (scalars and nested lists), excluding timing and identity fields."""
    fields: dict[str, list[float]] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if not _SKIPPED_NUMERIC.search(str(key)):
                fields.update(_numeric_fields(item, f"{prefix}{key}."))
        return fields
    flat: list[float] = []

    def walk(item: Any) -> bool:
        if isinstance(item, bool) or item is None:
            return False
        if isinstance(item, (int, float)):
            flat.append(float(item))
            return True
        if isinstance(item, list):
            return all(walk(child) for child in item)
        return False

    if walk(value) and flat:
        fields[prefix.rstrip(".")] = flat
    return fields


def compare_numeric(candidate: dict, reference: dict, min_cosine: float = 0.999, rtol: float = 0.05,
                    fields: Sequence[str] | None = None, count_rtol: float = 0.0,
                    **_: Any) -> tuple[bool, str, str, str]:
    """Generic tensor parity over the numeric fields both observations report.

    Short integer lists (shapes, counts) must match exactly, except that a single count may differ by
    ``count_rtol`` (e.g. pixels above a threshold); other vectors need cosine similarity of at least
    ``min_cosine``; scalars must agree within ``rtol``.
    """
    left, right = _numeric_fields(candidate), _numeric_fields(reference)
    names = [name for name in (fields or sorted(left.keys() & right.keys()))]
    if not names:
        raise ValueError(f"no common numeric fields (candidate {sorted(left)[:6]}, reference {sorted(right)[:6]})")
    problems, checked = [], []
    for name in names:
        a, b = left.get(name), right.get(name)
        if a is None or b is None:
            problems.append(f"{name} missing")
            continue
        integral = all(v.is_integer() for v in a + b) and len(a) <= 8
        if len(a) != len(b):
            problems.append(f"{name} length {len(a)} vs {len(b)}")
        elif integral and len(a) == 1 and count_rtol:
            if abs(a[0] - b[0]) > count_rtol * max(abs(b[0]), 1.0):
                problems.append(f"{name} {a} vs {b}")
        elif integral:
            if a != b:
                problems.append(f"{name} {a} vs {b}")
        elif len(a) == 1:
            if abs(a[0] - b[0]) > rtol * max(abs(b[0]), 1e-6):
                problems.append(f"{name} {a[0]:.5g} vs {b[0]:.5g}")
        else:
            cosine = _cosine(a, b)
            checked.append(f"{name} cos {cosine:.5f}")
            if not math.isfinite(cosine) or cosine < min_cosine:
                problems.append(f"{name} cosine {cosine:.5f} < {min_cosine}")
    detail = "; ".join(problems or checked or [f"{len(names)} fields identical"])
    return not problems, detail[:300], ",".join(names)[:200], str(len(names))


def _psnr(left: list[float], right: list[float]) -> float:
    mse = sum((a - b) ** 2 for a, b in zip(left, right)) / max(len(left), 1)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def _ssim(left: list[float], right: list[float]) -> float:
    """Global SSIM of two equally sized grayscale images (luminance, contrast, structure)."""
    n = max(len(left), 1)
    mean_a, mean_b = sum(left) / n, sum(right) / n
    var_a = sum((a - mean_a) ** 2 for a in left) / n
    var_b = sum((b - mean_b) ** 2 for b in right) / n
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(left, right)) / n
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mean_a * mean_b + c1) * (2 * cov + c2)) / ((mean_a ** 2 + mean_b ** 2 + c1) * (var_a + var_b + c2))


def _gray(rgb: list[int]) -> list[float]:
    return [0.299 * rgb[i] + 0.587 * rgb[i + 1] + 0.114 * rgb[i + 2] for i in range(0, len(rgb), 3)]


def compare_image(candidate: dict, reference: dict, min_psnr: float = 5.0, min_ssim: float = 0.1,
                  **_: Any) -> tuple[bool, str, str, str]:
    """Generated media: same geometry and frame count; thumbnail PSNR and SSIM above the bounds."""
    left, right = candidate["media_digest"], reference["media_digest"]
    geometry = [(d["frames"], d["height"], d["width"]) for d in (left, right)]
    if geometry[0] != geometry[1]:
        return False, f"geometry {geometry[0]} vs {geometry[1]}", str(geometry[0]), str(geometry[1])
    psnr = min(_psnr(a, b) for a, b in zip(left["thumbnails"], right["thumbnails"]))
    ssim = min(_ssim(_gray(a), _gray(b)) for a, b in zip(left["thumbnails"], right["thumbnails"]))
    ok = psnr >= min_psnr and ssim >= min_ssim
    return ok, f"thumbnail PSNR {psnr:.2f} dB, SSIM {ssim:.3f}", f"{psnr:.2f}", f"{ssim:.3f}"


def compare_audio(candidate: dict, reference: dict, min_duration_ratio: float = 0.8, max_duration_ratio: float = 1.2,
                  min_rms_ratio: float = 0.5, max_rms_ratio: float = 2.0, max_log_spectral_distance: float = 3.0,
                  **_: Any) -> tuple[bool, str, str, str]:
    """Generated audio: duration and loudness ratios, and the distance of the level-normalized spectra."""
    left, right = candidate["audio_digest"], reference["audio_digest"]
    duration = left["seconds"] / max(right["seconds"], 1e-9)
    rms = left["rms"] / max(right["rms"], 1e-12)
    a, b = left["log_spectrum"], right["log_spectrum"]
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    lsd = math.sqrt(sum(((x - mean_a) - (y - mean_b)) ** 2 for x, y in zip(a, b)) / len(a))
    ok = (min_duration_ratio <= duration <= max_duration_ratio and min_rms_ratio <= rms <= max_rms_ratio
          and lsd <= max_log_spectral_distance)
    return ok, f"duration ratio {duration:.3f}, RMS ratio {rms:.3f}, spectral distance {lsd:.2f} dB", \
        f"{left['seconds']:.2f}s", f"{right['seconds']:.2f}s"


COMPARATORS = {"parity_scores": compare_scores, "parity_numeric": compare_numeric,
               "parity_image": compare_image, "parity_audio": compare_audio, "parity_token_exact": compare_token_exact, "parity_text": compare_text,
               "parity_answer_line": compare_answer_line, "parity_wer": compare_wer,
               "parity_edit_distance": compare_edit_distance, "parity_boxes": compare_boxes,
               "parity_mask": compare_mask,
               "parity_top1": compare_top1, "parity_vector": compare_vector}
