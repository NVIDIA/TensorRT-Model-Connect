# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scores of task outputs against gold labels, for absolute-accuracy suites.

A binary metric grades each sample right or wrong (both sides are then compared with the paired
score test, ``noninferiority.binary``). A corpus metric is a statistic over units (utterances, sentence
pairs): each side's units are scored, and a paired bootstrap over independent clusters of units decides
(``noninferiority.bootstrap``).
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import noninferiority

BOOTSTRAP_SAMPLES = 2000
# Corpus statistics that cost seconds per evaluation resample less.
BOOTSTRAP_SAMPLES_BY_METRIC = {"coco_map": 200, "miou": 200}


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


CODE_LIMITS = {"RLIMIT_CPU": int(CODE_TIMEOUT_S), "RLIMIT_AS": 4 * 2**30, "RLIMIT_CORE": 0,
               "RLIMIT_FSIZE": 16 * 2**20, "RLIMIT_NPROC": 64, "RLIMIT_NOFILE": 64}


def _limits() -> None:  # in the child: bounded CPU time, memory, file sizes, processes, and open files
    import resource

    for name, value in CODE_LIMITS.items():
        resource.setrlimit(getattr(resource, name), (value, value))


# human-eval's reliability guard (github.com/openai/human-eval, execution.py, MIT), run before the program:
# destructive and process-spawning functions are disabled. (``help`` through ``builtins``: in a ``-c``
# program ``__builtins__`` is the module.) Not a security boundary; the user, limits, and permissions are.
RELIABILITY_GUARD = """
def _reliability_guard():
    import builtins, faulthandler, os, shutil, subprocess, sys
    faulthandler.disable()
    builtins.exit = None
    builtins.quit = None
    builtins.help = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork", "forkpty",
                 "killpg", "rename", "renames", "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown",
                 "chroot", "lchflags", "lchmod", "lchown", "getcwd", "chdir", "setsid", "setpgid", "posix_spawn",
                 "posix_spawnp"):
        setattr(os, name, None)
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    subprocess.Popen = None
    for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[name] = None
_reliability_guard()
del _reliability_guard
"""


NOBODY = 65534
# Paths the generated programs must not write (the harness adds the environment's roots) or read.
PROTECTED_WRITE: set[str] = set()
PROTECTED_READ = ("/root",)
SETPRIV = ["setpriv", f"--reuid={NOBODY}", f"--regid={NOBODY}", "--clear-groups", "--no-new-privs",
           "--inh-caps=-all", "--bounding-set=-all"]


def protect(paths: Sequence[Any]) -> None:
    """Directories the generated programs must not be able to write (result, bundle, cache, data roots)."""
    PROTECTED_WRITE.update(str(path) for path in paths if path)


def _sandbox_verified(protected: Sequence[str]) -> None:
    """The DESIGN.md Section 8 boundary holds, checked before every program: programs run as ``nobody``
    with no capabilities, and that user can neither write the protected roots nor read root's home.
    Raises otherwise (fail closed)."""
    import subprocess

    for path in protected:
        if Path(path).exists() and subprocess.run([*SETPRIV, "test", "-w", path]).returncode == 0:
            raise RuntimeError(f"code sandbox: nobody can write {path}")
    for path in PROTECTED_READ:
        if Path(path).exists() and subprocess.run([*SETPRIV, "ls", path], capture_output=True).returncode == 0:
            raise RuntimeError(f"code sandbox: nobody can read {path}")
    if subprocess.run([*SETPRIV, "true"]).returncode != 0:
        raise RuntimeError("code sandbox: cannot switch to the nobody user (setpriv)")


def code_pass(gold: Any, observation: Mapping[str, Any] | None, task: str | None = None) -> tuple[bool, str]:
    """HumanEval / MBPP: the prompt plus the generated code passes the problem's unit tests. The program
    runs as ``nobody`` (no capabilities, an empty environment and a temporary home and directory) under
    human-eval's reliability guard and CPU, memory, file-size, process, open-file, and wall-time limits, in
    its own process group (killed whole after the run); it passes only when it exits 0 after printing a
    per-run nonce that follows the tests (an early ``exit(0)`` fails). Fails closed when the sandbox
    boundary does not hold."""
    import signal
    import subprocess
    import sys
    import tempfile
    import uuid

    _sandbox_verified(sorted(PROTECTED_WRITE))
    completion = _text(observation)
    stops = gold.get("stops") or CODE_STOPS
    cut = min((completion.find(stop) for stop in stops if completion.find(stop) > 0), default=len(completion))
    completion = completion[:cut]  # up to the benchmark's stop sequences
    nonce = uuid.uuid4().hex
    tests = gold.get("test_code") or f"{gold['test']}\n\ncheck({gold['entry_point']})"
    program = f"{RELIABILITY_GUARD}\n{gold['prompt']}{completion}\n\n{tests}\n\nprint({nonce!r})\n"
    with tempfile.TemporaryDirectory(prefix="trtmc-code-") as directory:
        os.chown(directory, NOBODY, NOBODY)
        process = subprocess.Popen([*SETPRIV, sys.executable, "-I", "-c", program], cwd=directory, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=_limits,
                                   start_new_session=True, env={"PATH": "/usr/bin:/bin", "HOME": directory})
        try:
            stdout, _ = process.communicate(timeout=CODE_TIMEOUT_S + 5)
            passed = process.returncode == 0 and stdout.rstrip().endswith(nonce)
        except subprocess.TimeoutExpired:
            passed = False
        finally:
            try:  # the program and anything it started
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
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


# ---------------- conversion parity: (TRTMC observation, native observation, gate) -> (match, reason) -------
# ``match`` is None when a side's output is missing or unreadable: missing evidence (an error), never a failure.

def vector_parity(candidate: Mapping[str, Any] | None, native: Mapping[str, Any] | None,
                  gate: Mapping[str, Any]) -> tuple[bool | None, str]:
    """An encoder output vector (``values``) against the native one: equal length, finite values, cosine
    at least ``min_cosine`` and relative L2 error at most ``max_relative_l2`` (magnitude, not only direction)."""
    mine = [float(value) for value in (candidate or {}).get("values") or []]
    theirs = [float(value) for value in (native or {}).get("values") or []]
    if not mine or not theirs:
        return None, f"no output vector ({len(mine)} vs {len(theirs)} values)"
    if len(mine) != len(theirs):
        return False, f"shape differs: {len(mine)} vs {len(theirs)} values"
    if not all(math.isfinite(value) for value in mine):
        return False, "non-finite values"
    cosine = _cosine(mine, theirs)
    norm = math.sqrt(sum(value * value for value in theirs)) or 1.0
    relative = math.sqrt(sum((a - b) ** 2 for a, b in zip(mine, theirs))) / norm
    ok = cosine >= float(gate.get("min_cosine", 0.999)) and relative <= float(gate.get("max_relative_l2", 0.02))
    return ok, f"cosine {cosine:.5f}, relative L2 {relative:.4f}"


def geometry_parity(candidate: Mapping[str, Any] | None, native: Mapping[str, Any] | None,
                    gate: Mapping[str, Any]) -> tuple[bool | None, str]:
    """Monocular geometry (the ``depth_artifact`` / ``valid_mask_artifact`` files both sides write): equal
    size, valid-mask IoU at least ``min_mask_iou``, and the median relative depth error over the pixels
    valid on both sides at most ``max_median_relative_depth``."""
    import numpy as np

    candidate, native = candidate or {}, native or {}
    shapes = [(int(side.get("height") or 0), int(side.get("width") or 0)) for side in (candidate, native)]
    if not all(shapes[0]) or not all(shapes[1]):
        return None, f"no geometry size: {shapes[0][0]}x{shapes[0][1]} vs {shapes[1][0]}x{shapes[1][1]}"
    try:  # each side's files at its own size: missing evidence first, a size difference after
        depths = [np.fromfile(side["depth_artifact"], dtype="<f4").reshape(shape) for side, shape in zip((candidate, native), shapes)]
        masks = [np.fromfile(side["valid_mask_artifact"], dtype=np.uint8).reshape(shape).astype(bool)
                 for side, shape in zip((candidate, native), shapes)]
    except (KeyError, OSError, ValueError) as error:
        return None, f"geometry artifacts unreadable: {error}"
    if shapes[0] != shapes[1]:
        return False, f"size differs: {shapes[0][0]}x{shapes[0][1]} vs {shapes[1][0]}x{shapes[1][1]}"
    union = np.logical_or(*masks).sum()
    iou = float(np.logical_and(*masks).sum() / union) if union else 1.0
    both = np.logical_and(*masks) & np.isfinite(depths[0]) & (depths[1] > 0)
    relative = float(np.median(np.abs(depths[0][both] - depths[1][both]) / depths[1][both])) if both.any() else 1.0
    ok = iou >= float(gate.get("min_mask_iou", 0.99)) and relative <= float(gate.get("max_median_relative_depth", 0.01))
    return ok, f"mask IoU {iou:.4f}, median relative depth error {relative:.4f}"


def action_parity(candidate: Mapping[str, Any] | None, native: Mapping[str, Any] | None,
                  gate: Mapping[str, Any]) -> tuple[bool | None, str]:
    """A robot action chunk (``actions``, row-major [step, component]) against the native one: equal shape,
    finite values, and the largest absolute error at most ``max_error_of_range`` of the native chunk's range."""
    mine = [float(value) for value in (candidate or {}).get("actions") or []]
    theirs = [float(value) for value in (native or {}).get("actions") or []]
    if not mine or not theirs:
        return None, f"no action chunk ({len(mine)} vs {len(theirs)} values)"
    if len(mine) != len(theirs):
        return False, f"shape differs: {len(mine)} vs {len(theirs)} values"
    if not all(math.isfinite(value) for value in mine):
        return False, "non-finite actions"
    span = (max(theirs) - min(theirs)) or 1.0
    error = max(abs(a - b) for a, b in zip(mine, theirs))
    return error <= float(gate.get("max_error_of_range", 1e-3)) * span, f"max abs error {error:.3g} (range {span:.3g})"


def disparity_parity(candidate: Mapping[str, Any] | None, native: Mapping[str, Any] | None,
                     gate: Mapping[str, Any]) -> tuple[bool | None, str]:
    """A stereo disparity map (the ``disparity_artifact`` file both sides write) against the native one: equal
    size and a mean end-point error between the sides of at most ``max_mean_epe`` pixels."""
    import numpy as np

    candidate, native = candidate or {}, native or {}
    shapes = [(int(side.get("height") or 0), int(side.get("width") or 0)) for side in (candidate, native)]
    if not all(shapes[0]) or not all(shapes[1]):
        return None, f"no disparity size: {shapes[0][0]}x{shapes[0][1]} vs {shapes[1][0]}x{shapes[1][1]}"
    try:  # each side's file at its own size: missing evidence first, a size difference after
        mine, theirs = (np.fromfile(side["disparity_artifact"], dtype="<f4").reshape(shape)
                        for side, shape in zip((candidate, native), shapes))
    except (KeyError, OSError, ValueError) as error:
        return None, f"disparity artifacts unreadable: {error}"
    if shapes[0] != shapes[1]:
        return False, f"size differs: {shapes[0][0]}x{shapes[0][1]} vs {shapes[1][0]}x{shapes[1][1]}"
    if not np.isfinite(mine).all():
        return False, "non-finite disparities"
    error = float(np.abs(mine - theirs).mean())
    return error <= float(gate.get("max_mean_epe", 0.1)), f"mean end-point error {error:.4f} px"


def audio_parity(candidate: Mapping[str, Any] | None, native: Mapping[str, Any] | None,
                 gate: Mapping[str, Any]) -> tuple[bool | None, str]:
    """Generated speech against the native output (the ``audio_digest`` the servers attach): duration and RMS
    ratios and the log-spectral distance within the gate (the plugins' ``parity_audio`` comparator)."""
    from trtmc_aiperf_plugins.accuracy import COMPARATORS

    try:
        match, reason, _, _ = COMPARATORS["parity_audio"](candidate or {}, native or {}, **dict(gate))
    except (KeyError, TypeError, ValueError) as error:  # a side's audio digest is missing or malformed
        return None, f"not comparable: {error}"
    return match, reason


PARITY: dict[str, Callable[..., tuple[bool | None, str]]] = {"vector_parity": vector_parity, "geometry_parity": geometry_parity,
                                                      "action_parity": action_parity, "disparity_parity": disparity_parity,
                                                      "audio_parity": audio_parity}
# Parity metrics that read the output files the servers write (their artifacts are kept).
ARTIFACT_METRICS = {"geometry_parity", "disparity_parity"}


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
    """Per query: nDCG@10 of its fixed candidate documents (BM25 top-20), ranked by the model's scores."""
    units = []
    for index, problem in enumerate(problems):
        scores = (observations.get(index) or {}).get("scores") or []
        units.append(_ndcg(scores, list(problem["gold"]), 10))
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


def corpus_retrieval_units(problems: Sequence[Mapping[str, Any]], observations: Mapping[int, Any]) -> list[float]:
    """Per query: nDCG@10 of the whole corpus (every ``document`` sample) ranked by embedding cosine with the
    query (``query`` samples; their gold lists the relevant documents' positions among the documents)."""
    import numpy as np

    queries = [index for index, problem in enumerate(problems) if problem.get("task") == "query"]
    documents = [index for index, problem in enumerate(problems) if problem.get("task") == "document"]

    def matrix(indices: Sequence[int]) -> Any:
        rows = [np.asarray((observations.get(index) or {}).get("values") or [], dtype=np.float32) for index in indices]
        width = max((row.size for row in rows), default=0)
        stacked = np.stack([row if row.size == width else np.zeros(width, np.float32) for row in rows]) if rows else \
            np.zeros((0, 0), np.float32)
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        return stacked / np.where(norms > 0, norms, 1.0)

    if not queries or not documents:
        return [0.0] * len(queries)
    scores = matrix(queries) @ matrix(documents).T if matrix(queries).shape[1] == matrix(documents).shape[1] else \
        np.zeros((len(queries), len(documents)), np.float32)
    return [_ndcg(scores[row].tolist(), list(problems[query]["gold"]), 10) for row, query in enumerate(queries)]


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
    """Per window: (squared error of the scaled values summed over the horizon, values). Values are
    row-major [time, column], scaled by each column's training mean and standard deviation; a missing
    forecast counts its full squared scaled target as error."""
    units = []
    for index, problem in enumerate(problems):
        gold = problem["gold"]
        values, mean, std = [float(value) for value in gold["values"]], gold["mean"], gold["std"]
        forecast = point_forecast(observations.get(index), len(values))
        scaled = [((value - mean[i % len(mean)]) / (std[i % len(std)] or 1.0)) for i, value in enumerate(values)]
        predicted = ([((value - mean[i % len(mean)]) / (std[i % len(std)] or 1.0)) for i, value in enumerate(forecast)]
                     if forecast else [0.0] * len(values))
        units.append((sum((a - b) ** 2 for a, b in zip(predicted, scaled)), float(len(values))))
    return units


def mse(units: Sequence[tuple[float, float]]) -> float:
    count = sum(n for _, n in units)
    return sum(error for error, _ in units) / count if count else 0.0


def _chrf_metric() -> Any:
    from sacrebleu.metrics import CHRF

    return CHRF(word_order=2)


def translation_units(problems: Sequence[Mapping[str, Any]], observations: Mapping[int, Any]) -> list[list[float]]:
    """Per sentence: its chrF++ statistics (character and word n-gram matches) against the reference."""
    hypotheses = [_text(observations.get(index)).strip() for index in range(len(problems))]
    references = [str(problem["gold"]) for problem in problems]
    return [list(stats) for stats in _chrf_metric()._extract_corpus_statistics(hypotheses, [references])]


def chrf(units: Sequence[Sequence[float]]) -> float:
    """Corpus chrF++ (sacreBLEU 2.5, word order 2) from summed sentence statistics, as corpus_chrf."""
    if not units:
        return 0.0
    return _chrf_metric()._compute_score_from_stats([sum(column) for column in zip(*units)]).score


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
    categories, areas), the model's detections, and the label space they are compared in."""
    field = LABEL_FIELDS[params.get("label_space", "coco-contiguous-80")]
    units = []
    for index, problem in enumerate(problems):
        objects = problem["gold"]
        areas = objects.get("area") or [None] * len(objects["bbox"])
        annotations = [{"box": [float(v) for v in box], "category_index": int(category),
                        "category_id": COCO_IDS[int(category)], **({"area": float(area)} if area is not None else {})}
                       for box, category, area in zip(objects["bbox"], objects["category"], areas)]
        output = observations.get(index) or {"boxes": [], "scores": [], "class_ids": []}
        units.append(({"sample_id": problem.get("sample_id", str(index)), "annotations": annotations}, output, field))
    return units


def coco_map(units: Sequence[tuple[dict, dict, str]]) -> float:
    """COCO mAP@[.5:.95] x 100 by pycocotools' COCOeval (all images, at most 100 detections each)."""
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not units:
        return 0.0
    field = units[0][2]
    images, annotations, detections = [], [], []
    for image_id, (sample, output, _) in enumerate(units, start=1):
        images.append({"id": image_id})
        for annotation in sample["annotations"]:
            x0, y0, x1, y1 = annotation["box"]
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": annotation[field],
                                "bbox": [x0, y0, x1 - x0, y1 - y0],
                                "area": annotation.get("area", (x1 - x0) * (y1 - y0)), "iscrowd": 0})
        boxes = output.get("boxes") or []
        if boxes and not isinstance(boxes[0], (list, tuple)):  # a flat [n x 4] array (the worker's layout)
            boxes = [boxes[index:index + 4] for index in range(0, len(boxes), 4)]
        for box, score, label in zip(boxes, output.get("scores") or [], output.get("class_ids") or []):
            x0, y0, x1, y1 = (float(value) for value in box)
            detections.append({"image_id": image_id, "category_id": int(label), "bbox": [x0, y0, x1 - x0, y1 - y0],
                               "score": float(score)})
    if not detections:
        return 0.0
    categories = sorted({item["category_id"] for item in annotations + detections})
    with contextlib.redirect_stdout(io.StringIO()):  # COCOeval prints its progress and summary
        gold = COCO()
        gold.dataset = {"info": {}, "images": images, "annotations": annotations,
                        "categories": [{"id": category} for category in categories]}
        gold.createIndex()
        evaluation = COCOeval(gold, gold.loadRes(detections), "bbox")
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    return 100.0 * max(0.0, float(evaluation.stats[0]))


def precomputed_units(problems: Sequence[Mapping[str, Any]], observations: Mapping[int, Any]) -> list[float]:
    """Per sample: a score a check computed already (``value``, in points), e.g. CLIP-I to a target."""
    return [float((observations.get(index) or {}).get("value") or 0.0) for index in range(len(problems))]


def mean(units: Sequence[float]) -> float:
    return sum(units) / len(units) if units else 0.0


# name -> (units of a side, statistic over units, higher is better).
def _finite(values: Any) -> bool:
    try:
        return bool(values) and all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _text_answer(problem: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    return isinstance(observation.get("text"), str)  # an empty transcript or translation is an answer


def _vector_answer(problem: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    return _finite(observation.get("values"))


def _scores_answer(problem: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    return _finite(observation.get("scores"))


def _forecast_answer(problem: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    return _finite(point_forecast(observation, len(problem["gold"]["values"])))


def _detections_answer(problem: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    return isinstance(observation.get("boxes"), list)  # no detections is an answer


def _planes(observation: Mapping[str, Any], field: str) -> tuple[Any, int] | None:
    """The finite values of ``field`` and the plane size (positive height x width), or None."""
    import numpy as np

    values, height, width = observation.get(field), observation.get("height"), observation.get("width")
    if not isinstance(values, list) or not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        return None
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return (array, height * width) if array.ndim == 1 and bool(np.isfinite(array).all()) else None


def _masks_answer(problem: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    planes = _planes(observation, "masks")  # whole H x W planes (none: nothing found, an answer)
    return planes is not None and planes[0].size % planes[1] == 0


def _label_map_answer(problem: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    planes = _planes(observation, "mask")  # exactly one H x W label plane
    return planes is not None and planes[0].size == planes[1]


def _value_answer(problem: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    return _finite([observation.get("value")])


# Per corpus metric: whether an output answers its problem (the field the metric reads, well formed). An
# output that does not is a missing answer (an error), never scored as zeros or an empty prediction.
ANSWERS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], bool]] = {
    "chrf": _text_answer, "wer": _text_answer, "coco_map": _detections_answer, "forecast_mse": _forecast_answer,
    "miou": _label_map_answer, "mask_iou": _masks_answer, "precomputed_mean": _value_answer,
    "sts_spearman": _vector_answer, "rerank_ndcg": _scores_answer, "retrieval_ndcg": _vector_answer,
    "retrieval_ndcg10": _vector_answer}


def answered(metric: str, problems: Sequence[Mapping[str, Any]], observations: Mapping[int, Any]) -> dict[int, Any]:
    """The observations that answer their problem under ``metric`` (``ANSWERS``)."""
    check = ANSWERS[metric]
    return {index: observation for index, observation in observations.items()
            if index < len(problems) and isinstance(observation, Mapping) and check(problems[index], observation)}


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
                                                      "retrieval_ndcg": (retrieval_units, mean_percent, True),
                                                      "retrieval_ndcg10": (corpus_retrieval_units, mean_percent, True)}


def compare_corpus(metric: str, problems: Sequence[Mapping[str, Any]], candidate: Mapping[int, Any],
                   native: Mapping[int, Any], gate: Mapping[str, Any], params: Mapping[str, Any] | None = None,
                   resamples: int | None = None) -> dict[str, Any]:
    """Both sides' statistic and the non-inferiority outcome: a paired bootstrap over the problems'
    ``cluster`` (one unit per problem; else each unit is its own cluster), or over moving blocks of
    consecutive time-series windows when the problems carry their ``series`` position."""
    units_of, statistic, higher = CORPUS[metric]
    if metric in PARAMETRIC:
        mine, theirs = units_of(problems, candidate, params or {}), units_of(problems, native, params or {})
    else:
        mine, theirs = units_of(problems, candidate), units_of(problems, native)
    clusters = ([problem.get("cluster", index) for index, problem in enumerate(problems)]
                if len(mine) == len(problems) else list(range(len(mine))))

    def both(indices: Sequence[int]) -> tuple[float, float]:
        return statistic([mine[i] for i in indices]), statistic([theirs[i] for i in indices])

    series = [problem.get("series") for problem in problems] if len(mine) == len(problems) else []
    if series and all(series):
        clusters, block = [item["position"] for item in series], int(series[0]["block"])
    else:
        block = None
    outcome = noninferiority.bootstrap(both, clusters, dict(gate), higher,
                                       resamples or BOOTSTRAP_SAMPLES_BY_METRIC.get(metric, BOOTSTRAP_SAMPLES), block=block)
    return {"trtmc": statistic(mine), "native": statistic(theirs), "higher_is_better": higher, "units": len(mine),
            **outcome}
