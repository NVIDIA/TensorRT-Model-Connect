# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired non-inferiority of TRTMC against the native model.

The regression ``R`` is positive when TRTMC is worse, in the metric's points. An entry ``pass``es when
the one-sided 95% upper bound of ``R`` is below the margin, ``fail``s (a regression established) when
the one-sided 95% lower bound exceeds it, and is ``inconclusive`` otherwise.

- Binary metrics: Tango's paired score test on the discordant counts, with a cutoff that is exact for
  the problem count and margin (``critical``).
- Sampled models: per-problem seed-mean differences, one-sided Student-t bounds over problems.
- Corpus metrics: paired percentile bootstrap over independent clusters of ``R - margin(native)``.
"""

from __future__ import annotations

import functools
import math
import random
import statistics
from typing import Any, Callable, Hashable, Sequence

ALPHA = 0.05
BOOTSTRAP_SEED = 0
# Score-test cutoffs searched, from the asymptotic one-sided 95% quantile up.
CUTOFFS = tuple(1.645 + 0.005 * step for step in range(120))
# The exact size is evaluated on a discordance grid; between grid points it may rise slightly, so the
# cutoff keeps the grid maximum at or below this.
GRID_ALPHA = 0.049
# Discordance (both kinds of disagreement) covered: from the margin itself up to margin + 0.4.
DISCORDANCE_SPAN = 0.4


def margin(gate: dict[str, Any], native_score: float | None) -> float:
    """The margin in points: ``margin``, or ``relative_margin`` of the native score when that is larger."""
    value = float(gate.get("margin", 0.0))
    if gate.get("relative_margin") and native_score is not None:
        value = max(value, float(gate["relative_margin"]) * abs(float(native_score)))
    return value


def score_z(native_only: int, trtmc_only: int, n: int, delta: float) -> float:
    """Tango's score statistic for ``R = delta`` (``delta`` as a fraction): negative when R < delta."""
    b, c = float(native_only), float(trtmc_only)
    a, bb, cc = 2.0 * n, -(b + c) + (2.0 * n - b + c) * delta, -c * delta * (1.0 - delta)
    t = (-bb + math.sqrt(max(bb * bb - 4.0 * a * cc, 0.0))) / (2.0 * a)  # restricted MLE of P(TRTMC only)
    variance = n * (2.0 * t + delta * (1.0 - delta))
    if variance <= 0.0:  # delta = 0 and no discordance: R is exactly 0
        return 0.0 if b == c else math.copysign(math.inf, b - c)
    return (b - c - n * delta) / math.sqrt(variance)


def _score_z_array(b: Any, c: Any, n: int, delta: float) -> Any:
    a, bb = 2.0 * n, -(b + c) + (2.0 * n - b + c) * delta
    t = (-bb + (bb * bb + 4.0 * a * c * delta * (1.0 - delta)) ** 0.5) / (2.0 * a)
    return (b - c - n * delta) / (n * (2.0 * t + delta * (1.0 - delta))) ** 0.5


@functools.lru_cache(maxsize=None)
def critical(n: int, delta: float) -> float:
    """The smallest cutoff in CUTOFFS whose exact pass and fail probabilities at the margin (R = delta)
    stay at or below GRID_ALPHA on a dense discordance grid from ``delta`` (no TRTMC-only answers) to
    ``delta + DISCORDANCE_SPAN`` (steps of 0.0005 up to delta + 0.05, 0.005 beyond): the trinomial
    distribution of (b, c) enumerated where it has mass."""
    import numpy as np

    if n <= 0 or not 0.0 < delta < 1.0:
        return CUTOFFS[-1]
    log_factorial = np.array([math.lgamma(k + 1) for k in range(n + 1)])
    cutoffs = np.array(CUTOFFS)
    worst_pass, worst_fail = np.zeros(len(cutoffs)), np.zeros(len(cutoffs))
    near = np.arange(delta, delta + 0.05, 0.0005)
    far = np.arange(delta + 0.05, min(delta + DISCORDANCE_SPAN, 1.0) + 1e-12, 0.005)
    grid = sorted({*near, *far})
    for q in grid:
        p12, p21 = (q + delta) / 2.0, max((q - delta) / 2.0, 0.0)
        if p12 + p21 > 1.0:
            continue
        top_b = int(min(n, n * p12 + 12 * math.sqrt(n * p12 + 1) + 20))
        top_c = int(min(n, n * p21 + 12 * math.sqrt(n * p21 + 1) + 20))
        b, c = np.arange(top_b + 1)[:, None].astype(float), np.arange(top_c + 1)[None, :].astype(float)
        rest = n - b - c
        valid = rest >= 0
        rest_index = np.where(valid, rest, 0).astype(int)
        log_p = (log_factorial[n] - log_factorial[b.astype(int)] - log_factorial[c.astype(int)]
                 - log_factorial[rest_index] + rest_index * math.log(max(1.0 - p12 - p21, 1e-300)))
        log_p = log_p + (b * math.log(p12) if p12 > 0 else np.where(b == 0, 0.0, -np.inf))
        log_p = log_p + (c * math.log(p21) if p21 > 0 else np.where(c == 0, 0.0, -np.inf))
        probability = np.where(valid, np.exp(log_p), 0.0).ravel()
        z = np.broadcast_to(_score_z_array(b, c, n, delta), log_p.shape).ravel()
        order = np.argsort(z)
        z, cumulative = z[order], np.cumsum(probability[order])
        below = np.searchsorted(z, -cutoffs, side="left")
        above = np.searchsorted(z, cutoffs, side="right")
        worst_pass = np.maximum(worst_pass, np.where(below > 0, cumulative[np.maximum(below - 1, 0)], 0.0))
        worst_fail = np.maximum(worst_fail, cumulative[-1] - np.where(above > 0, cumulative[np.maximum(above - 1, 0)], 0.0))
    controlled = (worst_pass <= GRID_ALPHA) & (worst_fail <= GRID_ALPHA)
    return float(cutoffs[int(np.argmax(controlled))]) if controlled.any() else float(cutoffs[-1])


def binary(native_only: int, trtmc_only: int, n: int, delta_points: float) -> dict[str, Any]:
    """Outcome of right/wrong pairs: ``b`` problems only the native model answers right, ``c`` only TRTMC."""
    delta = delta_points / 100.0
    # One statistic at the margin, two tails: Z < -k rejects R >= delta (non-inferior), Z > k rejects
    # R <= delta (a regression beyond the margin).
    z = score_z(native_only, trtmc_only, n, delta) if n else 0.0
    k = critical(n, round(delta, 6))
    outcome = "pass" if z < -k else "fail" if z > k else "inconclusive"
    return {"outcome": outcome, "regression_points": 100.0 * (native_only - trtmc_only) / n if n else 0.0,
            "z": z, "critical": k}


def t_quantile(probability: float, df: float) -> float:
    """Student-t quantile (fractional degrees of freedom allowed)."""
    from scipy.stats import t

    return float(t.ppf(probability, df))


def paired_means(differences: Sequence[float], delta_points: float) -> dict[str, Any]:
    """Outcome from per-problem regressions (points, already averaged over seeds within each problem)."""
    n = len(differences)
    if n < 2:
        return {"outcome": "inconclusive", "regression_points": statistics.fmean(differences) if n else 0.0}
    mean = statistics.fmean(differences)
    half = t_quantile(1 - ALPHA, n - 1) * statistics.stdev(differences) / math.sqrt(n)
    outcome = "pass" if mean + half < delta_points else "fail" if mean - half > delta_points else "inconclusive"
    return {"outcome": outcome, "regression_points": mean, "upper": mean + half, "lower": mean - half}


def _moving_blocks(order: Sequence[int], length: int, generator: random.Random) -> list[int]:
    """One moving-block resample: blocks of ``length`` consecutive units (in ``order``) from uniform starts,
    concatenated and cut to the sample's size."""
    length = max(1, min(length, len(order)))
    picked: list[int] = []
    while len(picked) < len(order):
        start = generator.randrange(len(order) - length + 1)
        picked += order[start:start + length]
    return picked[:len(order)]


def bootstrap(statistic: Callable[[Sequence[int]], tuple[float, float]], clusters: Sequence[Hashable], gate: dict[str, Any],
              higher_is_better: bool, resamples: int, block: int | None = None) -> dict[str, Any]:
    """Outcome of a corpus metric. ``statistic(indices)`` returns (TRTMC score, native score) on the units
    at ``indices`` (repeats allowed); ``clusters`` gives each unit's independent group, or with ``block``
    each unit's position in a series, resampled as moving blocks of ``block`` consecutive units."""
    groups: dict[Hashable, list[int]] = {}
    for index, cluster in enumerate(clusters):
        groups.setdefault(cluster, []).append(index)
    keys = list(groups)
    order = sorted(range(len(clusters)), key=lambda index: clusters[index])

    def excess(indices: Sequence[int]) -> tuple[float, float, float]:
        mine, theirs = statistic(indices)
        regression = theirs - mine if higher_is_better else mine - theirs
        return regression - margin(gate, theirs), regression, theirs

    observed, regression, native = excess(range(len(clusters)))
    if gate.get("relative_margin") and native == 0.0:
        return {"outcome": "not-comparable", "regression_points": regression,
                "reason": "the native score is 0, so a relative margin is undefined"}
    generator = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(resamples):
        if block:
            picked = _moving_blocks(order, block, generator)
        else:
            picked = [index for _ in keys for index in groups[keys[generator.randrange(len(keys))]]]
        draws.append(excess(picked)[0])
    draws.sort()
    low, high = draws[int(ALPHA * resamples)], draws[int((1 - ALPHA) * resamples) - 1]
    outcome = "pass" if high < 0.0 else "fail" if low > 0.0 else "inconclusive"
    return {"outcome": outcome, "regression_points": regression, "margin_points": margin(gate, native),
            "excess_interval90": [low, high], "clusters": len(keys), "resamples": resamples, "observed_excess": observed,
            **({"block": block} if block else {})}
