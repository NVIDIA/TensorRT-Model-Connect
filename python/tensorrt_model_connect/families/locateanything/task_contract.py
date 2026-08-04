# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing prompt and structured-output contract for LocateAnything."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
import re
from typing import Callable, Literal, Sequence


LocalizationKind = Literal["box", "point"]


@dataclass(frozen=True)
class Localization:
    """One normalized LocateAnything box or point in the 0..1000 space."""

    kind: LocalizationKind
    coordinates: tuple[int, ...]


_REF_RE = re.compile(r"<ref>.+?</ref>", re.DOTALL)
_BOX_GROUP_RE = re.compile(r"<box>\s*((?:<[0-9]{1,4}>\s*)+)</box>")
_COORD_RE = re.compile(r"<([0-9]{1,4})>")


def detection_prompt(categories: Sequence[str]) -> str:
    """Build the official object-detection/document-layout prompt."""
    normalized = [str(category).strip() for category in categories if str(category).strip()]
    if not normalized:
        raise ValueError("LocateAnything detection requires at least one category")
    return (
        "Locate all the instances that matches the following description: "
        + "</c>".join(normalized)
        + "."
    )


def ground_single_prompt(phrase: str) -> str:
    return f"Locate a single instance that matches the following description: {_phrase(phrase)}."


def ground_multi_prompt(phrase: str) -> str:
    return f"Locate all the instances that match the following description: {_phrase(phrase)}."


def ground_text_prompt(phrase: str) -> str:
    return f"Please locate the text referred as {_phrase(phrase)}."


def detect_text_prompt() -> str:
    return "Detect all the text in box format."


def ground_gui_prompt(phrase: str, *, output_type: LocalizationKind = "box") -> str:
    if output_type == "point":
        return point_prompt(phrase)
    if output_type != "box":
        raise ValueError("LocateAnything GUI output_type must be 'box' or 'point'")
    return f"Locate the region that matches the following description: {_phrase(phrase)}."


def point_prompt(phrase: str) -> str:
    return f"Point to: {_phrase(phrase)}."


def _phrase(value: str) -> str:
    normalized = str(value).strip().rstrip(".")
    if not normalized:
        raise ValueError("LocateAnything task phrase must not be empty")
    return normalized


def parse_localizations(
    answer: str,
    *,
    require_reference: bool = False,
) -> tuple[Localization, ...] | None:
    """Parse a complete box-or-point answer, rejecting malformed or mixed groups."""
    text = str(answer)
    matches = list(_BOX_GROUP_RE.finditer(text))
    if (
        not matches
        or text.count("<box>") != len(matches)
        or text.count("</box>") != len(matches)
        or (require_reference and _REF_RE.search(text) is None)
    ):
        return None

    localizations: list[Localization] = []
    kinds: set[LocalizationKind] = set()
    for match in matches:
        coordinates = tuple(int(value) for value in _COORD_RE.findall(match.group(1)))
        if len(coordinates) == 2:
            kind: LocalizationKind = "point"
        elif len(coordinates) == 4:
            kind = "box"
        else:
            return None
        if any(value < 0 or value > 1000 for value in coordinates):
            return None
        if kind == "box" and (
            coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]
        ):
            return None
        kinds.add(kind)
        localizations.append(Localization(kind=kind, coordinates=coordinates))
    return tuple(localizations) if len(kinds) == 1 else None


def box_iou(left: Localization, right: Localization) -> float:
    if left.kind != "box" or right.kind != "box":
        return 0.0
    lx1, ly1, lx2, ly2 = left.coordinates
    rx1, ry1, rx2, ry2 = right.coordinates
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def point_distance(left: Localization, right: Localization) -> float:
    if left.kind != "point" or right.kind != "point":
        return float("inf")
    return hypot(left.coordinates[0] - right.coordinates[0], left.coordinates[1] - right.coordinates[1])


def matched_box_iou(
    left: Sequence[Localization], right: Sequence[Localization]
) -> float:
    """Return the best order-independent minimum IoU for two box sets."""
    if not left or len(left) != len(right):
        return 0.0
    scores = [[box_iou(a, b) for b in right] for a in left]
    for threshold in sorted({score for row in scores for score in row}, reverse=True):
        if _has_perfect_matching(scores, lambda score: score >= threshold):
            return threshold
    return 0.0


def matched_point_distance(
    left: Sequence[Localization], right: Sequence[Localization]
) -> float:
    """Return the best order-independent maximum distance for two point sets."""
    if not left or len(left) != len(right):
        return float("inf")
    distances = [[point_distance(a, b) for b in right] for a in left]
    for threshold in sorted({distance for row in distances for distance in row}):
        if _has_perfect_matching(distances, lambda distance: distance <= threshold):
            return threshold
    return float("inf")


def _has_perfect_matching(
    values: Sequence[Sequence[float]], predicate: Callable[[float], bool]
) -> bool:
    matches = [-1] * len(values)

    def assign(left_index: int, visited: set[int]) -> bool:
        for right_index, value in enumerate(values[left_index]):
            if right_index in visited or not predicate(value):
                continue
            visited.add(right_index)
            if matches[right_index] < 0 or assign(matches[right_index], visited):
                matches[right_index] = left_index
                return True
        return False

    return all(assign(left_index, set()) for left_index in range(len(values)))
