# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for fixed-canvas image seams (issue #1546).

A family that resizes an image onto a fixed engine canvas and fills the rest with a
constant feeds non-image pixels to the network. Unless the engine consumes a mask (or
the runtime rejects shapes that need padding), results silently depend on image aspect.

Two controls live here:

* every family runtime that matches the padded-canvas predicate must be classified in
  ``canvas_padding_classification.json``; confirmed families must carry their guard;
* the canvas a benchmark builds must equal the canvas of the premerge manifest of the
  same model, so the accuracy lane exercises the engine that premerge validates.

The scan is a discovery predicate over source text, not proof of numerical behaviour.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
CLASSIFICATION = Path(__file__).with_name("canvas_padding_classification.json")
STATUSES = {
    "CONFIRMED_SAME_MECHANISM",
    "CANDIDATE_NEEDS_VALIDATION",
    "INTENTIONAL_EXCEPTION",
    "NOT_APPLICABLE",
}
PAD = re.compile(
    r"zero-?pad|pad_center|aspect_preserve|letterbox|\bpad_value\b|\bcanvas\b|\bkPad\w*"
    r"|pixel_values\s*\([^;]*,\s*0\.0F\s*\)|pixel_values\.assign\([^;]*0\.0F",
    re.IGNORECASE,
)
IMG = re.compile(
    r"DecodedImage|image_pixels|input_image_h|pixel_values|image_height|img_chw",
    re.IGNORECASE,
)
RSZ = re.compile(r"stbir|resize|rescale|letterbox|\bscale\b", re.IGNORECASE)
NATIVE_SUFFIXES = {".cpp", ".h", ".cu", ".cuh"}


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def padded_canvas_families(root: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted((root / "families").glob("*/runtime/**/*")):
        if path.suffix not in NATIVE_SUFFIXES or not path.is_file():
            continue
        text = _read(path)
        if PAD.search(text) and IMG.search(text) and RSZ.search(text):
            family = path.relative_to(root / "families").parts[0]
            found.setdefault(family, []).append(path.relative_to(root).as_posix())
    return found


def classification_violations(root: Path, families: dict) -> list[str]:
    problems: list[str] = []
    for family, files in sorted(padded_canvas_families(root).items()):
        if family not in families:
            problems.append(f"{family}: UNCLASSIFIED padded-canvas seam in {files}")
    for family, entry in sorted(families.items()):
        status = entry.get("status")
        if status not in STATUSES:
            problems.append(f"{family}: invalid status {status!r}")
        elif not str(entry.get("evidence", "")).strip():
            problems.append(f"{family}: classification needs evidence")
        if status != "CONFIRMED_SAME_MECHANISM" or not (root / "families" / family).is_dir():
            continue
        guard = entry.get("guard") or {}
        text = "".join(_read(root / name) for name in guard.get("files", []))
        if not any(re.search(pattern, text) for pattern in guard.get("any_of", [])):
            problems.append(f"{family}: confirmed mechanism without its guard")
    return problems


def canvas_mismatches(root: Path) -> list[str]:
    problems: list[str] = []
    for path in sorted((root / "families").glob("*/tests/benchmark/*.yaml")):
        document = yaml.safe_load(_read(path)) or {}
        build = (document.get("candidate") or {}).get("build") or {}
        family = path.relative_to(root / "families").parts[0]
        model = document.get("model")
        manifest_path = root / "families" / family / "tests/manifests" / f"{model}.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(_read(manifest_path))
        for key in ("image_height", "image_width"):
            if key in build and key in manifest and build[key] != manifest[key]:
                problems.append(
                    f"{family}/{model}: benchmark build {key}={build[key]} "
                    f"!= premerge manifest {key}={manifest[key]}"
                )
    return problems


def test_padded_canvas_seams_are_classified_and_confirmed_ones_guarded() -> None:
    document = json.loads(CLASSIFICATION.read_text(encoding="utf-8"))
    assert classification_violations(REPO, document["families"]) == []


def test_detr_fails_closed_when_resize_needs_padding() -> None:
    seam = _read(REPO / "families/detr/runtime/image_preprocess_seam.cpp")
    model = _read(REPO / "families/detr/model.py")
    tests = _read(REPO / "families/detr/tests/cpp/test_image_preprocess_seam.cpp")
    has_mask = re.search(r"add_input\(\s*[\"']pixel_mask[\"']", model) is not None
    rejects = re.search(r"resized_h\s*!=\s*out_h\s*\|\|\s*resized_w\s*!=\s*out_w", seam)
    assert has_mask or rejects, "DETR accepts shapes needing padding without a pixel_mask input"
    covered = re.search(r"require padding|pixel_mask", tests) is not None
    assert has_mask or covered, "no test covers the DETR padding path"
    assert has_mask or not re.search(r"keeps_padding_zero|padding stays zero", tests)


def test_benchmark_canvas_matches_premerge_manifest() -> None:
    assert canvas_mismatches(REPO) == []


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_unclassified_padded_canvas_family_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "families/newdet/runtime/seam.cpp",
        "std::vector<float> pixel_values(3U * plane, 0.0F);  // resize then canvas\n"
        "auto image_pixels = resize(src);\n",
    )
    problems = classification_violations(tmp_path, {})
    assert problems == [
        "newdet: UNCLASSIFIED padded-canvas seam in ['families/newdet/runtime/seam.cpp']"
    ]


def test_confirmed_family_without_guard_is_reported(tmp_path: Path) -> None:
    (tmp_path / "families/fam").mkdir(parents=True)
    _write(tmp_path / "families/fam/model.py", "pass\n")
    entry = {
        "status": "CONFIRMED_SAME_MECHANISM",
        "evidence": "example",
        "guard": {"files": ["families/fam/model.py"], "any_of": ["require padding"]},
    }
    assert classification_violations(tmp_path, {"fam": entry}) == [
        "fam: confirmed mechanism without its guard"
    ]
    _write(tmp_path / "families/fam/model.py", "# require padding\n")
    assert classification_violations(tmp_path, {"fam": entry}) == []


def test_benchmark_manifest_canvas_drift_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path / "families/fam/tests/benchmark/m.yaml",
        "model: m\ncandidate:\n  build:\n    image_height: 1333\n    image_width: 1333\n",
    )
    _write(
        tmp_path / "families/fam/tests/manifests/m.json",
        json.dumps({"image_height": 796, "image_width": 1333}),
    )
    assert canvas_mismatches(tmp_path) == [
        "fam/m: benchmark build image_height=1333 != premerge manifest image_height=796"
    ]
