#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score MiniMax-H3 candidate videos with the pinned AVGen-Bench Vis metric."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from PIL import Image


AVGEN_REVISION = "1049eabac472d479fe5feeb1ee202961f8e0982a"
QALIGN_TREE = "70a31768f1eaf48a53f31c6d51c63c63b6e8c439"
QALIGN_SCORER_SHA256 = "397d7763447b2c8b18bf2bb2e42cf3b0ee7dd43ab40bdc633cfb3af113360f98"
QALIGN_LICENSE_SHA256 = "53fe0bdf6a7e86c30b0cbcbe0ca8db820c5e75c5d9b140e711252d9f16d33a4f"
QALIGN_MODEL = "q-future/one-align"
QALIGN_MODEL_REVISION = "dcc603b95aa0ebd82afa696d4a1e20d11fc80ddb"
QALIGN_LICENSE_ACCEPTANCE_ENV = "TRTMC_ACCEPT_QALIGN_SLAB_1_0"
EXPECTED_SAMPLE_COUNT = 235
EXPECTED_SHAPE = [124, 768, 1344, 3]
EXPECTED_RETAINED_FRAME_INDICES = [0, 24, 48, 72, 96]
EXPECTED_FRAME_SIZE = (1344, 768)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(root: Path, revision: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", revision],
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode or not value:
        detail = completed.stderr.strip() or value or "unresolved"
        raise ValueError(f"could not resolve AVGen-Bench {revision!r}: {detail}")
    return value


def validate_evaluator_checkout(root: Path) -> Path:
    """Bind scoring to the exact Q-Align source shipped by pinned AVGen-Bench."""

    if root.is_symlink():
        raise ValueError("AVGen-Bench evaluator root must not be a symlink")
    root = root.resolve(strict=True)
    if _git_value(root, "HEAD") != AVGEN_REVISION:
        raise ValueError(f"AVGen-Bench evaluator must be checked out at {AVGEN_REVISION}")
    if _git_value(root, f"{AVGEN_REVISION}:eval/Q-Align") != QALIGN_TREE:
        raise ValueError("AVGen-Bench Q-Align tree does not match the pinned revision")
    qalign_root = root / "eval" / "Q-Align"
    expected_files = {
        qalign_root / "q_align" / "evaluate" / "scorer.py": QALIGN_SCORER_SHA256,
        qalign_root / "S-Lab-LICENSE": QALIGN_LICENSE_SHA256,
    }
    for path, expected in expected_files.items():
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"pinned AVGen-Bench evaluator file mismatch: {path}")
    return qalign_root


def _stage_data(response: Mapping[str, Any]) -> Mapping[str, Any]:
    stage_output = response.get("stage_output")
    if not isinstance(stage_output, Mapping):
        raise ValueError("prediction has no serialized stage_output")
    data = stage_output.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("prediction stage_output has no data object")
    return data


def _candidate_frames(response: Mapping[str, Any]) -> list[Image.Image]:
    data = _stage_data(response)
    if int(data.get("returncode", 1)) != 0:
        raise ValueError(f"candidate returned {data.get('returncode')}")
    receipt = data.get("receipt")
    if not isinstance(receipt, Mapping) or receipt.get("status") != "passed":
        raise ValueError("candidate has no passed native receipt")
    if receipt.get("shape") != EXPECTED_SHAPE:
        raise ValueError(f"candidate shape is not {EXPECTED_SHAPE}")
    if receipt.get("retained_frame_indices") != EXPECTED_RETAINED_FRAME_INDICES:
        raise ValueError("candidate did not retain the official AVGen 1 fps frame subset")
    paths = data.get("frame_paths")
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)):
        raise ValueError("candidate frame_paths is not a sequence")
    if len(paths) != len(EXPECTED_RETAINED_FRAME_INDICES):
        raise ValueError("candidate does not contain exactly five retained frames")

    frames = []
    for value in paths:
        path = Path(str(value))
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"candidate retained frame is missing or a symlink: {path}")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != EXPECTED_FRAME_SIZE:
                raise ValueError(
                    f"candidate retained frame has mode/size {image.mode}/{image.size}"
                )
            frames.append(image.copy())
    return frames


def score_avgen_vis_predictions(
    predictions: Mapping[str, Any],
    answers: Mapping[str, Any],
    *,
    scorer: Callable[[list[Image.Image]], float],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all rows structurally, score valid rows, and apply aggregate gates."""

    responses = predictions.get("responses")
    requests = answers.get("requests")
    if not isinstance(responses, list) or not isinstance(requests, list):
        raise ValueError("predictions and answers must contain lists")
    if len(responses) != len(requests):
        raise ValueError(f"prediction/request length mismatch: {len(responses)} != {len(requests)}")

    samples = []
    scores = []
    for index, (response, request) in enumerate(zip(responses, requests, strict=True)):
        if not isinstance(response, Mapping) or not isinstance(request, Mapping):
            raise ValueError(f"AVGen-Bench row {index} must contain objects")
        expected_id = str(request.get("sample_id", ""))
        actual_id = str(response.get("sample_id", ""))
        if not expected_id or actual_id != expected_id:
            raise ValueError(
                f"AVGen-Bench sample id mismatch at {index}: {expected_id!r} != {actual_id!r}"
            )
        sample = {
            "sample_id": expected_id,
            "category": request.get("source_category", ""),
            "source_index": request.get("source_index", index),
        }
        try:
            frames = _candidate_frames(response)
            score = float(scorer(frames))
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Q-Align returned invalid score {score!r}")
            score = float(score)
            scores.append(score)
            sample.update({"status": "passed", "avgen_vis": score})
        except Exception as error:
            sample.update(
                {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        samples.append(sample)

    sample_count = len(samples)
    valid_count = len(scores)
    structural_pass_rate = valid_count / sample_count if sample_count else 0.0
    avgen_vis_mean = sum(scores) / valid_count if valid_count else 0.0
    avgen_vis_min = min(scores) if scores else 0.0
    avgen_vis_max = max(scores) if scores else 0.0
    required_sample_count = int(gates.get("required_sample_count", EXPECTED_SAMPLE_COUNT))
    min_structural_pass_rate = float(gates.get("min_structural_pass_rate", 1.0))
    min_avgen_vis_mean = float(gates.get("min_avgen_vis_mean", 0.8))
    gate_failures = []
    if sample_count != required_sample_count:
        gate_failures.append(
            {
                "gate": "required_sample_count",
                "actual": sample_count,
                "required": required_sample_count,
            }
        )
    if structural_pass_rate < min_structural_pass_rate:
        gate_failures.append(
            {
                "gate": "min_structural_pass_rate",
                "actual": structural_pass_rate,
                "required": min_structural_pass_rate,
            }
        )
    if avgen_vis_mean < min_avgen_vis_mean:
        gate_failures.append(
            {
                "gate": "min_avgen_vis_mean",
                "actual": avgen_vis_mean,
                "required": min_avgen_vis_mean,
            }
        )
    return {
        "status": "passed" if not gate_failures else "failed",
        "sample_count": sample_count,
        "valid_count": valid_count,
        "passed_count": valid_count,
        "structural_pass_rate": structural_pass_rate,
        "avgen_vis_mean": avgen_vis_mean,
        "avgen_vis_min": avgen_vis_min,
        "avgen_vis_max": avgen_vis_max,
        "gates": {
            "required_sample_count": required_sample_count,
            "min_structural_pass_rate": min_structural_pass_rate,
            "min_avgen_vis_mean": min_avgen_vis_mean,
        },
        "gate_failures": gate_failures,
        "samples": samples,
    }


def _load_official_scorer(
    evaluator_root: Path,
    *,
    model_id: str,
    model_revision: str,
    device: str,
) -> tuple[Callable[[list[Image.Image]], float], dict[str, Any]]:
    if os.environ.get(QALIGN_LICENSE_ACCEPTANCE_ENV) != "1":
        raise PermissionError(
            "Q-Align is S-Lab License 1.0 (non-commercial by default); set "
            f"{QALIGN_LICENSE_ACCEPTANCE_ENV}=1 only after confirming authorization"
        )
    qalign_root = validate_evaluator_checkout(evaluator_root)
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            model_id,
            revision=model_revision,
            local_files_only=True,
        )
    ).resolve(strict=True)
    sys.path.insert(0, str(qalign_root))
    from q_align import QAlignVideoScorer

    scorer = QAlignVideoScorer(pretrained=str(snapshot), device=device)

    def score(frames: list[Image.Image]) -> float:
        values = scorer([frames]).tolist()
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError("Q-Align must return exactly one score per video")
        return float(values[0])

    return score, {
        "repository": "https://github.com/NVIDIA/AVGen-Bench.git",
        "revision": AVGEN_REVISION,
        "qalign_tree": QALIGN_TREE,
        "qalign_model": model_id,
        "qalign_model_revision": model_revision,
        "frame_sampling": "1 fps at source fps=24: [0,24,48,72,96]",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--model-id", default=QALIGN_MODEL)
    parser.add_argument("--model-revision", default=QALIGN_MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--required-sample-count", type=int, default=EXPECTED_SAMPLE_COUNT)
    parser.add_argument("--min-structural-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-avgen-vis-mean", type=float, default=0.8)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scorer, provenance = _load_official_scorer(
        args.evaluator_root,
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
    )
    summary = score_avgen_vis_predictions(
        json.loads(args.predictions.read_text(encoding="utf-8")),
        json.loads(args.answers.read_text(encoding="utf-8")),
        scorer=scorer,
        gates={
            "required_sample_count": args.required_sample_count,
            "min_structural_pass_rate": args.min_structural_pass_rate,
            "min_avgen_vis_mean": args.min_avgen_vis_mean,
        },
    )
    summary["benchmark_provenance"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
