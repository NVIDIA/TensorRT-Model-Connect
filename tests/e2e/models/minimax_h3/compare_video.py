# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed visual parity comparator for decoded MiniMax-H3 frame arrays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
from tensorrt_model_connect.families.minimax_h3.provenance import (
    atomic_write_json,
    file_identity,
    stable_file_record,
    validate_source_revision,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--reference-receipt", required=True)
    parser.add_argument("--candidate-receipt", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    source_revision = validate_source_revision(args.source_revision)
    reference_path = Path(args.reference).resolve(strict=True)
    candidate_path = Path(args.candidate).resolve(strict=True)
    reference_receipt_path = Path(args.reference_receipt).resolve(strict=True)
    candidate_receipt_path = Path(args.candidate_receipt).resolve(strict=True)
    thresholds_path = Path(args.thresholds).resolve(strict=True)
    paths = {
        "reference": reference_path,
        "candidate": candidate_path,
        "reference_receipt": reference_receipt_path,
        "candidate_receipt": candidate_receipt_path,
        "thresholds": thresholds_path,
    }
    identities_before = {label: file_identity(path) for label, path in paths.items()}
    reference_raw = np.load(reference_path, allow_pickle=False)
    candidate_raw = np.load(candidate_path, allow_pickle=False)
    reference_receipt = json.loads(reference_receipt_path.read_text())
    candidate_receipt = json.loads(candidate_receipt_path.read_text())

    def normalized(array):
        if np.issubdtype(array.dtype, np.integer):
            return array.astype(np.float32) / np.iinfo(array.dtype).max
        return array.astype(np.float32)

    reference = normalized(reference_raw)
    candidate = normalized(candidate_raw)
    if reference.shape != candidate.shape:
        raise ValueError(f"frame shape mismatch: {reference.shape} != {candidate.shape}")
    if not np.isfinite(candidate).all():
        raise ValueError("candidate video contains non-finite pixels")
    thresholds = json.loads(thresholds_path.read_text())["threshold_overrides"]
    input_records = {}
    for label, path in paths.items():
        input_records[label], identity_after = stable_file_record(path, label)
        if identity_after != identities_before[label]:
            raise ValueError(f"MiniMax-H3 {label} changed while it was being read")

    for label, receipt, frames_label in (
        ("reference", reference_receipt, "reference"),
        ("candidate", candidate_receipt, "candidate"),
    ):
        if not isinstance(receipt, dict) or receipt.get("status") != "passed":
            raise ValueError(f"MiniMax-H3 {label} receipt is not a passed run")
        if receipt.get("source_revision") != source_revision:
            raise ValueError(f"MiniMax-H3 {label} receipt has a different source revision")
        if receipt.get("frames") != input_records[frames_label]:
            raise ValueError(f"MiniMax-H3 {label} frames do not match their receipt")

    def workload(receipt: dict) -> dict:
        value = receipt.get("workload", receipt.get("request"))
        if not isinstance(value, dict):
            raise ValueError("MiniMax-H3 comparison receipt has no workload")
        return value

    reference_workload = workload(reference_receipt)
    candidate_workload = workload(candidate_receipt)
    for field in ("prompt", "seed", "height", "width", "num_frames", "num_inference_steps"):
        if reference_workload.get(field) != candidate_workload.get(field):
            raise ValueError(f"MiniMax-H3 comparison workloads differ in {field}")

    def inventory_sha256(receipt: dict) -> str | None:
        if isinstance(receipt.get("checkpoint_inventory_sha256"), str):
            return receipt["checkpoint_inventory_sha256"]
        snapshot = receipt.get("checkpoint_snapshot")
        return snapshot.get("inventory_sha256") if isinstance(snapshot, dict) else None

    reference_inventory = inventory_sha256(reference_receipt)
    candidate_inventory = inventory_sha256(candidate_receipt)
    for label, digest in (("reference", reference_inventory), ("candidate", candidate_inventory)):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"MiniMax-H3 {label} receipt has no valid checkpoint inventory")
    if reference_inventory != candidate_inventory:
        raise ValueError("MiniMax-H3 comparison receipts use different checkpoint inventories")
    error = candidate - reference
    mse = float(np.mean(np.square(error), dtype=np.float64))
    mae = float(np.mean(np.abs(error), dtype=np.float64))
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    expected_shape = [
        int(thresholds["exact_num_frames"]),
        int(thresholds["exact_video_height"]),
        int(thresholds["exact_video_width"]),
        3,
    ]
    shape_matches = list(reference.shape) == expected_shape
    receipt = {
        "source_revision": source_revision,
        **input_records,
        "shape": list(reference.shape),
        "expected_shape": expected_shape,
        "shape_matches": shape_matches,
        "mse": mse,
        "mean_absolute_error": mae,
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "psnr_db": psnr,
        "minimum_psnr_db": float(thresholds["minimum_psnr_db"]),
        "maximum_mean_absolute_error": float(thresholds["maximum_mean_absolute_error"]),
    }
    receipt["passed"] = (
        shape_matches
        and psnr >= receipt["minimum_psnr_db"]
        and mae <= receipt["maximum_mean_absolute_error"]
    )
    atomic_write_json(Path(args.output), receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
