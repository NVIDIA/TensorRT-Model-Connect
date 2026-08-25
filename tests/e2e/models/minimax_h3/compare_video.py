# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed human-visible quality comparator for MiniMax-H3 frame arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from tensorrt_model_connect.families.minimax_h3.provenance import (
    atomic_write_json,
    file_identity,
    stable_file_record,
    validate_source_revision,
)
from visual_metrics import (
    compute_decoded_visual_metrics,
    evaluate_visual_quality,
    visual_block_size,
    visual_quality_passed,
)
from audio_metrics import (
    audio_quality_passed,
    compute_decoded_audio_metrics,
    evaluate_audio_quality,
    read_float32_wav,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--candidate-audio", required=True)
    parser.add_argument("--reference-wav", required=True)
    parser.add_argument("--candidate-wav", required=True)
    parser.add_argument("--reference-receipt", required=True)
    parser.add_argument("--candidate-receipt", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    source_revision = validate_source_revision(args.source_revision)
    reference_path = Path(args.reference).resolve(strict=True)
    candidate_path = Path(args.candidate).resolve(strict=True)
    reference_audio_path = Path(args.reference_audio).resolve(strict=True)
    candidate_audio_path = Path(args.candidate_audio).resolve(strict=True)
    reference_wav_path = Path(args.reference_wav).resolve(strict=True)
    candidate_wav_path = Path(args.candidate_wav).resolve(strict=True)
    reference_receipt_path = Path(args.reference_receipt).resolve(strict=True)
    candidate_receipt_path = Path(args.candidate_receipt).resolve(strict=True)
    thresholds_path = Path(args.thresholds).resolve(strict=True)
    paths = {
        "reference": reference_path,
        "candidate": candidate_path,
        "reference_audio": reference_audio_path,
        "candidate_audio": candidate_audio_path,
        "reference_wav": reference_wav_path,
        "candidate_wav": candidate_wav_path,
        "reference_receipt": reference_receipt_path,
        "candidate_receipt": candidate_receipt_path,
        "thresholds": thresholds_path,
    }
    identities_before = {label: file_identity(path) for label, path in paths.items()}
    reference_receipt = json.loads(reference_receipt_path.read_text())
    candidate_receipt = json.loads(candidate_receipt_path.read_text())
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
        audio_label = f"{label}_audio"
        wav_label = f"{label}_wav"
        if receipt.get("audio") != input_records[audio_label]:
            raise ValueError(f"MiniMax-H3 {label} audio does not match its receipt")
        if receipt.get("audio_wav") != input_records[wav_label]:
            raise ValueError(f"MiniMax-H3 {label} audio WAV does not match its receipt")

        audio = np.load(paths[audio_label], mmap_mode="r", allow_pickle=False)
        wav = read_float32_wav(paths[wav_label])
        if audio.shape != wav.samples.shape or not np.array_equal(audio, wav.samples):
            raise ValueError(f"MiniMax-H3 {label} WAV does not preserve its audio array")
        if (
            receipt.get("audio_shape") != [int(value) for value in audio.shape]
            or receipt.get("audio_num_samples_per_channel") != audio.shape[1]
            or receipt.get("audio_sample_rate_hz") != wav.sample_rate
            or receipt.get("audio_all_finite") is not True
            or receipt.get("audio_layout") != "channel_major"
            or receipt.get("audio_encoding") != "float32"
            or receipt.get("audio_wav_encoding") != "ieee_float32le"
        ):
            raise ValueError(f"MiniMax-H3 {label} audio metadata is inconsistent")

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
    if "plan_sha256" in candidate_receipt:
        if candidate_receipt.get("backend") != "tensorrt_native_single_device":
            raise ValueError(
                "MiniMax-H3 native candidate is not the qualified single-device backend"
            )
        if candidate_receipt.get("world_size") != 1:
            raise ValueError("MiniMax-H3 native candidate did not run with world_size=1")
        if candidate_receipt.get("collective_transport") != "none":
            raise ValueError("MiniMax-H3 single-device candidate unexpectedly used a collective")
    decoded = compute_decoded_visual_metrics(
        reference_path,
        candidate_path,
        block_size=visual_block_size(thresholds),
    )
    gates = evaluate_visual_quality(decoded, thresholds)
    decoded_audio = compute_decoded_audio_metrics(
        reference_audio_path,
        candidate_audio_path,
        reference_sample_rate=int(reference_receipt["audio_sample_rate_hz"]),
        candidate_sample_rate=int(candidate_receipt["audio_sample_rate_hz"]),
    )
    audio_gates = evaluate_audio_quality(decoded_audio, thresholds)
    expected_shape = [
        int(thresholds["exact_num_frames"]),
        int(thresholds["exact_video_height"]),
        int(thresholds["exact_video_width"]),
        3,
    ]
    shape_matches = list(decoded.shape) == expected_shape
    receipt = {
        "source_revision": source_revision,
        **input_records,
        "quality_contract": "human_visible_video_and_direct_stereo_waveform_parity",
        "pixel_metrics_gating": False,
        "shape": list(decoded.shape),
        "expected_shape": expected_shape,
        "shape_matches": shape_matches,
        "mse": decoded.mse,
        "mean_absolute_error": decoded.mean_absolute_error,
        "maximum_absolute_error": decoded.maximum_absolute_error,
        "psnr_db": decoded.psnr_db,
        "audio_shape": list(decoded_audio.candidate_shape),
        "expected_audio_shape": [
            int(thresholds["exact_audio_channels"]),
            int(thresholds["exact_audio_num_samples"]),
        ],
        "audio_sample_rate_hz": decoded_audio.candidate_sample_rate,
        "audio_duration_s": decoded_audio.candidate_duration_s,
        "metrics": {name: result.to_dict() for name, result in {**gates, **audio_gates}.items()},
    }
    receipt["passed"] = visual_quality_passed(gates) and audio_quality_passed(audio_gates)
    atomic_write_json(Path(args.output), receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
