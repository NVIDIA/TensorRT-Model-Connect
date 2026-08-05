#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a fixed, stratified Full-Duplex-Bench validation slice.

The public benchmark remains the source of truth.  This tool selects an
outcome-independent subset, normalizes each input once, and records the
benchmark annotations needed to score the same inputs through HF and TRTMC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Sequence


FDB_REVISION = "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
SELECTION_SEED = (
    f"full-duplex-bench-v1@{FDB_REVISION}:trtmc-validate-v1"
)
SAMPLE_RATE = 24_000
FRAME_RATE = 12.5
CATEGORY_COUNTS = {
    "candor_pause_handling": 216,
    "candor_turn_taking": 119,
    "icc_backchannel": 55,
    "synthetic_pause_handling": 137,
    "synthetic_user_interruption": 200,
}
CATEGORY_LICENSES = {
    "candor_pause_handling": (
        "CC BY-NC 4.0; upstream CANDOR terms also apply"
    ),
    "candor_turn_taking": (
        "CC BY-NC 4.0; upstream CANDOR terms also apply"
    ),
    "icc_backchannel": "CC BY-NC 4.0; upstream ICC terms also apply",
    "synthetic_pause_handling": "MIT",
    "synthetic_user_interruption": "MIT",
}


def load_source_manifest(source_root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = source_root / "DATASET_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{source_root}: missing DATASET_MANIFEST.json; stage the managed "
            "Full-Duplex-Bench source manifest before deriving a validation slice"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("upstream_revision") != FDB_REVISION:
        raise ValueError(
            f"{manifest_path}: expected upstream revision {FDB_REVISION}, "
            f"found {manifest.get('upstream_revision')!r}"
        )
    if int(manifest.get("sample_count", -1)) != sum(CATEGORY_COUNTS.values()):
        raise ValueError(f"{manifest_path}: source sample count is not 727")
    subsets = manifest.get("subsets")
    if not isinstance(subsets, dict):
        raise ValueError(f"{manifest_path}: missing subsets metadata")
    for category, expected_count in CATEGORY_COUNTS.items():
        subset = subsets.get(category)
        if not isinstance(subset, dict):
            raise ValueError(f"{manifest_path}: missing subset {category}")
        if int(subset.get("samples", -1)) != expected_count:
            raise ValueError(
                f"{manifest_path}: {category} sample count is not "
                f"{expected_count}"
            )
        if subset.get("license") != CATEGORY_LICENSES[category]:
            raise ValueError(
                f"{manifest_path}: {category} license does not match the "
                "managed source contract"
            )
    return manifest, manifest_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _numeric_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return 2**63 - 1, path.name


def discover_samples(source_root: Path) -> dict[str, list[Path]]:
    discovered: dict[str, list[Path]] = {}
    for category, expected_count in CATEGORY_COUNTS.items():
        category_root = source_root / category
        samples = sorted(
            (
                path
                for path in category_root.iterdir()
                if path.is_dir() and (path / "input.wav").is_file()
            ),
            key=_numeric_key,
        )
        if len(samples) != expected_count:
            raise ValueError(
                f"expected {expected_count} {category} samples under "
                f"{source_root}, found {len(samples)}"
            )
        discovered[category] = samples
    return discovered


def select_samples(
    discovered: dict[str, list[Path]],
    *,
    samples_per_category: int,
    seed: str = SELECTION_SEED,
) -> dict[str, list[Path]]:
    if samples_per_category <= 0:
        raise ValueError("samples_per_category must be positive")
    selected: dict[str, list[Path]] = {}
    for category, samples in discovered.items():
        if samples_per_category > len(samples):
            raise ValueError(
                f"requested {samples_per_category} {category} samples, "
                f"but only {len(samples)} are available"
            )
        selected[category] = sorted(
            samples,
            key=lambda path: (
                hashlib.sha256(
                    f"{seed}:{category}:{path.name}".encode("utf-8")
                ).hexdigest(),
                _numeric_key(path),
            ),
        )[:samples_per_category]
    return selected


def _prepare_audio(source: Path, destination: Path) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    audio, source_rate = sf.read(
        str(source), dtype="float32", always_2d=True
    )
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    if source_rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(source_rate), SAMPLE_RATE)
        mono = np.asarray(
            resample_poly(
                mono,
                up=SAMPLE_RATE // divisor,
                down=int(source_rate) // divisor,
            ),
            dtype=np.float32,
        )
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"invalid audio in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.wav")
    # libsndfile adds a wall-clock timestamp to the PEAK chunk of FLOAT WAVs,
    # which makes byte-identical input audio hash differently on every
    # preparation. Write a minimal IEEE-float WAV with a deterministic header.
    _write_float32_wav(temporary, mono)
    os.replace(temporary, destination)
    frame_count = max(
        1,
        int(math.ceil(mono.shape[0] / SAMPLE_RATE * FRAME_RATE - 1.0e-9)),
    )
    return {
        "source_sample_rate": int(source_rate),
        "source_channels": int(audio.shape[1]),
        "prepared_samples": int(mono.shape[0]),
        "max_frames": frame_count,
    }


def _write_float32_wav(path: Path, audio: Any) -> None:
    import numpy as np

    samples = np.asarray(audio, dtype="<f4").reshape(-1)
    payload = samples.tobytes()
    fmt = struct.pack(
        "<HHIIHH",
        3,  # WAVE_FORMAT_IEEE_FLOAT
        1,
        SAMPLE_RATE,
        SAMPLE_RATE * 4,
        4,
        32,
    )
    fact = struct.pack("<I", samples.size)
    riff_size = 4 + (8 + len(fmt)) + (8 + len(fact)) + (8 + len(payload))
    with path.open("wb") as stream:
        stream.write(b"RIFF")
        stream.write(struct.pack("<I", riff_size))
        stream.write(b"WAVE")
        for name, content in ((b"fmt ", fmt), (b"fact", fact), (b"data", payload)):
            stream.write(name)
            stream.write(struct.pack("<I", len(content)))
            stream.write(content)


def _annotation_payload(sample: Path, category: str) -> dict[str, Any]:
    if category == "candor_turn_taking":
        turns = json.loads(
            (sample / "turn_taking.json").read_text(encoding="utf-8")
        )
        return {"input_end_seconds": float(turns[0]["timestamp"][0])}
    if category == "synthetic_user_interruption":
        interruptions = json.loads(
            (sample / "interrupt.json").read_text(encoding="utf-8")
        )
        return {"input_end_seconds": float(interruptions[0]["timestamp"][1])}
    return {}


def prepare_dataset(
    *,
    source_root: Path,
    output_root: Path,
    icc_distribution_path: Path,
    samples_per_category: int = 30,
) -> Path:
    _source_manifest, source_manifest_path = load_source_manifest(source_root)
    discovered = discover_samples(source_root)
    selected = select_samples(
        discovered,
        samples_per_category=samples_per_category,
    )
    icc_distributions = json.loads(
        icc_distribution_path.read_text(encoding="utf-8")
    )
    if set(icc_distributions) != {
        sample.name for sample in discovered["icc_backchannel"]
    }:
        raise ValueError(
            f"{icc_distribution_path} does not cover the 55 ICC samples"
        )

    by_category: dict[str, list[dict[str, Any]]] = {}
    for category, samples in selected.items():
        requests = []
        for sample in samples:
            source_audio = sample / "input.wav"
            relative_audio = Path("audio") / category / sample.name / "input.wav"
            prepared_audio = output_root / relative_audio
            audio_metadata = _prepare_audio(source_audio, prepared_audio)
            scoring = _annotation_payload(sample, category)
            if category == "icc_backchannel":
                scoring["ground_truth_distribution"] = icc_distributions[
                    sample.name
                ]
            requests.append(
                {
                    "sample_id": f"fdb_{category}_{sample.name}",
                    "stage": "full_generation",
                    "category": category,
                    "source_sample_id": sample.name,
                    "source_relative_path": str(
                        source_audio.relative_to(source_root)
                    ),
                    "source_sha256": _sha256(source_audio),
                    "prepared_sha256": _sha256(prepared_audio),
                    "scoring": scoring,
                    "inputs": {
                        "audio": str(relative_audio),
                        "speech_test_max_frames": audio_metadata["max_frames"],
                        "max_new_tokens": audio_metadata["max_frames"],
                        "disable_teacher_replay": True,
                        "reference_mode": "official_greedy",
                    },
                    "preprocessing": {
                        **audio_metadata,
                        "target_sample_rate": SAMPLE_RATE,
                        "channel_conversion": "mean-to-mono-float32",
                        "resampler": "scipy.signal.resample_poly",
                        "output_subtype": "FLOAT",
                    },
                }
            )
        by_category[category] = requests

    requests = [
        by_category[category][index]
        for index in range(samples_per_category)
        for category in CATEGORY_COUNTS
    ]
    manifest = {
        "schema_version": "trtmc.full-duplex-bench-validation/v1",
        "dataset": "Full-Duplex-Bench v1.0 stratified validation slice",
        "source": "DanielLin94144/Full-Duplex-Bench",
        "source_repository": (
            "https://github.com/DanielLin94144/Full-Duplex-Bench"
        ),
        "source_revision": FDB_REVISION,
        "source_manifest": {
            "path": "../DATASET_MANIFEST.json",
            "sha256": _sha256(source_manifest_path),
        },
        "paper": "https://arxiv.org/abs/2503.04721",
        "source_sample_count": sum(CATEGORY_COUNTS.values()),
        "licenses": CATEGORY_LICENSES,
        "usage_notes": [
            "This slice is for internal research and QA.",
            (
                "CANDOR and ICC are non-commercial subsets; their upstream "
                "terms must be reviewed before other use."
            ),
        ],
        "sampling": {
            "method": "fixed SHA-256 rank within each benchmark category",
            "seed": SELECTION_SEED,
            "samples_per_category": samples_per_category,
            "category_counts": {
                category: len(rows) for category, rows in by_category.items()
            },
            "outcome_independent": True,
        },
        "request_count": len(requests),
        "requests": requests,
    }
    manifest_path = output_root / "dataset.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--icc-distribution", type=Path, required=True)
    parser.add_argument("--samples-per-category", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output = prepare_dataset(
        source_root=arguments.source_root,
        output_root=arguments.output_root,
        icc_distribution_path=arguments.icc_distribution,
        samples_per_category=arguments.samples_per_category,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
