#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score aligned HF/TRTMC outputs with Full-Duplex-Bench metric definitions.

Metric provenance: https://github.com/DanielLin94144/Full-Duplex-Bench at the
immutable ``FDB_REVISION`` below. This is a paired TRTMC implementation for
Dev/QA comparison; it does not execute or claim byte-equivalence with the
upstream evaluator scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FDB_REVISION = "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
SELECTION_SEED = (
    f"full-duplex-bench-v1@{FDB_REVISION}:trtmc-validate-v1"
)
ASR_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
ASR_REVISION = "ae9ad07059c7c739ffaf932226a8fe64ae2620b0"
ASR_FILENAME = "parakeet-tdt-0.6b-v2.nemo"
METRIC_FIELDS = {
    "synthetic_pause_handling": ("tor",),
    "candor_pause_handling": ("tor",),
    "icc_backchannel": ("tor", "frequency", "jsd"),
    "candor_turn_taking": ("tor",),
    "synthetic_user_interruption": ("tor",),
}
VALIDATION_SAMPLES_PER_CATEGORY = 30


def validate_requests_manifest(answers: Mapping[str, Any]) -> list[dict[str, Any]]:
    if answers.get("schema_version") != (
        "trtmc.full-duplex-bench-validation/v1"
    ):
        raise ValueError("unsupported Full-Duplex-Bench validation schema")
    if answers.get("source_revision") != FDB_REVISION:
        raise ValueError("Full-Duplex-Bench source revision is not pinned")
    sampling = answers.get("sampling")
    if not isinstance(sampling, Mapping) or sampling.get("seed") != SELECTION_SEED:
        raise ValueError("Full-Duplex-Bench selection seed is not pinned")
    requests = answers.get("requests")
    if not isinstance(requests, list):
        raise ValueError("Full-Duplex-Bench answers must contain requests")
    counts = {category: 0 for category in METRIC_FIELDS}
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("Full-Duplex-Bench requests must be objects")
        category = str(request.get("category", "") or "")
        if category not in counts:
            raise ValueError(
                f"unsupported Full-Duplex-Bench category {category!r}"
            )
        counts[category] += 1
    expected = {
        category: VALIDATION_SAMPLES_PER_CATEGORY for category in METRIC_FIELDS
    }
    if counts != expected:
        raise ValueError(
            "formal Full-Duplex-Bench validation requires exactly 30 samples "
            f"per category, found {counts}"
        )
    return requests


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_rows(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("responses", [])
    if not isinstance(rows, list):
        raise ValueError("prediction payload must contain a responses list")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("prediction rows must be objects")
        sample_id = str(row.get("sample_id", "") or "")
        if not sample_id or sample_id in indexed:
            raise ValueError(f"invalid or duplicate prediction sample_id {sample_id!r}")
        indexed[sample_id] = row
    return indexed


def _stage_data(row: Mapping[str, Any]) -> Mapping[str, Any]:
    stage_output = row.get("stage_output", {})
    if not isinstance(stage_output, Mapping):
        return {}
    data = stage_output.get("data", {})
    return data if isinstance(data, Mapping) else {}


def _wav_path(row: Mapping[str, Any]) -> Path:
    value = row.get("wav_path") or _stage_data(row).get("wav_path")
    path = Path(str(value or ""))
    if not path.is_file():
        raise FileNotFoundError(f"prediction audio does not exist: {path}")
    return path


def _request_input(request: Mapping[str, Any]) -> Path:
    inputs = request.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise ValueError("Full-Duplex-Bench request inputs must be an object")
    path = Path(str(inputs.get("audio", "") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"benchmark input does not exist: {path}")
    expected_sha256 = str(request.get("prepared_sha256", "") or "")
    if len(expected_sha256) != 64 or _sha256(path) != expected_sha256:
        raise ValueError(f"benchmark input checksum does not match: {path}")
    return path


def _aligned_output(path: Path, input_path: Path):
    import numpy as np
    import soundfile as sf

    output, output_rate = sf.read(str(path), dtype="float32", always_2d=True)
    output = np.asarray(output.mean(axis=1), dtype=np.float32)
    input_info = sf.info(str(input_path))
    input_rate = int(input_info.samplerate)
    if output_rate != input_rate:
        import torch
        import torchaudio

        waveform = torch.from_numpy(np.ascontiguousarray(output[None, :]))
        output = (
            torchaudio.functional.resample(
                waveform,
                int(output_rate),
                input_rate,
            )
            .squeeze(0)
            .numpy()
            .astype(np.float32, copy=False)
        )
        output_rate = input_rate
    target_samples = int(input_info.frames)
    if output.shape[0] > target_samples:
        output = output[:target_samples]
    elif output.shape[0] < target_samples:
        output = np.pad(output, (0, target_samples - output.shape[0]))
    return output, int(output_rate)


def _load_asr(*, local_files_only: bool):
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import hf_hub_download

    model_file = hf_hub_download(
        repo_id=ASR_MODEL,
        filename=ASR_FILENAME,
        revision=ASR_REVISION,
        local_files_only=local_files_only,
    )
    return nemo_asr.models.ASRModel.restore_from(restore_path=model_file).cuda()


def _transcribe(
    *,
    model: Any,
    wav_path: Path,
    input_path: Path,
    offset: float,
    cache_path: Path,
) -> dict[str, Any]:
    import soundfile as sf

    cache_identity = {
        "wav_sha256": _sha256(wav_path),
        "input_sha256": _sha256(input_path),
        "offset": offset,
        "asr_model": ASR_MODEL,
        "asr_revision": ASR_REVISION,
    }
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if all(cached.get(key) == value for key, value in cache_identity.items()):
            return cached

    waveform, sample_rate = _aligned_output(wav_path, input_path)
    waveform = waveform[int(round(offset * sample_rate)) :]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        sf.write(str(temporary_path), waveform, sample_rate)
        result = model.transcribe([str(temporary_path)], timestamps=True)[0]
    finally:
        temporary_path.unlink(missing_ok=True)
    chunks = [
        {
            "text": word["word"],
            "timestamp": [
                float(word["start"]) + offset,
                float(word["end"]) + offset,
            ],
        }
        for word in result.timestamp["word"]
    ]
    payload = {
        **cache_identity,
        "text": " ".join(str(chunk["text"]) for chunk in chunks).strip(),
        "chunks": chunks,
    }
    _write_json(cache_path, payload)
    return payload


def _tor(chunks: list[dict[str, Any]]) -> int:
    if not chunks:
        return 0
    first = float(chunks[0]["timestamp"][0])
    last_raw = chunks[-1]["timestamp"][1]
    last = float(last_raw if last_raw is not None else first)
    return 0 if last - first < 1.0 and len(chunks) <= 3 else 1


def _overlap_words(
    chunks: Iterable[Mapping[str, Any]], start: float, end: float
) -> list[str]:
    words = []
    for chunk in chunks:
        word_start, word_end = chunk["timestamp"]
        if word_start is None or (word_end is None and float(word_start) < end):
            continue
        word_start = float(word_start)
        word_end = word_start if word_end is None else float(word_end)
        if (
            (word_start >= start and word_end <= end)
            or (word_start <= end and word_end > end)
            or (word_start <= start and word_end > start)
        ):
            words.append(str(chunk["text"]))
    return words


def _score_backchannel(
    *,
    wav_path: Path,
    input_path: Path,
    transcription: Mapping[str, Any],
    ground_truth: Sequence[float],
    vad_model: Any,
) -> dict[str, float]:
    import numpy as np
    import torch
    import torchaudio
    from silero_vad import get_speech_timestamps

    audio, sample_rate = _aligned_output(wav_path, input_path)
    waveform = torch.from_numpy(np.ascontiguousarray(audio[None, :]))
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        sample_rate = 16_000
    segments = get_speech_timestamps(waveform, vad_model, return_seconds=True)
    duration = waveform.shape[-1] / sample_rate
    return _backchannel_metrics(
        segments=segments,
        chunks=transcription["chunks"],
        duration=duration,
        ground_truth=ground_truth,
    )


def _backchannel_metrics(
    *,
    segments: Iterable[Mapping[str, Any]],
    chunks: Iterable[Mapping[str, Any]],
    duration: float,
    ground_truth: Sequence[float],
) -> dict[str, float]:
    """Compute the benchmark's TOR, frequency, and timing-distribution metrics."""
    import numpy as np

    if duration <= 0.0:
        raise ValueError("backchannel audio duration must be positive")

    chunks = list(chunks)
    predicted: list[list[float]] = []
    tor = 0
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        segment_duration = end - start
        if segment_duration > 3.0:
            tor = 1
            break
        words = _overlap_words(chunks, start, end)
        if len(words) > 3:
            tor = 1
        elif segment_duration < 1.0:
            tor = 0 if len(words) <= 2 else 1
        else:
            tor = 1
        predicted.append([start, end])
    frequency = len(predicted) / duration
    if not predicted:
        jsd = 1.0
    else:
        intervals = np.zeros(int(duration / 0.2) + 1, dtype=np.float64)
        for start, end in predicted:
            for position in range(int(start / 0.2), int(end / 0.2) + 1):
                if position < len(intervals):
                    intervals[position] += 1
        intervals += 1.0e-10
        intervals /= intervals.sum()
        ground_truth_array = np.asarray(ground_truth, dtype=np.float64)
        resized = np.interp(
            np.linspace(0.0, 1.0, len(intervals)),
            np.linspace(0.0, 1.0, len(ground_truth_array)),
            ground_truth_array,
        )
        resized /= resized.sum()
        midpoint = 0.5 * (intervals + resized)
        left = np.sum(
            intervals[intervals > 0.0]
            * np.log(intervals[intervals > 0.0] / midpoint[intervals > 0.0])
        )
        right = np.sum(
            resized[resized > 0.0]
            * np.log(resized[resized > 0.0] / midpoint[resized > 0.0])
        )
        jsd = float(np.sqrt(0.5 * (left + right)))
    return {"tor": float(tor), "frequency": frequency, "jsd": jsd}


def score_backend(
    *,
    backend: str,
    predictions: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    cache_root: Path,
    asr_model: Any,
    vad_model: Any,
) -> dict[str, Any]:
    rows = _prediction_rows(predictions)
    expected_ids = {
        str(request.get("sample_id", "") or "") for request in requests
    }
    if set(rows) != expected_ids:
        missing = sorted(expected_ids - set(rows))
        extra = sorted(set(rows) - expected_ids)
        raise ValueError(
            f"{backend} predictions do not match selected requests: "
            f"missing={missing}, extra={extra}"
        )
    cases: list[dict[str, Any]] = []
    for request in requests:
        sample_id = str(request.get("sample_id", "") or "")
        if sample_id not in rows:
            raise ValueError(f"{backend} has no prediction for {sample_id}")
        category = str(request.get("category", "") or "")
        if category not in METRIC_FIELDS:
            raise ValueError(f"unsupported Full-Duplex-Bench category {category!r}")
        scoring = request.get("scoring", {})
        if not isinstance(scoring, Mapping):
            raise ValueError(f"{sample_id} scoring must be an object")
        input_path = _request_input(request)
        output_path = _wav_path(rows[sample_id])
        offset = (
            float(scoring.get("input_end_seconds", 0.0) or 0.0)
            if category == "synthetic_user_interruption"
            else 0.0
        )
        transcription = _transcribe(
            model=asr_model,
            wav_path=output_path,
            input_path=input_path,
            offset=offset,
            cache_path=cache_root / backend / sample_id / "asr.json",
        )
        if category == "icc_backchannel":
            ground_truth = scoring.get("ground_truth_distribution")
            if not isinstance(ground_truth, list) or not ground_truth:
                raise ValueError(f"{sample_id} has no ICC ground-truth distribution")
            metrics = _score_backchannel(
                wav_path=output_path,
                input_path=input_path,
                transcription=transcription,
                ground_truth=ground_truth,
                vad_model=vad_model,
            )
        else:
            metrics = {"tor": float(_tor(transcription["chunks"]))}
        cases.append(
            {
                "sample_id": sample_id,
                "category": category,
                **metrics,
                "transcript": transcription["text"],
            }
        )

    metrics: dict[str, dict[str, float | int]] = {}
    for category, fields in METRIC_FIELDS.items():
        category_rows = [row for row in cases if row["category"] == category]
        for field in fields:
            values = [float(row[field]) for row in category_rows]
            key = f"{category}.{field}"
            metrics[key] = {
                "value": statistics.fmean(values),
                "stddev": statistics.pstdev(values),
                "sample_count": len(values),
            }
    return {
        "backend": backend,
        "metrics": metrics,
        "cases": cases,
    }


def compare_scores(
    hf_score: Mapping[str, Any],
    trtmc_score: Mapping[str, Any],
    *,
    gates: Mapping[str, float],
) -> dict[str, Any]:
    hf_metrics = hf_score["metrics"]
    trtmc_metrics = trtmc_score["metrics"]
    hf_cases = {
        (str(row["category"]), str(row["sample_id"])): row
        for row in hf_score["cases"]
    }
    trtmc_cases = {
        (str(row["category"]), str(row["sample_id"])): row
        for row in trtmc_score["cases"]
    }
    if set(hf_cases) != set(trtmc_cases):
        raise ValueError("HF and TRTMC score cases are not aligned")
    metric_rows: dict[str, dict[str, Any]] = {}
    gate_failures = []
    for category, fields in METRIC_FIELDS.items():
        for field in fields:
            name = f"{category}.{field}"
            hf_value = float(hf_metrics[name]["value"])
            trtmc_value = float(trtmc_metrics[name]["value"])
            if field == "tor":
                gate_name = "max_tor_abs_delta"
            elif field == "frequency":
                gate_name = "max_backchannel_frequency_abs_delta"
            else:
                gate_name = "max_backchannel_jsd_abs_delta"
            threshold = float(gates[gate_name])
            delta = trtmc_value - hf_value
            passed = abs(delta) <= threshold
            paired_abs_deltas = [
                abs(
                    float(trtmc_cases[key][field])
                    - float(hf_case[field])
                )
                for key, hf_case in hf_cases.items()
                if key[0] == category
            ]
            row = {
                "hf": hf_value,
                "trtmc": trtmc_value,
                "trtmc_minus_hf": delta,
                "abs_delta": abs(delta),
                "threshold": threshold,
                "passed": passed,
                "sample_count": int(hf_metrics[name]["sample_count"]),
                "paired_changed_count": sum(
                    value > 1.0e-12 for value in paired_abs_deltas
                ),
                "paired_mean_abs_delta": statistics.fmean(
                    paired_abs_deltas
                ),
                "paired_max_abs_delta": max(paired_abs_deltas),
            }
            metric_rows[name] = row
            if not passed:
                gate_failures.append(
                    {
                        "metric": name,
                        "value": abs(delta),
                        "threshold": threshold,
                        "operator": "<=",
                    }
                )

    disagreements = []
    for failure in gate_failures:
        category, field = str(failure["metric"]).rsplit(".", 1)
        candidates = []
        for key, hf_case in hf_cases.items():
            if key[0] != category:
                continue
            trtmc_case = trtmc_cases[key]
            candidates.append(
                (
                    abs(float(trtmc_case[field]) - float(hf_case[field])),
                    key[1],
                    hf_case,
                    trtmc_case,
                )
            )
        if candidates:
            difference, sample_id, hf_case, trtmc_case = max(candidates)
            disagreements.append(
                {
                    "sample_id": sample_id,
                    "category": category,
                    "metric": field,
                    "case_abs_delta": difference,
                    "hf": hf_case[field],
                    "trtmc": trtmc_case[field],
                }
            )
    passed_count = len(metric_rows) - len(gate_failures)
    return {
        "status": "passed" if not gate_failures else "failed",
        "sample_count": len(hf_score["cases"]),
        "valid_count": len(hf_score["cases"]),
        "metric_gate_count": len(metric_rows),
        "passed_count": passed_count,
        "metric_gate_pass_rate": passed_count / len(metric_rows),
        "metrics": metric_rows,
        "gates": dict(gates),
        "gate_failures": gate_failures,
        "disagreements": disagreements,
        "hf_score": hf_score,
        "trtmc_score": trtmc_score,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-predictions", type=Path, required=True)
    parser.add_argument("--trtmc-predictions", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tor-abs-delta", type=float, required=True)
    parser.add_argument(
        "--max-backchannel-frequency-abs-delta", type=float, required=True
    )
    parser.add_argument(
        "--max-backchannel-jsd-abs-delta", type=float, required=True
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    hf_predictions = json.loads(arguments.hf_predictions.read_text(encoding="utf-8"))
    trtmc_predictions = json.loads(
        arguments.trtmc_predictions.read_text(encoding="utf-8")
    )
    answers = json.loads(arguments.requests.read_text(encoding="utf-8"))
    requests = validate_requests_manifest(answers)
    asr_model = _load_asr(local_files_only=arguments.local_files_only)
    from silero_vad import load_silero_vad

    vad_model = load_silero_vad()
    hf_score = score_backend(
        backend="hf",
        predictions=hf_predictions,
        requests=requests,
        cache_root=arguments.cache_root,
        asr_model=asr_model,
        vad_model=vad_model,
    )
    trtmc_score = score_backend(
        backend="trtmc",
        predictions=trtmc_predictions,
        requests=requests,
        cache_root=arguments.cache_root,
        asr_model=asr_model,
        vad_model=vad_model,
    )
    summary = compare_scores(
        hf_score,
        trtmc_score,
        gates={
            "max_tor_abs_delta": arguments.max_tor_abs_delta,
            "max_backchannel_frequency_abs_delta": (
                arguments.max_backchannel_frequency_abs_delta
            ),
            "max_backchannel_jsd_abs_delta": (
                arguments.max_backchannel_jsd_abs_delta
            ),
        },
    )
    summary.update(
        {
            "schema_version": "trtmc.full-duplex-bench-comparison/v1",
            "dataset": "Full-Duplex-Bench v1.0 stratified validation slice",
            "dataset_source_revision": answers["source_revision"],
            "dataset_selection_seed": answers["sampling"]["seed"],
            "samples_per_category": VALIDATION_SAMPLES_PER_CATEGORY,
            "metric_definition_revision": FDB_REVISION,
            "metric_implementation": "trtmc_paired_full_duplex_bench_v1",
            "asr_model": ASR_MODEL,
            "asr_revision": ASR_REVISION,
        }
    )
    _write_json(arguments.output, summary)
    print(
        f"status={summary['status']} "
        f"passed={summary['passed_count']}/{summary['metric_gate_count']} "
        f"samples={summary['sample_count']} output={arguments.output}"
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
