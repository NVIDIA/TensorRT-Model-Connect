# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Omni-multimodal comparator — compare TRT vs reference multi-branch outputs.

Metrics per branch: thinker token agreement, vision/audio embedding cosine,
talker audio validity and duration, and e2e text edit distance.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)
from ._helpers import cosine_similarity, normalized_edit_distance

logger = logging.getLogger(__name__)

_SIMPLE_WAVEFORM_FALLBACK = "no Code2Wav engine, generating simple waveform"


def _read_wav(path: Path) -> tuple[dict[str, Any], str]:
    """Read the PCM16 reference or IEEE-float32 product WAV."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return {}, f"invalid WAV: {exc}"

    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        return {}, "invalid WAV: missing RIFF/WAVE header"

    fmt: bytes | None = None
    audio: bytes | None = None
    offset = 12
    while offset + 8 <= len(payload):
        chunk_name = payload[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(payload):
            return {}, f"truncated WAV chunk {chunk_name!r}"
        if chunk_name == b"fmt ":
            fmt = payload[chunk_start:chunk_end]
        elif chunk_name == b"data":
            audio = payload[chunk_start:chunk_end]
        offset = chunk_end + (chunk_size & 1)

    if fmt is None or len(fmt) < 16 or audio is None:
        return {}, "invalid WAV: missing fmt or data chunk"

    audio_format, channels, sample_rate, _byte_rate, block_align, bits_per_sample = (
        struct.unpack_from("<HHIIHH", fmt)
    )
    sample_width = bits_per_sample // 8
    expected_block_align = channels * sample_width
    encoding = {
        (1, 16): "pcm_s16le",
        (3, 32): "ieee_float32le",
    }.get((audio_format, bits_per_sample), "unsupported")
    if (
        channels != 1
        or sample_rate < 1
        or block_align != expected_block_align
        or expected_block_align < 1
        or not audio
        or len(audio) % expected_block_align != 0
        or encoding == "unsupported"
    ):
        return {
            "channels": channels,
            "sample_width_bytes": sample_width,
            "sample_rate_hz": sample_rate,
            "num_samples": len(audio) // expected_block_align if expected_block_align > 0 else 0,
            "encoding": encoding,
        }, "WAV must be non-empty mono PCM16 or IEEE float32"

    num_samples = len(audio) // expected_block_align
    if encoding == "pcm_s16le":
        samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
    else:
        samples = np.frombuffer(audio, dtype="<f4").astype(np.float32)
        if not np.all(np.isfinite(samples)):
            return {}, "IEEE float32 WAV contains non-finite samples"
    return {
        "channels": channels,
        "sample_width_bytes": sample_width,
        "encoding": encoding,
        "sample_rate_hz": sample_rate,
        "num_samples": num_samples,
        "duration_s": num_samples / sample_rate,
        "rms": float(np.sqrt(np.mean(np.square(samples)))),
        "peak": float(np.max(np.abs(samples))),
        "samples": samples,
    }, ""


def _is_invariant_only(ref: StageOutput) -> bool:
    return (
        bool((ref.data or {}).get("_invariant_only"))
        or ref.metadata.get("source") == "invariant_only"
    )


def _token_agreement(a: List[int], b: List[int]) -> float:
    """Fraction of tokens that match between two sequences."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    min_len = min(len(a), len(b))
    matches = sum(1 for i in range(min_len) if a[i] == b[i])
    return matches / max(len(a), len(b))


class OmniComparator:
    """Compare TRT vs reference omni-multimodal outputs.

    Evaluates stage-specific metrics depending on the stage name.
    """

    @property
    def task_strategy(self) -> str:
        return "omni_multimodal"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        stage_name = stage.name
        th = threshold.metrics

        if _is_invariant_only(ref):
            return self._compare_invariants(trt, ref, th, stage)

        # Dispatch to stage-specific comparison
        if stage_name == "thinker_decode":
            return self._compare_thinker(trt, ref, th, stage)
        elif stage_name in ("vision_encode", "audio_encode"):
            return self._compare_encoder(trt, ref, th, stage)
        elif stage_name == "talker_decode":
            return self._compare_talker(trt, ref, th, stage)
        elif stage_name == "end_to_end":
            return self._compare_e2e(trt, ref, th, stage)
        else:
            return self._compare_generic(trt, ref, th, stage)

    def _compare_thinker(
        self,
        trt: StageOutput,
        ref: StageOutput,
        th: Dict[str, float],
        stage: StageSpec,
    ) -> CompareResult:
        """Compare thinker text decoding output."""
        trt_tokens = trt.data.get("token_ids", [])
        ref_tokens = ref.data.get("token_ids", [])

        metrics: Dict[str, MetricResult] = {}

        if trt_tokens and ref_tokens:
            agreement = _token_agreement(trt_tokens, ref_tokens)
            thresh = th.get("thinker_token_agreement", 0.8)
            metrics["thinker_token_agreement"] = MetricResult(
                value=agreement, threshold=thresh, operator=">=", passed=agreement >= thresh
            )

        trt_text = trt.text or ""
        ref_text = ref.text or ""
        if trt_text or ref_text:
            ned = normalized_edit_distance(trt_text, ref_text)
            thresh = th.get("thinker_text_edit_distance", 0.3)
            metrics["thinker_text_edit_distance"] = MetricResult(
                value=ned, threshold=thresh, operator="<=", passed=ned <= thresh
            )

        passed = all(m.passed for m in metrics.values()) if metrics else False
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"Thinker comparison: {len(metrics)} metrics",
        )

    def _compare_encoder(
        self,
        trt: StageOutput,
        ref: StageOutput,
        th: Dict[str, float],
        stage: StageSpec,
    ) -> CompareResult:
        """Compare vision/audio encoder embedding output."""
        trt_emb = trt.data.get("embedding", [])
        ref_emb = ref.data.get("embedding", [])

        if not trt_emb or not ref_emb:
            missing = []
            if not trt_emb:
                missing.append("TRT")
            if not ref_emb:
                missing.append("ref")
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"Missing embedding for {stage.name} from {', '.join(missing)}",
            )

        cosine = cosine_similarity(np.asarray(trt_emb), np.asarray(ref_emb))
        # Use canonical name from threshold defaults (e.g. "vision_embedding_cosine")
        branch = stage.name.replace("_encode", "")
        metric_name = f"{branch}_embedding_cosine"
        # Look up threshold: canonical name -> stage-based name -> generic fallback
        thresh = th.get(
            metric_name,
            th.get(f"{stage.name}_embedding_cosine", th.get("encoder_embedding_cosine", 0.95)),
        )
        metrics: Dict[str, MetricResult] = {
            metric_name: MetricResult(
                value=cosine, threshold=thresh, operator=">=", passed=cosine >= thresh
            ),
        }

        passed = all(m.passed for m in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"{stage.name} embedding cosine={cosine:.6f}",
        )

    def _compare_talker(
        self,
        trt: StageOutput,
        ref: StageOutput,
        th: Dict[str, float],
        stage: StageSpec,
    ) -> CompareResult:
        """Compare talker decoding output (tokens and/or audio)."""
        metrics: Dict[str, MetricResult] = {}

        trt_tokens = trt.data.get("token_ids", [])
        ref_tokens = ref.data.get("token_ids", [])

        if trt_tokens and ref_tokens:
            agreement = _token_agreement(trt_tokens, ref_tokens)
            thresh = th.get("talker_token_match", 0.7)
            metrics["talker_token_match"] = MetricResult(
                value=agreement, threshold=thresh, operator=">=", passed=agreement >= thresh
            )

        passed = all(m.passed for m in metrics.values()) if metrics else True
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"Talker comparison: {len(metrics)} metrics",
        )

    def _compare_e2e(
        self,
        trt: StageOutput,
        ref: StageOutput,
        th: Dict[str, float],
        stage: StageSpec,
    ) -> CompareResult:
        """Compare end-to-end omni output (text edit distance)."""
        trt_text = trt.text or ""
        ref_text = ref.text or ""

        ned = normalized_edit_distance(trt_text, ref_text)
        thresh = th.get("e2e_text_edit_distance", 0.3)
        metrics: Dict[str, MetricResult] = {
            "e2e_text_edit_distance": MetricResult(
                value=ned, threshold=thresh, operator="<=", passed=ned <= thresh
            ),
        }

        passed = all(m.passed for m in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"E2E text edit distance={ned:.4f}",
        )

    def _compare_generic(
        self,
        trt: StageOutput,
        ref: StageOutput,
        th: Dict[str, float],
        stage: StageSpec,
    ) -> CompareResult:
        """Fallback: compare any stage with available data."""
        # Try embedding comparison
        trt_emb = trt.data.get("embedding", [])
        ref_emb = ref.data.get("embedding", [])
        if trt_emb and ref_emb:
            return self._compare_encoder(trt, ref, th, stage)

        # Try text comparison
        if trt.text and ref.text:
            return self._compare_e2e(trt, ref, th, stage)

        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.SKIPPED.value,
            metrics={},
            message=f"No comparable data for stage {stage.name} (skipped)",
        )

    def _compare_invariants(
        self,
        trt: StageOutput,
        ref: StageOutput,
        th: Dict[str, float],
        stage: StageSpec,
    ) -> CompareResult:
        """L4 invariants, calibrated by pinned HF audio shape when available."""
        metrics: Dict[str, MetricResult] = {}

        if stage.name == "talker_decode":
            audio_path = str(trt.metadata.get("audio_output_path", "") or "")
            audio_size = 0
            wav_evidence: dict[str, Any] = {}
            wav_error = "audio artifact is missing"
            if audio_path:
                path = Path(audio_path)
                if path.is_file():
                    audio_size = path.stat().st_size
                    wav_evidence, wav_error = _read_wav(path)
            bytes_min = th.get("audio_artifact_bytes_min", 44.0)
            metrics["audio_artifact_bytes"] = MetricResult(
                value=float(audio_size),
                threshold=bytes_min,
                operator=">=",
                passed=audio_size >= bytes_min,
                note="generated audio artifact includes a WAV header and PCM payload",
            )
            metrics["audio_wav_valid"] = MetricResult(
                value=1.0 if not wav_error else 0.0,
                threshold=1.0,
                operator="==",
                passed=not wav_error,
                note=wav_error or "audio artifact is a complete PCM WAV",
            )

            fallback_used = _SIMPLE_WAVEFORM_FALLBACK in str(trt.metadata.get("stderr", "") or "")
            metrics["simple_waveform_fallback_absent"] = MetricResult(
                value=0.0 if fallback_used else 1.0,
                threshold=1.0,
                operator="==",
                passed=not fallback_used,
                note="Code2Wav synthetic fallback must never certify",
            )

            ref_thinker_text = str((ref.data or {}).get("decoded_text", "") or "")
            if ref_thinker_text:
                trt_thinker_text = str((trt.data or {}).get("thinker_text", "") or "")
                thinker_text_matches = trt_thinker_text == ref_thinker_text
                metrics["thinker_text_exact"] = MetricResult(
                    value=1.0 if thinker_text_matches else 0.0,
                    threshold=1.0,
                    operator="==",
                    passed=thinker_text_matches,
                    note="must exactly match the pinned official-HF Thinker response",
                )

            if wav_evidence:
                channels = int(wav_evidence.get("channels", 0))
                encoding = str(wav_evidence.get("encoding", ""))
                sample_rate = int(wav_evidence.get("sample_rate_hz", 0))
                num_samples = int(wav_evidence.get("num_samples", 0))
                duration_s = float(wav_evidence.get("duration_s", 0.0))
                rms = float(wav_evidence.get("rms", 0.0))
                peak = float(wav_evidence.get("peak", 0.0))
                ref_sample_rate = int((ref.data or {}).get("sample_rate", 0))
                ref_num_samples = int((ref.data or {}).get("num_samples", 0))
                duration_ratio = num_samples / ref_num_samples if ref_num_samples > 0 else 0.0
                duration_min = th.get("audio_duration_s_min", 0.5)
                duration_ratio_min = th.get("audio_reference_duration_ratio_min", 0.5)
                rms_min = th.get("audio_rms_min", 0.005)
                peak_min = th.get("audio_peak_min", 0.02)
                metrics.update(
                    {
                        "audio_channels": MetricResult(
                            value=float(channels),
                            threshold=1.0,
                            operator="==",
                            passed=channels == 1,
                        ),
                        "audio_encoding_supported": MetricResult(
                            value=1.0 if encoding in ("pcm_s16le", "ieee_float32le") else 0.0,
                            threshold=1.0,
                            operator="==",
                            passed=encoding in ("pcm_s16le", "ieee_float32le"),
                            note=f"decoded {encoding}",
                        ),
                        "audio_sample_rate_hz": MetricResult(
                            value=float(sample_rate),
                            threshold=float(ref_sample_rate),
                            operator="==",
                            passed=ref_sample_rate > 0 and sample_rate == ref_sample_rate,
                            note="must match the pinned official-HF reference",
                        ),
                        "audio_duration_s": MetricResult(
                            value=duration_s,
                            threshold=duration_min,
                            operator=">=",
                            passed=duration_s >= duration_min,
                        ),
                        "audio_reference_duration_ratio": MetricResult(
                            value=duration_ratio,
                            threshold=duration_ratio_min,
                            operator=">=",
                            passed=duration_ratio >= duration_ratio_min,
                            note=f"TRT samples={num_samples}, HF samples={ref_num_samples}",
                        ),
                        "audio_num_samples_exact": MetricResult(
                            value=float(num_samples),
                            threshold=float(ref_num_samples),
                            operator="==",
                            passed=ref_num_samples > 0 and num_samples == ref_num_samples,
                            note="must exactly match the pinned official-HF audio shape",
                        ),
                        "audio_rms": MetricResult(
                            value=rms,
                            threshold=rms_min,
                            operator=">=",
                            passed=rms >= rms_min,
                        ),
                        "audio_peak": MetricResult(
                            value=peak,
                            threshold=peak_min,
                            operator=">=",
                            passed=peak >= peak_min,
                        ),
                    }
                )
                waveform_cosine_min = th.get("audio_reference_waveform_cosine_min", 0.25)
                reference_path = Path(str((ref.data or {}).get("wav_path", "") or ""))
                waveform_cosine = 0.0
                waveform_note = "pinned official-HF waveform is missing or invalid"
                if reference_path.is_file():
                    reference_evidence, reference_error = _read_wav(reference_path)
                    if not reference_error and reference_evidence:
                        actual_samples = np.asarray(wav_evidence["samples"], dtype=np.float32)
                        reference_samples = np.asarray(
                            reference_evidence["samples"], dtype=np.float32
                        )
                        common_samples = min(actual_samples.size, reference_samples.size)
                        actual_common = actual_samples[:common_samples]
                        reference_common = reference_samples[:common_samples]
                        denominator = float(
                            np.linalg.norm(actual_common) * np.linalg.norm(reference_common)
                        )
                        waveform_cosine = (
                            float(np.dot(actual_common, reference_common) / denominator)
                            if denominator > 0.0
                            else 0.0
                        )
                        waveform_note = (
                            f"compared {common_samples} aligned PCM samples against the "
                            "pinned official-HF waveform"
                        )
                    else:
                        waveform_note = f"invalid pinned official-HF waveform: {reference_error}"
                metrics["audio_reference_waveform_cosine"] = MetricResult(
                    value=waveform_cosine,
                    threshold=waveform_cosine_min,
                    operator=">=",
                    passed=waveform_cosine >= waveform_cosine_min,
                    note=waveform_note,
                )
        elif stage.name in ("thinker_decode", "end_to_end"):
            text = trt.text or str(trt.data.get("text", "") or "")
            if not text:
                text = str(trt.data.get("raw_output", "") or "")
            has_text = bool(text.strip())
            metrics["non_empty_text"] = MetricResult(
                value=1.0 if has_text else 0.0,
                threshold=1.0,
                operator="==",
                passed=has_text,
                note="generated text/raw output is non-empty",
            )
        else:
            has_output = bool(trt.text) or any(
                value not in ("", None, [], {}) for value in (trt.data or {}).values()
            )
            metrics["non_empty_output"] = MetricResult(
                value=1.0 if has_output else 0.0,
                threshold=1.0,
                operator="==",
                passed=has_output,
                note="stage produced observable output",
            )

        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all invariant metrics must pass",
            message=f"{stage.name} invariant check",
        )


plugin = OmniComparator()
