# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned receipt for the public VoiceChat model-card reproduction."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec

_ROOT = Path(__file__).resolve().parents[5]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _persist_reference_audio(case: E2ECase, ctx: RunContext, filename: str) -> Path:
    reference_value = case.inputs.get("reference_audio", "")
    if not isinstance(reference_value, str) or not reference_value:
        raise RuntimeError("VoiceChat manifest is missing reference_audio")
    reference_path = _ROOT / reference_value
    if not reference_path.is_file():
        raise RuntimeError(f"VoiceChat reference audio is unavailable: {reference_path}")
    expected_sha = str(case.metadata.get("reference_audio_sha256", ""))
    actual_sha = _sha256(reference_path)
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError(
            f"VoiceChat reference audio SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    if not ctx.artifacts_dir:
        raise RuntimeError("VoiceChat reference audio requires an artifacts directory")
    artifact_dir = Path(ctx.artifacts_dir) / case.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / filename
    shutil.copyfile(reference_path, artifact_path)
    return artifact_path


class VoiceChatPinnedModelCardReference:
    """Return the immutable acceptance values from the pinned public recipe."""

    @property
    def backend_name(self) -> str:
        return "voicechat_pinned_model_card"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "model_card_general_conversation":
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"unsupported VoiceChat stage: {stage.name}"},
            )
        artifact_path = _persist_reference_audio(
            case, ctx, "model_card_sample_general_reference.flac"
        )
        fields = (
            "speech_source_revision",
            "speech_source_sha256",
            "speech_source_sample_rate",
            "speech_source_num_samples",
            "text_model_revision",
            "expected_output_sample_rate",
            "expected_output_num_samples",
            "expected_output_samples_per_frame",
            "expected_output_codec_frames",
            "expected_response_text",
            "required_response_terms",
        )
        return StageOutput(
            stage_name=stage.name,
            data={
                **{field: case.metadata[field] for field in fields},
                "audio_output_path": str(artifact_path),
            },
            metadata={
                "source": "pinned_public_model_card_reproduction",
                "hf_id": case.hf_id,
                "hf_revision": case.hf_revision,
            },
        )


class VoiceChatLifecycleInvariantReference:
    """Declare the required primitive evidence for the native lifecycle probe."""

    @property
    def backend_name(self) -> str:
        return "voicechat_lifecycle_invariants"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "native_full_duplex_lifecycle":
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"unsupported VoiceChat lifecycle stage: {stage.name}"},
            )
        return StageOutput(
            stage_name=stage.name,
            data={
                "schema_version": 3,
                "speech_source_sha256": case.metadata["speech_source_sha256"],
                "function_speech_source_sha256": case.metadata["function_speech_source_sha256"],
                "expected_output_sample_rate": case.metadata["expected_output_sample_rate"],
                "expected_output_num_samples": case.metadata["expected_output_num_samples"],
                "expected_output_samples_per_frame": case.metadata[
                    "expected_output_samples_per_frame"
                ],
                "expected_output_codec_frames": case.metadata["expected_output_codec_frames"],
                "expected_response_text": case.metadata["expected_response_text"],
                "control_latency_limit_ms": 500.0,
                "tail_completion_limit_ms": 15000.0,
                "audio_min_rms": 0.001,
                "audio_min_peak": 0.01,
                "response_truncate_min_discarded_samples": case.metadata[
                    "expected_output_samples_per_frame"
                ],
                "expected_sequence_sessions_checked": 13,
                "expected_media_segments_checked": 16,
                "required_sections": [
                    "baseline",
                    "irregular_chunking",
                    "barge_in",
                    "cancel",
                    "reset_vs_fresh",
                    "processed_input_clear",
                    "response_cancel_recovery",
                    "response_truncate_recovery",
                    "partial_finish_tail",
                    "sequence_continuity",
                    "media_continuity",
                    "normal_multiturn",
                    "function_channel",
                    "backpressure_concurrency",
                ],
                "audio_output_path": str(
                    _persist_reference_audio(
                        case, ctx, "native_full_duplex_lifecycle_reference.flac"
                    )
                ),
            },
            metadata={"source": "model_owned_l4_lifecycle_invariants"},
        )


reference = (
    VoiceChatPinnedModelCardReference(),
    VoiceChatLifecycleInvariantReference(),
)
