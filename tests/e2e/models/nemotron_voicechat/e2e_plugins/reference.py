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
                "VoiceChat reference audio SHA256 mismatch: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        if not ctx.artifacts_dir:
            raise RuntimeError("VoiceChat reference audio requires an artifacts directory")
        artifact_dir = Path(ctx.artifacts_dir) / case.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "model_card_sample_general_reference.flac"
        shutil.copyfile(reference_path, artifact_path)
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


reference = VoiceChatPinnedModelCardReference()
