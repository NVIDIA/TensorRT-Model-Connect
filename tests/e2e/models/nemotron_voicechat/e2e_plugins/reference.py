# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned receipt for the public VoiceChat model-card reproduction."""

from __future__ import annotations

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec


class VoiceChatPinnedModelCardReference:
    """Return the immutable acceptance values from the pinned public recipe."""

    @property
    def backend_name(self) -> str:
        return "voicechat_pinned_model_card"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        del ctx
        if stage.name != "model_card_general_conversation":
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"unsupported VoiceChat stage: {stage.name}"},
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
            data={field: case.metadata[field] for field in fields},
            metadata={
                "source": "pinned_public_model_card_reproduction",
                "hf_id": case.hf_id,
                "hf_revision": case.hf_revision,
            },
        )


reference = VoiceChatPinnedModelCardReference()
