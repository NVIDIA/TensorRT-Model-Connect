# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin registry adapter for the model-owned VoiceChat implementation."""

from __future__ import annotations

from . import model


class NemotronVoiceChatPlugin:
    name = "nemotron_voicechat"
    runtime_strategy = model.runtime_strategy
    matches_config = staticmethod(model.matches)

    @staticmethod
    def matches(model_type: str) -> bool:
        normalized = model_type.lower().replace("-", "_").replace(".", "_")
        return normalized in {
            "nemotron_voicechat",
            "nemotronlabs_voicechat",
            "nvidia_nemotronlabs_voicechat_11b",
        }

    @staticmethod
    def load_weights(*_args: object, **_kwargs: object) -> None:
        raise NotImplementedError("VoiceChat uses its model-owned build entrypoint")

    @staticmethod
    def build_engine(*_args: object, **_kwargs: object) -> None:
        raise NotImplementedError("VoiceChat uses its model-owned build entrypoint")


plugin = NemotronVoiceChatPlugin()
plugin.build = model.build
