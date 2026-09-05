# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the family-owned Qwen3-Omni TensorRT build policy."""

from pathlib import Path


FAMILY_ROOT = Path(__file__).resolve().parent.parent


def test_thinker_preserves_mainline_level_without_destabilizing_audio_builders() -> None:
    thinker = (FAMILY_ROOT / "model.py").read_text(encoding="utf-8")
    talker = (FAMILY_ROOT / "talker_builder.py").read_text(encoding="utf-8")
    code2wav = (FAMILY_ROOT / "code2wav_builder.py").read_text(encoding="utf-8")

    assert thinker.count("create_builder_config()") == 1
    assert thinker.count("builder_optimization_level = 3") == 1
    assert talker.count("create_builder_config()") == 2
    assert talker.count("builder_optimization_level = 1") == 2
    assert code2wav.count("create_builder_config()") == 1
    assert code2wav.count("builder_optimization_level = 1") == 1
