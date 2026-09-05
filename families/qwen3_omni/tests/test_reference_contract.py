# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path


def test_e2e_gates_only_thinker_text_and_final_waveform() -> None:
    source = Path(__file__).with_name("test_e2e.py").read_text(encoding="utf-8")
    assert 'native_text["text"] == reference_text' in source
    assert 'native_text["token_ids"]' not in source
    assert "_teacher_forced_code2wav" not in source
    assert "duration_ratio_max" not in source
    assert "actual.size == reference.size" in source
    assert "waveform_cosine_min" in source
