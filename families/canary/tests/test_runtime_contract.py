# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


FAMILY_ROOT = Path(__file__).resolve().parents[1]


def test_builder_and_runtime_use_the_same_mel_normalization_field() -> None:
    model = (FAMILY_ROOT / "model.py").read_text(encoding="utf-8")
    runtime = (FAMILY_ROOT / "runtime/plugin.cpp").read_text(encoding="utf-8")

    assert '"mel_normalize": "per_feature"' in model
    assert 'config.at("mel_normalize").get<std::string>() == "per_feature"' in runtime
    assert 'config.at("mel_normalize_per_feature")' not in runtime


def test_reference_keeps_the_main_cpu_and_pcm16_oracle() -> None:
    source = (FAMILY_ROOT / "tests/test_e2e.py").read_text(encoding="utf-8")
    assert "from scipy.signal import resample" in source
    assert 'map_location="cpu"' in source
    assert "model.cpu().eval()" in source
    assert "audio_i16" in source
    assert "resample_poly" not in source
