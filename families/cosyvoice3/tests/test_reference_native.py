# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent CPU DSP comparisons for the native reference frontend."""

import json
import os
import subprocess

import numpy as np
import pytest


@pytest.mark.parametrize("rate", [16000, 24000, 44100, 48000])
def test_native_reference_features(tmp_path, rate):
    binary = os.environ.get("COSYVOICE3_CPP_TEST")
    if not binary:
        pytest.skip("COSYVOICE3_CPP_TEST required for native DSP comparison")
    import soundfile as sf
    from families.cosyvoice3.reference import coefficients
    from families.cosyvoice3.tts import reference_features

    samples = np.random.default_rng(913).normal(0, 0.1, rate // 5).astype(np.float32)
    audio = tmp_path / "reference.wav"
    sf.write(audio, samples, rate, subtype="FLOAT")
    fixture, result = tmp_path / "input.json", tmp_path / "output.json"
    fixture.write_text(
        json.dumps(dict(samples=samples.tolist(), sample_rate=rate, coefficients=coefficients()))
    )
    subprocess.run([binary, "--features", str(fixture), str(result)], check=True, timeout=60)
    actual = json.loads(result.read_text())
    expected = reference_features(audio)
    for name, key in [
        ("speaker", "speaker_features"),
        ("tokens", "token_features"),
        ("mel", "mel"),
    ]:
        values = expected[key].numpy().reshape(-1)
        np.testing.assert_allclose(actual[name], values, atol=1e-4, rtol=1e-5, err_msg=name)
