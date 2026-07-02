# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

from tests.e2e_harness.contracts import E2ECase
from tests.e2e_harness.model_selection import case_matches_e2e_model
from tests.test_e2e import _case_matches_e2e_model, _parse_e2e_model_filters


_SPEECH_E2E_TEST = (
    "tests/e2e/models/nemotron_speech_streaming/"
    "test_nemotron_speech_streaming_e2e.py"
)


def test_parse_e2e_model_filters_supports_repeat_and_csv():
    assert _parse_e2e_model_filters(["decoder_family, bark", "flux"]) == {
        "decoder_family",
        "bark",
        "flux",
    }


def test_case_matches_e2e_model_by_name_family_and_strategy():
    case = E2ECase(
        name="example-decoder-fp16",
        hf_id="example-org/example-decoder",
        family="example_decoder",
        runtime_strategy="example_decoder_decoder_kv_cache",
    )

    assert _case_matches_e2e_model(case, {"example-decoder-fp16"})
    assert _case_matches_e2e_model(case, {"example_decoder"})
    assert _case_matches_e2e_model(case, {"example_decoder_decoder_kv_cache"})
    assert not _case_matches_e2e_model(case, {"bark"})


def test_case_matches_e2e_model_does_not_match_shared_hf_id_basename():
    base = E2ECase(
        name="example-decoder",
        hf_id="example-org/example-decoder",
        family="example_decoder",
        runtime_strategy="example_decoder_decoder_kv_cache",
    )
    probe = E2ECase(
        name="example-decoder-probe01",
        hf_id="example-org/example-decoder",
        family="example_decoder",
        runtime_strategy="example_decoder_decoder_kv_cache",
        metadata={"ci_tier": "nightly_only", "l0_replacement": "example-decoder"},
    )

    assert _case_matches_e2e_model(base, {"example-decoder"})
    assert not _case_matches_e2e_model(probe, {"example-decoder"})
    assert _case_matches_e2e_model(probe, {"example-decoder-probe01"})
    assert case_matches_e2e_model(base, {"example-decoder"})
    assert not case_matches_e2e_model(probe, {"example-decoder"})


def _collect_speech_cases(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            _SPEECH_E2E_TEST,
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_collection_guard_does_not_expand_shared_hf_id_to_probes():
    result = _collect_speech_cases(
        "--e2e-model", "nemotron-speech-streaming-en-0.6b"
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[nemotron-speech-streaming-en-0.6b]" in result.stdout
    assert "-asr-probe" not in result.stdout


def test_collection_guard_applies_exact_models_file_after_aliases(tmp_path):
    models_file = tmp_path / "models.txt"
    models_file.write_text(
        "nemotron-speech-streaming-en-0.6b\n",
        encoding="utf-8",
    )

    result = _collect_speech_cases(
        "--e2e-model",
        "nemotron_speech_streaming",
        "--e2e-models-file",
        str(models_file),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[nemotron-speech-streaming-en-0.6b]" in result.stdout
    assert "-asr-probe" not in result.stdout
