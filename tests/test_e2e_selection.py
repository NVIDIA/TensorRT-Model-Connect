# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tests.e2e.models.nemotron_speech_streaming import runner as speech_runner
from tests.e2e_harness.contracts import E2ECase
from tests.test_e2e import _case_matches_e2e_model, _parse_e2e_model_filters


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


def test_model_runner_does_not_expand_shared_hf_id_to_probes():
    class Config:
        @staticmethod
        def getoption(name, default=None):
            if name == "--e2e-model":
                return ["nemotron-speech-streaming-en-0.6b"]
            return default

    selected = speech_runner.model_case_names(Config())

    assert selected == ["nemotron-speech-streaming-en-0.6b"]
