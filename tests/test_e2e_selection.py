# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

from tests.e2e.models.nemotron_speech_streaming import runner as speech_runner
from tests.e2e_harness.contracts import E2ECase
from tests.test_e2e import _case_matches_e2e_model, _parse_e2e_model_filters


_MODEL_RUNNERS = Path(__file__).parent / "e2e" / "models"


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


def test_model_runner_applies_exact_models_file_before_aliases(tmp_path):
    models_file = tmp_path / "models.txt"
    models_file.write_text(
        "nemotron-speech-streaming-en-0.6b\n",
        encoding="utf-8",
    )

    class Config:
        @staticmethod
        def getoption(name, default=None):
            options = {
                "--e2e-model": ["nemotron_speech_streaming"],
                "--e2e-models-file": str(models_file),
            }
            return options.get(name, default)

    selected = speech_runner.model_case_names(Config())

    assert selected == ["nemotron-speech-streaming-en-0.6b"]


def test_all_model_runners_enforce_exact_models_file_selection():
    missing = []
    for runner in sorted(_MODEL_RUNNERS.glob("*/runner.py")):
        source = runner.read_text(encoding="utf-8")
        calls = {
            node.func.id
            for node in ast.walk(ast.parse(source, filename=str(runner)))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        if calls.isdisjoint(
            {"select_cases_from_models_file", "model_case_names_for_dir"}
        ):
            missing.append(runner.relative_to(Path(__file__).parents[1]).as_posix())

    assert not missing, f"model runners without exact models-file selection: {missing}"
