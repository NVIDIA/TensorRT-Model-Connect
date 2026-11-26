# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the E2E harness: the checkpoint source, the case selection and the recorded checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub.constants as hub_constants
import pytest
from huggingface_hub import HfApi

from families.voxtral.tests import test_e2e as e2e

_PREMERGE = "voxtral-mini-3b-2507-recording-en"


def _premerge_cache(root: Path, manifest: dict) -> Path:
    """The cache the pre-merge provides: the pinned snapshot's files and its ref, no tree listing."""
    repo = root / ("models--" + manifest["hf_id"].replace("/", "--"))
    snapshot = repo / "snapshots" / manifest["hf_revision"]
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (repo / "refs").mkdir()
    (repo / "refs" / manifest["hf_revision"]).write_text(manifest["hf_revision"], encoding="utf-8")
    return snapshot


@pytest.fixture
def offline_hub(tmp_path, monkeypatch) -> Path:
    """An empty offline Hugging Face cache in which any Hub API call fails the test."""
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def hub_call(*args, **kwargs):
        raise AssertionError("the E2E reached for the Hub API")

    for name in ("list_repo_tree", "repo_info", "model_info"):
        monkeypatch.setattr(HfApi, name, hub_call)
    return cache


def test_every_manifest_pins_the_original_checkpoint() -> None:
    manifests = {manifest["name"]: manifest for _, manifest, _ in e2e.CASES.values()}
    assert {manifest["hf_id"] for manifest in manifests.values()} == {"mistralai/Voxtral-Mini-3B-2507"}
    assert {manifest["hf_revision"] for manifest in manifests.values()} == {"3060fe34b35ba5d44202ce9ff3c097642914f8f3"}
    assert {manifest["precision"] for manifest in manifests.values()} == {"fp16", "bf16", "fp32"}
    assert any(case.get("premerge") for _, _, case in e2e.CASES.values())


def test_checkpoint_comes_from_the_premerge_cache_without_a_hub_call(offline_hub) -> None:
    _, manifest, _ = e2e.CASES[_PREMERGE]
    snapshot = _premerge_cache(offline_hub, manifest)

    assert e2e._model_dir(manifest).resolve() == snapshot.resolve()


def test_missing_checkpoint_names_the_download_instead_of_reaching_the_hub(offline_hub) -> None:
    _, manifest, _ = e2e.CASES[_PREMERGE]

    with pytest.raises(AssertionError, match="is not in the Hugging Face cache"):
        e2e._model_dir(manifest)


class _Config:
    def __init__(self, **options: object) -> None:
        self._options = {f"--{name.replace('_', '-')}": value for name, value in options.items()}

    def getoption(self, name: str, default: object = None) -> object:
        return self._options.get(name, default)


def _outcomes(collected: set[str] | None = None, **options: object) -> dict[str, str]:
    """run / skip / fail of every case under a command-line selection, in a session that collected the E2E cases
    ``collected`` (by default this family's only)."""
    collected = set(e2e.CASES) if collected is None else collected
    outcomes = {}
    for case_name in sorted(e2e.CASES):
        try:
            e2e._require_selected(case_name, _Config(**options), collected)
        except pytest.skip.Exception:
            outcomes[case_name] = "skip"
        except pytest.fail.Exception:
            outcomes[case_name] = "fail"
        else:
            outcomes[case_name] = "run"
    return outcomes


def test_without_a_selection_every_case_skips() -> None:
    assert set(_outcomes().values()) == {"skip"}


def test_a_selection_runs_exactly_its_cases() -> None:
    assert set(_outcomes(e2e_model=["voxtral"]).values()) == {"run"}
    fp32 = _outcomes(e2e_model=["voxtral-mini-3b-2507-fp32"])
    assert {name for name, outcome in fp32.items() if outcome == "run"} == {
        name for name in e2e.CASES if name.startswith("voxtral-mini-3b-2507-fp32-")}
    assert {outcome for name, outcome in fp32.items() if "-fp32-" not in name} == {"skip"}
    only = _outcomes(e2e_model=["voxtral"], e2e_testcase=[_PREMERGE])
    assert only[_PREMERGE] == "run" and set(only.values()) == {"run", "skip"}
    assert list(_outcomes(e2e_testcase=[_PREMERGE]).values()).count("run") == 1


def test_a_testcase_that_selects_no_case_fails() -> None:
    typo = "voxtral-mini-3b-2507-recordng-en"
    assert set(_outcomes(e2e_model=["voxtral"], e2e_testcase=[typo]).values()) == {"fail"}
    # A manifest name selects that manifest's cases; they fail, the other manifests' cases are not selected.
    fp16 = _outcomes(e2e_model=["voxtral-mini-3b-2507"], e2e_testcase=[typo])
    assert {name for name, outcome in fp16.items() if outcome == "fail"} == {
        name for name, (_, manifest, _) in e2e.CASES.items() if manifest["name"] == "voxtral-mini-3b-2507"}
    assert set(fp16.values()) == {"fail", "skip"}
    # Without --e2e-model, a testcase must name an E2E case that this session collected, of any family.
    assert set(_outcomes(e2e_testcase=["voxtral-mini-3b-2507-recordng-en"]).values()) == {"fail"}
    with_other = set(e2e.CASES) | {"other-case"}
    assert set(_outcomes(with_other, e2e_testcase=["other-case"]).values()) == {"skip"}
    # Another family's selection skips this family's cases.
    assert set(_outcomes(e2e_model=["whisper"], e2e_testcase=[_PREMERGE]).values()) == {"skip"}


def _item(originalname: str, **parameters: object) -> SimpleNamespace:
    if not parameters:
        return SimpleNamespace(originalname=originalname)
    return SimpleNamespace(originalname=originalname, callspec=SimpleNamespace(params=parameters))


def test_the_collected_cases_are_the_case_parameters_of_e2e_tests() -> None:
    items = [
        _item("test_e2e", case_name=_PREMERGE),
        _item("test_official_checkpoint_e2e", case_name="other-case", execution="fp16"),
        _item("test_manifest_fields", case_name="not-an-e2e-case"),
        _item("test_without_a_selection_every_case_skips"),
    ]

    assert e2e._collected_cases(items) == {_PREMERGE, "other-case"}


def test_token_differences_count_positions_and_length() -> None:
    assert e2e._token_differences([1, 2, 3], [1, 2, 3]) == 0
    assert e2e._token_differences([1, 2, 4], [1, 2, 3]) == 1
    assert e2e._token_differences([1, 2], [1, 2, 3]) == 1
    assert e2e._token_differences([], [7, 8]) == 2


def _failed(checks: list[dict]) -> set[str]:
    return {check["name"] for check in checks if not check["passed"]}


def test_recorded_checks_fail_with_the_test() -> None:
    same = e2e._comparison_checks("fp32", [1, 2], "a", [1, 2], "a", None, 0, "a")
    assert _failed(same) == set()
    assert _failed(e2e._comparison_checks("fp32", [1, 3], "a", [1, 2], "a", None, 0, "a")) == {
        "token_id_differences"}
    assert _failed(e2e._comparison_checks("fp32", [1, 2], "a", [1, 2], "b", None, 0, "a")) == {"transcript_differs"}
    assert _failed(e2e._comparison_checks("fp32", [1, 2], "a", [1, 2], "a", None, 1, "a")) == {"sdk_differences"}
    allowance = e2e.NEAR_TIE_LOGIT_GAP
    assert _failed(e2e._comparison_checks("fp16", [1], "a", [2], "b", [0.0, allowance], 0, "a")) == set()
    assert _failed(e2e._comparison_checks("bf16", [1], "a", [1], "a", [0.0, allowance + 0.01], 0, "a")) == {
        "largest_teacher_forced_gap"}
    # The native text must be the official processor's decoding of the native ids, at every precision.
    for precision in ("fp32", "fp16"):
        checks = e2e._comparison_checks(precision, [1], "a", [1], "a", [0.0], 0, "a b")
        assert "native_decoding_differs" in _failed(checks)
    assert "empty_transcript" in _failed(e2e._comparison_checks("fp32", [], " ", [], " ", None, 0, " "))
    for check in same:
        assert set(check) == {"name", "label", "scope", "actual", "operator", "expected", "passed"}
        assert check["scope"] in {"contract", "independent_reference"}


def test_a_non_finite_gap_never_passes() -> None:
    nan = float("nan")
    assert max([0.0, nan]) == 0.0  # why the largest gap is not taken with max() alone
    for gaps in ([0.0, nan], [nan, 0.0], [0.0, float("inf")]):
        with pytest.raises(ValueError, match="non-finite"):
            e2e._largest_gap(gaps)
        with pytest.raises(ValueError, match="non-finite"):
            e2e._comparison_checks("bf16", [1, 2], "a", [1, 2], "a", gaps, 0, "a")
    assert e2e._largest_gap([]) == 0.0 and e2e._largest_gap([0.0, 0.125]) == 0.125


def test_the_ten_chunk_case_reaches_the_largest_encoder_profile(tmp_path) -> None:
    """The case meant to run the largest profile fills every chunk the default bundle accepts and stays within the
    cache with its token budget."""
    from families.voxtral.config import DEFAULT_MAX_SEQUENCE_LENGTH, max_audio_chunks, parse_config
    from families.voxtral.tests.test_support import CONFIG

    path, manifest, case = e2e.CASES["voxtral-mini-3b-2507-fp32-ten-chunks"]
    assert path.is_file() and manifest["max_sequence_length"] == DEFAULT_MAX_SEQUENCE_LENGTH
    # The E2E fails the case when the native run stops before its budget, so the late positions are really reached.
    assert case["stops_at_budget"] is True
    wav = e2e._input_audio(case, tmp_path)
    samples = len(e2e._read_pcm16(wav))
    chunks = -(-samples // (e2e.SAMPLE_RATE * 30))
    config = parse_config(CONFIG)
    # Prompt framing of the automatic-language request: three prefix and two suffix tokens.
    assert chunks == max_audio_chunks(manifest["max_sequence_length"], config, 3 + 5)
    prompt = 3 + chunks * config.audio_tokens_per_chunk + 2
    assert prompt + case["max_new_tokens"] <= manifest["max_sequence_length"]
    assert prompt + case["max_new_tokens"] > manifest["max_sequence_length"] - 64
