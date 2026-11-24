# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the E2E harness: the checkpoint source, the case selection, the trace checks' inputs and the
short answers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub.constants as hub_constants
import numpy as np
import pytest
from huggingface_hub import HfApi

from families.laguna.tests import test_e2e as e2e

_PREMERGE = "laguna-xs-2-beyond-window"


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


def test_checkpoint_comes_from_the_premerge_cache_without_a_hub_call(offline_hub) -> None:
    manifest, _ = e2e._CASES[_PREMERGE]
    snapshot = _premerge_cache(offline_hub, manifest)

    assert e2e._checkpoint(manifest).resolve() == snapshot.resolve()


def test_missing_checkpoint_names_the_download_instead_of_reaching_the_hub(offline_hub) -> None:
    manifest, _ = e2e._CASES[_PREMERGE]

    with pytest.raises(AssertionError, match="is not in the Hugging Face cache"):
        e2e._checkpoint(manifest)


class _Config:
    def __init__(self, **options: object) -> None:
        self._options = {f"--{name.replace('_', '-')}": value for name, value in options.items()}

    def getoption(self, name: str, default: object = None) -> object:
        return self._options.get(name, default)


def _outcomes(monkeypatch, collected: set[str] | None = None, **options: object) -> dict[str, str]:
    """run / skip / fail of every case under a command-line selection, in a session that collected the E2E cases
    `collected` (by default this family's only)."""
    monkeypatch.delenv("TRTMC_E2E", raising=False)
    collected = set(e2e._CASES) if collected is None else collected
    outcomes = {}
    for case_name, (manifest, _) in sorted(e2e._CASES.items()):
        try:
            e2e._require_selected(case_name, manifest, _Config(**options), collected)
        except pytest.skip.Exception:
            outcomes[case_name] = "skip"
        except pytest.fail.Exception:
            outcomes[case_name] = "fail"
        else:
            outcomes[case_name] = "run"
    return outcomes


def test_a_testcase_that_selects_no_case_fails(monkeypatch) -> None:
    for model in ("laguna", "laguna-xs-2"):
        outcomes = _outcomes(monkeypatch, e2e_model=[model], e2e_testcase=["laguna-xs-2-chta"])
        assert set(outcomes.values()) == {"fail"}, (model, outcomes)
    assert _outcomes(monkeypatch, e2e_model=["laguna-xs-2-chat"], e2e_testcase=["laguna-xs-2-code"]) == {
        "laguna-xs-2-beyond-window": "skip",
        "laguna-xs-2-chat": "fail",
        "laguna-xs-2-code": "skip",
    }
    # Without --e2e-model, a testcase must name an E2E case that this session collected, of any family.
    assert set(_outcomes(monkeypatch, e2e_testcase=["laguna-xs-2-chta"]).values()) == {"fail"}
    assert set(_outcomes(monkeypatch, e2e_testcase=["qwen3-0.6b"]).values()) == {"fail"}
    with_qwen = set(e2e._CASES) | {"qwen3-0.6b"}
    assert set(_outcomes(monkeypatch, with_qwen, e2e_testcase=["qwen3-0.6b"]).values()) == {"skip"}
    assert set(_outcomes(monkeypatch, with_qwen, e2e_model=["qwen"], e2e_testcase=["qwen3-0.6b"]).values()) == {
        "skip"
    }
    assert set(_outcomes(monkeypatch, e2e_model=["qwen"], e2e_testcase=["laguna-xs-2-chat"]).values()) == {"skip"}


def _item(originalname: str, **parameters: object) -> SimpleNamespace:
    if not parameters:
        return SimpleNamespace(originalname=originalname)
    return SimpleNamespace(originalname=originalname, callspec=SimpleNamespace(params=parameters))


def test_the_collected_cases_are_the_case_parameters_of_e2e_tests() -> None:
    items = [
        _item("test_e2e", case_name="laguna-xs-2-chat"),
        _item("test_official_checkpoint_e2e", case_name="qwen3-0.6b", execution="fp16"),
        _item("test_manifest_fields", case_name="not-an-e2e-case"),
        _item("test_prefill_logits_must_be_finite"),
    ]

    assert e2e._collected_cases(items) == {"laguna-xs-2-chat", "qwen3-0.6b"}


def test_a_testcase_narrows_the_selected_family_to_that_case(monkeypatch) -> None:
    only_chat = {"laguna-xs-2-beyond-window": "skip", "laguna-xs-2-chat": "run", "laguna-xs-2-code": "skip"}

    assert _outcomes(monkeypatch, e2e_model=["laguna"], e2e_testcase=["laguna-xs-2-chat"]) == only_chat
    assert _outcomes(monkeypatch, e2e_testcase=["laguna-xs-2-chat"]) == only_chat
    assert set(_outcomes(monkeypatch, e2e_model=["laguna"]).values()) == {"run"}
    premerge = _outcomes(monkeypatch, e2e_testcase=[_PREMERGE])
    assert {case for case, outcome in premerge.items() if outcome == "run"} == {_PREMERGE}


def test_a_case_marked_past_the_window_must_reach_it() -> None:
    beyond_window, chat = e2e._CASES[_PREMERGE][1], e2e._CASES["laguna-xs-2-chat"][1]
    assert beyond_window["past_sliding_window"] is True

    with pytest.raises(AssertionError):
        e2e._trace_segments(500, 512, beyond_window, e2e._THRESHOLDS)
    assert set(e2e._trace_segments(617, 512, beyond_window, e2e._THRESHOLDS)) == {"all_rows", "rows_past_window"}
    assert set(e2e._trace_segments(69, 512, chat, e2e._THRESHOLDS)) == {"all_rows"}


def test_prefill_logits_must_be_finite() -> None:
    reference = np.array([0.0, 1.0, 0.5])

    with pytest.raises(AssertionError, match="not finite"):
        e2e._prefill_prediction(np.array([0.0, np.inf, 0.5], dtype=np.float32), reference)
    token, cosine = e2e._prefill_prediction(np.array([0.0, 2.0, 1.0], dtype=np.float32), reference)
    assert token == 1
    assert cosine == pytest.approx(1.0)


@pytest.mark.parametrize("flag", ["exact_answer", "past_sliding_window"])
@pytest.mark.parametrize("value", ["yes", 1, None])
def test_a_case_flag_must_be_a_boolean(tmp_path, flag, value) -> None:
    manifest, case = e2e._CASES["laguna-xs-2-chat"]
    path = tmp_path / "laguna-xs-2.json"
    path.write_text(json.dumps({**manifest, "testcases": [{**case, flag: True}]}), encoding="utf-8")
    assert e2e._load_cases(tmp_path)["laguna-xs-2-chat"][1][flag] is True

    path.write_text(json.dumps({**manifest, "testcases": [{**case, flag: value}]}), encoding="utf-8")
    with pytest.raises(AssertionError, match=flag):
        e2e._load_cases(tmp_path)


def test_stop_ids_come_from_the_generation_config(tmp_path) -> None:
    config = tmp_path / "generation_config.json"
    config.write_text(json.dumps({"eos_token_id": [2, 24]}), encoding="utf-8")
    assert e2e._stop_ids(tmp_path) == {2, 24}
    config.write_text(json.dumps({"eos_token_id": 24}), encoding="utf-8")
    assert e2e._stop_ids(tmp_path) == {24}

    for value in (True, None, "24", [], [2, 1.0], [2, -1]):
        config.write_text(json.dumps({"eos_token_id": value}), encoding="utf-8")
        with pytest.raises(AssertionError):
            e2e._stop_ids(tmp_path)


class _Tokenizer:
    """Decodes through a fixed vocabulary in which, as in the checkpoint, only the EOS token is special."""

    _VOCAB = {2: "<|EOS|>", 7: " is", 13: ".", 24: "</assistant>", 55: "X", 111: "Jup", 268: "\n", 21789: "iter"}

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(self._VOCAB[i] for i in ids if not (skip_special_tokens and i == 2))


def test_a_short_answer_must_be_exactly_the_expected_text() -> None:
    assert {name for name, (_, case) in e2e._CASES.items() if case.get("exact_answer")} == {
        "laguna-xs-2-chat",
        _PREMERGE,
    }
    expected = e2e._normalized(e2e._CASES["laguna-xs-2-chat"][1]["expected_continuation_text"])
    tokenizer, stop_ids = _Tokenizer(), {2, 24}

    def answer(*ids: int) -> str:
        return e2e._answer(tokenizer, list(ids), stop_ids)

    # The recorded answer, "\nJupiter\n</assistant>", and the same word ending differently.
    for ids in ((268, 111, 21789, 268, 24), (111, 21789, 24), (111, 21789, 13, 268, 24), (111, 21789, 2)):
        assert answer(*ids) == expected, ids
    # "JupiterX" passes the edit distance and the substring check, but is not the answer.
    native, reference = "\nJupiterX\n</assistant>", "\nJupiter\n</assistant>"
    assert e2e._normalized_edit_distance(native, reference) <= e2e._THRESHOLDS["normalized_text_edit_distance"]
    assert answer(268, 111, 21789, 55, 268, 24) == "jupiterx" != expected
    # An answer cut by the token budget has no stop token and keeps all its text.
    assert answer(111, 21789, 7) == "jupiter is" != expected
