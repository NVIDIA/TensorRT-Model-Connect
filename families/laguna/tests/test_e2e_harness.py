# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the E2E harness: the checkpoint source, the case selection and the trace checks' inputs."""

from __future__ import annotations

from pathlib import Path

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


def _outcomes(monkeypatch, **options: object) -> dict[str, str]:
    """run / skip / fail of every case under a command-line selection."""
    monkeypatch.delenv("TRTMC_E2E", raising=False)
    outcomes = {}
    for case_name, (manifest, _) in sorted(e2e._CASES.items()):
        try:
            e2e._require_selected(case_name, manifest, _Config(**options))
        except pytest.skip.Exception:
            outcomes[case_name] = "skip"
        except pytest.fail.Exception:
            outcomes[case_name] = "fail"
        else:
            outcomes[case_name] = "run"
    return outcomes


def test_a_testcase_that_names_no_case_of_the_selected_family_fails(monkeypatch) -> None:
    for model in ("laguna", "laguna-xs-2"):
        outcomes = _outcomes(monkeypatch, e2e_model=[model], e2e_testcase=["laguna-xs-2-chta"])
        assert set(outcomes.values()) == {"fail"}, (model, outcomes)
    assert _outcomes(monkeypatch, e2e_model=["laguna-xs-2-chat"], e2e_testcase=["laguna-xs-2-code"]) == {
        "laguna-xs-2-beyond-window": "skip",
        "laguna-xs-2-chat": "fail",
        "laguna-xs-2-code": "skip",
    }
    # Without this family named, a testcase it does not own may belong to another family's file.
    assert set(_outcomes(monkeypatch, e2e_testcase=["laguna-xs-2-chta"]).values()) == {"skip"}
    assert set(_outcomes(monkeypatch, e2e_model=["qwen"], e2e_testcase=["qwen3-0.6b"]).values()) == {"skip"}
    assert set(_outcomes(monkeypatch, e2e_model=["qwen"], e2e_testcase=["laguna-xs-2-chat"]).values()) == {"skip"}


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
