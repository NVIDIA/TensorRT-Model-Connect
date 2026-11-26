# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E harness checks without a GPU: case selection and manifest value types."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from families.nemotron3_diarization.tests import test_e2e

CONVERSATION = "nemotron-3-diarization-conversation"


class _Config:
    def __init__(self, options: dict[str, list[str]]) -> None:
        self._options = options

    def getoption(self, name: str):
        return self._options.get(name)


def _select(**options: list[str]) -> tuple[list[str], bool]:
    names = {"model": "--e2e-model", "testcase": "--e2e-testcase"}
    return test_e2e._selected_cases(_Config({names[key]: value for key, value in options.items()}))


@pytest.mark.parametrize(
    "model", ["nemotron3_diarization", "nemotron-3-diarization", CONVERSATION]
)
def test_naming_the_family_with_an_unknown_case_fails(model):
    with pytest.raises(pytest.UsageError, match="matches none of its cases"):
        _select(model=[model], testcase=["nemotron-3-diarization-conversaton"])


def test_known_case_is_selected():
    assert _select(model=["nemotron3_diarization"], testcase=[CONVERSATION]) == (
        [CONVERSATION],
        True,
    )


def test_selection_of_another_family_skips_this_file():
    assert _select(model=["whisper"], testcase=["whisper-tiny-en"]) == ([], True)
    assert _select(testcase=["whisper-tiny-en"]) == ([], True)


class _Metafunc:
    fixturenames = ("case_name",)

    def __init__(self, options: dict[str, list[str]]) -> None:
        self.config = _Config(options)
        self.calls = []

    def parametrize(self, name, parameters, ids):
        self.calls.append((name, list(parameters), list(ids)))


def _item(name: str, case_name: str):
    return SimpleNamespace(name=name, originalname=name,
                           callspec=SimpleNamespace(params={"case_name": case_name}))


def _request(testcases: list[str], items: list) -> SimpleNamespace:
    return SimpleNamespace(config=_Config({"--e2e-testcase": testcases}),
                           session=SimpleNamespace(items=items))


def test_a_testcase_only_selection_of_no_own_case_runs_one_placeholder():
    metafunc = _Metafunc({"--e2e-testcase": ["nemotron-3-diarization-conversaton"]})
    test_e2e.pytest_generate_tests(metafunc)
    assert metafunc.calls == [("case_name", [test_e2e.UNMATCHED], [test_e2e.UNMATCHED])]


def test_a_selection_of_another_family_keeps_its_empty_parametrization():
    metafunc = _Metafunc({"--e2e-model": ["whisper"], "--e2e-testcase": ["whisper-tiny-en"]})
    test_e2e.pytest_generate_tests(metafunc)
    assert metafunc.calls == [("case_name", [], [])]


def test_an_unmatched_testcase_fails_when_no_collected_e2e_owns_it():
    typo = "nemotron-3-diarization-conversaton"
    items = [_item("test_e2e", test_e2e.UNMATCHED),
             _item("test_official_checkpoint_e2e", "whisper-tiny-en"),
             _item("test_selection", typo)]
    with pytest.raises(pytest.fail.Exception, match="matches no collected E2E case"):
        test_e2e._resolve_unmatched(_request([typo], items))


def test_an_unmatched_testcase_skips_when_another_family_owns_it():
    items = [_item("test_e2e", test_e2e.UNMATCHED),
             _item("test_official_checkpoint_e2e", "whisper-tiny-en")]
    with pytest.raises(pytest.skip.Exception, match="another collected family"):
        test_e2e._resolve_unmatched(_request(["whisper-tiny-en"], items))


def test_without_a_selector_every_case_is_listed_but_disabled():
    names, enabled = _select()
    assert CONVERSATION in names and not enabled


@pytest.mark.parametrize("value", [True, "1e-3", None])
def test_manifest_numbers_are_not_coerced(value):
    with pytest.raises(TypeError, match="logit_tolerance"):
        test_e2e._manifest_value({"logit_tolerance": value}, "logit_tolerance", (int, float))


def test_manifest_values_of_their_own_type_pass():
    assert test_e2e._manifest_value({"logit_tolerance": 0.001}, "logit_tolerance",
                                    (int, float)) == 0.001
    assert test_e2e._manifest_value({"tensor_parallel_size": 1}, "tensor_parallel_size",
                                    (int,)) == 1
    with pytest.raises(TypeError):
        test_e2e._manifest_value({"tensor_parallel_size": 1.0}, "tensor_parallel_size", (int,))
    with pytest.raises(TypeError):
        test_e2e._manifest_value({"bundle": 3}, "bundle", (str,))
