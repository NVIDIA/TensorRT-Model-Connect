# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the E2E's own rules: case selection and the recorded comparison."""

from __future__ import annotations

import json
import math
import sys
import types

import pytest

from tools import e2e_evidence

from . import test_e2e as e2e

_FP16 = sorted(name for name, (manifest, _) in e2e._CASES.items() if manifest["precision"] == "fp16")
_FP32 = sorted(name for name, (manifest, _) in e2e._CASES.items() if manifest["precision"] == "fp32")
_FP32_MANIFEST = next(manifest["name"] for manifest, _ in e2e._CASES.values() if manifest["precision"] == "fp32")


class _Config:
    def __init__(self, **options):
        self.options = options

    def getoption(self, name):
        return self.options.get(name)


class _Metafunc:
    fixturenames = ("tmp_path", "tmp_path_factory", "case_name")

    def __init__(self, **options):
        self.config = _Config(**options)
        self.parametrized = []

    def parametrize(self, name, values, ids):
        self.parametrized.append((name, values, ids))


def _generate(**options):
    metafunc = _Metafunc(**options)
    e2e.pytest_generate_tests(metafunc)
    (name, values, ids), = metafunc.parametrized
    assert name == "case_name"
    return values, ids


def test_the_manifests_hold_both_precisions():
    assert _FP16 and _FP32 and set(_FP16) | set(_FP32) == set(e2e._CASES)


@pytest.mark.parametrize("options", [
    {"--e2e-model": [e2e._FAMILY], "--e2e-testcase": ["minicpm-v-4.6-car-colour"]},
    {"--e2e-model": [_FP32_MANIFEST], "--e2e-testcase": [_FP16[0]]},
    {"--e2e-testcase": [f"{_FP16[0]}x"], "--e2e-model": [f"{e2e._FAMILY},other"]},
])
def test_a_selection_naming_the_family_that_matches_no_case_is_an_error(options):
    # Pytest would skip an empty parametrization silently, and the run would pass
    # without testing anything.
    with pytest.raises(pytest.UsageError, match="matches none of its cases"):
        _generate(**options)


def test_a_testcase_narrows_the_family_selection():
    values, ids = _generate(**{"--e2e-model": [e2e._FAMILY], "--e2e-testcase": [_FP16[0]]})
    assert values == ids == [_FP16[0]]


def test_a_manifest_name_selects_its_cases_only():
    values, _ = _generate(**{"--e2e-model": [_FP32_MANIFEST]})
    assert values == _FP32


def test_the_family_name_selects_every_case():
    values, _ = _generate(**{"--e2e-model": [f" {e2e._FAMILY} ,"]})
    assert values == sorted(e2e._CASES)


def test_a_models_file_selects_like_the_option(tmp_path):
    listing = tmp_path / "models.txt"
    listing.write_text(f"# comment\n\n  {_FP32_MANIFEST}\n", encoding="utf-8")
    values, _ = _generate(**{"--e2e-models-file": str(listing)})
    assert values == _FP32
    with pytest.raises(pytest.UsageError, match="does not exist"):
        _generate(**{"--e2e-models-file": str(tmp_path / "missing.txt")})


@pytest.mark.parametrize("options", [
    {"--e2e-model": ["another_family"]},
    {"--e2e-model": ["another_family"], "--e2e-testcase": ["another-family-case"]},
    # The model filter selects another family; this family's case name does not select it back.
    {"--e2e-model": ["another_family"], "--e2e-testcase": [_FP16[0]]},
])
def test_a_selection_of_other_families_leaves_this_file_empty(options):
    assert _generate(**options) == ([], [])


def test_a_testcase_only_selection_of_no_own_case_runs_the_session_check():
    check = e2e._SELECTION_CHECK
    assert _generate(**{"--e2e-testcase": ["another-family-case"]}) == ([check], [check])


class _Item:
    def __init__(self, originalname, case_name=None):
        self.originalname = originalname
        if case_name is not None:
            self.callspec = type("CallSpec", (), {"params": {"case_name": case_name}})()


class _Request:
    def __init__(self, testcases, items):
        self.config = _Config(**{"--e2e-testcase": testcases})
        self.session = type("Session", (), {"items": items})()


def test_a_testcase_another_collected_family_owns_skips_here():
    request = _Request(["other-case"], [_Item("test_e2e", e2e._SELECTION_CHECK), _Item("test_family_e2e", "other-case")])
    with pytest.raises(pytest.skip.Exception, match="another family"):
        e2e._foreign_testcases(request)


@pytest.mark.parametrize("items", [
    [_Item("test_e2e", "selection-check")],
    [_Item("test_e2e", "selection-check"), _Item("test_e2e", "another-case")],
    [_Item("test_e2e", "selection-check"), _Item("test_contract", "typo-case"), _Item("test_e2e")],
])
def test_a_testcase_no_collected_e2e_owns_fails(items):
    # Neither this family's check, nor another case, nor a non-E2E test or one without cases owns it.
    with pytest.raises(pytest.fail.Exception, match="matches none of the session's E2E cases"):
        e2e._foreign_testcases(_Request(["typo-case"], items))


@pytest.mark.parametrize("case", [
    {"oracle": "token_exact"},
    {"oracle": "token_exact", "expected_answer": "white", "expected_mentions": ["white"]},
    {"oracle": "token_exact", "expected_answer": " "},
    {"oracle": "token_exact", "expected_mentions": []},
    {"oracle": "token_exact", "expected_answers": ["white"]},
])
def test_each_case_has_exactly_one_content_check(tmp_path, monkeypatch, case):
    manifest = {"name": "m", "family": e2e._FAMILY, "task": "images_text_to_text", "precision": "fp16",
                "max_sequence_length": 64, "tensor_parallel_size": 1, "testcases": [{"name": "c", **case}]}
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(e2e, "_TEST_DIR", tmp_path)
    with pytest.raises(AssertionError):
        e2e._load_cases()


def test_without_a_selection_the_cases_run_only_with_trtmc_e2e(monkeypatch):
    monkeypatch.delenv("TRTMC_E2E", raising=False)
    values, ids = _generate()
    assert ids == sorted(e2e._CASES)
    assert all(value.marks and value.marks[0].name == "skip" for value in values)
    monkeypatch.setenv("TRTMC_E2E", "1")
    assert _generate() == (sorted(e2e._CASES), sorted(e2e._CASES))


def _recorder(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return e2e_evidence.Evidence(
        tmp_path / "evidence", family=e2e._FAMILY, case="contract-check",
        source_revision="a" * 40, roots=(tmp_path,),
    )


_EOS = 248046


def _oracle(recorder, tmp_path, gaps, argmax_ids, thresholds=None, oracle="teacher_forced",
            token_ids=(5603, _EOS), budget=8, text="white", decoded=None, reference_text=None, content=None):
    case = {"oracle": oracle, "max_new_tokens": budget, **(content or {"expected_answer": "white"})}
    payload = {"text": text, "token_ids": list(token_ids)}
    reference = {"reference_ids": list(token_ids), "native_logit_gaps": gaps,
                 "teacher_forced_argmax_ids": argmax_ids, "eos_token_ids": [_EOS],
                 "native_ids_decoded": text if decoded is None else decoded,
                 "reference_text": text if reference_text is None else reference_text}
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        e2e._assert_oracle(case, thresholds or {}, payload, reference, tmp_path)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    return recorder.data


def _check(data):
    record = data["reference_comparison"]
    assert record["scope"] == "independent_reference" and record["enforced"] is True
    native, reference = record["native"], record["reference"]
    assert native["artifact"] != reference["artifact"]
    assert native["size_bytes"] > 0 and reference["size_bytes"] > 0
    check, = record["checks"]
    return check


def test_a_teacher_forced_pass_records_a_complete_comparison(tmp_path):
    check = _check(_oracle(_recorder(tmp_path), tmp_path, [0.0, 0.0556], [5603, 248045]))
    assert (check["name"], check["scope"], check["operator"]) == (
        "max_logit_gap", "independent_reference", "<=")
    assert (check["actual"], check["expected"], check["passed"]) == (0.0556, e2e._LOGIT_TOLERANCE, True)


def test_a_teacher_forced_failure_is_recorded_before_the_assertion(tmp_path):
    recorder = _recorder(tmp_path)
    with pytest.raises(AssertionError, match="trails the reference argmax"):
        _oracle(recorder, tmp_path, [0.0, 0.358], [5603, 248045])
    check = _check(recorder.data)
    assert (check["actual"], check["passed"]) == (0.358, False)


def test_a_token_exact_case_records_no_library_comparison(tmp_path):
    # The badge comes from the asserted token equality; a comparison record would replace it.
    data = _oracle(_recorder(tmp_path), tmp_path, [0.0, 0.0], [5603, 248046], oracle="token_exact")
    assert "reference_comparison" not in data


@pytest.mark.parametrize("gaps", [[0.0, math.nan], [math.inf, 0.0]])
def test_nonfinite_gaps_are_rejected_before_the_comparison(tmp_path, gaps):
    recorder = _recorder(tmp_path)
    with pytest.raises(AssertionError):
        _oracle(recorder, tmp_path, gaps, [5603, 248045])
    assert "reference_comparison" not in recorder.data


@pytest.mark.parametrize("argmax_ids", [[5603], [5603, 248045, 1]])
def test_mismatched_lengths_are_rejected(tmp_path, argmax_ids):
    with pytest.raises(AssertionError):
        _oracle(_recorder(tmp_path), tmp_path, [0.0, 0.0], argmax_ids)


@pytest.mark.parametrize("value", [-0.1, math.nan, math.inf, True, "0.3", None])
def test_a_logit_tolerance_must_be_a_finite_nonnegative_number(tmp_path, value):
    with pytest.raises(AssertionError):
        _oracle(_recorder(tmp_path), tmp_path, [0.0, 0.0], [5603, 248046], {"logit_tolerance": value})


def test_a_token_exact_case_rejects_a_threshold(tmp_path):
    with pytest.raises(AssertionError, match="take no threshold"):
        _oracle(_recorder(tmp_path), tmp_path, [0.0, 0.0], [5603, 248046], {"logit_tolerance": 0.3},
                oracle="token_exact")


@pytest.mark.parametrize("text", [
    '{"threshold_overrides": {"logit_tolerance": 0.3, "logit_tolerance": 0.5}}',
    '{"threshold_overrides": {"max_ned": 0.1}}',
    '{"threshold_overrides": {}}',
    '{"threshold_overrides": {"logit_tolerance": 0.3}, "note": "x"}',
])
def test_a_threshold_sidecar_takes_only_its_known_key_once(tmp_path, monkeypatch, text):
    (tmp_path / "thresholds").mkdir()
    (tmp_path / "thresholds" / "case.json").write_text(text, encoding="utf-8")
    monkeypatch.setattr(e2e, "_TEST_DIR", tmp_path)
    with pytest.raises(AssertionError):
        e2e._thresholds("case")


def test_a_threshold_sidecar_is_optional(tmp_path, monkeypatch):
    monkeypatch.setattr(e2e, "_TEST_DIR", tmp_path)
    assert e2e._thresholds("case") == {}
    (tmp_path / "thresholds").mkdir()
    (tmp_path / "thresholds" / "case.json").write_text(
        '{"threshold_overrides": {"logit_tolerance": 0.3}}', encoding="utf-8")
    assert e2e._tolerance(e2e._thresholds("case")) == 0.3


@pytest.mark.parametrize("oracle", ["teacher_forced", "token_exact"])
def test_an_output_that_stops_early_without_an_end_token_fails(tmp_path, oracle):
    # One correct token of an eight-token budget, without the end token, is a truncated answer.
    with pytest.raises(AssertionError, match="stopped early"):
        _oracle(_recorder(tmp_path), tmp_path, [0.0], [5603], oracle=oracle, token_ids=[5603])


@pytest.mark.parametrize("oracle", ["teacher_forced", "token_exact"])
def test_an_output_that_continues_after_an_end_token_fails(tmp_path, oracle):
    with pytest.raises(AssertionError, match="continues after an end token"):
        _oracle(_recorder(tmp_path), tmp_path, [0.0, 0.0], [_EOS, 5603], oracle=oracle,
                token_ids=[_EOS, 5603])


def test_an_output_that_spends_the_budget_needs_no_end_token(tmp_path):
    check = _check(_oracle(_recorder(tmp_path), tmp_path, [0.0, 0.1], [5603, 5604], token_ids=[5603, 5605],
                           budget=2))
    assert check["passed"] is True


@pytest.mark.parametrize("oracle", ["teacher_forced", "token_exact"])
@pytest.mark.parametrize("text", ["whitE", "white ", "white\ufffd", ""])
def test_native_text_must_equal_the_reference_decoding_of_its_ids(tmp_path, oracle, text):
    # Matching ids and an answer substring are not enough: the detokenized text must be whole.
    with pytest.raises(AssertionError):
        _oracle(_recorder(tmp_path), tmp_path, [0.0, 0.0], [5603, _EOS], oracle=oracle, text=text,
                decoded="white")


@pytest.mark.parametrize("text", ["White", " white\n", "WHITE"])
def test_a_short_answer_may_differ_in_case_and_white_space_only(tmp_path, text):
    _oracle(_recorder(tmp_path), tmp_path, [0.0, 0.0], [5603, _EOS], oracle="token_exact", text=text)


@pytest.mark.parametrize(("text", "reference_text"), [
    ("whiteX", "whiteX"), ("white car", "white car"), ("White.", "White."), ("white", "black"), ("black", "white"),
])
def test_a_short_answer_is_the_whole_text_of_both_runs(tmp_path, text, reference_text):
    with pytest.raises(AssertionError):
        _oracle(_recorder(tmp_path), tmp_path, [0.0, 0.0], [5603, _EOS], oracle="token_exact", text=text,
                reference_text=reference_text)


_OCR = {"expected_mentions": ["DeepSeek-OCR-2"]}


def test_a_mention_may_stand_anywhere_as_a_whole_word(tmp_path):
    text = '""DeepSeek-OCR-2 family plugin.'
    _oracle(_recorder(tmp_path), tmp_path, [0.0, 0.0], [5603, _EOS], oracle="token_exact", text=text,
            content=_OCR)


@pytest.mark.parametrize(("text", "reference_text"), [
    ("DeepSeek-OCR-2X plugin", "DeepSeek-OCR-2X plugin"),
    ("DeepSeek-OCR-23 plugin", "DeepSeek-OCR-23 plugin"),
    ("the DeepSeek-OCR-2 plugin", "the OCR plugin"),
    ("the OCR plugin", "the DeepSeek-OCR-2 plugin"),
])
def test_a_mention_inside_a_longer_name_or_missing_from_either_run_fails(tmp_path, text, reference_text):
    with pytest.raises(AssertionError):
        _oracle(_recorder(tmp_path), tmp_path, [0.0, 0.0], [5603, _EOS], oracle="token_exact", text=text,
                reference_text=reference_text, content=_OCR)


class _Stop(Exception):
    pass


def _stop():
    raise _Stop


@pytest.mark.parametrize("case_name", sorted(e2e._CASES))
def test_the_evidence_of_a_case_attaches_only_the_images_it_feeds(tmp_path, tmp_path_factory, monkeypatch,
                                                                    case_name):
    # The report shows a case's first recorded image as its input, so the media are the files the case feeds, and no
    # fixture or other case's image named in a recorded value. The E2E runs from the repository root, where fixture
    # paths resolve; this check stops it where the runtime would start.
    pytest.importorskip("PIL")
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence", family=e2e._FAMILY, case="contract-check",
        source_revision="a" * 40, roots=(e2e._REPO, tmp_path),
    )
    monkeypatch.chdir(e2e._REPO)
    monkeypatch.setattr(e2e, "_environment", lambda: (None, None, None))
    monkeypatch.setattr(e2e, "_checkpoint", lambda manifest: tmp_path)
    monkeypatch.setattr(e2e, "_bundle", lambda *args: tmp_path / "bundle")
    monkeypatch.setattr(e2e, "_run_cli", lambda *args: _stop())
    work = tmp_path / "case"
    work.mkdir()
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(_Stop):
            e2e.test_e2e(case_name, work, tmp_path_factory, None)
    finally:
        e2e_evidence._ACTIVE.reset(token)
    _, case = e2e._CASES[case_name]
    fed = [work / f"input-{index}.png" for index in range(len(case.get("images", [])))]
    artifacts = recorder.data["artifacts"]
    assert [artifact["label"] for artifact in artifacts] == [path.name for path in fed]
    for artifact, path in zip(artifacts, fed):
        assert (recorder.directory / artifact["path"]).read_bytes() == path.read_bytes()


_MANIFEST = {"hf_id": "openbmb/MiniCPM-V-4.6", "hf_revision": "97328f3c2d70c4645b6a35594807eb6932068635"}


def _snapshot(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


def _hub(monkeypatch, snapshot_download):
    """Replace huggingface_hub, which _checkpoint imports when it runs, by a module with this snapshot_download."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))


def test_the_checkpoint_is_read_from_the_local_cache_only(tmp_path, monkeypatch):
    calls = []
    snapshot = _snapshot(tmp_path, _MANIFEST["hf_revision"])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _hub(monkeypatch, lambda **kwargs: calls.append(kwargs) or str(snapshot))
    assert e2e._checkpoint(_MANIFEST) == snapshot
    assert calls == [{"repo_id": _MANIFEST["hf_id"], "revision": _MANIFEST["hf_revision"], "local_files_only": True}]


def test_the_checkpoint_must_be_the_pinned_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _hub(monkeypatch, lambda **kwargs: str(_snapshot(tmp_path, "main")))
    with pytest.raises(AssertionError):
        e2e._checkpoint(_MANIFEST)


def test_the_checkpoint_is_resolved_offline_only(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    _hub(monkeypatch, lambda **kwargs: pytest.fail("resolved online"))
    with pytest.raises(AssertionError, match="offline"):
        e2e._checkpoint(_MANIFEST)


def test_a_snapshot_cached_without_its_file_listing_resolves_offline(tmp_path, monkeypatch):
    # A cache as CI stages it: the commit's snapshot without a listing of the revision's files. Offline, the
    # checkpoint resolves from the snapshot; asking the hub for the revision lists its files and fails offline.
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import constants

    snapshot = tmp_path / "models--openbmb--MiniCPM-V-4.6" / "snapshots" / _MANIFEST["hf_revision"]
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    assert e2e._checkpoint(_MANIFEST) == snapshot


def _torch_with_tf32():
    """The parts of torch that the reference reads: both TF32 switches on, FP32 matmul precision "highest"."""
    return types.SimpleNamespace(
        backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
                                       cudnn=types.SimpleNamespace(allow_tf32=True)),
        get_float32_matmul_precision=lambda: "highest", float32="float32",
    )


@pytest.mark.parametrize("failing", ["AutoProcessor", "AutoModelForImageTextToText"])
def test_the_reference_restores_the_tf32_switches_when_loading_fails(tmp_path, monkeypatch, failing):
    def fail(*args, **kwargs):
        raise RuntimeError("load failed")

    classes = {name: types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: types.SimpleNamespace())
               for name in ("AutoProcessor", "AutoModelForImageTextToText")}
    classes[failing] = types.SimpleNamespace(from_pretrained=fail)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(**classes))
    torch = _torch_with_tf32()
    with pytest.raises(RuntimeError, match="load failed"):
        e2e._reference(tmp_path, {}, [], [], torch)
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
