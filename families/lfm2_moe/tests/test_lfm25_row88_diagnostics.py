# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU controls for passive capture, exact replay history, and verdict isolation."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from families.lfm2_moe.tests import lfm25_row88_diagnostics as diagnostic


def target():
    return json.loads(diagnostic._FIXTURE.read_text())


def capture():
    fixture = target()
    result = diagnostic.Row88Capture()
    result.prompt_ids = fixture["prompt_ids"]
    result.calls = len(fixture["reference_prefix"])
    logits = np.zeros(128000, dtype=np.float32)
    logits[fixture["reference_prefix"][88]] = 1
    result.report.update(status="captured", logits=logits)
    return result


def test_passive_hook_records_only_row88_and_never_replaces_outputs(monkeypatch):
    torch = pytest.importorskip("torch")
    observed = []
    monkeypatch.setattr(
        diagnostic, "record_evidence", lambda name, value: observed.append((name, value))
    )

    class Model(torch.nn.Module):
        def forward(self, value):
            return SimpleNamespace(logits=value)

    model = Model()
    probe = diagnostic.Row88Capture()
    probe.attach(model)
    value = torch.zeros((1, 1, 7), dtype=torch.bfloat16)
    for index in range(94):
        value.fill_(index)
        result = model(value)
        assert result.logits is value
    probe.finish([1, 2])
    assert probe.calls == 94
    assert not model._forward_hooks
    assert probe.report["status"] == "captured"
    assert probe.report["dtype"] == "torch.bfloat16"
    assert probe.report["logits"].tolist() == [88.0] * 7
    assert len(observed) == 1 and observed[0][0] == "reference_row88"


def test_hook_failure_and_missing_row_do_not_raise(monkeypatch):
    monkeypatch.setattr(diagnostic, "record_evidence", lambda *args: None)
    probe = diagnostic.Row88Capture()
    probe.attach(SimpleNamespace())
    probe.finish([1])
    assert probe.report["status"] == "capture_error"
    probe = diagnostic.Row88Capture()
    assert probe._observe(None, (), SimpleNamespace()) is None
    probe.finish([1])
    assert probe.report["status"] == "not_captured"
    probe = diagnostic.Row88Capture()
    probe.calls = 88
    assert probe._observe(None, (), SimpleNamespace()) is None
    probe.finish([1])
    assert probe.report["status"] == "capture_error"


def test_only_exact_pinned_full_card_case_is_observed():
    fixture = target()
    manifest = {key: fixture[key] for key in ("hf_id", "hf_revision")}
    case = {"name": fixture["case"]}
    assert diagnostic.capture_for(manifest, case) is not None
    assert diagnostic.capture_for({**manifest, "hf_revision": "other"}, case) is None
    assert diagnostic.capture_for(manifest, {"name": "other"}) is None


def replay(monkeypatch, tmp_path, *, wrong_prompt=False, wrong_position=False):
    fixture = target()
    calls = []
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(tmp_path / "build"))

    def run(command, **options):
        calls.append((command, options))
        if command[0] == "cmake":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "--encode-prompt" in command:
            ids = fixture["prompt_ids"] + ([17] if wrong_prompt else [])
            return SimpleNamespace(returncode=0, stdout=json.dumps({"prompt_ids": ids}), stderr="")
        requests = [json.loads(line) for line in options["input"].splitlines()]
        assert requests[0]["tokens"] == fixture["prompt_ids"]
        assert requests[0]["reset"] and requests[0]["decode_phase"] is False
        assert requests[1]["tokens"] == fixture["reference_prefix"][:88]
        assert requests[1]["decode_phase"] is True
        logits = np.zeros(128000, dtype=np.float32)
        logits[fixture["native_prefix"][88]] = 1
        logits.tofile(requests[1]["output"])
        responses = [
            {"ready": True},
            {"position": 16, "vocab_size": 128000},
            {"position": 105 if wrong_position else 104, "vocab_size": 128000},
        ]
        return SimpleNamespace(
            returncode=0, stdout="\n".join(map(json.dumps, responses)), stderr=""
        )

    monkeypatch.setattr(diagnostic.subprocess, "run", run)
    report = {}
    diagnostic._collect_native(
        report,
        capture(),
        tmp_path / "bundle",
        "What is C. elegans?",
        fixture["native_prefix"],
        fixture["reference_prefix"],
        tmp_path,
    )
    return report, calls


def test_exact_native_prompt_and_phase_history_admission(monkeypatch, tmp_path):
    report, calls = replay(monkeypatch, tmp_path)
    assert report["status"] == "admitted"
    assert all(report["admission"].values())
    assert len(calls) == 3
    assert len(report["shared_history_ids"]) == 104
    assert "not intercepted" in report["prompt_scope"]


@pytest.mark.parametrize("wrong_prompt,wrong_position", [(True, False), (False, True)])
def test_wrong_prompt_or_state_cannot_be_admitted(
    monkeypatch, tmp_path, wrong_prompt, wrong_position
):
    report, calls = replay(
        monkeypatch, tmp_path, wrong_prompt=wrong_prompt, wrong_position=wrong_position
    )
    assert report["status"] == "target_not_reproduced"
    assert len(calls) == (2 if wrong_prompt else 3)


@pytest.mark.parametrize("which", ["reference", "native", "missing_capture", "calls"])
def test_wrong_target_prefix_or_capture_never_launches_native(monkeypatch, tmp_path, which):
    fixture = target()
    reference, native = fixture["reference_prefix"][:], fixture["native_prefix"][:]
    probe = capture()
    if which == "reference":
        reference[0] += 1
    elif which == "native":
        native[0] += 1
    elif which == "missing_capture":
        probe.report["status"] = "not_captured"
    else:
        probe.calls += 1
    monkeypatch.setattr(
        diagnostic.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("no replay on wrong target"),
    )
    report = {}
    diagnostic._collect_native(
        report, probe, tmp_path / "bundle", "prompt", native, reference, tmp_path
    )
    assert report["status"] == "target_not_reproduced"


def test_diagnostic_exception_persists_without_masking_existing_semantic_failure(
    monkeypatch, tmp_path
):
    from families.lfm2_moe.tests import test_e2e as e2e
    from tools import e2e_evidence

    fixture = target()
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence/lfm2_moe/row88",
        family="lfm2_moe",
        case="row88",
        source_revision="fixture",
    )
    monkeypatch.setattr(e2e, "_require_selected", lambda *args: None)
    monkeypatch.setattr(
        e2e, "_required_environment", lambda *args: (Path("native"), tmp_path, None)
    )
    monkeypatch.setattr(e2e, "_checkpoint", lambda *args: tmp_path)
    monkeypatch.setattr(e2e, "_build_bundle", lambda *args: None)
    monkeypatch.setattr(e2e, "_assert_rank_sections", lambda *args: None)
    native = {"token_ids": fixture["native_prefix"], "text": "full native output retained"}
    monkeypatch.setattr(e2e, "_run_native", lambda *args: native)
    monkeypatch.setattr(
        e2e,
        "_hf_reference",
        lambda *args, **kwargs: (fixture["reference_prefix"], "full reference", "decoded"),
    )

    def broken(*args):
        raise RuntimeError("diagnostic setup failed")

    def semantic_failure(*args):
        assert recorder.data["paired_row88"]["status"] == "diagnostic_error"
        raise AssertionError("unchanged semantic failure")

    monkeypatch.setattr(diagnostic, "_collect_native", broken)
    monkeypatch.setattr(e2e, "_assert_correctness", semantic_failure)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(AssertionError, match="unchanged semantic failure"):
            e2e.test_e2e(fixture["case"], SimpleNamespace(config=None), tmp_path)
        recorder.finish("failed", failure="unchanged semantic failure")
    finally:
        e2e_evidence._ACTIVE.reset(token)
    saved = json.loads((recorder.directory / "evidence.json").read_text())
    assert saved["paired_row88"]["error_type"] == "RuntimeError"
    assert saved["native"]["text"] == native["text"]
    assert saved["reference"]["reference_text"] == "full reference"
    assert saved["failure_stage"] == "compare"


@pytest.mark.parametrize("fails", [False, True])
def test_existing_generate_arguments_and_reference_failure_are_preserved(
    monkeypatch, tmp_path, fails
):
    import sys
    from types import ModuleType
    from families.lfm2_moe.tests import test_e2e as e2e

    torch = pytest.importorskip("torch")
    fixture, observed, calls = target(), [], []
    monkeypatch.setattr(
        diagnostic, "record_evidence", lambda name, value: observed.append((name, value))
    )

    class Model(torch.nn.Module):
        device, dtype = "cuda", torch.bfloat16

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()

        def to(self, device):
            assert device == "cuda"
            return self

        def forward(self):
            return SimpleNamespace(logits=torch.ones((1, 1, 7), dtype=torch.bfloat16))

        def generate(self, **kwargs):
            calls.append(kwargs)
            for _ in range(89):
                self()
            if fails:
                raise RuntimeError("original generate failure")
            return np.array([fixture["prompt_ids"] + fixture["reference_prefix"]])

    class Inputs(dict):
        def to(self, device):
            return self

    model = Model()
    inputs = Inputs(input_ids=np.array([fixture["prompt_ids"]]), attention_mask=np.ones((1, 16)))
    monkeypatch.setattr(e2e, "_render_prompt", lambda *args: inputs)
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: model
    )
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(
            decode=lambda *args, **kwargs: "text"
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    probe = diagnostic.Row88Capture()
    case = {"max_new_tokens": 8192, "repetition_penalty": 1.0, "use_chat_template": True}
    if fails:
        with pytest.raises(RuntimeError, match="original generate failure"):
            e2e._hf_reference(
                tmp_path, {"precision": "bf16"}, case, "prompt", [1], torch, diagnostic=probe
            )
    else:
        result = e2e._hf_reference(
            tmp_path, {"precision": "bf16"}, case, "prompt", [1], torch, diagnostic=probe
        )
        assert result[0] == fixture["reference_prefix"]
    assert len(calls) == 1
    assert calls[0]["input_ids"] is inputs["input_ids"]
    assert calls[0]["attention_mask"] is inputs["attention_mask"]
    assert {key: value for key, value in calls[0].items() if key not in inputs} == {
        "max_new_tokens": 8192,
        "do_sample": False,
        "repetition_penalty": 1.0,
    }
    assert not model._forward_hooks
    assert observed[0][1]["status"] == "captured"
