# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

import tools.perf_matrix as perf
from apps.benchmark.performance.baselines import (
    hf_transformers,
    lance_reference,
    sana_wm_reference,
    task_reference,
)


REPO = Path(__file__).resolve().parents[4]
SUITE = REPO / "apps/benchmark/performance/release.yaml"


def test_reference_config_preserves_explicit_values_without_defaults() -> None:
    original = {
        "source_text": "Hello",
        "source_language": "eng_Latn",
        "config": {
            "temperature": 0.0,
            "seed": 0,
            "use_chat_template": False,
            "labels": [],
            "suffix": "",
        },
    }
    flattened = hf_transformers.flatten_config(original)
    assert flattened == {
        "source_text": "Hello",
        "source_language": "eng_Latn",
        "temperature": 0.0,
        "seed": 0,
        "use_chat_template": False,
        "labels": [],
        "suffix": "",
    }
    assert "config" in original
    assert hf_transformers.flatten_config({"prompt": "Hello"}) == {"prompt": "Hello"}
    with pytest.raises(ValueError, match="duplicate"):
        hf_transformers.flatten_config({"seed": 0, "config": {"seed": 1}})
    with pytest.raises(ValueError, match="object"):
        hf_transformers.flatten_config({"config": [1]})


def test_translation_languages_use_tokenizer_controls_and_preserve_absence() -> None:
    class Tokenizer:
        src_lang = "default"
        unk_token_id = 99

        def convert_tokens_to_ids(self, token):
            return {"eng_Latn": 10, "fra_Latn": 11}.get(token, 99)

        def convert_ids_to_tokens(self, token):
            return {10: "eng_Latn", 11: "fra_Latn"}[token]

    tokenizer = Tokenizer()
    assert hf_transformers._translation_controls(tokenizer, {}) == {}
    assert tokenizer.src_lang == "default"
    assert hf_transformers._translation_controls(
        tokenizer, {"source_language": "eng_Latn", "target_language": "fra_Latn"}
    ) == {"forced_bos_token_id": 11}
    assert tokenizer.src_lang == "eng_Latn"
    assert hf_transformers._translation_controls(
        tokenizer, {"source_language_token_id": 10, "forced_bos_token_id": 0}
    ) == {"forced_bos_token_id": 0}
    with pytest.raises(ValueError, match="disagrees"):
        hf_transformers._translation_controls(
            tokenizer, {"target_language": "fra_Latn", "forced_bos_token_id": 10}
        )
    with pytest.raises(ValueError, match="recognize"):
        hf_transformers._translation_controls(tokenizer, {"target_language": "invalid"})
    fixed = SimpleNamespace(source_lang="en", target_lang="ru")
    assert (
        hf_transformers._translation_controls(
            fixed, {"source_language": "en", "target_language": "ru"}
        )
        == {}
    )
    with pytest.raises(ValueError, match="target language"):
        hf_transformers._translation_controls(fixed, {"target_language": "de"})
    with pytest.raises(ValueError, match="nonnegative integer"):
        hf_transformers._translation_controls(tokenizer, {"forced_bos_token_id": -1.0})


def test_translation_and_config_reach_reference_generate(monkeypatch) -> None:
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        @property
        def shape(self):
            return self.values.shape

        def __getitem__(self, key):
            return Tensor(self.values[key])

        def to(self, *_args, **_kwargs):
            return self

        def detach(self):
            return self

        def tolist(self):
            return self.values.tolist()

    class Tokenizer:
        pad_token_id = 0
        unk_token_id = 99
        src_lang = "default"

        def convert_tokens_to_ids(self, language):
            return {"eng_Latn": 10, "fra_Latn": 11}.get(language, 99)

        def __call__(self, prompt, **_kwargs):
            assert prompt == "source text"
            return {"input_ids": Tensor([[self.convert_tokens_to_ids(self.src_lang), 17]])}

        def decode(self, tokens, **_kwargs):
            return "translated:" + str(tokens)

    captured = {}

    class Model:
        config = SimpleNamespace(decoder_start_token_id=0, eos_token_id=3)

        def generate(self, **kwargs):
            captured.update(kwargs)
            return Tensor([[0, 8, 3]])

    fake = ModuleType("torch")
    fake.float16, fake.float32, fake.bfloat16, fake.int64 = "fp16", "fp32", "bf16", "i64"
    fake.inference_mode = nullcontext
    fake.autocast = lambda **_kwargs: nullcontext()
    monkeypatch.setitem(sys.modules, "torch", fake)
    invoke, summarize = hf_transformers._generation_call(
        Tokenizer(),
        Model(),
        {
            "source_text": "source text",
            "source_language": "eng_Latn",
            "target_language": "fra_Latn",
            "config": {"max_new_tokens": 5, "temperature": 0.0, "repetition_penalty": 1.1},
        },
        "seq2seq-lm",
        "strip-start-and-eos",
        "fp32",
        "generate",
    )
    output = summarize(invoke())
    assert captured["input_ids"].tolist() == [[10, 17]]
    assert captured["forced_bos_token_id"] == 11
    assert captured["max_new_tokens"] == 5 and captured["do_sample"] is False
    assert captured["repetition_penalty"] == 1.1
    assert output["token_ids"] == [8]


@pytest.mark.parametrize(
    "entry_id,task,operation",
    [
        ("m2m_100.generate", "text_translation", "translate"),
        ("marian.generate", "text_translation", "translate"),
        ("patchtst.solve", "series_to_regression_distribution", "regress"),
        ("patchtst.solve", "series_to_regression_values", "regress"),
    ],
)
def test_release_entry_survives_semantic_primary_switch(
    tmp_path, monkeypatch, entry_id, task, operation
):
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(row for row in entries if row["id"] == entry_id)
    original_resolve = perf.ManifestCatalog.resolve

    def resolve(catalog, selector):
        return replace(original_resolve(catalog, selector), task=task)

    monkeypatch.setattr(perf.ManifestCatalog, "resolve", resolve)
    entry = perf.resolve_entries([spec], environment)[0]
    assert entry.spec["id"] == spec["id"]
    assert entry.spec["operation"] == entry.case.operation == operation
    assert entry.spec["measurement"] == spec["measurement"]
    assert entry.spec["equivalence_margin_percent"] == spec["equivalence_margin_percent"]
    assert spec["operation"] != operation
    if operation == "translate":
        assert entry.case.request["source_text"]
        assert perf._baseline_task(entry) == "seq2seq-lm"
        assert perf._contract_name(entry) == "exact-token-ids"


def test_seq2seq_reference_choice_is_an_existing_entry_field_not_a_family_registry(tmp_path):
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    for name in ("bart", "m2m_100", "marian", "t5"):
        spec = next(row for row in entries if row["id"] == name + ".generate")
        assert spec["baseline"]["task"] == "seq2seq-lm"
        entry = perf.resolve_entries([spec], environment)[0]
        assert (
            perf._baseline_task(replace(entry, model=replace(entry.model, family="new_owner")))
            == "seq2seq-lm"
        )


def test_family_secondary_task_reaches_candidate_reference_and_output_contract(tmp_path, monkeypatch):
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(row for row in entries if row["id"] == "patchtst.solve")
    original_resolve = perf.ManifestCatalog.resolve

    def resolve(catalog, selector):
        model = original_resolve(catalog, selector)
        cases = tuple({**case, "selected_task": "series_to_regression_values"} for case in model.testcases)
        return replace(model, task="series_to_point_forecast", testcases=cases)

    monkeypatch.setattr(perf.ManifestCatalog, "resolve", resolve)
    entry, = perf.resolve_entries([spec], environment)
    assert entry.case.effective_task == "series_to_regression_values"
    assert entry.model.task == entry.case.worker_request()["expected_task"] == "series_to_point_forecast"
    assert entry.spec["operation"] == "regress"
    assert perf._contract_name(entry) == "regression-values"
    candidate = perf.candidate_command(entry, environment, tmp_path / "candidate")
    reference = perf.baseline_command(entry, environment, tmp_path / "reference.json")
    assert candidate[candidate.index("--task") + 1] == "series_to_regression_values"
    assert reference[reference.index("--selected-task") + 1] == "series_to_regression_values"
    assert reference[reference.index("--manifest") + 1] == str(entry.model.manifest_path)
    request = json.loads(reference[reference.index("--request-json") + 1])
    assert "selected_task" not in request
    assert entry.spec["measurement"] == spec["measurement"]
    assert entry.spec["equivalence_margin_percent"] == spec["equivalence_margin_percent"]


def test_reference_selection_does_not_rewrite_manifest_or_silently_choose_default(tmp_path):
    manifest = tmp_path / "model.json"
    manifest.write_text('{"task":"series_to_point_forecast"}')
    original = manifest.read_bytes()
    arguments = SimpleNamespace(manifest=manifest, selected_task="series_to_quantile_forecast")
    assert task_reference._selected_task(arguments) == "series_to_quantile_forecast"
    assert manifest.read_bytes() == original
    arguments.selected_task = None
    assert task_reference._selected_task(arguments) == "series_to_point_forecast"
    for invalid in ("", " ", " series_to_point_forecast", 7):
        arguments.selected_task = invalid
        with pytest.raises(ValueError, match="selected_task"):
            task_reference._selected_task(arguments)


@pytest.mark.parametrize("payload", [{"token_ids": []}, {"token_ids": [7], "prompt": "hello"}])
def test_text_only_reference_loaders_never_replace_token_input_with_empty_text(payload):
    with pytest.raises(ValueError, match="token_ids"):
        hf_transformers._batch_prompt(payload)
    arguments = SimpleNamespace(timing_contract_json=json.dumps({
        "timing_scope": "task-model-call-wall", "input_preparation_included": False,
        "asset_loading_included": False,
    }))
    with pytest.raises(ValueError, match="token_ids"):
        task_reference._load_embedding(arguments, payload, {})


class _SummaryTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    def numel(self):
        return self.values.size

    def isfinite(self):
        return _SummaryTensor(np.isfinite(self.values))

    def all(self):
        return _SummaryTensor(self.values.all())

    def item(self):
        return self.values.item()

    def __getitem__(self, key):
        return _SummaryTensor(self.values[key])

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()


def test_forecast_reference_names_axes_without_arbitrary_squeeze_or_sorting():
    point = task_reference._forecast_summary(
        _SummaryTensor(np.zeros((1, 4, 3))), "series_to_point_forecast"
    )
    assert point["shape"] == [4, 3] and point["axes"] == ["horizon", "channel"]
    assert point["horizon_steps"] == [1, 2, 3, 4] and point["forecast_elements"] == 12
    quantile = task_reference._forecast_summary(
        _SummaryTensor(np.zeros((1, 3, 4))), "series_to_quantile_forecast", [0.1, 0.5, 0.9]
    )
    assert quantile["shape"] == [3, 4, 1]
    assert quantile["axes"] == ["quantile", "horizon", "channel"]
    assert quantile["quantile_levels"] == [0.1, 0.5, 0.9]
    with pytest.raises(ValueError, match="one-series"):
        task_reference._forecast_summary(
            _SummaryTensor(np.zeros((2, 4, 3))), "series_to_point_forecast"
        )
    with pytest.raises(ValueError, match="quantile levels"):
        task_reference._forecast_summary(
            _SummaryTensor(np.zeros((1, 3, 4))), "series_to_quantile_forecast", []
        )


def test_semantic_forecast_comparison_rejects_swapped_or_missing_axes():
    entry = SimpleNamespace(
        spec={"baseline": {"output_contract": "forecast-shape"}},
        model=SimpleNamespace(task="series_to_point_forecast"),
        case=SimpleNamespace(selected_task=None),
    )
    summary = {
        "forecast_elements": 12,
        "shape": [4, 3],
        "axes": ["horizon", "channel"],
        "horizon_steps": [1, 2, 3, 4],
    }
    assert perf._output_contract(
        entry, {"output_summary": summary}, {"output_summary": dict(summary)}
    )[0]
    for patch in (
        {"shape": [3, 4]},
        {"axes": ["channel", "horizon"]},
        {"axes": None},
        {"horizon_steps": [1, 3, 5, 7]},
    ):
        assert not perf._output_contract(
            entry, {"output_summary": summary}, {"output_summary": {**summary, **patch}}
        )[0]
    entry.model.task = "series_to_quantile_forecast"
    quantiles = {"forecast_elements": 12, "shape": [3, 4, 1],
                 "axes": ["quantile", "horizon", "channel"], "horizon_steps": [1, 2, 3, 4],
                 "quantile_levels": [0.1, 0.5, 0.9]}
    assert perf._output_contract(entry, {"output_summary": quantiles},
                                 {"output_summary": dict(quantiles)})[0]
    missing_levels = {key: value for key, value in quantiles.items() if key != "quantile_levels"}
    assert not perf._output_contract(entry, {"output_summary": missing_levels},
                                     {"output_summary": dict(missing_levels)})[0]


def test_regression_values_reference_retains_single_target_axis():
    summary = task_reference._regression_values_summary(_SummaryTensor([[1.5, -2.0]]))
    assert summary["values"] == [1.5, -2.0]
    assert summary["axes"] == ["target"] and summary["target_count"] == 2
    assert summary["target_names"] == [] and summary["target_units"] == []
    assert "distribution" not in summary and "horizon_steps" not in summary
    entry = SimpleNamespace(spec={"baseline": {"output_contract": "regression-values"}})
    assert perf._output_contract(entry, {"output_summary": summary},
                                 {"output_summary": dict(summary)})[0]
    assert not perf._output_contract(
        entry, {"output_summary": {**summary, "target_names": ["x", "y"]}},
        {"output_summary": {**summary, "target_names": ["y", "x"]}})[0]
    for invalid in ({**summary, "axes": ["horizon"]}, {**summary, "target_count": 1},
                    {**summary, "values": [float("nan"), 2]},
                    {**summary, "target_names": ["one"]}):
        assert not perf._output_contract(entry, {"output_summary": invalid},
                                         {"output_summary": dict(invalid)})[0]
    for shape in ((2,), (2, 2), (1, 0), (1, 2, 1)):
        with pytest.raises(ValueError, match="one batch"):
            task_reference._regression_values_summary(_SummaryTensor(np.zeros(shape)))
    with pytest.raises(ValueError, match="finite"):
        task_reference._regression_values_summary(_SummaryTensor([[float("inf")]]))


def test_regression_distribution_keeps_parameter_names_and_target_axis():
    values = (_SummaryTensor([[1.0, 2.0]]), _SummaryTensor([[0.5, 0.7]]))
    summary = task_reference._regression_summary(values, "normal", ["loc", "scale"])
    assert summary["target_count"] == 2 and summary["axes"] == ["target"]
    assert summary["parameters"] == [
        {"name": "location", "values": [1.0, 2.0]},
        {"name": "scale", "values": [0.5, 0.7]},
    ]
    entry = SimpleNamespace(spec={"baseline": {"output_contract": "regression-distribution"}})
    assert perf._output_contract(
        entry, {"output_summary": summary}, {"output_summary": dict(summary)}
    )[0]
    assert not perf._output_contract(
        entry,
        {"output_summary": summary},
        {"output_summary": {**summary, "distribution": "student_t"}},
    )[0]
    with pytest.raises(ValueError, match="same target axis"):
        task_reference._regression_summary(
            (_SummaryTensor([[1, 2]]), _SummaryTensor([[1]])), "normal", ["loc", "scale"]
        )
    student = task_reference._regression_summary(
        (_SummaryTensor([[4.0]]), _SummaryTensor([[1.0]]), _SummaryTensor([[0.5]])),
        "student_t", ["df", "loc", "scale"],
    )
    assert [parameter["name"] for parameter in student["parameters"]] == [
        "degrees_of_freedom", "location", "scale"
    ]
    with pytest.raises(ValueError, match="parameter names"):
        task_reference._regression_summary(values, "normal", ["loc", "location"])


def test_timeseries_reference_preserves_observed_masks_and_unobserved_padding():
    request = {"past_values": [1, 2, 3], "observed_mask": [1, 0, 1]}
    observed = task_reference._observed_values(request, 3)
    assert observed == [1.0, 0.0, 1.0]
    assert task_reference._align(observed, 5, 0.0) == [0, 0, 1, 0, 1]
    assert task_reference._observed_values({"past_values": [1, 2]}, 2) == [1, 1]
    assert task_reference._observed_values({"observed_mask": []}, 2) == [1, 1]
    with pytest.raises(ValueError, match="match past_values"):
        task_reference._observed_values({"observed_mask": [1]}, 2)


def _voicechat_reference_fixture(tmp_path, monkeypatch):
    events = []
    reference = tmp_path / "speech-reference"
    utility = reference / "nemo/collections/speechlm2/inference/utils/offline_voicechat.py"
    utility.parent.mkdir(parents=True)
    utility.write_text("# official utility location fixture\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    audio_path = tmp_path / "request.wav"
    audio_path.write_bytes(b"fixture; decoded by the mocked official loader")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"task":"offline_speech_dialogue"}', encoding="utf-8")
    arguments = SimpleNamespace(
        model=str(checkpoint),
        revision="checkpoint-revision",
        manifest=manifest,
        family="nemotron_voicechat",
        precision="fp32",
        operation="speech_dialogue",
        local_files_only=True,
    )
    request = {"audio_path": str(audio_path)}
    options = {"reference_repo": str(reference)}
    state = {
        "kwargs": [],
        "prompts": [],
        "events": events,
        "metadata": SimpleNamespace(samplerate=16000, channels=1, frames=4),
    }

    class Tensor(_SummaryTensor):
        def __init__(self, values, label="output"):
            super().__init__(values)
            self.label = label

        def to(self, device):
            events.append(("transfer", self.label, device))
            return self

        def __getitem__(self, key):
            return Tensor(self.values[key], self.label)

        def cpu(self):
            events.append(("materialize", self.label))
            return self

        def numpy(self):
            return np.asarray(self.values, dtype=np.float32)

    class Model:
        source_sample_rate = 16000
        target_sample_rate = 22050

        def float(self):
            events.append("float32")
            return self

    model = Model()
    signal = Tensor([[0.1, 0.2, 0.3, 0.4]], "signal")
    lengths = Tensor([4], "lengths")
    upstream = ModuleType("nemo.collections.speechlm2.inference.utils.offline_voicechat")
    upstream.__file__ = str(utility)

    def build_model(path, *, device):
        events.append(("build", path, device))
        return model

    def load_wav(path, *, device):
        events.append(("decode", path, device))
        return signal, signal, lengths

    def encode_prompt(received_model, prompt, *, device):
        assert received_model is model
        events.append(("prompt", prompt, device))
        state["prompts"].append(prompt)
        return "prompt-token-buffer", "prompt-length-buffer"

    def inference(received_model, **kwargs):
        assert received_model is model
        events.append("infer")
        state["kwargs"].append(kwargs)
        # The padded NaN must not be included in the valid audio result.
        return state.get(
            "result",
            {
                "text": ["Full spoken answer."],
                "audio": Tensor([[0.1, 0.2, 0.3, float("nan")]]),
                "audio_len": Tensor([3]),
            },
        )

    upstream.build_model = build_model
    upstream.load_wav_16k_mono = load_wav
    upstream.encode_system_prompt = encode_prompt
    upstream.run_offline_inference = inference
    sf = ModuleType("soundfile")
    sf.info = lambda _path: state["metadata"]

    def write(path, values, rate, **kwargs):
        events.append(("write_audio", str(path), rate, kwargs))
        state["written_audio"] = np.array(values, copy=True)

    sf.write = write
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    monkeypatch.setitem(sys.modules, "soundfile", sf)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setattr(
        task_reference, "_seed_all", lambda _torch, seed: events.append(("seed", seed))
    )
    monkeypatch.setattr(task_reference, "_synchronize", lambda: events.append("barrier"))
    return arguments, request, options, state, upstream, Tensor


def test_voicechat_reference_builds_once_and_times_real_offline_calls(tmp_path, monkeypatch):
    arguments, request, options, state, _upstream, _tensor = _voicechat_reference_fixture(
        tmp_path, monkeypatch
    )
    session = task_reference._load_voicechat(arguments, request, options)
    assert session.timing_scope == "task-pipeline-call-wall"
    assert session.input_preparation_included is True and session.asset_loading_included is False
    assert state["events"] == [
        ("build", arguments.model, "cuda"),
        "float32",
        ("decode", request["audio_path"], "cpu"),
    ]
    samples, output = task_reference._measure(session, warmup=1, iterations=2)
    assert len(samples) == 2 and all(value > 0 for value in samples)
    assert sum(isinstance(event, tuple) and event[0] == "build" for event in state["events"]) == 1
    assert sum(isinstance(event, tuple) and event[0] == "decode" for event in state["events"]) == 1
    assert state["events"].count("infer") == 3
    assert state["events"].count(("seed", 0)) == 3
    assert state["events"].count(("transfer", "signal", "cuda")) == 3
    assert state["prompts"] == [task_reference.VOICECHAT_SYSTEM_PROMPT] * 3
    assert output["text"] == "Full spoken answer."
    assert output["audio_samples"] == output["num_samples"] == 3
    assert output["sample_rate"] == 22050 and output["channels"] == 1
    assert output["input_samples"] == 4 and output["input_sample_rate"] == 16000
    np.testing.assert_array_equal(
        output["_audio_f32"], np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    )
    for kwargs in state["kwargs"]:
        assert kwargs["decode_audio"] is True and kwargs["input_pad_len"] == 0
        assert kwargs["prompt_tokens"] == "prompt-token-buffer"
        assert kwargs["prompt_token_lens"] == "prompt-length-buffer"
        assert not (
            {"temperature", "top_p", "repetition_penalty", "function_calls"} & kwargs.keys()
        )


def test_voicechat_reference_preserves_explicit_prompt_sampling_and_seed(tmp_path, monkeypatch):
    arguments, request, options, state, _upstream, _tensor = _voicechat_reference_fixture(
        tmp_path, monkeypatch
    )
    session = task_reference._load_voicechat(
        arguments,
        {
            **request,
            "system_prompt": "  An explicit spoken prompt.  ",
            "config": {
                "seed": 17,
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.2,
                "presence_penalty": 0.0,
            },
        },
        options,
    )
    session.invoke()
    kwargs = state["kwargs"][0]
    assert state["prompts"] == ["  An explicit spoken prompt.  "]
    assert ("seed", 17) in state["events"]
    assert {
        key: kwargs[key]
        for key in ("temperature", "top_p", "repetition_penalty", "presence_penalty")
    } == {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.2, "presence_penalty": 0.0}


@pytest.mark.parametrize(
    "patch",
    [
        {"tools": []},
        {"tail_frames": 1},
        {"finish_tail_frames": -1},
        {"max_new_tokens": 256},
        {"system_prompt": "   "},
        {"seed": -1},
        {"temperature": True},
    ],
)
def test_voicechat_reference_rejects_unmatched_scope_before_loading(tmp_path, monkeypatch, patch):
    arguments, request, options, state, _upstream, _tensor = _voicechat_reference_fixture(
        tmp_path, monkeypatch
    )
    with pytest.raises(ValueError):
        task_reference._load_voicechat(arguments, {**request, **patch}, options)
    assert state["events"] == []


def test_voicechat_reference_does_not_relabel_live_dialogue_as_offline(tmp_path, monkeypatch):
    arguments, request, options, state, _upstream, _tensor = _voicechat_reference_fixture(
        tmp_path, monkeypatch
    )
    arguments.manifest.write_text('{"task":"duplex_speech_dialogue"}', encoding="utf-8")
    with pytest.raises(ValueError, match="live or tool"):
        task_reference._load_voicechat(arguments, request, options)
    assert state["events"] == []


def test_voicechat_empty_prompt_matches_native_default_selection(tmp_path, monkeypatch):
    arguments, request, options, state, _upstream, _tensor = _voicechat_reference_fixture(
        tmp_path, monkeypatch)
    session = task_reference._load_voicechat(arguments, {**request, "system_prompt": ""}, options)
    session.invoke()
    assert state["prompts"] == [task_reference.VOICECHAT_SYSTEM_PROMPT]


def test_voicechat_reference_rejects_unmatched_audio_and_stale_import(tmp_path, monkeypatch):
    arguments, request, options, state, upstream, _tensor = _voicechat_reference_fixture(
        tmp_path, monkeypatch
    )
    state["metadata"] = SimpleNamespace(samplerate=48000, channels=2, frames=4)
    with pytest.raises(ValueError, match="16 kHz mono"):
        task_reference._load_voicechat(arguments, request, options)
    assert state["events"] == []
    upstream.__file__ = str(tmp_path / "another-installation.py")
    with pytest.raises(RuntimeError, match="different NeMo installation"):
        task_reference._load_voicechat(arguments, request, options)
    assert state["events"] == []


def test_voicechat_reference_rejects_invalid_live_output(tmp_path, monkeypatch):
    arguments, request, options, state, _upstream, tensor = _voicechat_reference_fixture(
        tmp_path, monkeypatch
    )
    session = task_reference._load_voicechat(arguments, request, options)
    state["result"] = {
        "text": ["real text"],
        "audio": tensor([[0.1, 0.2]]),
        "audio_len": tensor([3]),
    }
    with pytest.raises(RuntimeError, match="audio_len"):
        session.invoke()
    state["result"] = {
        "text": ["real text"],
        "audio": tensor([[float("nan")]]),
        "audio_len": tensor([1]),
    }
    with pytest.raises(RuntimeError, match="nonfinite"):
        session.invoke()


def test_voicechat_runner_writes_audio_after_measurement(tmp_path, monkeypatch):
    arguments, request, options, state, _upstream, _tensor = _voicechat_reference_fixture(
        tmp_path, monkeypatch
    )
    arguments.adapter = "nemo-voicechat"
    arguments.request_json = json.dumps(request)
    arguments.adapter_options_json = json.dumps(options)
    arguments.timing_contract_json = json.dumps(
        {
            "timing_scope": "task-pipeline-call-wall",
            "input_preparation_included": True,
            "asset_loading_included": False,
        }
    )
    arguments.mode = "pytorch-eager"
    arguments.padding = "longest"
    arguments.warmup, arguments.iterations = 1, 2
    arguments.case_name = "offline-voicechat"
    arguments.output = tmp_path / "reference.json"
    monkeypatch.setattr(task_reference, "_environment", lambda: {})
    assert task_reference.run(arguments) == 0
    result = json.loads(arguments.output.read_text(encoding="utf-8"))
    assert result["model_load_included"] is False
    assert result["measurement_policy"]["input_preparation_included"] is True
    assert result["output_summary"]["text"] == "Full spoken answer."
    assert result["output_summary"]["audio_artifact"].endswith("reference.audio.wav")
    assert "_audio_f32" not in result["output_summary"]
    assert state["events"][-1] == (
        "write_audio",
        str(tmp_path / "reference.audio.wav"),
        22050,
        {"subtype": "FLOAT"},
    )
    np.testing.assert_array_equal(
        state["written_audio"], np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    )


def _moge_reference_fixture(tmp_path, monkeypatch):
    events = []
    reference = tmp_path / "moge-reference"
    module_path = reference / "moge/model/v2.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# upstream location fixture\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"fixture weights")
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"fixture decoded by mocked PIL")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"task":"image_to_metric_geometry"}', encoding="utf-8")
    arguments = SimpleNamespace(
        model=str(checkpoint),
        revision="checkpoint-revision",
        manifest=manifest,
        family="moge",
        operation="geometry",
        precision="fp32",
        local_files_only=True,
    )
    state = {
        "events": events,
        "inputs": [],
        "inference": [],
        "pixels": np.asarray([[[255, 0, 127], [0, 255, 64]]], dtype=np.uint8),
        "checkpoint": {"model_config": {"encoder": "fixture"}, "model": {"weight": "fixture"}},
        "arrays": {
            "points": np.asarray([[[0.25, -0.5, 2], [np.inf, np.inf, np.inf]]], dtype=np.float32),
            "depth": np.asarray([[2, np.inf]], dtype=np.float32),
            "mask": np.asarray([[True, False]]),
            "intrinsics": np.asarray([[0.8, 0, 0.5], [0, 1.1, 0.5], [0, 0, 1]], dtype=np.float32),
        },
    }

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def permute(self, *axes):
            events.append(("permute", axes))
            return Tensor(self.values.transpose(axes))

        def to(self, device):
            events.append(("input_transfer", device))
            return self

        def detach(self):
            return self

        def cpu(self):
            events.append("materialize")
            return self

        def numpy(self):
            return self.values

    class Model:
        def __init__(self, **kwargs):
            events.append(("construct", kwargs))
            state["model"] = self

        def load_state_dict(self, weights, *, strict):
            events.append(("load_state", weights, strict))
            return state.get("state_mismatch", ([], []))

        def eval(self):
            events.append("eval")
            return self

        def float(self):
            events.append("float32")
            return self

        def to(self, device):
            events.append(("model_transfer", device))
            return self

        def infer(self, tensor, **kwargs):
            events.append("infer")
            state["inputs"].append(np.array(tensor.values, copy=True))
            state["inference"].append(kwargs)
            return {name: Tensor(value) for name, value in state["arrays"].items()}

    module = ModuleType("moge.model.v2")
    module.__file__ = str(module_path)
    module.MoGeModel = Model
    attention = ModuleType("torch.nn.attention")
    attention.SDPBackend = SimpleNamespace(MATH=object())

    class MathContext:
        def __enter__(self):
            events.append("math_enter")

        def __exit__(self, *_args):
            events.append("math_exit")

    def sdpa_kernel(backends):
        assert backends == [attention.SDPBackend.MATH]
        events.append("math_backend")
        return MathContext()

    attention.sdpa_kernel = sdpa_kernel
    fake_torch = ModuleType("torch")

    def load(path, **kwargs):
        events.append(("load_checkpoint", str(path), kwargs))
        return state["checkpoint"]

    def from_numpy(values):
        events.append("prepare_tensor")
        return Tensor(values)

    fake_torch.load = load
    fake_torch.from_numpy = from_numpy
    fake_torch.backends = SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        cudnn=SimpleNamespace(allow_tf32=True),
    )

    class Image:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            events.append("close_image")

        def convert(self, mode):
            events.append(("color", mode))
            return self

        def __array__(self, dtype=None, copy=None):
            return np.asarray(state["pixels"], dtype=dtype)

    def open_image(path):
        events.append(("decode_image", str(path)))
        return Image()

    pil = ModuleType("PIL")
    pil.Image = SimpleNamespace(open=open_image)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, attention.__name__, attention)
    monkeypatch.setattr(task_reference, "_synchronize", lambda: events.append("barrier"))
    return (
        arguments,
        {"image_path": str(image_path)},
        {"reference_repo": str(reference), "num_tokens": 1800},
        state,
    )


def test_moge_reference_uses_public_upstream_apis_and_loads_once(tmp_path, monkeypatch):
    arguments, request, options, state = _moge_reference_fixture(tmp_path, monkeypatch)
    session = task_reference._load_moge(arguments, request, options)
    assert session.timing_scope == "task-pipeline-call-wall"
    assert session.input_preparation_included and not session.asset_loading_included
    assert (
        "load_checkpoint",
        str(Path(arguments.model) / "model.pt"),
        {"map_location": "cpu", "weights_only": True, "mmap": True},
    ) in state["events"]
    assert state["model"].onnx_compatible_mode is True
    assert "prepare_tensor" not in state["events"] and "infer" not in state["events"]
    samples, summary = task_reference._measure(session, warmup=1, iterations=2)
    assert len(samples) == 2 and all(value > 0 for value in samples)
    assert (
        sum(isinstance(event, tuple) and event[0] == "load_checkpoint" for event in state["events"])
        == 1
    )
    assert (
        sum(isinstance(event, tuple) and event[0] == "decode_image" for event in state["events"])
        == 1
    )
    assert state["events"].count("prepare_tensor") == 3
    assert (
        state["events"].count("infer")
        == state["events"].count("math_backend")
        == state["events"].count("math_enter")
        == state["events"].count("math_exit")
        == 3
    )
    for controls in state["inference"]:
        assert controls == {
            "num_tokens": 1800,
            "use_fp16": False,
            "force_projection": True,
            "apply_mask": True,
        }
    expected = (state["pixels"].astype(np.float32) / 255.0).transpose(2, 0, 1)
    for values in state["inputs"]:
        np.testing.assert_array_equal(values, expected)
    assert summary["point_shape"] == [1, 2, 3] and summary["units"] == "meters"
    assert summary["requested_fov_x"] is None
    assert np.isposinf(summary["_geometry_arrays"]["depth"][0, 1])


def test_moge_reference_preserves_explicit_fov_degrees(tmp_path, monkeypatch):
    arguments, request, options, state = _moge_reference_fixture(tmp_path, monkeypatch)
    session = task_reference._load_moge(arguments, {**request, "config": {"fov_x": 72.5}}, options)
    summary = session.invoke()
    assert state["inference"][0]["fov_x"] == 72.5
    assert summary["requested_fov_x"] == 72.5


@pytest.mark.parametrize(
    "request_update,options_update",
    [
        ({}, {"num_tokens": None}),
        ({"num_tokens": 3600}, {}),
        ({"num_tokens": 1800}, {"num_tokens": 3600}),
        ({"fov_x": 0}, {}),
        ({"fov_x": 180}, {}),
        ({"fov_x": True}, {}),
        ({"apply_mask": False}, {}),
    ],
)
def test_moge_reference_rejects_unqualified_controls_before_model_load(
    tmp_path, monkeypatch, request_update, options_update
):
    arguments, request, options, state = _moge_reference_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        task_reference._load_moge(
            arguments, {**request, **request_update}, {**options, **options_update}
        )
    assert not any(
        isinstance(event, tuple) and event[0] == "load_checkpoint" for event in state["events"]
    )


def test_moge_reference_preserves_strict_checkpoint_contract(tmp_path, monkeypatch):
    arguments, request, options, state = _moge_reference_fixture(tmp_path, monkeypatch)
    state["state_mismatch"] = (["uninitialized_weight"], [])
    with pytest.raises(ValueError, match="state mismatch"):
        task_reference._load_moge(arguments, request, options)
    assert "infer" not in state["events"]
    state["checkpoint"]["unexpected"] = 1
    with pytest.raises(ValueError, match="top-level"):
        task_reference._load_moge(arguments, request, options)


def test_moge_geometry_artifacts_preserve_masked_infinity_and_calibration(tmp_path, monkeypatch):
    arguments, request, options, state = _moge_reference_fixture(tmp_path, monkeypatch)
    summary = dict(task_reference._load_moge(arguments, request, options).invoke())
    task_reference._write_geometry_artifacts(summary, tmp_path / "reference.json")
    assert "_geometry_arrays" not in summary
    assert summary["height"] == 1 and summary["width"] == 2 and summary["valid_pixels"] == 1
    np.testing.assert_array_equal(
        np.fromfile(summary["points_artifact"], dtype="<f4").reshape(1, 2, 3),
        state["arrays"]["points"],
    )
    np.testing.assert_array_equal(
        np.fromfile(summary["depth_artifact"], dtype="<f4").reshape(1, 2), state["arrays"]["depth"]
    )
    np.testing.assert_array_equal(np.fromfile(summary["valid_mask_artifact"], dtype="u1"), [1, 0])
    calibration = json.loads(Path(summary["intrinsics_artifact"]).read_text(encoding="utf-8"))
    assert (
        calibration["normalized"] is True
        and calibration["height"] == 1
        and calibration["width"] == 2
    )
    np.testing.assert_array_equal(
        np.asarray(calibration["intrinsics"], dtype=np.float32), state["arrays"]["intrinsics"]
    )
    invalid = dict(summary, _geometry_arrays={**state["arrays"], "mask": np.asarray([[1, 2]])})
    with pytest.raises(RuntimeError, match="arrays"):
        task_reference._write_geometry_artifacts(invalid, tmp_path / "invalid.json")
    invalid = dict(summary, _geometry_arrays={**state["arrays"], "depth": np.asarray([[2, 0]])})
    with pytest.raises(RuntimeError, match="positive infinity"):
        task_reference._write_geometry_artifacts(invalid, tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.geometry.points.f32").exists()


def test_moge_runner_writes_geometry_only_after_measurement(tmp_path, monkeypatch):
    arguments, request, options, state = _moge_reference_fixture(tmp_path, monkeypatch)
    arguments.adapter, arguments.mode = "upstream-moge", "pytorch-eager"
    arguments.request_json = json.dumps(request)
    arguments.adapter_options_json = json.dumps(options)
    arguments.timing_contract_json = json.dumps(
        {
            "timing_scope": "task-pipeline-call-wall",
            "input_preparation_included": True,
            "asset_loading_included": False,
        }
    )
    arguments.padding = "longest"
    arguments.warmup, arguments.iterations = 1, 2
    arguments.case_name, arguments.output = "moge-geometry", tmp_path / "reference.json"
    monkeypatch.setattr(task_reference, "_environment", lambda: {})
    writer = task_reference._write_geometry_artifacts

    def write(summary, path):
        state["events"].append("write_geometry")
        writer(summary, path)

    monkeypatch.setattr(task_reference, "_write_geometry_artifacts", write)
    assert task_reference.run(arguments) == 0
    result = json.loads(arguments.output.read_text(encoding="utf-8"))
    assert result["model_load_included"] is False and result["measurement"]["iterations"] == 2
    assert state["events"][-1] == "write_geometry"
    assert state["events"].count("infer") == 3
    summary = result["output_summary"]
    assert summary["camera_axes"] == ["right", "down", "forward"]
    assert summary["intrinsics_coordinates"] == "normalized_uv"
    assert "_geometry_arrays" not in summary and Path(summary["depth_artifact"]).is_file()


def _environment(tmp_path: Path) -> tuple[Path, perf.Environment]:
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("bench", "worker", "hf.py", "task.py"):
        path = tools / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in (
        "libtrtmc_runtime.so",
        "libtrtmc_backend_trt.so",
        "libtrtmc_model_gpt2.so",
        "libtrtmc_model_lance.so",
    ):
        (runtime / name).write_bytes(b"")
    references = {}
    for name in perf.REFERENCE_FIELDS:
        path = tmp_path / name
        path.mkdir()
        references[name] = str(path)
    value = {
        "schema_version": perf.ENVIRONMENT_SCHEMA,
        "name": "test",
        "tools": {
            "trtmc_bench": str(tools / "bench"),
            "trtmc_worker": str(tools / "worker"),
            "hf_transformers_runner": str(tools / "hf.py"),
            "task_reference_runner": str(tools / "task.py"),
        },
        "references": references,
        "storage": {
            "results_root": str(tmp_path / "results"),
            "scratch_root": str(tmp_path / "scratch"),
            "bundle_cache": str(tmp_path / "bundles"),
            "bundle_roots": [],
            "runtime_root": str(runtime),
            "bundle_retention": "retain",
        },
        "execution": {"local_files_only": True, "timeout_seconds": 10},
    }
    path = tmp_path / "environment.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path, perf.load_environment(path)


def _fake_measurement_runner(
    environment,
    entry,
    *,
    candidate_samples=(),
    candidate_tokens=(),
    candidate_exit_codes=(),
    record_bundle=False,
):
    state = {"candidate_runs": 0, "commands": [], "environments": []}

    def run_command(arguments, *, stdout_path, stderr_path, env=None, **_kwargs):
        state["commands"].append(list(arguments))
        state["environments"].append(dict(env or {}))
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        output = Path(arguments[arguments.index("--output") + 1])
        if Path(arguments[0]) == environment.trtmc_bench:
            index = state["candidate_runs"]
            state["candidate_runs"] += 1
            exit_code = candidate_exit_codes[index] if index < len(candidate_exit_codes) else 0
            if exit_code:
                return {"argv": list(arguments), "exit_code": exit_code}
            samples = candidate_samples[index] if index < len(candidate_samples) else [10.0] * 10
            tokens = candidate_tokens[index] if index < len(candidate_tokens) else [1, 2]
            bundles = (
                [{"model": entry.model.name, "bundle": str(entry.case.bundle_path)}]
                if record_bundle
                else []
            )
            output.mkdir(parents=True)
            (output / "result.json").write_text(
                json.dumps(
                    {
                        "schema_version": "trtmc.benchmark-run/v2",
                        "status": "completed",
                        "preparation": {"bundles": bundles},
                        "cells": [
                            {
                                "status": "completed",
                                "metrics": {"latency_ms": {"p50": float(np.median(samples))}},
                                "samples_ms": samples,
                                "output_summary": {
                                    "token_ids": tokens,
                                    "output_tokens": len(tokens),
                                },
                                "timing_scope": "public_task_call_wall",
                                "asset_loading_included": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        else:
            output.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "precision": entry.reference_precision,
                        "metrics": {"latency_ms": {"p50": 10.1}},
                        "samples_ms": [10.1] * 10,
                        "output_summary": {"token_ids": [1, 2], "output_tokens": 2},
                        "measurement_policy": dict(entry.baseline_timing),
                    }
                ),
                encoding="utf-8",
            )
        return {
            "argv": list(arguments),
            "exit_code": 0,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

    return state, run_command


def test_environment_enforces_storage_root_and_per_entry_cache_policy(tmp_path: Path) -> None:
    environment_path, _ = _environment(tmp_path)
    value = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    storage_root = tmp_path / "managed"
    storage_root.mkdir()
    value["storage"]["storage_root"] = str(storage_root)
    value["execution"].update({"hf_cache_mode": "per_entry", "hf_cache_retention": "delete_always"})
    environment_path.write_text(yaml.safe_dump(value), encoding="utf-8")
    environment = perf.load_environment(environment_path)

    assert environment.storage_root == storage_root
    assert environment.hf_cache_mode == "per_entry"
    assert environment.hf_cache_retention == "delete_always"
    with pytest.raises(perf.PerfMatrixError, match="results_root must stay below storage_root"):
        perf.preflight((), environment, require_runtime=False)


def test_per_entry_hf_cache_is_private_and_follows_retention(tmp_path: Path, monkeypatch) -> None:
    environment_path, _ = _environment(tmp_path)
    value = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    value["execution"].update(
        {"hf_cache_mode": "per_entry", "hf_cache_retention": "delete_on_pass"}
    )
    environment_path.write_text(yaml.safe_dump(value), encoding="utf-8")
    environment = perf.load_environment(environment_path)
    monkeypatch.setenv("HF_HUB_CACHE", "/shared/hub")
    monkeypatch.setenv("HF_MODULES_CACHE", "/shared/modules")
    monkeypatch.setenv("TRANSFORMERS_CACHE", "/shared/transformers")
    work = environment.scratch_root / "entry" / "attempt-1"
    (work / "hf-cache").mkdir(parents=True)

    command_environment = perf._entry_command_environment(environment, work)
    assert command_environment["HF_HOME"] == str((work / "hf-cache").resolve())
    assert "HF_HUB_CACHE" not in command_environment
    assert "HF_MODULES_CACHE" not in command_environment
    assert "TRANSFORMERS_CACHE" not in command_environment
    assert perf._cleanup_entry_work(work, environment, passed=False)["status"] == "retained"
    assert perf._cleanup_entry_work(work, environment, passed=True)["status"] == "deleted"
    assert not work.exists()


def test_shared_hf_cache_cannot_be_deleted(tmp_path: Path) -> None:
    environment_path, _ = _environment(tmp_path)
    value = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    value["execution"].update({"hf_cache_mode": "shared", "hf_cache_retention": "delete_always"})
    environment_path.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(perf.PerfMatrixError, match="shared Hugging Face cache"):
        perf.load_environment(environment_path)


def test_checked_in_environments_have_no_dead_gpu_headroom_setting() -> None:
    root = REPO / "apps/benchmark/performance/environments"
    for path in root.glob("*.yaml"):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "minimum_gpu_free_fraction" not in value["execution"], path


def test_explicit_empty_model_selection_fails_closed(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text('{"families": []}\n', encoding="utf-8")

    with pytest.raises(perf.PerfMatrixError, match="matches no release entries"):
        perf.select_entries(
            [{"id": "a", "family": "alpha", "model": "model-a"}], model_selection=selection
        )


@pytest.mark.parametrize("entry_id", (".", ".."))
def test_entry_slug_cannot_escape_its_root(entry_id: str) -> None:
    assert perf._entry_slug(entry_id) == "entry"


def test_release_suite_expands_profiles_and_covers_ready_catalog() -> None:
    name, entries, excluded = perf.load_suite(SUITE)
    assert name == "release-family-performance"
    profile = next(entry for entry in entries if entry["id"] == "gpt2.generate@gpt2-125m")
    assert profile["workload"]["testcase"] == "gpt2-125m"
    vision_ids = {
        "timm_densenet.classify",
        "timm_efficientnet.classify",
        "timm_inception.classify",
        "timm_mnasnet.classify",
        "timm_mobilenetv2.classify",
        "timm_mobilenetv3.classify",
        "timm_repvgg.classify",
        "timm_resnet.classify",
        "timm_vgg.classify",
        "timm_vit.classify",
    }
    vision_entries = {entry["id"]: entry for entry in entries if entry["id"] in vision_ids}
    assert set(vision_entries) == vision_ids
    assert all(
        entry["baseline"]["adapter"] == "hf-transformers-vision"
        for entry in vision_entries.values()
    )
    perf._coverage(entries, excluded)


@pytest.mark.parametrize(
    (
        "family",
        "expected_scope",
        "input_preparation_included",
        "calls_after_load",
        "calls_after_invoke",
    ),
    [
        ("bert", "task-pipeline-call-wall", True, [], ["tokenize", "model"]),
        ("eagle_vlm", "task-model-call-wall", False, ["tokenize"], ["tokenize", "model"]),
        ("bert", "task-model-call-wall", False, ["tokenize"], ["tokenize", "model"]),
        ("eagle_vlm", "task-pipeline-call-wall", True, [], ["tokenize", "model"]),
        ("renamed_embedding", "task-model-call-wall", False, ["tokenize"], ["tokenize", "model"]),
        ("renamed_embedding", "task-pipeline-call-wall", True, [], ["tokenize", "model"]),
    ],
)
def test_embedding_reference_measures_the_declared_timing_contract(
    monkeypatch,
    family,
    expected_scope,
    input_preparation_included,
    calls_after_load,
    calls_after_invoke,
) -> None:
    calls: list[str] = []

    class FakeTensor:
        shape = (1, 2)
        dtype = "fp32"

        def to(self, *_args, **_kwargs):
            return self

        def unsqueeze(self, _dimension):
            return self

        def sum(self, **_kwargs):
            return self

        def clamp(self, **_kwargs):
            return self

        def numel(self):
            return 2

        def isfinite(self):
            return self

        def all(self):
            return self

        def item(self):
            return True

        def __mul__(self, _other):
            return self

        def __truediv__(self, _other):
            return self

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, *_args, **_kwargs):
            calls.append("tokenize")
            return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

    class FakeModel:
        config = SimpleNamespace(_commit_hash="model-revision")

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def eval(self):
            return self

        def to(self, *_args, **_kwargs):
            return self

        def __call__(self, **_kwargs):
            calls.append("model")
            return SimpleNamespace(last_hidden_state=FakeTensor())

    fake_torch = ModuleType("torch")
    fake_torch.device = lambda value: value
    fake_torch.float16 = "fp16"
    fake_torch.float32 = "fp32"
    fake_torch.bfloat16 = "bf16"
    fake_torch.inference_mode = nullcontext
    fake_torch.ones = lambda *_args, **_kwargs: FakeTensor()
    fake_torch.nn = SimpleNamespace(
        functional=SimpleNamespace(normalize=lambda value, **_kwargs: value)
    )
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModel = FakeModel
    fake_transformers.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    arguments = SimpleNamespace(
        family=family,
        model="sentence-transformers/all-MiniLM-L6-v2",
        precision="fp32",
        revision="model-revision",
        trust_remote_code=False,
        local_files_only=True,
        timing_contract_json=json.dumps({
            "timing_scope": expected_scope,
            "input_preparation_included": input_preparation_included,
            "asset_loading_included": False,
        }),
    )

    session = task_reference.LOADERS["hf-transformers-embedding"](
        arguments,
        {"prompt": "The quick brown fox"},
        {},
    )

    assert calls == calls_after_load
    assert session.timing_scope == expected_scope
    assert session.input_preparation_included is input_preparation_included
    assert session.asset_loading_included is False
    assert session.invoke()["embedding_vectors"] == 1
    assert calls == calls_after_invoke


def test_check_resolves_selected_entry_with_one_runtime_root(tmp_path: Path, capsys) -> None:
    environment_path, _ = _environment(tmp_path)
    assert (
        perf.main(
            [
                "check",
                str(SUITE),
                "--environment",
                str(environment_path),
                "--entry",
                "gpt2.generate",
            ]
        )
        == 0
    )
    assert "Ready: 1" in capsys.readouterr().out


def test_candidate_and_reference_commands_use_current_contract(tmp_path: Path) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["id"] == "gpt2.generate"]
    resolved = perf.resolve_entries(selected, environment)[0]
    candidate = perf.candidate_command(resolved, environment, tmp_path / "candidate", no_build=True)
    assert "--runtime-root" in candidate
    assert "--operation" in candidate
    assert "--no-build" in candidate
    reference = perf.baseline_command(resolved, environment, tmp_path / "reference.json")
    assert "--case-name" in reference
    assert "--task" in reference
    assert ("--revision" in reference) is bool(resolved.model.hf_revision)


def test_lerobot_reference_is_family_owned_and_has_a_closed_contract(tmp_path: Path) -> None:
    _, environment = _environment(tmp_path)
    source = Path(environment.references["lerobot_repo"])
    entrypoint = source / "lerobot/common/policies/act/modeling_act.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("", encoding="utf-8")
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["id"] == "lerobot_act.control"]
    resolved = perf.resolve_entries(selected, environment)[0]
    command = perf.baseline_command(resolved, environment, tmp_path / "reference.json")
    parsed = task_reference.build_parser().parse_args(command[2:])
    assert parsed.adapter == "pytorch-lerobot-act"
    assert json.loads(parsed.adapter_options_json) == {"source_root": str(source)}

    candidate = {
        "action_steps": 100,
        "action_dim": 14,
        "action_values": 1400,
        "within_training_bounds": True,
    }
    reference = {
        "action_steps": 100,
        "action_dim": 14,
        "action_values": 1400,
        "finite": True,
    }
    assert perf._output_contract(
        resolved,
        {"output_summary": candidate},
        {"output_summary": reference},
    ) == (True, "", None)


def test_comparison_preserves_output_gate_and_three_performance_states(
    tmp_path: Path,
) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["id"] == "gpt2.generate"]
    entry = perf.resolve_entries(selected, environment)[0]

    def value(candidate_ms: float, reference_ms: float, tokens: list[int]):
        candidate = {
            "metrics": {"latency_ms": {"p50": candidate_ms}},
            "output_summary": {"token_ids": tokens, "output_tokens": len(tokens)},
            "timing_scope": "public_task_call_wall",
            "asset_loading_included": False,
        }
        reference = {
            "status": "completed",
            "precision": entry.reference_precision,
            "metrics": {"latency_ms": {"p50": reference_ms}},
            "output_summary": {"token_ids": tokens, "output_tokens": len(tokens)},
            "measurement_policy": dict(entry.baseline_timing),
        }
        return candidate, reference

    candidate, reference = value(10.0, 12.0, [1, 2])
    assert perf.compare(entry, candidate, reference)[0] == "green"
    candidate, reference = value(10.0, 10.2, [1, 2])
    assert perf.compare(entry, candidate, reference)[0] == "yellow"
    candidate, reference = value(12.0, 10.0, [1, 2])
    assert perf.compare(entry, candidate, reference)[0] == "red"
    reference["output_summary"]["token_ids"] = [9]
    assert perf.compare(entry, candidate, reference)[0] == "contract-mismatch"


@pytest.mark.parametrize(
    ("samples", "status"),
    (
        ([100.0, 101.0, 99.0, 100.0, 100.0, 101.0, 100.0, 99.0, 100.0, 100.0], "stable"),
        ([3.7, 3.4, 3.0, 2.7, 2.3, 1.9, 1.6, 1.4, 1.2, 1.0], "unstable"),
        ([10.0, 11.0], "not_evaluated"),
    ),
)
def test_timing_stability_preserves_the_ten_sample_contract(samples, status) -> None:
    assert perf._timing_stability(samples)["status"] == status


@pytest.mark.parametrize(
    ("second_samples", "expected_status", "stability_status"),
    (([10.0] * 10, "yellow", "stable_after_retry"), (None, "white", "measurement_inconclusive")),
)
def test_unstable_measurement_is_retried_once(
    tmp_path: Path,
    monkeypatch,
    second_samples,
    expected_status,
    stability_status,
) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(entry for entry in entries if entry["id"] == "gpt2.generate")
    entry = perf.resolve_entries((spec,), environment)[0]
    falling = [3.7, 3.4, 3.0, 2.7, 2.3, 1.9, 1.6, 1.4, 1.2, 1.0]
    second = falling if second_samples is None else second_samples
    state, run_command = _fake_measurement_runner(
        environment,
        entry,
        candidate_samples=(falling, second),
    )
    monkeypatch.setattr(perf, "run_command", run_command)
    row = perf._execute_entry(
        entry,
        environment,
        tmp_path / "run",
        no_build=True,
        verbose=False,
        attempt=1,
    )

    assert len(state["commands"]) == 4
    assert row["status"] == expected_status
    assert row["measurement_stability"]["status"] == stability_status
    assert set(row["commands"]) == {
        "candidate",
        "reference",
        "candidate_measurement_2",
        "reference_measurement_2",
    }


def test_scratch_is_run_scoped_and_success_cleans_all_entry_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    _, environment = _environment(tmp_path)
    environment = replace(
        environment,
        hf_cache_mode="per_entry",
        hf_cache_retention="delete_on_pass",
    )
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(value for value in entries if value["id"] == "gpt2.generate")
    entry = perf.resolve_entries((spec,), environment)[0]
    run = tmp_path / "run-a"
    entry_work = environment.scratch_root / "run-a" / "gpt2.generate"
    (entry_work / "attempt-1" / "hf-cache").mkdir(parents=True)
    state, run_command = _fake_measurement_runner(environment, entry)
    monkeypatch.setattr(perf, "run_command", run_command)

    row = perf._execute_entry(
        entry,
        environment,
        run,
        no_build=True,
        verbose=False,
        attempt=2,
    )

    assert row["status"] == "yellow"
    assert not entry_work.exists()
    expected_cache = str((entry_work / "attempt-2" / "hf-cache").resolve())
    assert {value["HF_HOME"] for value in state["environments"]} == {expected_cache}


def test_existing_artifact_attempt_is_skipped_in_one_execution(tmp_path: Path, monkeypatch) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(value for value in entries if value["id"] == "gpt2.generate")
    entry = perf.resolve_entries((spec,), environment)[0]
    run = tmp_path / "run"
    (run / "artifacts" / "gpt2.generate" / "attempt-1").mkdir(parents=True)
    _, run_command = _fake_measurement_runner(environment, entry)
    monkeypatch.setattr(perf, "run_command", run_command)

    row = perf._execute_entry(
        entry,
        environment,
        run,
        no_build=True,
        verbose=False,
        attempt=1,
    )

    assert row["attempts"] == 2
    assert row["artifact_dir"] == "artifacts/gpt2.generate/attempt-2"


def test_failed_command_records_the_scanned_artifact_attempt(tmp_path: Path, monkeypatch) -> None:
    entry = SimpleNamespace(
        spec={"id": "first", "operation": "generate"},
        model=SimpleNamespace(name="model", family="family"),
        case=SimpleNamespace(testcase_name="case"),
    )
    run = tmp_path / "run"
    artifact_root = run / "artifacts" / "first"
    (artifact_root / "attempt-1").mkdir(parents=True)
    attempts = []

    def execute(_entry, _environment, _run, *, attempt, **_kwargs):
        attempts.append(attempt)
        (artifact_root / f"attempt-{attempt}").mkdir(exist_ok=True)
        raise perf.PerfMatrixError("candidate command failed")

    results = {
        "schema_version": perf.RESULT_SCHEMA,
        "status": "running",
        "selected_entry_ids": ["first"],
        "rows": [],
    }
    monkeypatch.setattr(perf, "_execute_entry", execute)
    monkeypatch.setattr(perf, "_write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(perf, "write_report", lambda *_args, **_kwargs: {})

    assert (
        perf._run_rows(
            run,
            results,
            (entry,),
            SimpleNamespace(),
            no_build=True,
            verbose=False,
        )
        == 1
    )
    assert results["rows"][0]["attempts"] == 2

    assert (
        perf._run_rows(
            run,
            results,
            (entry,),
            SimpleNamespace(),
            no_build=True,
            verbose=False,
        )
        == 1
    )
    assert attempts == [2, 3]
    assert results["rows"][0]["attempts"] == 3


def test_second_measurement_contract_mismatch_discards_first_stability(
    tmp_path: Path, monkeypatch
) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(value for value in entries if value["id"] == "gpt2.generate")
    entry = perf.resolve_entries((spec,), environment)[0]
    falling = [3.7, 3.4, 3.0, 2.7, 2.3, 1.9, 1.6, 1.4, 1.2, 1.0]
    _, run_command = _fake_measurement_runner(
        environment,
        entry,
        candidate_samples=(falling, [10.0] * 10),
        candidate_tokens=([1, 2], [9]),
    )
    monkeypatch.setattr(perf, "run_command", run_command)

    row = perf._execute_entry(
        entry,
        environment,
        tmp_path / "run",
        no_build=True,
        verbose=False,
        attempt=1,
    )

    assert row["status"] == "contract-mismatch"
    assert "measurement_stability" not in row


def test_failed_remeasurement_is_not_treated_as_a_pass(tmp_path: Path, monkeypatch) -> None:
    _, environment = _environment(tmp_path)
    environment = replace(environment, bundle_retention="delete_on_pass")
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(value for value in entries if value["id"] == "gpt2.generate")
    entry = perf.resolve_entries((spec,), environment)[0]
    entry.case.bundle_path.parent.mkdir(parents=True)
    entry.case.bundle_path.write_bytes(b"bundle")
    falling = [3.7, 3.4, 3.0, 2.7, 2.3, 1.9, 1.6, 1.4, 1.2, 1.0]
    _, run_command = _fake_measurement_runner(
        environment,
        entry,
        candidate_samples=(falling,),
        candidate_exit_codes=(0, 1),
        record_bundle=True,
    )
    monkeypatch.setattr(perf, "run_command", run_command)

    with pytest.raises(perf.PerfMatrixError, match="candidate command failed"):
        perf._execute_entry(
            entry,
            environment,
            tmp_path / "run",
            no_build=True,
            verbose=False,
            attempt=1,
        )

    assert entry.case.bundle_path.is_file()


def test_delete_always_cleans_declared_bundle_when_candidate_process_fails(
    tmp_path: Path, monkeypatch
) -> None:
    _, environment = _environment(tmp_path)
    environment = replace(environment, bundle_retention="delete_always")
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(value for value in entries if value["id"] == "gpt2.generate")
    entry = perf.resolve_entries((spec,), environment)[0]
    entry.case.bundle_path.parent.mkdir(parents=True)
    entry.case.bundle_path.write_bytes(b"bundle")
    _, run_command = _fake_measurement_runner(
        environment,
        entry,
        candidate_exit_codes=(1,),
    )
    monkeypatch.setattr(perf, "run_command", run_command)

    with pytest.raises(perf.PerfMatrixError, match="candidate command failed"):
        perf._execute_entry(
            entry,
            environment,
            tmp_path / "run",
            no_build=True,
            verbose=False,
            attempt=1,
        )

    assert not entry.case.bundle_path.exists()

    external_bundle = tmp_path / "external.bundle"
    external_bundle.write_bytes(b"external")
    external_entry = replace(
        entry,
        case=entry.case.with_values(bundle_path=external_bundle),
    )
    _, run_command = _fake_measurement_runner(
        environment,
        external_entry,
        candidate_exit_codes=(1,),
    )
    monkeypatch.setattr(perf, "run_command", run_command)
    with pytest.raises(perf.PerfMatrixError, match="candidate command failed"):
        perf._execute_entry(
            external_entry,
            environment,
            tmp_path / "external-run",
            no_build=True,
            verbose=False,
            attempt=1,
        )
    assert external_bundle.is_file()


def test_prepare_aggregates_public_builder_receipts(tmp_path: Path, monkeypatch) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["id"] == "gpt2.generate"]
    entry = perf.resolve_entries(selected, environment)[0]

    def run_command(arguments, *, stdout_path, stderr_path, **_kwargs):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(
            json.dumps(
                {
                    "bundles": [
                        {
                            "model": "distilgpt2",
                            "bundle": str(tmp_path / "distilgpt2.bundle"),
                            "status": "built",
                            "included_in_performance_metrics": False,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {
            "argv": list(arguments),
            "exit_code": 0,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

    monkeypatch.setattr(perf, "run_command", run_command)
    output = tmp_path / "preparation.json"
    assert perf.prepare_entries((entry,), environment, output, verbose=False) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == perf.PREPARATION_SCHEMA
    assert len(receipt["bundles"]) == 1


def test_report_uses_selected_ids_and_shows_pending_and_stability(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    results = {
        "schema_version": perf.RESULT_SCHEMA,
        "status": "completed",
        "suite": "test",
        "environment": "test",
        "selected_entry_ids": ["gpt2.generate", "pending.generate"],
        "rows": [
            {
                "id": "gpt2.generate",
                "model": "distilgpt2",
                "operation": "generate",
                "status": "yellow",
                "measurement_stability": {"status": "stable_after_retry"},
                "comparison": {
                    "candidate_p50_ms": 1.0,
                    "reference_p50_ms": 1.01,
                },
            }
        ],
    }
    report = perf.write_report(run, results)
    assert report["summary"]["comparable"] == 1
    assert report["summary"]["selected"] == 2
    assert report["summary"]["pending"] == 1
    assert (run / "report.json").is_file()
    html = (run / "report.html").read_text(encoding="utf-8")
    assert "pending: 1" in html
    assert "stable_after_retry" in html


def test_multi_entry_progress_publishes_only_completed_rows(tmp_path: Path, monkeypatch) -> None:
    entries = tuple(
        SimpleNamespace(
            spec={"id": entry_id, "operation": "generate"},
            model=SimpleNamespace(name=entry_id, family="gpt2"),
            case=SimpleNamespace(testcase_name=entry_id),
        )
        for entry_id in ("first", "second")
    )
    results = {
        "schema_version": perf.RESULT_SCHEMA,
        "status": "running",
        "selected_entry_ids": ["first", "second"],
        "rows": [],
    }
    snapshots = []

    def execute(entry, *_args, attempt, **_kwargs):
        return {"id": entry.spec["id"], "status": "green", "attempts": attempt}

    def write_json(path, value):
        if path.name == "results.json":
            snapshots.append([row["id"] for row in value["rows"]])

    monkeypatch.setattr(perf, "_execute_entry", execute)
    monkeypatch.setattr(perf, "_write_json", write_json)
    monkeypatch.setattr(perf, "write_report", lambda *_args, **_kwargs: {})

    assert (
        perf._run_rows(
            tmp_path / "run",
            results,
            entries,
            SimpleNamespace(),
            no_build=True,
            verbose=False,
        )
        == 0
    )
    assert snapshots[:2] == [["first"], ["first", "second"]]


def test_contract_mismatch_is_finished_but_keeps_run_non_green(tmp_path: Path, monkeypatch) -> None:
    entry = SimpleNamespace(spec={"id": "first"})
    results = {
        "schema_version": perf.RESULT_SCHEMA,
        "status": "running",
        "selected_entry_ids": ["first"],
        "rows": [{"id": "first", "status": "contract-mismatch", "attempts": 1}],
    }
    monkeypatch.setattr(
        perf,
        "_execute_entry",
        lambda *_args, **_kwargs: pytest.fail("finished contract mismatch was rerun"),
    )
    monkeypatch.setattr(perf, "_write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(perf, "write_report", lambda *_args, **_kwargs: {})

    assert (
        perf._run_rows(
            tmp_path / "run",
            results,
            (entry,),
            SimpleNamespace(),
            no_build=True,
            verbose=False,
        )
        == 1
    )
    assert results["status"] == "failed"


@pytest.mark.parametrize("stored_ids", (None, ["removed.entry"]))
def test_resume_fails_when_stored_selection_is_missing(tmp_path: Path, capsys, stored_ids) -> None:
    environment_path, _ = _environment(tmp_path)
    run = tmp_path / "resume-run"
    run.mkdir()
    results = {
        "schema_version": perf.RESULT_SCHEMA,
        "status": "failed",
        "suite_path": str(SUITE),
        "environment_path": str(environment_path),
        "rows": [],
    }
    if stored_ids is not None:
        results["selected_entry_ids"] = stored_ids
    (run / "results.json").write_text(json.dumps(results), encoding="utf-8")

    assert perf.main(["resume", str(run)]) == 2
    expected = (
        "matrix results has no selected entry IDs"
        if stored_ids is None
        else "selected entries are missing from the suite: removed.entry"
    )
    assert expected in capsys.readouterr().err


def test_run_executes_candidate_then_reference_and_publishes_report(
    tmp_path: Path, monkeypatch
) -> None:
    environment_path, environment = _environment(tmp_path)

    def run_command(arguments, *, stdout_path, stderr_path, **_kwargs):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        output = Path(arguments[arguments.index("--output") + 1])
        if Path(arguments[0]) == environment.trtmc_bench:
            output.mkdir(parents=True)
            (output / "result.json").write_text(
                json.dumps(
                    {
                        "schema_version": "trtmc.benchmark-run/v2",
                        "status": "completed",
                        "preparation": {"bundles": []},
                        "cells": [
                            {
                                "status": "completed",
                                "metrics": {"latency_ms": {"p50": 10.0}},
                                "samples_ms": [10.0] * 10,
                                "output_summary": {
                                    "token_ids": [1, 2],
                                    "output_tokens": 2,
                                },
                                "timing_scope": "public_task_call_wall",
                                "asset_loading_included": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        else:
            output.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "precision": "fp32",
                        "metrics": {"latency_ms": {"p50": 10.1}},
                        "samples_ms": [10.1] * 10,
                        "output_summary": {
                            "token_ids": [1, 2],
                            "output_tokens": 2,
                        },
                        "measurement_policy": {
                            "timing_scope": "public_operation_call_wall",
                            "input_preparation_included": True,
                            "asset_loading_included": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
        return {
            "argv": list(arguments),
            "exit_code": 0,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

    monkeypatch.setattr(perf, "run_command", run_command)
    assert (
        perf.main(
            [
                "run",
                str(SUITE),
                "--environment",
                str(environment_path),
                "--entry",
                "gpt2.generate",
            ]
        )
        == 0
    )
    runs = list((tmp_path / "results").iterdir())
    assert len(runs) == 1
    result = json.loads((runs[0] / "results.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["rows"][0]["status"] == "yellow"
    assert (runs[0] / "report.json").is_file()


def test_reference_runner_dependencies_are_baseline_owned() -> None:
    root = REPO / "apps/benchmark/performance/baselines"
    required = {
        "audio_reference.py",
        "elf_reference.py",
        "lance_reference.py",
        "reference_support.py",
        "sana_wm_reference.py",
    }
    assert all((root / name).is_file() for name in required)
    source = (root / "task_reference.py").read_text(encoding="utf-8")
    assert "from tools." not in source
    assert 'REPOSITORY / "tools/' not in source
    assert "tests/e2e" not in source
    assert "tensorrt_model_connect.families" not in source


def test_lance_command_requires_explicit_checkout_without_a_commit_gate(
    tmp_path: Path,
) -> None:
    _, environment = _environment(tmp_path)
    checkout = Path(environment.references["lance_repo"])
    (checkout / "inference_lance.py").write_text("", encoding="utf-8")
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["id"] == "lance.generate"]
    resolved = perf.resolve_entries(selected, environment)[0]
    command = perf.baseline_command(resolved, environment, tmp_path / "reference.json")
    parsed = task_reference.build_parser().parse_args(command[2:])
    options = json.loads(parsed.adapter_options_json)
    assert parsed.adapter == "upstream-lance"
    assert options == {
        "model_subdir": "Lance_3B",
        "reference_repo": str(checkout),
        "resolution": "image_768res",
        "vit_subdir": "Qwen2.5-VL-ViT",
    }

    direct = lance_reference.build_parser().parse_args(
        [
            "--reference-repo",
            str(checkout),
            "--model",
            "model",
            "--image",
            str(tmp_path / "image.png"),
            "--prompt",
            "describe",
            "--max-new-tokens",
            "4",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--output",
            str(tmp_path / "output.json"),
        ]
    )
    assert direct.reference_repo == checkout


def test_check_fails_fast_when_selected_reference_input_is_missing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    environment_path, _ = _environment(tmp_path)
    value = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    value["references"]["lance_repo"] = "${TRTMC_TEST_UNSET_LANCE_REPO}"
    environment_path.write_text(yaml.safe_dump(value), encoding="utf-8")
    monkeypatch.delenv("TRTMC_TEST_UNSET_LANCE_REPO", raising=False)
    assert (
        perf.main(
            [
                "check",
                str(SUITE),
                "--environment",
                str(environment_path),
                "--entry",
                "lance.generate",
            ]
        )
        == 2
    )
    assert "TRTMC_TEST_UNSET_LANCE_REPO" in capsys.readouterr().err


def test_timeseries_entries_use_current_forecast_request_schema(tmp_path: Path) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["operation"] == "solve"]
    assert len(selected) == 5
    resolved = perf.resolve_entries(selected, environment)
    for entry in resolved:
        assert entry.case.request["past_values"]
        assert not ({"branch_input", "field_input", "trunk_input"} & entry.case.request.keys())
    timesfm = next(entry for entry in resolved if entry.model.family == "timesfm")
    assert timesfm.case.request["frequency"] == 2
    source = (REPO / "apps/benchmark/performance/baselines/task_reference.py").read_text(
        encoding="utf-8"
    )
    assert '_numeric_values(request, "past_values")' in source
    assert '_numeric_values(request, "branch_input")' not in source
    assert '_numeric_values(request, "field_input")' not in source


def test_qwen3_omni_uses_the_text_generation_contract(tmp_path: Path) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["id"] == "qwen3_omni.generate"]
    resolved = perf.resolve_entries(selected, environment)[0]
    assert resolved.spec["operation"] == "generate"
    assert resolved.spec["baseline"]["output_contract"] == "exact-text"
    assert resolved.case.request["max_new_tokens"] == 16
    assert "talker_max_new_tokens" not in resolved.case.request
    command = perf.baseline_command(resolved, environment, tmp_path / "reference.json")
    assert command[command.index("--adapter") + 1] == "hf-qwen3-omni"
    assert command[command.index("--operation") + 1] == "generate"
    request = json.loads(command[command.index("--request-json") + 1])
    assert request["max_new_tokens"] == 16
    assert "talker_max_new_tokens" not in request
    source = (REPO / "apps/benchmark/performance/baselines/task_reference.py").read_text(
        encoding="utf-8"
    )
    assert '"enable_audio_output": False' in source
    assert "return_audio=False" in source


def test_sana_reference_reports_materialized_video_shape() -> None:
    video = np.stack(
        [
            np.zeros((24, 32, 3), dtype=np.uint8),
            np.ones((24, 32, 3), dtype=np.uint8),
        ]
    )
    summary = sana_wm_reference.media_summary(video)
    assert summary == {
        "media_type": "video",
        "media_count": 2,
        "num_frames": 2,
        "height": 24,
        "width": 32,
        "channels": 3,
    }
    assert perf._media_shape(summary) == (2, 24, 32, 3)
    source = (REPO / "apps/benchmark/performance/baselines/task_reference.py").read_text(
        encoding="utf-8"
    )
    assert 'request.get("action"' in source


def test_sana_world_request_preserves_official_camera_controls(tmp_path: Path) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["id"] == "sana_wm.generate_image"]
    request = perf.resolve_entries(selected, environment)[0].case.request
    assert request["translation_speed"] == 0.055
    assert request["rotation_speed_deg"] == 1.2
    assert request["fps"] == 16
    assert request["flow_shift"] == 9.8
    assert request["no_action_overlay"] is True


def test_sana_reference_calls_official_pipeline_with_exact_workload(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {"generate": []}
    reference_repo = tmp_path / "Sana"
    reference_repo.mkdir()
    model_dir = tmp_path / "model"
    (model_dir / "dit").mkdir(parents=True)
    (model_dir / "refiner/text_encoder").mkdir(parents=True)
    (model_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
    (model_dir / "dit/sana_wm_1600m_720p.safetensors").write_bytes(b"weights")
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("drive forward", encoding="utf-8")
    intrinsics = tmp_path / "intrinsics.npy"
    intrinsics.write_bytes(b"intrinsics")
    output = tmp_path / "result.json"

    class FakeImage:
        def convert(self, mode):
            captured["image_mode"] = mode
            return self

    pil = ModuleType("PIL")
    pil.Image = SimpleNamespace(open=lambda path: FakeImage())
    monkeypatch.setitem(sys.modules, "PIL", pil)

    synchronize_calls = []
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: synchronize_calls.append(True),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    pyrallis = ModuleType("pyrallis")

    def parse_config(**kwargs):
        captured["parse"] = kwargs
        return "config"

    pyrallis.parse = parse_config
    monkeypatch.setitem(sys.modules, "pyrallis", pyrallis)

    class RefinerSettings:
        def __init__(self, **kwargs):
            captured["refiner"] = kwargs

    class GenerationParams:
        def __init__(self, **kwargs):
            captured["generation"] = kwargs

    class Pipeline:
        def __init__(self, **kwargs):
            captured["pipeline"] = kwargs

        def generate(self, *args):
            captured["generate"].append(args)
            return {
                "video": np.zeros((321, 24, 32, 3), dtype=np.uint8),
                "c2w": "camera",
            }

    trajectory = np.zeros((321, 4, 4), dtype=np.float32)
    official = SimpleNamespace(
        InferenceConfig=object,
        RefinerSettings=RefinerSettings,
        GenerationParams=GenerationParams,
        SanaWMPipeline=Pipeline,
        action_string_to_c2w=lambda action, **kwargs: (
            captured.update(action=(action, kwargs)) or trajectory
        ),
        _snap_num_frames=lambda value, **kwargs: value,
        resize_and_center_crop=lambda value: ("cropped", (1, 1), (2, 2), (0, 0)),
        load_intrinsics=lambda path, frames: (
            captured.update(load_intrinsics=(path, frames)) or "raw-intrinsics"
        ),
        transform_intrinsics_for_crop=lambda value, *sizes: (
            captured.update(transform_intrinsics=(value, sizes)) or "intrinsics"
        ),
        apply_overlay=lambda *_: (_ for _ in ()).throw(
            AssertionError("no-action-overlay must skip overlay")
        ),
    )
    monkeypatch.setattr(sana_wm_reference, "_official_module", lambda path: official)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sana_wm_reference.py",
            "--reference-repo",
            str(reference_repo),
            "--image",
            str(image),
            "--model-dir",
            str(model_dir),
            "--prompt",
            str(prompt),
            "--action",
            "w-320",
            "--intrinsics",
            str(intrinsics),
            "--num_frames",
            "321",
            "--fps",
            "16",
            "--step",
            "60",
            "--cfg_scale",
            "5.0",
            "--flow_shift",
            "9.8",
            "--seed",
            "42",
            "--refiner_seed",
            "42",
            "--translation_speed",
            "0.055",
            "--rotation_speed_deg",
            "1.2",
            "--no_action_overlay",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--output",
            str(output),
        ],
    )

    assert sana_wm_reference.main() == 0
    assert captured["action"] == (
        "w-320",
        {"translation_speed": 0.055, "rotation_speed_deg": 1.2},
    )
    assert captured["refiner"] == {
        "root": model_dir / "refiner",
        "gemma_root": model_dir / "refiner/text_encoder",
        "seed": 42,
    }
    assert captured["generation"] == {
        "num_frames": 321,
        "fps": 16,
        "step": 60,
        "cfg_scale": 5.0,
        "flow_shift": 9.8,
        "seed": 42,
    }
    assert len(captured["generate"]) == 3
    assert len(synchronize_calls) == 5
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["output_summary"]["num_frames"] == 321


def test_sana_reference_requires_action_from_current_request(tmp_path: Path) -> None:
    checkout = tmp_path / "Sana"
    checkout.mkdir()
    arguments = SimpleNamespace(
        manifest=REPO / "families/sana_wm/tests/manifests/sana-wm-bidirectional.json"
    )
    with pytest.raises(ValueError, match="non-empty action"):
        task_reference._run_sana_wm(
            arguments,
            {"prompt": "drive", "image_path": "assets/demo_0.png"},
            {"reference_repo": str(checkout)},
        )


def test_sana_task_reference_uses_one_explicit_official_command(
    monkeypatch, tmp_path: Path
) -> None:
    checkout = tmp_path / "Sana"
    checkout.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "samples_ms": [1.0, 2.0],
                    "output_summary": {
                        "media_type": "video",
                        "media_count": 321,
                        "num_frames": 321,
                        "height": 704,
                        "width": 1280,
                        "channels": 3,
                    },
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(task_reference.subprocess, "run", run)
    arguments = SimpleNamespace(
        manifest=REPO / "families/sana_wm/tests/manifests/sana-wm-bidirectional.json",
        warmup=1,
        iterations=2,
    )
    result = task_reference._run_sana_wm(
        arguments,
        {
            "prompt": "drive forward",
            "image_path": "assets/demo_0.png",
            "action": "w-80,jw-40,w-40,lw-60,w-100",
            "translation_speed": 0.055,
            "rotation_speed_deg": 1.2,
            "num_frames": 321,
            "fps": 16,
            "num_steps": 60,
            "cfg_scale": 5.0,
            "flow_shift": 9.8,
            "seed": 42,
            "no_action_overlay": True,
        },
        {
            "reference_repo": str(checkout),
            "model_dir": str(model_dir),
            "intrinsics": "assets/demo_0_intrinsics.npy",
        },
    )

    command = captured["command"]
    assert command[command.index("--reference-repo") + 1] == str(checkout.resolve())
    assert command[command.index("--translation_speed") + 1] == "0.055"
    assert command[command.index("--rotation_speed_deg") + 1] == "1.2"
    assert command[command.index("--num_frames") + 1] == "321"
    assert command[command.index("--fps") + 1] == "16"
    assert command[command.index("--flow_shift") + 1] == "9.8"
    assert command[command.index("--refiner_seed") + 1] == "42"
    assert command[command.index("--warmup") + 1] == "1"
    assert command[command.index("--iterations") + 1] == "2"
    assert "--no_action_overlay" in command
    assert "env" not in captured["kwargs"]
    assert result[0] == [1.0, 2.0]
    assert result[1]["num_frames"] == 321


def test_output_contracts_are_closed_and_semantic(tmp_path: Path) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    selected = {
        entry["id"]: perf.resolve_entries([entry], environment)[0]
        for entry in entries
        if entry["id"]
        in {
            "canary.transcribe",
            "chronos_bolt.solve",
            "segformer.segment",
            "timm_vit.classify",
        }
    }
    assert perf._contract_name(selected["canary.transcribe"]) == "transcription-text"
    assert perf._contract_name(selected["chronos_bolt.solve"]) == "forecast-shape"
    assert perf._contract_name(selected["segformer.segment"]) == "segmentation-shape"
    assert perf._contract_name(selected["timm_vit.classify"]) == "classification-top-class"

    forecast = selected["chronos_bolt.solve"]
    candidate = {"output_summary": {"forecast_elements": 12, "shape": [1, 4, 3]}}
    reference = {"output_summary": {"element_count": 12, "shape": [1, 3, 4]}}
    assert perf._output_contract(forecast, candidate, reference)[0] is True
    reference["output_summary"]["element_count"] = 11
    assert perf._output_contract(forecast, candidate, reference)[0] is False

    bad_spec = {
        **forecast.spec,
        "baseline": {**forecast.spec["baseline"], "output_contract": "misspelled"},
    }
    bad = perf.ResolvedEntry(
        bad_spec,
        forecast.model,
        forecast.case,
        forecast.manifest,
        forecast.reference_precision,
        forecast.baseline_timing,
    )
    with pytest.raises(perf.PerfMatrixError, match="unsupported output contract"):
        perf._contract_name(bad)
