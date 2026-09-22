# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import sys
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from itertools import permutations
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from qualification_tests.benchmark_qualification.performance import matrix as perf
from qualification_tests.benchmark_qualification.performance.references import (
    hf_transformers,
    generic_reference,
)
from apps.benchmark.trtmc_benchmark.types import ModelDescriptor


REPO = Path(__file__).resolve().parents[4]
SUITE = REPO / "qualification_tests/benchmark_qualification/performance/config/release.yaml"


def test_shared_reference_runner_never_uses_family_identity_to_select_behavior() -> None:
    tree = ast.parse(Path(generic_reference.__file__).read_text(encoding="utf-8"))
    parents: dict[ast.AST, ast.AST] = {
        child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
    }
    family_reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "family"
        and isinstance(node.value, ast.Name)
        and node.value.id == "arguments"
    ]
    controls = [
        node.test
        for node in ast.walk(tree)
        if isinstance(node, (ast.If, ast.IfExp))
    ] + [node.subject for node in ast.walk(tree) if isinstance(node, ast.Match)]
    violations = [
        node.lineno
        for node in family_reads
        if any(node in ast.walk(control) for control in controls)
        or (isinstance(parents.get(node), ast.Call) and node in parents[node].args)
    ]
    assert violations == []


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


@pytest.mark.parametrize("selector,task,height,width,frames", [
    ("pixart-sigma-1024-l0", "text_to_image", 512, 512, 1),
    ("flux-schnell-l0", "text_to_image", 384, 384, 1),
    ("ltx-video-l0", "text_to_video", 256, 256, 9),
    ("wan21-t2v-1.3b-l0", "text_to_video", 384, 672, 5),
])
@pytest.mark.parametrize("controls,expected", [
    ({}, None),
    ({"height": 24, "width": 40, "num_frames": 3}, (24, 40, 3)),
    ({"height": 0, "width": 0, "num_frames": 0}, (None, None, None)),
    ({"config": {"height": 0, "width": 0, "num_frames": 0}}, (None, None, None)),
    ({"config": {"video_height": 24, "video_width": 40, "video_num_frames": 3}}, (24, 40, 3)),
    ({"config": {"height": 24, "video_height": 80, "width": 40, "video_width": 96,
                 "num_frames": 3, "video_num_frames": 7}}, (24, 40, 7)),
    ({"config": {"height": 0, "video_height": 24, "width": 0, "video_width": 40,
                 "num_frames": 3, "video_num_frames": 0}}, (None, None, None)),
])
@pytest.mark.parametrize("accepts_frames", [True, False])
def test_diffusers_build_dimensions_fill_only_absent_reference_arguments(
    monkeypatch, tmp_path: Path, selector: str, task: str, height: int, width: int,
    frames: int, controls: dict, expected: tuple | None, accepts_frames: bool,
) -> None:
    captured = {}

    class Pipeline:
        def to(self, device):
            assert device == "cuda"

        def __call__(self, prompt, height=None, width=None, num_frames=None):
            captured.update(height=height, width=width, num_frames=num_frames)
            return SimpleNamespace(images=[])

    class ImagePipeline(Pipeline):
        def __call__(self, prompt, height=None, width=None):
            captured.update(height=height, width=width)
            return SimpleNamespace(images=[])

    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    pil = ModuleType("PIL")
    pil.Image = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setattr(generic_reference, "_diffusion_pipeline", lambda *_: Pipeline() if accepts_frames else ImagePipeline())
    original = perf.ManifestCatalog(REPO / "families").resolve(selector)
    model = replace(original, testcases=({**original.testcases[0], **controls},))
    case = perf.resolve_case(model, tmp_path / "model.bundle", selected_task=task)
    candidate_request = json.loads(json.dumps(case.request))
    request = generic_reference.flatten_config(case.request)
    before = dict(request)
    arguments = SimpleNamespace(manifest=model.manifest_path, family=model.family, precision=model.precision)
    generic_reference._load_diffusers(arguments, request, {}).invoke()
    wanted_height, wanted_width, wanted_frames = expected if expected is not None else (height, width, frames)
    assert captured["height"] == wanted_height
    assert captured["width"] == wanted_width
    if accepts_frames:
        assert captured["num_frames"] == wanted_frames
    else:
        assert "num_frames" not in captured
    assert request == before
    assert case.request == candidate_request


def test_translation_languages_use_tokenizer_controls_and_preserve_absence() -> None:
    class Tokenizer:
        src_lang = "default"
        unk_token_id = 99

        def convert_tokens_to_ids(self, token):
            return {"eng_Latn": 10, "fra_Latn": 11}.get(token, 99)

        def convert_ids_to_tokens(self, token):
            return {10: "eng_Latn", 11: "fra_Latn"}[token]

    tokenizer = Tokenizer()
    assert hf_transformers._translation_controls(tokenizer, {}) == ({}, None)
    assert tokenizer.src_lang == "default"
    assert hf_transformers._translation_controls(
        tokenizer, {"source_language": "eng_Latn", "target_language": "fra_Latn"}
    ) == ({"forced_bos_token_id": 11}, None)
    assert tokenizer.src_lang == "eng_Latn"
    assert hf_transformers._translation_controls(
        tokenizer, {"source_language_token_id": 10, "forced_bos_token_id": 0}
    ) == ({"forced_bos_token_id": 0}, None)
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
        == ({}, None)
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
        generation_config = SimpleNamespace(num_beams=4)

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
    assert captured["num_beams"] == 1
    assert captured["repetition_penalty"] == 1.1
    assert output["token_ids"] == [8]


def test_reference_only_source_language_placement_reaches_hf_runner(
    tmp_path: Path, monkeypatch,
) -> None:
    _, environment = _environment(tmp_path)
    manifest = tmp_path / "synthetic" / "tests" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"task": "text_translation", "reference_precision": "fp16"}),
        encoding="utf-8",
    )
    model = ModelDescriptor(
        name="synthetic-translator",
        hf_id="example/synthetic-translator",
        hf_revision="a" * 40,
        bundle_name="synthetic-translator.bundle",
        family="synthetic",
        task="text_translation",
        precision="fp16",
        manifest_path=manifest,
        testcases=(
            {
                "name": "translate",
                "source_text": "Hello",
                "source_language": "eng_Latn",
                "target_language": "fra_Latn",
                "max_new_tokens": 8,
            },
        ),
        build_settings={"max_sequence_length": 32},
    )
    monkeypatch.setattr(perf.ManifestCatalog, "resolve", lambda *_args: model)
    spec = {
        "id": "synthetic.translate",
        "family": "synthetic",
        "operation": "translate",
        "model": model.name,
        "workload": {"testcase": "translate"},
        "measurement": {"warmup": 1, "iterations": 1},
        "baseline": {
            "runner": "hf-transformers",
            "task": "seq2seq-lm",
            "mode": "hf-eager",
            "timing_scope": "public_operation_call_wall",
            "input_preparation_included": True,
            "asset_loading_included": False,
            "source_language_placement": "replace-final-unk",
        },
    }
    entry = perf.resolve_entries([spec], environment)[0]

    command = perf.baseline_command(entry, environment, tmp_path / "reference.json")
    request = json.loads(command[command.index("--request-json") + 1])

    assert request["source_language_placement"] == "replace-final-unk"
    assert not any(
        "source_language_placement" in argument
        for argument in perf.candidate_command(
            entry, environment, tmp_path / "candidate", no_build=True
        )
    )


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
    assert generic_reference._selected_task(arguments) == "series_to_quantile_forecast"
    assert manifest.read_bytes() == original
    arguments.selected_task = None
    assert generic_reference._selected_task(arguments) == "series_to_point_forecast"
    for invalid in ("", " ", " series_to_point_forecast", 7):
        arguments.selected_task = invalid
        with pytest.raises(ValueError, match="selected_task"):
            generic_reference._selected_task(arguments)


@pytest.mark.parametrize("payload", [{"token_ids": []}, {"token_ids": [7], "prompt": "hello"}])
def test_text_only_reference_loaders_never_replace_token_input_with_empty_text(payload):
    with pytest.raises(ValueError, match="token_ids"):
        hf_transformers._batch_prompt(payload)
    arguments = SimpleNamespace(timing_contract_json=json.dumps({
        "timing_scope": "task-model-call-wall", "input_preparation_included": False,
        "asset_loading_included": False,
    }))
    with pytest.raises(ValueError, match="token_ids"):
        generic_reference._load_embedding(arguments, payload, {})


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
    point = generic_reference._forecast_summary(
        _SummaryTensor(np.zeros((1, 4, 3))), "series_to_point_forecast"
    )
    assert point["shape"] == [4, 3] and point["axes"] == ["horizon", "channel"]
    assert point["horizon_steps"] == [1, 2, 3, 4] and point["forecast_elements"] == 12
    quantile = generic_reference._forecast_summary(
        _SummaryTensor(np.zeros((1, 3, 4))), "series_to_quantile_forecast", [0.1, 0.5, 0.9]
    )
    assert quantile["shape"] == [3, 4, 1]
    assert quantile["axes"] == ["quantile", "horizon", "channel"]
    assert quantile["quantile_levels"] == [0.1, 0.5, 0.9]
    with pytest.raises(ValueError, match="one-series"):
        generic_reference._forecast_summary(
            _SummaryTensor(np.zeros((2, 4, 3))), "series_to_point_forecast"
        )
    with pytest.raises(ValueError, match="quantile levels"):
        generic_reference._forecast_summary(
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
    summary = generic_reference._regression_values_summary(_SummaryTensor([[1.5, -2.0]]))
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
            generic_reference._regression_values_summary(_SummaryTensor(np.zeros(shape)))
    with pytest.raises(ValueError, match="finite"):
        generic_reference._regression_values_summary(_SummaryTensor([[float("inf")]]))


def test_head_score_performance_contract_preserves_shape_and_representation():
    entry = SimpleNamespace(
        spec={"id": "scores", "operation": "head_scores", "baseline": {}},
        model=SimpleNamespace(task="text_to_head_scores"),
        case=SimpleNamespace(selected_task=None),
    )
    assert perf._contract_name(entry) == "head-scores-shape"
    value = {"shape": [1, 2, 2], "values": [-1.0, 2.0, 3.0, -4.0],
             "score_kind": "logit", "pooling": "none", "normalization": "none"}
    summary = {"output_summary": value}
    assert perf._output_contract(entry, summary, summary)[0]
    for changed in ({"shape": [1, 4]}, {"score_kind": "unbounded"},
                    {"pooling": "first"}, {"normalization": "l2"}):
        assert not perf._output_contract(entry, summary, {"output_summary": {**value, **changed}})[0]
    # This is a performance shape/representation contract, not an accuracy threshold.
    assert perf._output_contract(entry, summary, {"output_summary": {**value, "values": [4.0] * 4}})[0]


@pytest.mark.parametrize("changed", [
    {"shape": []}, {"shape": [0, 4]}, {"shape": [True, 4]}, {"shape": [1.0, 4]},
    {"shape": [2, 3]}, {"values": None}, {"values": [1, 2, 3]},
    {"values": [True, 2, 3, 4]}, {"values": [float("nan"), 2, 3, 4]},
    {"values": [float("inf"), 2, 3, 4]}, {"score_kind": "embedding"},
    {"score_kind": []}, {"score_kind": None},
    {"pooling": ""}, {"normalization": None},
])
def test_head_score_performance_contract_rejects_incomplete_outputs(changed):
    entry = SimpleNamespace(spec={"baseline": {"output_contract": "head-scores-shape"}})
    value = {"shape": [2, 2], "values": [1, 2, 3, 4], "score_kind": "logit",
             "pooling": "none", "normalization": "none", **changed}
    summary = {"output_summary": value}
    assert not perf._output_contract(entry, summary, summary)[0]


def test_regression_distribution_keeps_parameter_names_and_target_axis():
    values = (_SummaryTensor([[1.0, 2.0]]), _SummaryTensor([[0.5, 0.7]]))
    summary = generic_reference._regression_summary(values, "normal", ["loc", "scale"])
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
        generic_reference._regression_summary(
            (_SummaryTensor([[1, 2]]), _SummaryTensor([[1]])), "normal", ["loc", "scale"]
        )
    student = generic_reference._regression_summary(
        (_SummaryTensor([[4.0]]), _SummaryTensor([[1.0]]), _SummaryTensor([[0.5]])),
        "student_t", ["df", "loc", "scale"],
    )
    assert [parameter["name"] for parameter in student["parameters"]] == [
        "degrees_of_freedom", "location", "scale"
    ]
    with pytest.raises(ValueError, match="parameter names"):
        generic_reference._regression_summary(values, "normal", ["loc", "location"])


@pytest.mark.parametrize("distribution,parameters", [
    ("normal", [("location", [1.0, 2.0]), ("scale", [0.5, 0.7])]),
    ("student_t", [("degrees_of_freedom", [4.0, 5.0]),
                   ("location", [1.0, 2.0]), ("scale", [0.5, 0.7])]),
    ("negative_binomial", [("total_count", [2.0, 3.0]), ("logits", [-0.1, 0.2])]),
])
def test_regression_distribution_parameter_order_is_not_part_of_contract(distribution, parameters):
    entry = SimpleNamespace(spec={"baseline": {"output_contract": "regression-distribution"}})
    value = {"distribution": distribution, "target_count": 2, "axes": ["target"],
             "parameters": [{"name": name, "values": values} for name, values in parameters]}
    left = {"output_summary": value}
    for order in permutations(value["parameters"]):
        right = {"output_summary": {**value, "parameters": list(order)}}
        before = deepcopy((left, right))
        assert perf._output_contract(entry, left, right) == (True, "", None)
        assert perf._output_contract(entry, right, left) == (True, "", None)
        assert (left, right) == before


@pytest.mark.parametrize("changed", [
    {"distribution": "student_t"},
    {"distribution": "unknown"},
    {"target_count": 1},
    {"target_count": True},
    {"axes": ["parameter", "target"]},
    {"parameters": []},
    {"parameters": [{"name": "location", "values": [1.0, 2.0]}]},
    {"parameters": [{"name": "location", "values": [1.0, 2.0]},
                    {"name": "other", "values": [0.5, 0.7]}]},
    {"parameters": [{"name": "location", "values": [1.0, 2.0]},
                    {"name": "location", "values": [0.5, 0.7]}]},
    {"parameters": [{"name": "location", "values": [1.0]},
                    {"name": "scale", "values": [0.5, 0.7]}]},
    {"parameters": [{"name": "location", "values": [True, 2.0]},
                    {"name": "scale", "values": [0.5, 0.7]}]},
    {"parameters": [{"name": "location", "values": [float("nan"), 2.0]},
                    {"name": "scale", "values": [0.5, 0.7]}]},
    {"parameters": [{"name": "location", "values": [1.0, 2.0]},
                    {"name": "scale", "values": [0.5, float("inf")]}]},
])
def test_regression_distribution_order_fix_retains_mismatch_rejection(changed):
    entry = SimpleNamespace(spec={"baseline": {"output_contract": "regression-distribution"}})
    value = {"distribution": "normal", "target_count": 2, "axes": ["target"],
             "parameters": [{"name": "location", "values": [1.0, 2.0]},
                            {"name": "scale", "values": [0.5, 0.7]}]}
    valid = {"output_summary": value}
    invalid = {"output_summary": {**value, **changed}}
    assert not perf._output_contract(entry, valid, invalid)[0]
    assert not perf._output_contract(entry, invalid, valid)[0]


def test_timeseries_reference_preserves_observed_masks_and_unobserved_padding():
    request = {"past_values": [1, 2, 3], "observed_mask": [1, 0, 1]}
    observed = generic_reference._observed_values(request, 3)
    assert observed == [1.0, 0.0, 1.0]
    assert generic_reference._align(observed, 5, 0.0) == [0, 0, 1, 0, 1]
    assert generic_reference._observed_values({"past_values": [1, 2]}, 2) == [1, 1]
    assert generic_reference._observed_values({"observed_mask": []}, 2) == [1, 1]
    with pytest.raises(ValueError, match="match past_values"):
        generic_reference._observed_values({"observed_mask": [1]}, 2)


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
    _, suite_entries, _ = perf._load_suite_file(SUITE)
    reference_fields = {
        str(declaration["environment"])
        for entry in suite_entries
        for declaration in entry.get("baseline", {}).get("reference_inputs", {}).values()
    }
    references = {}
    for name in reference_fields:
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


def test_environment_preserves_reference_virtualenv_symlink(tmp_path: Path) -> None:
    environment_path, _ = _environment(tmp_path)
    value = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    virtualenv = tmp_path / "reference-venv"
    (virtualenv / "bin").mkdir(parents=True)
    python = virtualenv / "bin/python"
    python.symlink_to(Path(sys.executable))
    value["tools"]["reference_python"] = str(python)
    environment_path.write_text(yaml.safe_dump(value), encoding="utf-8")

    environment = perf.load_environment(environment_path)

    assert environment.reference_python == python
    assert environment.reference_python.is_symlink()


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
    root = REPO / "qualification_tests/benchmark_qualification/performance/config/environments"
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
    perf._coverage(entries, excluded)
    # Adapter identity belongs to the suite entry, not a shared family-name registry.
    _, builtin_entries, _ = perf._load_suite_file(SUITE)
    builtin_vision = {entry["id"]: entry for entry in builtin_entries if entry["id"] in vision_ids}
    assert set(builtin_vision) == vision_ids
    builtin_timm = [entry for entry in builtin_entries if entry["family"].startswith("timm_")]
    assert builtin_timm
    assert all(
        entry["baseline"]["adapter"] == "timm-classification"
        and entry["baseline"]["reference_backend"] == "timm"
        for entry in builtin_timm
    )


@pytest.mark.parametrize(
    (
        "family",
        "expected_scope",
        "input_preparation_included",
        "mode",
        "calls_after_load",
        "calls_after_invoke",
        "calls_after_summary",
        "compiled",
    ),
    [
        (
            "bert",
            "task-pipeline-call-wall",
            True,
            "hf-eager",
            [],
            ["tokenize", "model", "materialize"],
            ["tokenize", "model", "materialize", "validate"],
            False,
        ),
        (
            "bert",
            "task-pipeline-call-wall",
            True,
            "torch-compile",
            ["compile"],
            ["compile", "tokenize", "model", "materialize"],
            ["compile", "tokenize", "model", "materialize", "validate"],
            True,
        ),
        (
            "eagle_vlm",
            "task-model-call-wall",
            False,
            "hf-eager",
            ["tokenize"],
            ["tokenize", "model", "materialize"],
            ["tokenize", "model", "materialize", "validate"],
            False,
        ),
        (
            "bert",
            "task-model-call-wall",
            False,
            "hf-eager",
            ["tokenize"],
            ["tokenize", "model", "materialize"],
            ["tokenize", "model", "materialize", "validate"],
            False,
        ),
        (
            "eagle_vlm",
            "task-pipeline-call-wall",
            True,
            "hf-eager",
            [],
            ["tokenize", "model", "materialize"],
            ["tokenize", "model", "materialize", "validate"],
            False,
        ),
        (
            "renamed_embedding",
            "task-model-call-wall",
            False,
            "hf-eager",
            ["tokenize"],
            ["tokenize", "model", "materialize"],
            ["tokenize", "model", "materialize", "validate"],
            False,
        ),
        (
            "renamed_embedding",
            "task-pipeline-call-wall",
            True,
            "hf-eager",
            [],
            ["tokenize", "model", "materialize"],
            ["tokenize", "model", "materialize", "validate"],
            False,
        ),
    ],
)
def test_embedding_reference_measures_the_declared_timing_contract(
    monkeypatch,
    family,
    expected_scope,
    input_preparation_included,
    mode,
    calls_after_load,
    calls_after_invoke,
    calls_after_summary,
    compiled,
) -> None:
    calls: list[str] = []

    class FakeTensor:
        shape = (1, 2)
        dtype = "fp32"

        def to(self, *_args, **_kwargs):
            return self

        def detach(self):
            calls.append("materialize")
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
            calls.append("validate")
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
    compile_evidence = {"compiled_graph_count": 0}
    monkeypatch.setattr(
        generic_reference,
        "_compile_forward",
        lambda _model: calls.append("compile") or compile_evidence,
    )
    arguments = SimpleNamespace(
        family=family,
        model="sentence-transformers/all-MiniLM-L6-v2",
        mode=mode,
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

    session = generic_reference.LOADERS["hf-transformers-embedding"](
        arguments,
        {"prompt": "The quick brown fox"},
        {},
    )

    assert calls == calls_after_load
    assert session.timing_scope == expected_scope
    assert session.input_preparation_included is input_preparation_included
    assert session.asset_loading_included is False
    assert (session.compile_evidence is compile_evidence) is compiled
    vector = session.invoke()
    assert calls == calls_after_invoke
    assert session.summarize is not None
    assert session.summarize(vector)["embedding_vectors"] == 1
    assert calls == calls_after_summary


@pytest.mark.parametrize(("compiled", "expected_invocations"), [(False, 2), (True, 3)])
def test_zero_warmup_keeps_compilation_outside_timed_samples(
    monkeypatch, compiled, expected_invocations
) -> None:
    invocations = 0
    compile_evidence = {"compiled_graph_count": 0} if compiled else None

    def invoke():
        nonlocal invocations
        invocations += 1
        if compile_evidence is not None and invocations == 1:
            compile_evidence["compiled_graph_count"] = 1
        return {"value": invocations}

    monkeypatch.setattr(generic_reference, "_synchronize", lambda: None)
    samples, output = generic_reference._measure(
        generic_reference.Session(
            invoke=invoke,
            framework="test",
            compile_evidence=compile_evidence,
        ),
        warmup=0,
        iterations=2,
    )

    assert len(samples) == 2
    assert invocations == expected_invocations
    assert output == {"value": expected_invocations}


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


def test_reference_falls_back_when_compiled_process_fails(tmp_path: Path, monkeypatch) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(value for value in entries if value["id"] == "gpt2.generate")
    spec = {**spec, "baseline": {**spec["baseline"], "fallback": "hf-eager"}}
    entry = perf.resolve_entries((spec,), environment)[0]
    state, successful = _fake_measurement_runner(environment, entry)
    modes = []

    def run_command(arguments, **kwargs):
        if Path(arguments[0]) != environment.trtmc_bench:
            mode = arguments[arguments.index("--mode") + 1]
            modes.append(mode)
            if mode == "torch-compile":
                kwargs["stdout_path"].parent.mkdir(parents=True, exist_ok=True)
                kwargs["stdout_path"].write_text("", encoding="utf-8")
                kwargs["stderr_path"].write_text("compile failed", encoding="utf-8")
                return {"argv": list(arguments), "exit_code": 1}
        return successful(arguments, **kwargs)

    monkeypatch.setattr(perf, "run_command", run_command)
    row = perf.execute_entry(
        entry,
        environment,
        tmp_path / "run",
        no_build=True,
        verbose=False,
        attempt=1,
    )

    assert row["status"] in perf.TERMINAL_COMPARISONS
    assert modes == ["torch-compile", "hf-eager"]
    assert row["reference_attempts"] == [
        {
            "measurement_attempt": 1,
            "mode": "torch-compile",
            "fallback": False,
            "exit_code": 1,
            "fallback_reason": "reference command failed",
        },
        {"measurement_attempt": 1, "mode": "hf-eager", "fallback": True, "exit_code": 0},
    ]
    assert state["candidate_runs"] == 1


def test_compiled_output_mismatch_tries_the_configured_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(value for value in entries if value["id"] == "gpt2.generate")
    spec = {**spec, "baseline": {**spec["baseline"], "fallback": "hf-eager"}}
    entry = perf.resolve_entries((spec,), environment)[0]
    state, successful = _fake_measurement_runner(environment, entry, candidate_tokens=([9],))

    monkeypatch.setattr(perf, "run_command", successful)

    row = perf.execute_entry(
        entry,
        environment,
        tmp_path / "run",
        no_build=True,
        verbose=False,
        attempt=1,
    )

    assert row["status"] == "contract-mismatch"
    reference_commands = [
        command for command in state["commands"] if Path(command[0]) != environment.trtmc_bench
    ]
    assert [command[command.index("--mode") + 1] for command in reference_commands] == [
        "torch-compile",
        "hf-eager",
    ]
    assert row["reference_attempts"][0]["fallback_reason"] == "output contract mismatch"



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
    row = perf.execute_entry(
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

    row = perf.execute_entry(
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

    row = perf.execute_entry(
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
    monkeypatch.setattr(perf, "execute_entry", execute)
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

    row = perf.execute_entry(
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
        perf.execute_entry(
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
        perf.execute_entry(
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
        perf.execute_entry(
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

    monkeypatch.setattr(perf, "execute_entry", execute)
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
        "execute_entry",
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
    root = REPO / "qualification_tests/benchmark_qualification/performance/references"
    required = {
        "audio_reference.py",
    }
    assert all((root / name).is_file() for name in required)
    source = (root / "generic_reference.py").read_text(encoding="utf-8")
    assert "from tools." not in source
    assert 'REPOSITORY / "tools/' not in source
    assert "tests/e2e" not in source
    assert "tensorrt_model_connect.families" not in source



def test_check_fails_fast_when_selected_reference_input_is_missing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    environment_path, _ = _environment(tmp_path)
    value = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    value["references"]["lance_repo"] = "${TRTMC_TEST_UNSET_LANCE_REPO}"
    environment_path.write_text(yaml.safe_dump(value), encoding="utf-8")
    monkeypatch.delenv("TRTMC_TEST_UNSET_LANCE_REPO", raising=False)
    # Keep effective coverage; only this test's Lance route must stay builtin.
    _, entries, excluded = perf.load_suite(SUITE)
    _, builtin_entries, _ = perf._load_suite_file(SUITE)
    builtin = next(entry for entry in builtin_entries if entry["id"] == "lance.generate")
    builtin_suite = tmp_path / "builtin-release.yaml"
    builtin_suite.write_text(yaml.safe_dump({
        "schema_version": perf.SUITE_SCHEMA, "name": "builtin-lance-fixture",
        "entries": [builtin if entry["id"] == builtin["id"] else entry for entry in entries],
        "excluded_profiles": [
            {"model": model, "reason": "existing effective exclusion"} for model in sorted(excluded)
        ],
    }))
    assert (
        perf.main(
            [
                "check",
                str(builtin_suite),
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
    source = (REPO / "qualification_tests/benchmark_qualification/performance/references/generic_reference.py").read_text(
        encoding="utf-8"
    )
    assert '_numeric_values(request, "past_values")' in source
    assert '_numeric_values(request, "branch_input")' not in source
    assert '_numeric_values(request, "field_input")' not in source



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


def test_detection_output_contract_matches_by_class_and_iou() -> None:
    entry = SimpleNamespace(
        spec={
            "baseline": {
                "output_contract": "detection-parity",
                "min_box_iou": 0.9,
                "max_score_abs_error": 0.05,
            }
        }
    )
    candidate = {
        "output_summary": {
            "boxes": [10.0, 10.0, 50.0, 50.0],
            "scores": [0.91],
            "class_ids": [7],
        }
    }
    reference = {
        "output_summary": {
            "boxes": [[10.5, 10.0, 50.0, 50.0]],
            "scores": [0.9],
            "class_ids": [7],
        }
    }

    matched, reason, evidence = perf._output_contract(entry, candidate, reference)

    assert matched is True
    assert reason == ""
    assert evidence["minimum_box_iou"] >= 0.9
    reference["output_summary"]["class_ids"] = [8]
    assert perf._output_contract(entry, candidate, reference)[0] is False


def test_prompted_mask_contract_matches_binary_semantics() -> None:
    entry = SimpleNamespace(
        spec={"baseline": {"output_contract": "prompted-mask-parity", "min_mask_iou": 0.7}}
    )
    candidate = {
        "output_summary": {
            "num_masks": 1,
            "height": 2,
            "width": 2,
            "mask_kind": "logits",
            "masks": [2.0, -1.0, -1.0, 3.0],
        }
    }
    reference = {
        "output_summary": {
            "num_masks": 1,
            "height": 2,
            "width": 2,
            "mask_kind": "binary",
            "masks": [1, 0, 0, 1],
        }
    }

    matched, reason, evidence = perf._output_contract(entry, candidate, reference)

    assert matched is True
    assert reason == ""
    assert evidence == {"masks": 1, "minimum_mask_iou": 1.0, "required_mask_iou": 0.7}
    reference["output_summary"]["masks"] = [0, 1, 1, 0]
    assert perf._output_contract(entry, candidate, reference)[0] is False


def test_instance_mask_contract_checks_masks_boxes_and_scores() -> None:
    entry = SimpleNamespace(
        spec={
            "baseline": {
                "output_contract": "instance-mask-parity",
                "min_mask_iou": 0.7,
                "min_box_iou": 0.9,
                "max_score_abs_error": 0.05,
            }
        }
    )
    value = {
        "num_masks": 1,
        "height": 2,
        "width": 2,
        "mask_kind": "binary",
        "masks": [1, 0, 0, 1],
        "iou_scores": [0.9],
        "boxes": [[0.0, 0.0, 2.0, 2.0]],
        "box_coordinates": "original_image_pixels_xyxy",
    }
    candidate = {"output_summary": value}
    reference = {"output_summary": dict(value)}

    matched, reason, evidence = perf._output_contract(entry, candidate, reference)

    assert matched is True
    assert reason == ""
    assert evidence["minimum_mask_iou"] == evidence["minimum_box_iou"] == 1.0
    reference["output_summary"]["iou_scores"] = [0.7]
    assert perf._output_contract(entry, candidate, reference)[0] is False


def test_vision_language_text_contract_allows_small_normalized_differences() -> None:
    entry = SimpleNamespace(
        spec={
            "baseline": {
                "output_contract": "vision-language-text",
                "max_normalized_edit_distance": 0.15,
            }
        }
    )
    candidate = {"output_summary": {"text": "Red."}}
    reference = {"output_summary": {"text": "red"}}

    matched, _, evidence = perf._output_contract(entry, candidate, reference)

    assert matched is True
    assert evidence["normalized_edit_distance"] <= 0.15
    candidate["output_summary"]["text"] = "blue"
    assert perf._output_contract(entry, candidate, reference)[0] is False
