# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path

import pytest

pytest.importorskip("aiperf")

from aiperf.accuracy.models import BenchmarkProblem  # noqa: E402

from trtmc_aiperf_plugins.benchmarks import TEMPLATE_MARGIN, select  # noqa: E402
from trtmc_aiperf_qual import absolute  # noqa: E402
from trtmc_aiperf_qual.config import ConfigError  # noqa: E402
from trtmc_aiperf_qual.models import _absolute  # noqa: E402
from trtmc_aiperf_qual.report import counted  # noqa: E402


def problem(task, prompt="q", size=5):
    return BenchmarkProblem(prompt=prompt, ground_truth=" A", task=task, metadata={"generation_size": size})


def test_mmlu_asks_for_the_letter_on_both_routes():
    """The answer-format instruction follows lighteval's on the completion prompt and in the first chat message;
    the answer budget fits a short answer statement."""
    from trtmc_aiperf_plugins.benchmarks import MMLU_ANSWER_INSTRUCTION, MMLU_GENERATION_SIZE, instructed

    head = "The following are multiple choice questions (with answers) about college biology.\n\n"
    question = "Question: Which is a cell?\nA. a\nB. b\nC. c\nD. d\nAnswer:"
    original = BenchmarkProblem(prompt=head + question, ground_truth=" B", task="college_biology",
                                metadata={"generation_size": 5, "stop_sequence": ["\n"]},
                                raw_messages=[{"role": "user", "content": head + question}])
    changed = instructed(original)
    expected = head.replace("biology.", "biology." + MMLU_ANSWER_INSTRUCTION) + question
    assert changed.prompt == expected and changed.raw_messages[0]["content"] == expected
    assert changed.metadata == {"generation_size": MMLU_GENERATION_SIZE, "stop_sequence": ["\n"]}
    assert original.prompt == head + question  # the input problem is left as it was
    with pytest.raises(ValueError):
        instructed(problem("college_biology", prompt=question))


def test_selection_keeps_the_first_problems_per_task_in_dataset_order():
    problems = [problem(task) for task in ("a", "a", "a", "b", "b", "c")]
    chosen = select(problems, {"TRTMC_ACCURACY_PER_TASK": "2"})
    assert [p.task for p in chosen] == ["a", "a", "b", "b", "c"]
    assert len(select(problems, {"TRTMC_ACCURACY_LIMIT": "4"})) == 4


def test_selection_caps_generation_and_drops_problems_that_do_not_fit_for_every_side():
    problems = [problem("a", "x" * 10, size=2048), problem("a", "x" * 900, size=2048)]
    chosen = select(problems, {"TRTMC_ACCURACY_MAX_NEW_TOKENS": "64", "TRTMC_ACCURACY_TOKEN_LIMIT": "512"},
                    count_tokens=len)
    assert len(chosen) == 1 and chosen[0].metadata["generation_size"] == 64
    assert 10 + TEMPLATE_MARGIN + 64 <= 512 < 900
    assert problems[0].metadata["generation_size"] == 2048  # the loader's problems are not mutated


def side(answers, exit_code=0):
    return {"records": {"greedy": {index: {"passed": ok, "unparsed": False, "actual": str(ok)}
                                   for index, ok in answers.items()}}, "exit": {"greedy": exit_code}}


ITEM = {"suite": "mmlu-0shot", "plugin": "trtmc_mmlu", "endpoint": "completions", "gate": {"margin": 1.0}}


def test_equal_scores_pass_and_report_both_accuracies_and_agreement():
    problems = [{"task": "t", "gold": " A"}] * 200
    answers = {index: index % 4 != 0 for index in range(200)}
    entry = absolute.judge({**ITEM, "gate": {"margin": 2.0}}, problems, side(answers), side(answers))
    assert entry["status"] == "pass" and entry["samples"] == 200
    assert entry["metrics"]["trtmc_score"] == entry["metrics"]["native_score"] == 75.0
    assert entry["metrics"]["answer_agreement"] == 1.0 and entry["metrics"]["test"]["outcome"] == "pass"
    # 200 problems cannot show a 1-point margin even without disagreement: inconclusive, not pass.
    assert absolute.judge(ITEM, problems, side(answers), side(answers))["status"] == "inconclusive"


def test_a_regression_beyond_the_margin_fails_and_a_missing_answer_is_an_error():
    problems = [{"task": "t", "gold": " A"}] * 200
    native = {index: True for index in range(200)}
    trtmc = {index: index >= 8 for index in range(200)}  # 8 answers lost: 4 points, all one-sided
    entry = absolute.judge(ITEM, problems, side(trtmc), side(native))
    assert entry["status"] == "fail" and entry["metrics"]["regression_points"] == 4.0
    assert len(entry["reasons"]) == 1 and entry["failures"][0]["sample_id"] == "t/0"
    within = absolute.judge({**ITEM, "gate": {"margin": 5.0}}, problems, side(trtmc), side(native))
    assert within["status"] == "inconclusive"  # below the margin, but not shown to be
    assert "TRTMC 96.00 vs native 100.00" in counted(entry)
    incomplete = absolute.judge(ITEM, problems, side({index: True for index in range(199)}), side(native))
    assert incomplete["status"] == "error" and incomplete["counts"]["missing_trtmc"] == 1


def test_rejudge_reapplies_a_gate_to_the_recorded_counts():
    entry = {"counts": {"native_only": 3, "trtmc_only": 0}, "samples": 1000, "expected_samples": 1000,
             "gate": {"margin": 1.0}, "metrics": {"native_score": 50.0}}
    entry["metrics"]["test"] = absolute.binary_test(entry)
    assert absolute.status(entry)[0] == "pass"
    tight = {**entry, "gate": {"margin": 0.2}}
    tight["metrics"] = {**entry["metrics"], "test": absolute.binary_test(tight)}
    assert absolute.status(tight)[0] == "inconclusive"


def test_sampled_models_are_judged_on_per_problem_seed_means():
    problems = [{"task": "t", "gold": " A"}] * 10
    candidate = {"records": {f"seed{s}": {i: {"passed": i < 6} for i in range(10)} for s in (1, 2)}, "exit": {}}
    native = {"records": {f"seed{s}": {i: {"passed": i < 7} for i in range(10)} for s in (1, 2)}, "exit": {}}
    entry = absolute.judge({**ITEM, "gate": {"margin": 15.0}}, problems, candidate, native)
    assert entry["metrics"]["regression_points"] == pytest.approx(10.0)
    assert len(entry["metrics"]["per_problem_regression"]) == 10 and entry["status"] == "inconclusive"
    assert absolute.judge({**ITEM, "gate": {"margin": 30.0}}, problems, candidate, native)["status"] == "pass"


DEFINITIONS = {"mmlu": {"plugin": "trtmc_mmlu", "suite": "mmlu-0shot", "gate": {"margin": 1.0},
                        "quantized_gate": {"margin": 2.0}, "sampled_gate": {"margin": 3.0}},
               "lambada": {"plugin": "trtmc_lambada", "suite": "lambada", "gate": {"margin": 0.5}}}


def test_benchmarks_take_the_route_and_gate_of_the_catalog_request():
    chat, = _absolute(["mmlu"], DEFINITIONS, {"use_chat_template": True}, quantized=False)
    assert chat["endpoint"] == "chat" and chat["gate"] == {"margin": 1.0} and "seeds" not in chat
    quantized, = _absolute(["mmlu"], DEFINITIONS, {}, quantized=True)
    assert quantized["endpoint"] == "completions" and quantized["gate"] == {"margin": 2.0}
    sampled, limited = _absolute(["mmlu", {"name": "lambada", "limit": 400}], DEFINITIONS,
                                 {"temperature": 0.7, "top_k": 50}, quantized=False)
    assert sampled["seeds"] == [1, 2, 3] and sampled["gate"] == {"margin": 3.0}
    assert limited["limit"] == 400 and limited["gate"] == {"margin": 0.5}
    greedy, = _absolute(["mmlu"], DEFINITIONS, {"temperature": 1.0, "top_k": 1}, quantized=False)
    assert "seeds" not in greedy
    with pytest.raises(ConfigError):
        _absolute(["hellaswag"], DEFINITIONS, {}, quantized=False)


def test_workload_perf_compares_problems_answered_with_the_same_length():
    raw = [{"metadata": {"session_num": index},
            "responses": [{"text": '{"usage": {"prompt_tokens": 500, "completion_tokens": %d}, '
                                   '"trtmc_timing": {"model_call_ms": %f}}' % (5, ms)}]}
           for index, ms in enumerate((100.0, 110.0, 120.0))]
    trtmc = absolute.timings(raw)
    native = {0: {"model_call_ms": 50.0, "completion_tokens": 5}, 1: {"model_call_ms": 60.0, "completion_tokens": 5},
              2: {"model_call_ms": 70.0, "completion_tokens": 4}}
    perf = absolute.workload_perf(trtmc, native)
    assert perf["pairs"] == 2 and perf["trtmc_p50_ms"] == 105.0 and perf["native_p50_ms"] == 55.0
    assert perf["light"] == "red" and perf["prompt_tokens_p50"] == 500
    assert absolute.workload_perf({}, native) == {"pairs": 0}


def test_lambada_grader_reads_the_first_word_of_the_continuation():
    import asyncio

    from trtmc_aiperf_plugins.benchmarks import FirstWordGrader

    grader = FirstWordGrader(run=None)
    grade = lambda text, gold: asyncio.run(grader.grade(text, gold))  # noqa: E731
    assert grade(" signs. And then", "signs").correct
    assert grade('signs," he said', "signs").correct
    assert not grade(" sign", "signs").correct and not grade(" Signs", "signs").correct
    assert grade("", "signs").unparsed


def test_a_benchmark_may_fix_its_route():
    plain, = _absolute(["lambada"], {"lambada": {"plugin": "trtmc_lambada", "suite": "lambada", "endpoint": "completions",
                                                 "gate": {"margin": 0.5}}},
                       {"use_chat_template": True}, quantized=False)
    assert plain["endpoint"] == "completions"


def test_choice_and_contains_grade_like_mmstar_and_ocrbench():
    from trtmc_aiperf_qual import gold_metrics

    assert gold_metrics.choice("B", {"text": "B"})[0] and gold_metrics.choice("B", {"text": "(B) a cat"})[0]
    assert gold_metrics.choice("B", {"text": "Answer: B."})[0] and not gold_metrics.choice("B", {"text": "A"})[0]
    assert not gold_metrics.choice("B", {"text": "Because"})[0]  # a word starting with B is no answer
    assert gold_metrics.contains(["CENTRE"], {"text": "The text reads centre."})[0]
    formula = "Handwritten Mathematical Expression Recognition"
    assert gold_metrics.contains(["x ^ { 2 }"], {"text": "x^{2}"}, formula)[0]
    assert not gold_metrics.contains(["x ^ { 2 }"], {"text": "x^{2}"})[0]


def test_corpus_metrics_and_their_paired_bootstrap():
    from trtmc_aiperf_qual import gold_metrics

    problems = [{"gold": "the cat sat"}, {"gold": "on the mat"}]
    units = gold_metrics.wer_units(problems, {0: {"text": "The cat sat."}, 1: {"text": "on a mat"}})
    assert units == [(0.0, 3.0), (1.0, 3.0)] and gold_metrics.wer(units) == pytest.approx(100 / 6)
    assert gold_metrics.spearman([(0.1, 1.0), (0.5, 2.0), (0.9, 3.0)]) == pytest.approx(100.0)
    pairs = [{"gold": score} for score in (1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)]
    same = {i: {"values": [1.0, 0.1 * (i // 2)] if i % 2 else [1.0, 0.0]} for i in range(8)}
    result = gold_metrics.compare_corpus("sts_spearman", pairs, same, same, {"margin": 0.5})
    assert result["regression_points"] == 0 and result["outcome"] == "pass" and result["units"] == 4


def test_corpus_entries_gate_on_the_margin_relative_to_the_native_score():
    from trtmc_aiperf_qual import noninferiority

    item = {"suite": "librispeech-test-clean", "metric": "wer", "gate": {"margin": 0.2, "relative_margin": 0.03}}
    problems = [{"gold": "a b c d e f g h i j", "task": "t"}] * 20
    native = {"observations": {"greedy": {i: {"text": "a b c d e f g h i x"} for i in range(20)}}, "exit": {},
              "timings": {"greedy": {}}}
    worse = {"observations": {"greedy": {i: {"text": "a b c d e f g h x x" if i < 10 else "a b c d e f g h i x"}
                                         for i in range(20)}}, "exit": {}, "timings": {"greedy": {}}}
    entry = absolute.judge(item, problems, worse, native)
    assert entry["metrics"]["native_score"] == 10.0 and entry["metrics"]["test"]["regression_points"] == 5.0
    assert entry["status"] == "fail" and entry["metrics"]["test"]["margin_points"] == pytest.approx(0.3)
    assert absolute.judge(item, problems, native, native)["status"] == "pass"
    missing = {**native, "observations": {"greedy": {i: {"text": "a"} for i in range(19)}}}
    assert absolute.judge(item, problems, missing, native)["status"] == "error"
    # 3% of a native WER of 10 is 0.3 points, more than the 0.2-point margin
    assert noninferiority.margin(item["gate"], 10.0) == pytest.approx(0.3)


def test_the_bootstrap_resamples_whole_clusters():
    from trtmc_aiperf_qual import noninferiority

    drawn = []

    def statistic(indices):
        drawn.append(sorted(indices))
        return 0.0, 0.0

    noninferiority.bootstrap(statistic, ["s1", "s1", "s2"], {"margin": 1.0}, True, 20)
    # every resample holds whole speakers: s1's two units together, never one alone
    assert all(draw.count(0) == draw.count(1) for draw in drawn[1:])


def test_rerank_retrieval_and_code_metrics():
    from trtmc_aiperf_qual import gold_metrics

    assert gold_metrics.rerank_units([{"gold": [2]}], {0: {"scores": [0.1, 0.2, 0.9]}}) == [1.0]
    assert gold_metrics.rerank_units([{"gold": [0]}], {0: {"scores": [0.1, 0.2, 0.9]}})[0] == pytest.approx(0.5)
    problems = [{"task": "query", "gold": [0]}, {"task": "document"}, {"task": "document"},
                {"task": "query", "gold": [0]}, {"task": "document"}, {"task": "document"}]
    vectors = {0: [1, 0], 1: [0, 1], 2: [1, 0.1], 3: [0, 1], 4: [0.1, 1], 5: [1, 0]}
    units = gold_metrics.retrieval_units(problems, {i: {"values": v} for i, v in vectors.items()})
    assert units[1] == 1.0 and units[0] == pytest.approx(1 / math.log2(3))
    import os
    import shutil

    if os.geteuid() != 0 or not shutil.which("setpriv"):
        return  # the code sandbox switches to nobody: root only (it fails closed elsewhere)
    gold = {"prompt": "def add(a, b):\n", "test": "def check(f):\n    assert f(1, 2) == 3\n", "entry_point": "add"}
    assert not gold_metrics.code_pass(gold, {"text": "    import sys; sys.exit(0)\n"})[0]  # an early exit(0) fails
    assert gold_metrics.code_pass(gold, {"text": "    return a + b\n\ndef unrelated():\n    raise SystemExit(1)\n"})[0]
    assert not gold_metrics.code_pass(gold, {"text": "    return a - b\n"})[0]
    assert not gold_metrics.code_pass(gold, {"text": "    while True:\n        pass\n"})[0]  # the time limit
    # human-eval's reliability guard: destructive calls are disabled inside the program
    assert not gold_metrics.code_pass(gold, {"text": "    import os; os.system('true'); return a + b\n"})[0]


def test_top1_is_the_argmax_class_index():
    from trtmc_aiperf_qual import gold_metrics

    assert gold_metrics.top1(2, {"scores": [0.1, 0.3, 0.9]}) == (True, "2")
    assert gold_metrics.top1(1, {"scores": [0.1, 0.3, 0.9]})[0] is False
    assert gold_metrics.top1(1, {}) == (False, "")
    assert gold_metrics.top1(0, {"top_class": 0, "top_score": 9.9})[0]  # classify reports only its top class


def test_coco_map_scores_detections_in_the_model_label_space():
    pytest.importorskip("pycocotools")
    from trtmc_aiperf_qual import gold_metrics

    problems = [{"sample_id": "a", "gold": {"bbox": [[10, 10, 50, 50]], "category": [2]}}]
    exact = {0: {"boxes": [[10, 10, 50, 50]], "scores": [0.9], "class_ids": [2]}}
    by_id = {0: {"boxes": [[10, 10, 50, 50]], "scores": [0.9], "class_ids": [3]}}  # car: index 2, id 3
    assert gold_metrics.coco_map(gold_metrics.coco_units(problems, exact, {})) == pytest.approx(100.0)
    assert gold_metrics.coco_map(gold_metrics.coco_units(problems, by_id, {})) == 0.0
    assert gold_metrics.coco_map(gold_metrics.coco_units(problems, by_id, {"label_space": "coco-category-id"})) == \
        pytest.approx(100.0)
    result = gold_metrics.compare_corpus("coco_map", problems, exact, exact, {"margin": 1.0}, {}, resamples=10)
    assert result["regression_points"] == 0 and result["outcome"] == "pass"


def test_forecasts_are_scored_on_the_point_or_median_quantile_forecast():
    from trtmc_aiperf_qual import gold_metrics

    assert gold_metrics.point_forecast({"values": [1.0, 2.0]}, 2) == [1.0, 2.0]
    quantiles = {"values": [0, 0, 1, 2, 9, 9], "shape": [1, 3, 2]}  # median row: [1, 2]
    assert gold_metrics.point_forecast(quantiles, 2) == [1.0, 2.0]
    gold = {"values": [1.0, 4.0], "mean": [0.0], "std": [1.0]}
    units = gold_metrics.forecast_units([{"gold": gold}], {0: {"values": [1.0, 2.0]}})
    assert units == [(4.0, 2.0)] and gold_metrics.mse(units) == 2.0
    # values are scaled by the training statistics; a missing forecast predicts the training mean
    scaled = {"values": [3.0], "mean": [1.0], "std": [2.0]}
    assert gold_metrics.forecast_units([{"gold": scaled}], {}) == [(1.0, 1.0)]
    # ... but the judge never scores one: an absent, malformed, or wrong-length forecast is a missing answer
    problems = [{"task": "etth1", "gold": gold}]
    native = {"observations": {"greedy": {0: {"values": [1.0, 2.0]}}}, "exit": {"greedy": 0}, "timings": {"greedy": {}}}
    masks = [{}, {"masks": [1], "height": 2, "width": 2}, {"masks": [1, 0, 0, float("nan")], "height": 2, "width": 2},
             {"masks": [], "height": 0, "width": 2}]
    assert [gold_metrics.ANSWERS["mask_iou"]({}, item) for item in masks] == [False] * 4
    assert gold_metrics.ANSWERS["mask_iou"]({}, {"masks": [], "height": 2, "width": 2})  # nothing found: an answer
    assert gold_metrics.ANSWERS["mask_iou"]({}, {"masks": [1, 0, 0, 1] * 2, "height": 2, "width": 2})
    assert not gold_metrics.ANSWERS["miou"]({}, {"mask": [1, 2, 3], "height": 2, "width": 2})
    assert gold_metrics.ANSWERS["miou"]({}, {"mask": [1, 2, 3, 4], "height": 2, "width": 2})
    for broken in ({}, {"shape": [1, 2]}, {"values": [1.0]}, {"values": [1.0, float("nan")]}):
        candidate = {**native, "observations": {"greedy": {0: broken}}}
        entry = absolute.judge_corpus({"suite": "etth1-mse", "metric": "forecast_mse", "gate": {"margin": 1.0}},
                                      problems, candidate, native)
        assert entry["status"] == "error" and "usable output" in entry["reasons"][0]


def test_translations_are_scored_with_corpus_chrf():
    pytest.importorskip("sacrebleu")
    from trtmc_aiperf_qual import gold_metrics

    problems = [{"gold": "Der Hund schläft."}, {"gold": "Die Katze spielt."}]
    perfect = {0: {"text": "Der Hund schläft."}, 1: {"text": "Die Katze spielt."}}
    wrong = {0: {"text": "Ein Vogel singt."}, 1: {"text": "Die Katze spielt."}}
    assert gold_metrics.chrf(gold_metrics.translation_units(problems, perfect)) == pytest.approx(100.0)
    result = gold_metrics.compare_corpus("chrf", problems, wrong, perfect, {"margin": 1.0}, resamples=50)
    assert result["regression_points"] > 20 and result["higher_is_better"]


def test_sentence_grader_ignores_spacing_around_punctuation():
    import asyncio

    from trtmc_aiperf_plugins.benchmarks import SentenceExactGrader

    grader = SentenceExactGrader(run=None)
    assert asyncio.run(grader.grade("An English film, television actor.", "An English film , television actor .")).correct
    assert not asyncio.run(grader.grade("An English stage actor.", "An English film , television actor .")).correct


def test_grounding_boxes_are_scored_on_the_normalized_grid():
    from trtmc_aiperf_qual import gold_metrics

    gold = {"value": [100.0, 50.0, 200.0, 100.0], "image_size": [1000, 500]}  # 0-1000 grid: 100,100,300,300
    assert gold_metrics.box_iou50(gold, {"text": "<ref>dog</ref><box><100><100><300><300></box>"})[0]
    assert not gold_metrics.box_iou50(gold, {"text": "<ref>dog</ref><box><400><400><600><600></box>"})[0]
    assert gold_metrics.box_iou50(gold, {"text": "no box"}) == (False, "")


def test_ade20k_miou_compares_predicted_classes_with_the_annotation():
    import base64
    import io

    import numpy as np
    from PIL import Image

    from trtmc_aiperf_qual import gold_metrics

    annotation = np.array([[1, 1], [2, 0]], dtype=np.uint8)  # classes 1 and 2; 0 is ignored
    buffer = io.BytesIO()
    Image.fromarray(annotation).save(buffer, format="PNG")
    problems = [{"gold": {"png_b64": base64.b64encode(buffer.getvalue()).decode()}}]
    exact = {0: {"mask": [0, 0, 1, 5], "height": 2, "width": 2}}  # class index = annotation - 1
    assert gold_metrics.miou(gold_metrics.miou_units(problems, exact)) == pytest.approx(100.0)
    wrong = {0: {"mask": [1, 1, 1, 1], "height": 2, "width": 2}}  # all class 2: IoU 0 for 1, 1/3 for 2
    assert gold_metrics.miou(gold_metrics.miou_units(problems, wrong)) == pytest.approx(100 / 6)
    assert gold_metrics.miou(gold_metrics.miou_units(problems, {})) == 0.0


def test_prompted_masks_are_scored_against_the_object_polygon():
    from trtmc_aiperf_qual import gold_metrics
    from trtmc_aiperf_qual.suites import _interior_point, polygon_mask

    polygon, size = [0, 0, 4, 0, 4, 4, 0, 4], [8, 4]  # the left half of an 8x4 image
    mask = polygon_mask(polygon, size)
    assert mask.shape == (4, 8) and mask[:, :4].all() and not mask[:, 6:].any()
    x, y = _interior_point(polygon, size)
    assert 0 < x < 0.6 and 0 < y < 1
    gold = {"polygon": polygon, "image_size": size}
    left = [1.0 if column < 5 else -1.0 for _ in range(4) for column in range(8)]
    right = [-1.0 if column < 5 else 1.0 for _ in range(4) for column in range(8)]
    best_second = {"masks": right + left, "iou_scores": [0.1, 0.9], "height": 4, "width": 8}
    units = gold_metrics.mask_iou_units([{"gold": gold}], {0: best_second})
    assert units[0] == pytest.approx(polygon_mask(polygon, size).sum() / (5 * 4))
    assert gold_metrics.mask_iou_units([{"gold": gold}], {}) == [0.0]


def test_geneval_runs_only_for_the_image_families():
    from trtmc_aiperf_qual.runner import applies, expected_suites

    check = {"check": "geneval", "only_families": ["flux"]}
    assert applies(check, {"family": "flux"}) and not applies(check, {"family": "wan_t2v"})
    assert "geneval" in expected_suites({"family": "flux", "supplementary": [check]})
    assert "geneval" not in expected_suites({"family": "wan_t2v", "supplementary": [check]})


def test_vbench_object_dimensions_become_geneval_style_requirements():
    from trtmc_aiperf_qual.suites import vbench_object_records

    rows = [{"prompt_en": "a red bicycle", "dimension": ["color"], "auxiliary_info": {"color": {"color": "red"}}},
            {"prompt_en": "a bicycle on the left of a car, front view", "dimension": ["spatial_relationship"],
             "auxiliary_info": {"spatial_relationship": {"spatial_relationship": {
                 "object_a": "bicycle", "object_b": "car", "relationship": "on the left of"}}}},
            {"prompt_en": "a bird and a cat", "dimension": ["multiple_objects"],
             "auxiliary_info": {"multiple_objects": {"object": "bird and cat"}}},
            {"prompt_en": "a beautiful sunset", "dimension": ["aesthetic_quality"]}]
    color, spatial, both = vbench_object_records(rows)
    assert color["label"]["include"] == [{"class": "bicycle", "count": 1, "color": "red"}]
    assert spatial["label"]["include"][1] == {"class": "bicycle", "count": 1, "position": ["left of", 0]}
    assert [item["class"] for item in both["label"]["include"]] == ["bird", "cat"] and both["task"] == "multiple_objects"


def test_precomputed_scores_compare_as_a_corpus_mean():
    item = {"suite": "edit-similarity", "metric": "precomputed_mean", "gate": {"margin": 3.0}}
    problems = [{"task": "edit", "gold": "make it red"}] * 4
    side = lambda values: {"observations": {"greedy": {i: {"value": v} for i, v in enumerate(values)}},  # noqa: E731
                           "exit": {}, "timings": {"greedy": {}}}
    entry = absolute.judge(item, problems, side([80, 82, 84, 86]), side([81, 83, 85, 87]))
    assert entry["metrics"]["test"]["regression_points"] == 1.0 and entry["status"] == "pass"
    tight = absolute.judge({**item, "gate": {"margin": 1.0}}, problems, side([80, 82, 84, 86]), side([81, 83, 85, 87]))
    assert tight["status"] == "inconclusive"  # exactly at the margin


def test_failed_requests_are_missing_answers_not_wrong_ones(tmp_path):
    from types import SimpleNamespace

    from unittest.mock import patch

    from trtmc_aiperf_qual.config import Environment

    raw = [{"metadata": {"session_num": 0}, "status": 200, "responses": []},
           {"metadata": {"session_num": 1}, "status": 422,
            "error": {"message": "TensorRT enqueue failed"}, "responses": []}]
    graded = [{"session_num": 0, "passed": True}, {"session_num": 1, "passed": False, "actual": ""}]
    run = SimpleNamespace(raw_records=lambda: raw, accuracy_records=lambda: graded, exit_code=1)
    item = {**ITEM, "suite": "mmlu-0shot"}
    model = {"candidate": {"max_sequence_length": 4096, "checkpoint": "m"}, "reference": {}}
    with patch.object(absolute, "run_aiperf", return_value=run):
        side_result = absolute.run_side(Environment({"hf_datasets_cache": str(tmp_path)}), {"url": "u"}, model, item,
                                        [{}, {}], tmp_path)
    assert list(side_result["records"]["greedy"]) == [0] and "enqueue failed" in side_result["failed"]["greedy"]
    native = {"records": {"greedy": {0: {"passed": True}, 1: {"passed": True}}}, "exit": {}}
    entry = absolute.judge(item, [{"task": "t", "gold": "A"}] * 2, side_result, native)
    assert entry["status"] == "error" and "enqueue failed" in entry["reasons"][0]


def test_the_native_side_tries_the_next_precision(tmp_path):
    from contextlib import contextmanager
    from unittest.mock import patch

    from trtmc_aiperf_qual.config import Environment

    started = []

    @contextmanager
    def serving_replicas(environment, model, backend, out, *, count, **kwargs):
        started.append((backend, kwargs["precision"], count))
        if kwargs["precision"] == "fp16":
            raise RuntimeError("fp16 overflow")
        yield {"url": "http://unused", "replicas": count}

    model = {"operation": "transcribe", "absolute": [{"suite": "s"}],
             "reference": {"backend": "reference", "perf_precision": "fp16", "precision": "fp32"}}
    with patch.object(absolute, "serving_replicas", serving_replicas), \
            patch.object(absolute, "run_side", return_value={"ok": 1}):
        native = absolute.run_native(Environment({"native_replicas": 4}), model, "python", {"s": []}, tmp_path)
    assert native["precision"] == "fp32" and native["runs"] == {"s": {"ok": 1}} and "overflow" in native["fallback_from"]
    assert started == [("reference", "fp16", 4), ("reference", "fp32", 4)]
    started.clear()
    with patch.object(absolute, "serving_replicas", serving_replicas), \
            patch.object(absolute, "run_side", return_value={"ok": 1}):
        absolute.run_native(Environment({"native_replicas": 4, "smoke": True}), model, "python", {"s": []}, tmp_path)
    assert started[-1] == ("reference", "fp32", 1)  # smoke mode: one copy


def test_native_copies_fit_the_free_gpu_memory():
    from trtmc_aiperf_qual.services import replicas_that_fit

    gib = 1024
    # 76 GiB used by other tenants of a 250 GiB GPU; one copy loads 20 GiB (30 GiB at its peak).
    assert replicas_that_fit((76 * gib, 250 * gib), (96 * gib, 250 * gib), 4) == 4
    # A 84 GiB copy leaves no room for a second one at its peak.
    assert replicas_that_fit((76 * gib, 250 * gib), (160 * gib, 250 * gib), 4) == 1
    assert replicas_that_fit((76 * gib, 250 * gib), (116 * gib, 250 * gib), 4) == 2
    assert replicas_that_fit(None, None, 4) == 1 and replicas_that_fit((0, 1), (0, 1), 1) == 1


def test_aiperf_spreads_requests_over_every_copy_one_at_a_time():
    one = absolute._targets({"url": "http://h:8900"}, "/v1/tasks/detect")
    assert one == ["--url", "http://h:8900/v1/tasks/detect", "--concurrency", "1"]
    two = absolute._targets({"url": "http://h:8900", "urls": ["http://h:8900", "http://h:8911"]})
    assert two == ["--url", "http://h:8900", "--url", "http://h:8911", "--concurrency", "2"]


def test_workload_timings_of_concurrent_native_copies_are_informational_only():
    item = {"suite": "s", "plugin": "p", "endpoint": "chat", "gate": {"margin": 1.0}}
    problems = [{"task": "t", "gold": "A"}, {"task": "t", "gold": "B"}]
    side = {"records": {"greedy": {0: {"passed": True, "unparsed": False}, 1: {"passed": False, "unparsed": False}}},
            "exit": {"greedy": 0},
            "timings": {"greedy": {0: {"model_call_ms": 10.0, "completion_tokens": 1},
                                   1: {"model_call_ms": 10.0, "completion_tokens": 1}}}}
    model = {"absolute": [item], "candidate": {}}
    native = {"backend": "reference", "precision": "fp16", "runs": {"s": side}, "replicas": 4}
    [entry] = absolute.entries(model, {"s": problems}, {"s": side}, native, None)
    assert entry["native"]["replicas"] == 4 and entry["workload_perf"]["light"] == "white"
    assert "not comparable" in entry["workload_perf"]["note"] and entry["status"] == "inconclusive"


def test_an_adapter_checkpoint_without_a_tokenizer_uses_the_base_models(monkeypatch):
    monkeypatch.setattr(absolute, "_has_tokenizer", lambda name, revision, trust: name != "org/adapter")
    adapter = {"candidate": {"checkpoint": "org/adapter", "revision": "a1"},
               "reference": {"model": "org/base", "revision": "b1"}}
    assert absolute.tokenizer_source(adapter) == ("org/base", "b1")
    full = {"candidate": {"checkpoint": "org/full", "revision": "f1"}, "reference": {"model": "org/base"}}
    assert absolute.tokenizer_source(full) == ("org/full", "f1")
    assert absolute.tokenizer_source({"candidate": {"checkpoint": "org/adapter"}, "reference": {}}) == ("org/adapter", None)


def test_an_empty_answer_is_a_wrong_answer_not_a_failed_request():
    empty = {"status": 200, "error": {"type": "InvalidInferenceResultError", "message": "no content"}}
    assert not absolute.unanswered(empty)  # AIPerf grades it (passed False)
    assert absolute.unanswered({"status": 422, "error": {"type": "HTTPError"}})
    assert absolute.unanswered({"status": 200, "error": {"type": "ClientConnectionError"}})
    assert not absolute.unanswered({"status": 200})
    assert "rejected" in absolute.failed_reason([empty, {"status": 422, "error": {"message": "rejected"}}], 1)


def test_a_native_score_below_the_suitability_floor_is_not_comparable():
    def entry(native, gate):
        value = {"samples": 10, "expected_samples": 10, "gate": gate, "counts": {},
                 "metrics": {"trtmc_score": native, "native_score": native}}
        value["metrics"]["test"] = absolute.binary_test(value)
        return value

    status, reasons = absolute.status(entry(0.53, {"margin": 1.0, "min_native": 30.0}))  # gpt-oss on MMLU
    assert status == "not-comparable" and "does not fit" in reasons[0]
    assert absolute.status(entry(44.0, {"margin": 1.0, "min_native": 30.0}))[0] == "inconclusive"  # 10 problems
    assert absolute.status(entry(0.0, {"margin": 1.0}))[0] == "inconclusive"  # no floor declared


def test_unmerged_raw_records_still_show_failed_requests(tmp_path):
    import json

    from trtmc_aiperf_qual.aiperf_runner import AiperfRun

    (tmp_path / "raw_records").mkdir()
    record = {"metadata": {"benchmark_phase": "profiling", "session_num": 0}, "status": 400,
              "error": {"message": "multi-message chat requires a chat-template renderer"}}
    (tmp_path / "raw_records" / "raw_records_processor_a.jsonl").write_text(json.dumps(record) + "\n")
    records = AiperfRun(tmp_path, 1, []).raw_records()  # AIPerf merged no export: every request failed
    assert len(records) == 1 and absolute.unanswered(records[0])




def test_missing_parity_evidence_is_an_error_not_a_failure(tmp_path):
    from trtmc_aiperf_qual import gold_metrics

    side = {"height": 2, "width": 2, "disparity_artifact": str(tmp_path / "missing.f32")}
    assert gold_metrics.disparity_parity(side, side, {"max_mean_epe": 0.1})[0] is None
    assert gold_metrics.vector_parity({}, {"values": [1.0]}, {})[0] is None
    assert gold_metrics.vector_parity({"values": [1.0, 2.0]}, {"values": [1.0]}, {})[0] is False  # a wrong output
    problems = [{"sample_id": "0"}]
    observations = {"observations": {"greedy": {0: side}}, "exit": {"greedy": 0}, "timings": {"greedy": {}}}
    entry = absolute.judge_parity({"suite": "stereo", "metric": "disparity_parity", "gate": {"max_mean_epe": 0.1}},
                                  problems, observations, observations)
    assert entry["status"] == "error" and "readable evidence" in entry["reasons"][0]


def test_smoke_covers_one_problem_of_each_code_benchmark(monkeypatch):
    import datasets

    from trtmc_aiperf_qual import suites
    from trtmc_aiperf_qual.config import Environment

    rows = {"openai/openai_humaneval": [{"task_id": f"HumanEval/{i}", "prompt": "def f():\n", "test": "", "entry_point": "f"}
                                        for i in range(3)],
            "google-research-datasets/mbpp": [{"task_id": i, "text": "t", "test_list": ["assert True"]} for i in range(3)]}
    monkeypatch.setattr(datasets, "load_dataset", lambda name, *args, **kwargs: rows[name])
    smoke = suites._code_records({}, Environment({"smoke": True}))
    assert [record["task"] for record in smoke] == ["humaneval", "mbpp"]
    assert len(suites._code_records({}, Environment({}))) == 6


def test_raw_records_keep_unicode_line_separators_inside_json_strings(tmp_path):
    import json

    from trtmc_aiperf_qual.aiperf_runner import RAW_EXPORT, AiperfRun

    record = {"metadata": {"benchmark_phase": "profiling", "session_num": 0}, "text": "a b\x85c"}
    (tmp_path / RAW_EXPORT).write_text(json.dumps(record, ensure_ascii=False) + "\n")  # as AIPerf writes it
    assert [item["text"] for item in AiperfRun(tmp_path, 0, []).raw_records()] == ["a b\x85c"]



def test_trtmc_answers_come_from_copies_and_l1_times_one_server(tmp_path):
    """With candidate_replicas, the Acc answers come from copies of the TRTMC server (each one request at a time)
    and L1 then times a single server started after the copies stopped; smoke mode keeps one server."""
    from contextlib import contextmanager
    from unittest.mock import patch

    from trtmc_aiperf_qual import runner
    from trtmc_aiperf_qual.config import Environment

    events = []

    @contextmanager
    def copies(environment, model, backend, out, *, count, **options):
        events.append(("copies", backend, count))
        yield {"url": "u", "urls": ["u"] * count, "replicas": count}
        events.append(("copies stopped",))

    @contextmanager
    def single(environment, model, backend, out, **options):
        events.append(("single", backend))
        yield {"url": "u", "info": {}}

    def answers(environment, service, model, out, **runs):
        events.append(("answers", service.get("replicas", 1)))
        return [{"suite": "s"}]

    model, accuracy = {"absolute": [{"suite": "s"}]}, []
    with patch.object(runner, "serving_replicas", copies), patch.object(runner, "serving", single), \
            patch.object(absolute, "candidate_entries", answers):
        runner._candidate(Environment({"candidate_replicas": 4}), model, {"suite": {}}, [], {}, accuracy, [], tmp_path,
                          absolute_runs={"plans": {}})
        assert events == [("copies", "trtmc", 4), ("answers", 4), ("copies stopped",), ("single", "trtmc")]
        events.clear()
        runner._candidate(Environment({"candidate_replicas": 4, "smoke": True}), model, None, [], {}, accuracy, [],
                          tmp_path, absolute_runs={"plans": {}})
        assert events == [("single", "trtmc"), ("answers", 1)]
    assert len(accuracy) == 2


def test_concurrent_trtmc_copies_make_the_workload_times_incomparable():
    from unittest.mock import patch

    judged = {"suite": "s", "status": "pass", "workload_perf": {"pairs": 3, "light": "green"}}
    native = {"backend": "reference", "precision": "fp16", "replicas": 1, "runs": {"s": {}}}
    with patch.object(absolute, "judge", side_effect=lambda *args: {**judged}):
        alone = absolute.entries({"absolute": [{"suite": "s"}]}, {"s": []}, {"s": {}}, native, None)[0]
        copies = absolute.entries({"absolute": [{"suite": "s"}]}, {"s": []}, {"s": {}}, native, None,
                                  candidate_replicas=4)[0]
    assert alone["workload_perf"]["light"] == "green" and alone["candidate_replicas"] == 1
    assert copies["workload_perf"]["light"] == "white" and "TRTMC ran as 4" in copies["workload_perf"]["note"]
    assert copies["candidate_replicas"] == 4
    failed = absolute.entries({"absolute": [{"suite": "s", "gate": {"margin": 1.0}}]}, {"s": []}, {"s": {}}, native,
                              "native exploded",
                              candidate_replicas=4, candidate_mps=True)[0]
    assert failed["status"] == "error" and failed["candidate_replicas"] == 4 and failed["candidate_mps"]



class FakeMps:
    """nvidia-cuda-mps-control and nvidia-smi for the MPS tests: ``-d`` logs a control and a server pid (or fails
    after logging), ``quit`` ends them unless ``stubborn``; signals end them unless ``immortal``."""

    def __init__(self, directory, *, start_fails=False, stubborn=False, immortal=False):
        self.directory, self.start_fails, self.stubborn, self.immortal = directory, start_fails, stubborn, immortal
        self.calls, self.living, self.env = [], set(), None

    def run(self, command, **kwargs):
        import subprocess

        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, stdout="0, GPU-aaaa\n")
        self.calls.append("quit" if kwargs.get("input") else "start")
        if kwargs.get("input"):
            if not self.stubborn:
                self.living.clear()
            return subprocess.CompletedProcess(command, 1 if self.stubborn else 0)
        self.env = kwargs["env"]
        log = Path(self.env["CUDA_MPS_LOG_DIRECTORY"]) / "control.log"  # the attempt's own directory
        log.write_text("[t Control 101] Starting control daemon using socket x\n[t Control 102] Starting new server 102\n")
        self.living.update({101, 102})
        if self.start_fails:
            raise subprocess.TimeoutExpired(command, 60)
        return subprocess.CompletedProcess(command, 0)

    def kill(self, pid, sig):
        self.calls.append(f"kill {pid}")
        if not self.immortal:
            self.living.discard(pid)


def mps_patches(fake, monkeypatch):
    from trtmc_aiperf_qual import services

    monkeypatch.setattr(services.subprocess, "run", fake.run)
    monkeypatch.setattr(services.os, "kill", fake.kill)
    monkeypatch.setattr(services, "_alive", lambda pid: pid in fake.living)
    monkeypatch.setattr(services, "_ours", lambda pid, pipe: pid != 999)
    monkeypatch.setattr(services.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(services, "MPS_EXIT_S", 0)
    return services


def test_copies_share_the_gpu_through_their_own_mps_daemon_when_enabled(tmp_path, monkeypatch):
    """With acc_mps, every copy gets the private daemon's variables (the GPU by UUID), the daemon quits after the
    copies stop, and its exit is verified."""
    from contextlib import contextmanager

    from trtmc_aiperf_qual.config import Environment

    fake = FakeMps(tmp_path / "acc-mps")
    services = mps_patches(fake, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    events = []

    @contextmanager
    def serving(environment, model, backend, out, *, extra_env=None, port=None, **options):
        events.append(("start", out.name, dict(extra_env or {})))
        yield {"url": f"http://{out.name}"}
        events.append(("stop", out.name, fake.living == {101, 102}))

    environment = Environment({"acc_mps": True, "repo": str(tmp_path), "ports": {"candidate": 9000, "reference": 9100}})
    memory = iter([(0, 250 * 1024), (10 * 1024, 250 * 1024)])
    monkeypatch.setattr(services, "serving", serving)
    monkeypatch.setattr(services, "gpu_memory_mib", lambda: next(memory))
    with services.serving_replicas(environment, {}, "trtmc", tmp_path / "acc", count=2) as service:
        assert service["replicas"] == 2 and service["mps"]
    starts = [event for event in events if event[0] == "start"]
    assert len(starts) == 2 and all(event[2]["CUDA_VISIBLE_DEVICES"] == "GPU-aaaa" for event in starts)
    assert all(event[2]["CUDA_MPS_PIPE_DIRECTORY"] == str(tmp_path / "acc-mps" / "pipe") for event in starts)
    assert fake.env["CUDA_VISIBLE_DEVICES"] == "GPU-aaaa"
    assert all(event[2] for event in events if event[0] == "stop")  # the daemon outlived its clients
    assert fake.calls == ["start", "quit"] and not fake.living


def test_an_mps_daemon_that_fails_to_start_is_stopped_and_one_that_will_not_stop_fails_the_phase(tmp_path, monkeypatch):
    from trtmc_aiperf_qual.config import Environment
    from trtmc_aiperf_qual.services import ServiceError

    environment = Environment({"acc_mps": True, "repo": str(tmp_path)})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    fake = FakeMps(tmp_path / "late", start_fails=True)
    services = mps_patches(fake, monkeypatch)
    with services.mps(environment, tmp_path / "late") as variables:
        assert variables == {}  # time-sliced copies
    assert fake.calls == ["start", "quit"] and not fake.living
    assert (tmp_path / "late" / "mps-unavailable.txt").is_file()

    fake = FakeMps(tmp_path / "stuck", stubborn=True)
    services = mps_patches(fake, monkeypatch)
    with services.mps(environment, tmp_path / "stuck") as variables:
        assert variables["CUDA_MPS_PIPE_DIRECTORY"].endswith("pipe")
    assert fake.calls[:2] == ["start", "quit"] and "kill 101" in fake.calls and not fake.living  # signalled

    fake = FakeMps(tmp_path / "immortal", stubborn=True, immortal=True)
    services = mps_patches(fake, monkeypatch)
    with pytest.raises(services.GpuStateError, match="still run"):
        with services.mps(environment, tmp_path / "immortal"):
            pass
    assert not issubclass(services.GpuStateError, Exception)  # no phase or fallback handler catches it
    assert ServiceError  # (start failures stay ordinary service errors)

    fake = FakeMps(tmp_path / "retry")
    services = mps_patches(fake, monkeypatch)
    for _ in range(2):  # a retry's daemon logs in a directory of its own
        with services.mps(environment, tmp_path / "retry") as variables:
            pass
    assert variables["CUDA_MPS_LOG_DIRECTORY"] == str(tmp_path / "retry-2" / "log")


def test_a_gpu_left_in_an_unknown_state_stops_the_run(tmp_path):
    """Neither the phase runner's retries nor the native side's precision fallback absorb a GpuStateError."""
    from contextlib import contextmanager
    from unittest.mock import patch

    from trtmc_aiperf_qual import runner
    from trtmc_aiperf_qual.config import Environment
    from trtmc_aiperf_qual.services import GpuStateError

    def stuck():
        raise GpuStateError("MPS processes [1] still run")

    with pytest.raises(GpuStateError):
        runner._Phases(tmp_path).run("absolute_native", stuck, retries=1)

    @contextmanager
    def serving_replicas(*args, **kwargs):
        stuck()
        yield

    model = {"operation": "generate", "absolute": [{"suite": "s"}],
             "reference": {"backend": "reference", "perf_precision": "fp16", "precision": "fp32"}}
    with patch.object(absolute, "serving_replicas", serving_replicas), pytest.raises(GpuStateError):
        absolute.run_native(Environment({"native_replicas": 4}), model, "python", {"s": []}, tmp_path)


def test_ambiguous_gpu_ordinals_leave_the_copies_without_mps(tmp_path, monkeypatch):
    import subprocess

    from trtmc_aiperf_qual.config import Environment

    fake = FakeMps(tmp_path / "two")
    services = mps_patches(fake, monkeypatch)
    two = lambda command, **kwargs: (subprocess.CompletedProcess(command, 0, stdout="0, GPU-a\n1, GPU-b\n")  # noqa: E731
                                     if command[0] == "nvidia-smi" else fake.run(command, **kwargs))
    monkeypatch.setattr(services.subprocess, "run", two)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    with services.mps(Environment({"acc_mps": True, "repo": str(tmp_path)}), tmp_path / "two") as variables:
        assert variables == {}
    assert fake.calls == [] and "ambiguous" in (tmp_path / "two" / "mps-unavailable.txt").read_text()
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    with services.mps(Environment({"acc_mps": True, "repo": str(tmp_path)}), tmp_path / "two") as variables:
        assert variables["CUDA_VISIBLE_DEVICES"] == "GPU-b"



def test_only_the_runs_own_mps_processes_are_signalled(tmp_path, monkeypatch):
    """A pid the log names that now belongs to another process (reused, or a log naming it) is never signalled."""
    from trtmc_aiperf_qual import services

    (tmp_path / "log").mkdir()
    (tmp_path / "log" / "control.log").write_text("[t Control 999] Starting control daemon using socket x\n")
    killed = []
    monkeypatch.setattr(services.subprocess, "run", lambda command, **kwargs: services.subprocess.CompletedProcess(command, 0))
    monkeypatch.setattr(services.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(services, "_alive", lambda pid: True)
    monkeypatch.setattr(services.time, "sleep", lambda seconds: None)
    services._mps_stop({"CUDA_MPS_PIPE_DIRECTORY": str(tmp_path / "pipe")}, tmp_path)  # pid 999 is not an MPS program
    assert killed == []
    assert not services._ours(1, str(tmp_path / "pipe"))  # init: not an MPS program of this run


def test_a_selection_is_loaded_once_per_settings_and_shared(tmp_path):
    """The plan's selection is written once and read back by later loads under the same settings (each side's
    AIPerf run); any other setting is another selection; without TRTMC_ACCURACY_CACHE nothing is cached."""
    import asyncio

    from trtmc_aiperf_plugins import benchmarks

    calls = []

    class Fake:
        environ = None

        async def load_problems(self, tasks, n_shots, enable_cot):
            calls.append(dict(self.environ))
            return [problem("t", prompt=f"q{len(calls)}")]

    Fake.load_problems = benchmarks._cached(Fake.load_problems)
    loader = Fake()
    loader.environ = {"TRTMC_ACCURACY_CACHE": str(tmp_path), "TRTMC_ACCURACY_PER_TASK": "40"}
    first = asyncio.run(loader.load_problems(None, 0, False))
    again = asyncio.run(loader.load_problems(None, 0, False))
    assert len(calls) == 1 and [p.prompt for p in again] == [p.prompt for p in first] == ["q1"]
    loader.environ = {**loader.environ, "TRTMC_ACCURACY_PER_TASK": "20"}
    assert asyncio.run(loader.load_problems(None, 0, False))[0].prompt == "q2" and len(calls) == 2
    loader.environ = {"TRTMC_ACCURACY_PER_TASK": "40"}
    asyncio.run(loader.load_problems(None, 0, False))
    assert len(calls) == 3 and not list(tmp_path.glob("*.partial"))
    assert all(benchmarks.BENCHMARKS[name].load_problems.__wrapped__ for name in ("trtmc_mmlu", "trtmc_lambada"))


def test_the_other_copies_start_together_and_every_started_copy_stops(tmp_path, monkeypatch):
    import threading
    import time
    from contextlib import contextmanager

    from trtmc_aiperf_qual import services
    from trtmc_aiperf_qual.config import Environment

    events, lock = [], threading.Lock()

    def fake(fail=None):
        @contextmanager
        def serving(environment, model, backend, out, *, extra_env=None, port=None, **options):
            time.sleep(0.3)  # a copy loading
            if fail and out.name.endswith(fail[0]):
                raise fail[1]
            with lock:
                events.append(("start", out.name))
            try:  # as services.serving: the server stops whatever ends its use
                yield {"url": f"http://{out.name}"}
            finally:
                with lock:
                    events.append(("stop", out.name))
        return serving

    environment = Environment({"ports": {"candidate": 9000, "reference": 9100}})
    monkeypatch.setattr(services, "replicas_that_fit", lambda before, after, wanted, reserve_mib=0: wanted)
    monkeypatch.setattr(services, "gpu_memory_mib", lambda: (0, 1))
    monkeypatch.setattr(services, "serving", fake())
    began = time.time()
    with services.serving_replicas(environment, {}, "trtmc", tmp_path / "acc", count=4) as service:
        assert service["replicas"] == 4
    assert time.time() - began < 1.0  # 0.3 s for the first, 0.3 s for the other three together
    assert sum(event[0] == "stop" for event in events) == 4

    events.clear()
    monkeypatch.setattr(services, "serving", fake(("replica2", services.ServiceError("no memory"))))
    with services.serving_replicas(environment, {}, "trtmc", tmp_path / "acc", count=4) as service:
        assert service["replicas"] == 3
    events.clear()
    monkeypatch.setattr(services, "serving", fake(("replica2", OSError("disk"))))
    with pytest.raises(OSError):
        with services.serving_replicas(environment, {}, "trtmc", tmp_path / "acc", count=4):
            pass
    assert sorted(name for kind, name in events if kind == "stop") == sorted(name for kind, name in events
                                                                             if kind == "start")


def test_a_selection_is_cached_only_under_an_immutable_tokenizer(tmp_path):
    """The tokenizer's identity is part of the key: a local tokenizer by its contents, a hub one by its cached
    snapshot commit; one whose identity cannot be told is never cached."""
    import asyncio

    from trtmc_aiperf_plugins import benchmarks

    calls = []

    class Fake:
        environ = None

        async def load_problems(self, tasks, n_shots, enable_cot):
            calls.append(1)
            return [problem("t")]

    Fake.load_problems = benchmarks._cached(Fake.load_problems)
    loader = Fake()
    loader.environ = {"TRTMC_ACCURACY_CACHE": str(tmp_path / "cache"),
                      "TRTMC_ACCURACY_TOKENIZER": "no-such-org/no-such-tokenizer-2026"}
    asyncio.run(loader.load_problems(None, 0, False))
    asyncio.run(loader.load_problems(None, 0, False))
    assert len(calls) == 2 and not (tmp_path / "cache").exists()  # a mutable revision online: no cache
    loader.environ = {**loader.environ, "TRTMC_ACCURACY_TOKENIZER_REVISION": "a" * 40}
    asyncio.run(loader.load_problems(None, 0, False))
    asyncio.run(loader.load_problems(None, 0, False))
    assert len(calls) == 3  # a pinned commit: cached

    local = tmp_path / "tokenizer"
    local.mkdir()
    (local / "tokenizer.json").write_text("v1")
    loader.environ = {"TRTMC_ACCURACY_CACHE": str(tmp_path / "cache"), "TRTMC_ACCURACY_TOKENIZER": str(local)}
    asyncio.run(loader.load_problems(None, 0, False))
    asyncio.run(loader.load_problems(None, 0, False))
    assert len(calls) == 4  # cached under v1
    (local / "tokenizer.json").write_text("v2")
    asyncio.run(loader.load_problems(None, 0, False))
    assert len(calls) == 5 and len(list((tmp_path / "cache").glob("*.json"))) == 3  # pinned, v1, v2


def test_an_interrupt_while_copies_start_still_stops_every_started_copy(tmp_path, monkeypatch):
    """Interrupted while waiting on the copies' starts, the orchestration waits on until every start finished,
    registers every copy that started, and only then raises the interrupt: no copy outlives the call."""
    import concurrent.futures
    from contextlib import contextmanager

    from trtmc_aiperf_qual import services
    from trtmc_aiperf_qual.config import Environment

    events = []

    @contextmanager
    def serving(environment, model, backend, out, *, extra_env=None, port=None, **options):
        events.append(("start", out.name))
        try:
            yield {"url": f"http://{out.name}"}
        finally:
            events.append(("stop", out.name))

    waits = []

    from trtmc_aiperf_qual import cancel

    def interrupted_once(futures):
        waits.append(cancel.EVENT.is_set())
        if len(waits) == 1:
            raise KeyboardInterrupt  # Ctrl-C while the starts are under way
        return concurrent.futures.wait(futures)

    monkeypatch.setattr(services, "serving", serving)
    monkeypatch.setattr(services, "wait", interrupted_once)
    monkeypatch.setattr(services, "replicas_that_fit", lambda before, after, wanted, reserve_mib=0: wanted)
    monkeypatch.setattr(services, "gpu_memory_mib", lambda: (0, 1))
    with pytest.raises(KeyboardInterrupt):
        with services.serving_replicas(Environment({"ports": {"candidate": 9000}}), {}, "trtmc", tmp_path / "acc",
                                       count=3):
            pass
    assert waits == [False, True] and cancel.EVENT.is_set()  # the starts were told to give up; teardown is short
    cancel.EVENT.clear()
    assert sorted(name for kind, name in events if kind == "stop") == ["acc", "acc-replica1", "acc-replica2"]

    class Interrupted(concurrent.futures.ThreadPoolExecutor):
        def __exit__(self, *exc):
            super().__exit__(*exc)
            raise KeyboardInterrupt  # Ctrl-C while the pool shuts down

    events.clear()
    monkeypatch.setattr(services, "wait", concurrent.futures.wait)
    monkeypatch.setattr(services, "ThreadPoolExecutor", Interrupted)
    with pytest.raises(KeyboardInterrupt):
        with services.serving_replicas(Environment({"ports": {"candidate": 9000}}), {}, "trtmc", tmp_path / "acc",
                                       count=3):
            pass
    assert sorted(name for kind, name in events if kind == "stop") == ["acc", "acc-replica1", "acc-replica2"]



def test_the_plugin_selections_are_warmed_and_a_failure_is_left_to_the_run(monkeypatch):
    from trtmc_aiperf_qual import campaign
    from trtmc_aiperf_qual.config import Environment

    planned = []

    def plan(environment, model, item):
        planned.append(item["suite"])
        if item["suite"] == "broken":
            raise OSError("dataset unreachable")

    monkeypatch.setattr(absolute, "plan", plan)
    model = {"absolute": [{"suite": "mmlu-0shot", "plugin": "trtmc_mmlu"}, {"suite": "stsb", "metric": "spearman"},
                          {"suite": "broken", "plugin": "trtmc_lambada"}]}
    campaign.warm_selections(Environment({}), model)  # no exception: the run's own plan reports it
    assert planned == ["mmlu-0shot", "broken"]



def test_the_selection_tokenizer_is_pinned_to_the_commit_its_revision_resolves_to(tmp_path, monkeypatch):
    import huggingface_hub

    asked = []

    class Api:
        def model_info(self, name, revision=None):
            asked.append((name, revision))
            if name == "gated/model":
                raise OSError("401")
            return type("Info", (), {"sha": "b" * 40})()

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    absolute.pinned_revision.cache_clear()
    assert absolute.pinned_revision("org/model", None) == "b" * 40
    assert absolute.pinned_revision("org/model", "c" * 40) == "c" * 40  # already a commit
    assert absolute.pinned_revision(str(tmp_path), "main") == "main"  # a local directory
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda name, filename, revision=None: None)
    assert absolute.pinned_revision("gated/model", "main") == "main"  # unresolved: left as given (not cached)
    assert asked == [("org/model", None), ("gated/model", "main")]
    snapshot = tmp_path / "hub" / "snapshots" / ("d" * 40)
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda name, filename, revision=None: str(snapshot / filename))
    absolute.pinned_revision.cache_clear()
    assert absolute.pinned_revision("gated/model", "main") == "d" * 40  # offline: the cached snapshot's commit
    absolute.pinned_revision.cache_clear()


def test_both_sides_answer_at_once_trtmc_sized_after_the_native_copies(tmp_path, monkeypatch):
    """Under one MPS daemon the native copies start first, then TRTMC's, and both sides' requests overlap; a side
    that fails is reported for its own fallback while the other's answers stand."""
    import threading
    import time
    from contextlib import contextmanager

    from trtmc_aiperf_qual import services
    from trtmc_aiperf_qual.config import Environment

    events, lock = [], threading.Lock()

    def note(event):
        with lock:
            events.append(event)

    def fake_replicas(fail=None):
        @contextmanager
        def serving_replicas(environment, model, backend, out, *, count, mps_env=None, reserve_mib=0, **options):
            assert mps_env == {"CUDA_MPS_PIPE_DIRECTORY": "p"}  # one daemon for both sides
            if fail == backend:
                raise services.ServiceError(f"{backend} did not start")
            note(("up", backend, reserve_mib))
            yield {"url": backend, "replicas": count, "mps": True, "footprint_mib": 1000}
            note(("down", backend))
        return serving_replicas

    @contextmanager
    def daemon(environment, directory):
        yield {"CUDA_MPS_PIPE_DIRECTORY": "p"}

    def side(environment, service, model, item, problems, out):
        note(("answering", service["url"]))
        time.sleep(0.2)
        note(("answered", service["url"]))
        return {"records": {"greedy": {index: {} for index in range(len(problems))}}}

    monkeypatch.setattr(services, "mps", daemon)
    monkeypatch.setattr(absolute, "serving_replicas", fake_replicas())
    monkeypatch.setattr(absolute, "run_side", side)
    model = {"operation": "generate", "absolute": [{"suite": "s"}],
             "reference": {"backend": "reference", "perf_precision": "fp16", "precision": "fp32"}}
    environment = Environment({"native_replicas": 8, "candidate_replicas": 4})
    result = absolute.overlapped_acc(environment, model, "python", {"s": [{}, {}]}, tmp_path)
    up = [event for event in events if event[0] == "up"]
    assert up == [("up", "reference", 0), ("up", "trtmc", 4000)]  # native first; its growth (0.5 x 1000 x 8) held back
    assert events.index(("up", "trtmc", 4000)) < events.index(("answering", "reference"))  # native waited, idle
    assert events.index(("answering", "trtmc")) < events.index(("answered", "reference"))  # then both at once
    assert result["native"]["replicas"] == 8 and result["copies"] == {"candidate_replicas": 4, "candidate_mps": True}

    events.clear()
    monkeypatch.setattr(absolute, "serving_replicas", fake_replicas(fail="trtmc"))
    result = absolute.overlapped_acc(environment, model, "python", {"s": [{}]}, tmp_path)
    assert "trtmc did not start" in result["candidate_error"] and result["native"]["runs"]["s"]["records"]
    monkeypatch.setattr(absolute, "serving_replicas", fake_replicas(fail="reference"))
    result = absolute.overlapped_acc(environment, model, "python", {"s": [{}]}, tmp_path)
    assert "reference did not start" in result["native_error"] and "candidate" in result


def test_answers_given_alongside_are_judged_and_only_l1_starts_a_server(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from trtmc_aiperf_qual import runner
    from trtmc_aiperf_qual.config import Environment

    started = []

    @contextmanager
    def single(environment, model, backend, out, **options):
        started.append(out.name)
        yield {"url": "u", "info": {}}

    judged = {"suite": "s", "status": "pass", "workload_perf": {"pairs": 3, "light": "green"}}
    monkeypatch.setattr(runner, "serving", single)
    monkeypatch.setattr(absolute, "judge", lambda *args: dict(judged))
    accuracy = []
    native = {"backend": "reference", "precision": "fp16", "replicas": 8, "mps": True, "runs": {"s": {}}}
    runner._candidate(Environment({"candidate_replicas": 4}), {"absolute": [{"suite": "s"}]}, {"suite": {}}, [], {},
                      accuracy, [], tmp_path, absolute_runs={"plans": {"s": []}, "native": native, "native_error": None,
                                                             "answered": {"candidate": {"s": {}}, "candidate_replicas": 4,
                                                                          "candidate_mps": True}})
    assert started == ["candidate"]  # the L1 server only
    entry = accuracy[0]
    assert entry["sides_concurrent"] and entry["workload_perf"]["light"] == "white"
    assert "both sides answered at once" in entry["workload_perf"]["note"]



def test_a_side_that_did_not_answer_every_problem_is_incomplete():
    model, plans = {"absolute": [{"suite": "s"}, {"suite": "t"}]}, {"s": [{}, {}], "t": [{}]}
    whole = {"s": {"records": {"greedy": {0: {"correct": False}, 1: {"correct": True}}}},
             "t": {"observations": {"greedy": {0: {}}}}}
    assert absolute.incomplete(model, plans, whole) is None  # a wrong answer is an answer
    assert "1 of 2 answered" in absolute.incomplete(model, plans, {**whole, "s": {"records": {"greedy": {0: {}}}}})
    failed = {**whole, "s": {**whole["s"], "failed": {"greedy": "1 requests failed: HTTP 500"}}}
    assert "HTTP 500" in absolute.incomplete(model, plans, failed)
    assert "t: not run" == absolute.incomplete(model, plans, {"s": whole["s"]})


def test_the_two_sides_copies_never_share_a_port_and_a_foreign_server_is_refused(tmp_path, monkeypatch):
    from trtmc_aiperf_qual import services
    from trtmc_aiperf_qual.config import Environment

    environment = Environment({"ports": {"candidate": 8901, "reference": 8900}, "repo": str(tmp_path),
                               "bundle_root": str(tmp_path), "runtime_root": str(tmp_path), "worker": "w",
                               "serve_python": "python"})
    ports = {backend: {services.copy_port(environment, backend, index) for index in range(services.MAX_COPIES)}
             for backend in ("trtmc", "reference")}
    assert not ports["trtmc"] & ports["reference"] and len(ports["trtmc"]) == services.MAX_COPIES

    class Process:
        pid = 1

        def poll(self):
            return None

    stopped = []
    monkeypatch.setattr(services.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(services, "_wait_ready", lambda process, url, out: {"backend": "reference"})
    monkeypatch.setattr(services, "_stop", lambda process: stopped.append(process))
    monkeypatch.setattr(services, "_port_free", lambda port: True)
    model = {"catalog_profile": "m", "candidate": {"bundle": "b"}, "reference": {}}
    with pytest.raises(services.ServiceError, match="answered as 'reference'"):
        with services.serving(environment, model, "trtmc", tmp_path / "srv"):
            pass
    assert stopped  # the server this run started is stopped
    monkeypatch.setattr(services, "_port_free", lambda port: False)
    with pytest.raises(services.ServiceError, match="in use"):
        with services.serving(environment, model, "trtmc", tmp_path / "srv2"):
            pass


def test_a_cancelled_run_stops_its_aiperf_process_promptly(tmp_path):
    import sys
    import threading
    import time

    from trtmc_aiperf_qual import aiperf_runner, cancel

    timer = threading.Timer(0.5, cancel.EVENT.set)
    timer.start()
    began = time.time()
    try:
        with open(tmp_path / "log", "w") as log, pytest.raises(cancel.Cancelled):
            aiperf_runner._run([sys.executable, "-c", "import time; time.sleep(60)"], log, {}, 3600)
    finally:
        cancel.EVENT.clear()
    assert time.time() - began < 5



def test_a_blocking_request_is_released_by_cancellation_and_one_copy_is_measured(tmp_path, monkeypatch):
    import threading
    import time
    from contextlib import contextmanager

    from trtmc_aiperf_qual import cancel, services
    from trtmc_aiperf_qual.config import Environment

    timer = threading.Timer(0.5, cancel.EVENT.set)
    timer.start()
    began = time.time()
    try:
        with pytest.raises(cancel.Cancelled):
            cancel.wait_for(lambda: time.sleep(30))  # a probe whose server never answers
    finally:
        cancel.EVENT.clear()
    assert time.time() - began < 3
    assert cancel.wait_for(lambda: 42) == 42

    @contextmanager
    def serving(environment, model, backend, out, *, extra_env=None, port=None, **options):
        yield {"url": "u"}

    memory = iter([(1000, 250_000), (9000, 250_000)])
    monkeypatch.setattr(services, "serving", serving)
    monkeypatch.setattr(services, "gpu_memory_mib", lambda: next(memory))
    with services.serving_replicas(Environment({"ports": {"reference": 8900}}), {}, "reference", tmp_path / "n",
                                   count=1) as service:
        assert service["replicas"] == 1 and service["footprint_mib"] == 8000  # the other side reserves for it



def test_a_probe_stalled_on_its_error_body_is_released_by_cancellation(monkeypatch):
    import threading
    import time
    import urllib.error
    import urllib.request

    from trtmc_aiperf_qual import cancel

    class StalledBody:
        def read(self):
            time.sleep(30)  # error headers arrived; the body never does
            return b""

        def close(self):
            pass

    def urlopen(call, timeout):
        raise urllib.error.HTTPError(call.full_url, 500, "error", {}, StalledBody())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    timer = threading.Timer(0.5, cancel.EVENT.set)
    timer.start()
    began = time.time()
    try:
        with pytest.raises(cancel.Cancelled):
            absolute._probe({"url": "http://u"}, "generate", {"prompt": "p"})
    finally:
        cancel.EVENT.clear()
    assert time.time() - began < 3

    class Body:
        def read(self):
            return b"bad request"

        def close(self):
            pass

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda call, timeout: (_ for _ in ()).throw(urllib.error.HTTPError(call.full_url, 400, "e", {}, Body())))
    with pytest.raises(RuntimeError, match="probe rejected: bad request"):
        absolute._probe({"url": "http://u"}, "generate", {"prompt": "p"})



def test_waiting_for_aiperf_exports_observes_cancellation(tmp_path):
    import threading
    import time

    from trtmc_aiperf_qual import aiperf_runner, cancel

    timer = threading.Timer(0.5, cancel.EVENT.set)
    timer.start()
    began = time.time()
    try:
        with pytest.raises(cancel.Cancelled):
            aiperf_runner._wait_ready(tmp_path, timeout_s=60)  # AIPerf exited without its exports
    finally:
        cancel.EVENT.clear()
    assert time.time() - began < 3


@pytest.mark.parametrize("slower", ["reference", "trtmc"])
def test_no_copy_leaves_the_shared_daemon_while_the_other_side_answers(tmp_path, monkeypatch, slower):
    """Whichever side finishes first keeps its copies until the other side has answered (a client leaving the
    shared MPS server stalled the other side's copies on GB300)."""
    import threading
    import time
    from contextlib import contextmanager

    from trtmc_aiperf_qual import services
    from trtmc_aiperf_qual.config import Environment

    events, lock = [], threading.Lock()

    def note(event):
        with lock:
            events.append(event)

    @contextmanager
    def serving_replicas(environment, model, backend, out, *, count, mps_env=None, reserve_mib=0, **options):
        yield {"url": backend, "replicas": count, "mps": True, "footprint_mib": 1}
        note(("down", backend))

    @contextmanager
    def daemon(environment, directory):
        yield {"CUDA_MPS_PIPE_DIRECTORY": "p"}

    def side(environment, service, model, item, problems, out):
        time.sleep(1.5 if service["url"] == slower else 0.1)
        note(("answered", service["url"]))
        return {"records": {"greedy": {0: {}}}}

    monkeypatch.setattr(services, "mps", daemon)
    monkeypatch.setattr(absolute, "serving_replicas", serving_replicas)
    monkeypatch.setattr(absolute, "run_side", side)
    model = {"operation": "generate", "absolute": [{"suite": "s"}],
             "reference": {"backend": "reference", "perf_precision": "fp16", "precision": "fp32"}}
    result = absolute.overlapped_acc(Environment({"native_replicas": 2, "candidate_replicas": 2}), model, "python",
                                     {"s": [{}]}, tmp_path)
    assert "native" in result and "candidate" in result
    last_answer = max(events.index(("answered", backend)) for backend in ("reference", "trtmc"))
    assert all(events.index(("down", backend)) > last_answer for backend in ("reference", "trtmc"))



def test_queries_and_server_stops_are_short_once_cancelled(tmp_path, monkeypatch):
    import subprocess
    import sys
    import threading
    import time

    from trtmc_aiperf_qual import cancel, services

    timer = threading.Timer(0.5, cancel.EVENT.set)
    timer.start()
    began = time.time()
    try:
        with pytest.raises(cancel.Cancelled):
            cancel.output([sys.executable, "-c", "import time; time.sleep(60)"], 60)  # a stalled nvidia-smi
        assert time.time() - began < 3
        stubborn = subprocess.Popen([sys.executable, "-c", "import signal, time; signal.signal(signal.SIGINT, "
                                     "signal.SIG_IGN); time.sleep(600)"], start_new_session=True)
        time.sleep(0.5)
        began = time.time()
        services._stop(stubborn)  # ignores SIGINT: killed after the cancelled run's short grace
        assert stubborn.poll() is not None and time.time() - began < services.CANCELLED_STOP_S + 3
    finally:
        cancel.EVENT.clear()
    assert cancel.output([sys.executable, "-c", "print('ok')"], 10).strip() == "ok"
