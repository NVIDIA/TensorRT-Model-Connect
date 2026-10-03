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


def test_mcnemar_is_one_sided_against_trtmc():
    assert absolute.mcnemar_worse_p(0, 0) == 1.0
    assert absolute.mcnemar_worse_p(10, 0) == 1.0
    assert absolute.mcnemar_worse_p(0, 6) == pytest.approx(1 / 64)
    assert absolute.mcnemar_worse_p(5, 15) < 0.05 < absolute.mcnemar_worse_p(8, 12)


def side(answers, exit_code=0):
    return {"records": {"greedy": {index: {"passed": ok, "unparsed": False, "actual": str(ok)}
                                   for index, ok in answers.items()}}, "exit": {"greedy": exit_code}}


ITEM = {"suite": "mmlu-5shot", "plugin": "trtmc_mmlu", "endpoint": "completions", "gate": {"max_delta_points": 1.0}}


def test_equal_scores_pass_and_report_both_accuracies_and_agreement():
    problems = [{"task": "t", "gold": " A"}] * 200
    answers = {index: index % 4 != 0 for index in range(200)}
    entry = absolute.judge(ITEM, problems, side(answers), side(answers))
    assert entry["status"] == "pass" and entry["samples"] == 200
    assert entry["metrics"]["trtmc_accuracy"] == entry["metrics"]["native_accuracy"] == 75.0
    assert entry["metrics"]["answer_agreement"] == 1.0 and entry["metrics"]["mcnemar_p_worse"] == 1.0


def test_a_significant_or_large_drop_fails_and_a_missing_answer_is_an_error():
    problems = [{"task": "t", "gold": " A"}] * 200
    native = {index: True for index in range(200)}
    trtmc = {index: index >= 8 for index in range(200)}  # 8 answers lost: 4 points, p = 1/256
    entry = absolute.judge(ITEM, problems, side(trtmc), side(native))
    assert entry["status"] == "fail" and entry["metrics"]["delta_points"] == -4.0
    assert len(entry["reasons"]) == 1 and entry["failures"][0]["sample_id"] == "t/0"
    assert "significantly worse" in entry["notes"][0]
    within = absolute.judge({**ITEM, "gate": {"max_delta_points": 5.0}}, problems, side(trtmc), side(native))
    assert within["status"] == "pass" and within["notes"]  # significant but close enough: a note
    assert "TRTMC 96.0% vs native 100.0%" in counted(entry)
    incomplete = absolute.judge(ITEM, problems, side({index: True for index in range(199)}), side(native))
    assert incomplete["status"] == "error" and incomplete["counts"]["missing_trtmc"] == 1


def test_rejudge_reapplies_a_gate_to_the_recorded_scores():
    metrics = {"delta_points": -1.5, "mcnemar_p_worse": 0.2}
    assert absolute.status(metrics, {"max_delta_points": 1.0}, expected=100, paired=100)[0] == "fail"
    assert absolute.status(metrics, {"max_delta_points": 2.0}, expected=100, paired=100)[0] == "pass"


def test_sampled_models_compare_mean_accuracy_over_seeds():
    problems = [{"task": "t", "gold": " A"}] * 10
    candidate = {"records": {f"seed{s}": {i: {"passed": i < 6} for i in range(10)} for s in (1, 2)}, "exit": {}}
    native = {"records": {f"seed{s}": {i: {"passed": i < 7} for i in range(10)} for s in (1, 2)}, "exit": {}}
    entry = absolute.judge({**ITEM, "gate": {"max_delta_points": 15.0}}, problems, candidate, native)
    assert entry["metrics"]["delta_points"] == -10.0 and len(entry["metrics"]["per_seed"]) == 2
    assert "mcnemar_p_worse" not in entry["metrics"] and entry["status"] == "pass"


DEFINITIONS = {"mmlu": {"plugin": "trtmc_mmlu", "suite": "mmlu-5shot", "gate": {"max_delta_points": 1.0},
                        "quantized_gate": {"max_delta_points": 2.0}, "sampled_gate": {"max_delta_points": 3.0}},
               "gsm8k": {"plugin": "trtmc_gsm8k", "suite": "gsm8k", "gate": {"max_delta_points": 1.5}}}


def test_benchmarks_take_the_route_and_gate_of_the_catalog_request():
    chat, = _absolute(["mmlu"], DEFINITIONS, {"use_chat_template": True}, quantized=False)
    assert chat["endpoint"] == "chat" and chat["gate"] == {"max_delta_points": 1.0} and "seeds" not in chat
    quantized, = _absolute(["mmlu"], DEFINITIONS, {}, quantized=True)
    assert quantized["endpoint"] == "completions" and quantized["gate"] == {"max_delta_points": 2.0}
    sampled, limited = _absolute(["mmlu", {"name": "gsm8k", "limit": 400}], DEFINITIONS,
                                 {"temperature": 0.7, "top_k": 50}, quantized=False)
    assert sampled["seeds"] == [1, 2, 3] and sampled["gate"] == {"max_delta_points": 3.0}
    assert limited["limit"] == 400 and limited["gate"] == {"max_delta_points": 1.5}
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
                                                 "gate": {"max_delta_points": 1.0}}},
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
    result = gold_metrics.compare_corpus("sts_spearman", pairs, same, same)
    assert result["delta_points"] == 0 and not result["significantly_worse"] and result["units"] == 4


def test_corpus_entries_gate_on_the_difference_relative_to_the_native_score():
    item = {"suite": "librispeech-test-clean", "metric": "wer", "gate": {"max_delta_points": 0.2, "max_relative": 0.03}}
    problems = [{"gold": "a b c d e f g h i j", "task": "t"}] * 20
    native = {"observations": {"greedy": {i: {"text": "a b c d e f g h i x"} for i in range(20)}}, "exit": {},
              "timings": {"greedy": {}}}
    worse = {"observations": {"greedy": {i: {"text": "a b c d e f g h x x" if i < 10 else "a b c d e f g h i x"}
                                         for i in range(20)}}, "exit": {}, "timings": {"greedy": {}}}
    entry = absolute.judge(item, problems, worse, native)
    assert entry["metrics"]["native_score"] == 10.0 and entry["metrics"]["delta_points"] == 5.0
    assert entry["status"] == "fail" and entry["metrics"]["significantly_worse"] and entry["notes"]
    assert absolute.judge(item, problems, native, native)["status"] == "pass"
    missing = {**native, "observations": {"greedy": {i: {"text": "a"} for i in range(19)}}}
    assert absolute.judge(item, problems, missing, native)["status"] == "error"
    # a 3% relative gate: 0.25 points on a native WER of 10 is within max(0.2, 0.3)
    metrics = {"delta_points": 0.25, "native_score": 10.0}
    assert absolute.status(metrics, item["gate"], expected=1, paired=1)[0] == "pass"


def test_rerank_retrieval_and_code_metrics():
    from trtmc_aiperf_qual import gold_metrics

    assert gold_metrics.rerank_units([{"gold": [2]}], {0: {"scores": [0.1, 0.2, 0.9]}}) == [1.0]
    assert gold_metrics.rerank_units([{"gold": [0]}], {0: {"scores": [0.1, 0.2, 0.9]}})[0] == pytest.approx(0.5)
    problems = [{"task": "query", "gold": [0]}, {"task": "document"}, {"task": "document"},
                {"task": "query", "gold": [0]}, {"task": "document"}, {"task": "document"}]
    vectors = {0: [1, 0], 1: [0, 1], 2: [1, 0.1], 3: [0, 1], 4: [0.1, 1], 5: [1, 0]}
    units = gold_metrics.retrieval_units(problems, {i: {"values": v} for i, v in vectors.items()})
    assert units[1] == 1.0 and units[0] == pytest.approx(1 / math.log2(3))
    gold = {"prompt": "def add(a, b):\n", "test": "def check(f):\n    assert f(1, 2) == 3\n", "entry_point": "add"}
    assert gold_metrics.code_pass(gold, {"text": "    return a + b\n\ndef unrelated():\n    raise SystemExit(1)\n"})[0]
    assert not gold_metrics.code_pass(gold, {"text": "    return a - b\n"})[0]
    assert not gold_metrics.code_pass(gold, {"text": "    while True:\n        pass\n"})[0]  # the time limit


def test_top1_is_the_argmax_class_index():
    from trtmc_aiperf_qual import gold_metrics

    assert gold_metrics.top1(2, {"scores": [0.1, 0.3, 0.9]}) == (True, "2")
    assert gold_metrics.top1(1, {"scores": [0.1, 0.3, 0.9]})[0] is False
    assert gold_metrics.top1(1, {}) == (False, "")
    assert gold_metrics.top1(0, {"top_class": 0, "top_score": 9.9})[0]  # classify reports only its top class


def test_coco_map_scores_detections_in_the_model_label_space():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # qualification_tests (the family's AP)
    from trtmc_aiperf_qual import gold_metrics

    problems = [{"sample_id": "a", "gold": {"bbox": [[10, 10, 50, 50]], "category": [2]}}]
    exact = {0: {"boxes": [10, 10, 50, 50], "scores": [0.9], "class_ids": [2]}}
    by_id = {0: {"boxes": [10, 10, 50, 50], "scores": [0.9], "class_ids": [3]}}  # car: index 2, id 3
    assert gold_metrics.coco_map(gold_metrics.coco_units(problems, exact, {})) == pytest.approx(100.0)
    assert gold_metrics.coco_map(gold_metrics.coco_units(problems, by_id, {})) == 0.0
    assert gold_metrics.coco_map(gold_metrics.coco_units(problems, by_id, {"label_space": "coco-category-id"})) == \
        pytest.approx(100.0)
    result = gold_metrics.compare_corpus("coco_map", problems, exact, exact, {})
    assert result["delta_points"] == 0 and result["ci95"] is None


def test_forecasts_are_scored_on_the_point_or_median_quantile_forecast():
    from trtmc_aiperf_qual import gold_metrics

    assert gold_metrics.point_forecast({"values": [1.0, 2.0]}, 2) == [1.0, 2.0]
    quantiles = {"values": [0, 0, 1, 2, 9, 9], "shape": [1, 3, 2]}  # median row: [1, 2]
    assert gold_metrics.point_forecast(quantiles, 2) == [1.0, 2.0]
    units = gold_metrics.forecast_units([{"gold": [1.0, 4.0]}], {0: {"values": [1.0, 2.0]}})
    assert units == [(4.0, 2.0)] and gold_metrics.mse(units) == 2.0
    assert gold_metrics.forecast_units([{"gold": [1.0]}], {}) == [(1.0, 1.0)]  # a missing forecast scores 0


def test_translations_are_scored_with_corpus_chrf():
    pytest.importorskip("sacrebleu")
    from trtmc_aiperf_qual import gold_metrics

    problems = [{"gold": "Der Hund schläft."}, {"gold": "Die Katze spielt."}]
    perfect = {0: {"text": "Der Hund schläft."}, 1: {"text": "Die Katze spielt."}}
    wrong = {0: {"text": "Ein Vogel singt."}, 1: {"text": "Die Katze spielt."}}
    assert gold_metrics.chrf(gold_metrics.translation_units(problems, perfect)) == pytest.approx(100.0)
    result = gold_metrics.compare_corpus("chrf", problems, wrong, perfect)
    assert result["delta_points"] < -20 and result["higher_is_better"]


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
    item = {"suite": "edit-similarity", "metric": "precomputed_mean", "gate": {"max_delta_points": 1.0}}
    problems = [{"task": "edit", "gold": "make it red"}] * 4
    side = lambda values: {"observations": {"greedy": {i: {"value": v} for i, v in enumerate(values)}},  # noqa: E731
                           "exit": {}, "timings": {"greedy": {}}}
    entry = absolute.judge(item, problems, side([80, 82, 84, 86]), side([81, 83, 85, 87]))
    assert entry["metrics"]["delta_points"] == -1.0 and entry["status"] == "pass"


def test_failed_requests_are_missing_answers_not_wrong_ones(tmp_path):
    from types import SimpleNamespace

    from unittest.mock import patch

    raw = [{"metadata": {"session_num": 0}, "status": 200, "responses": []},
           {"metadata": {"session_num": 1}, "status": 422,
            "error": {"message": "TensorRT enqueue failed"}, "responses": []}]
    graded = [{"session_num": 0, "passed": True}, {"session_num": 1, "passed": False, "actual": ""}]
    run = SimpleNamespace(raw_records=lambda: raw, accuracy_records=lambda: graded, exit_code=1)
    item = {**ITEM, "suite": "mmlu-5shot"}
    model = {"candidate": {"max_sequence_length": 4096, "checkpoint": "m"}, "reference": {}}
    with patch.object(absolute, "run_aiperf", return_value=run):
        side_result = absolute.run_side({"hf_datasets_cache": str(tmp_path)}, {"url": "u"}, model, item, [{}, {}], tmp_path)
    assert list(side_result["records"]["greedy"]) == [0] and "enqueue failed" in side_result["failed"]["greedy"]
    native = {"records": {"greedy": {0: {"passed": True}, 1: {"passed": True}}}, "exit": {}}
    entry = absolute.judge(item, [{"task": "t", "gold": "A"}] * 2, side_result, native)
    assert entry["status"] == "error" and "enqueue failed" in entry["reasons"][0]


def test_a_capped_benchmark_length_builds_its_own_bundle(tmp_path):
    import shutil

    from trtmc_aiperf_qual.config import CONFIG_ROOT, Environment
    from trtmc_aiperf_qual.models import resolve_model

    root = tmp_path / "config"
    shutil.copytree(CONFIG_ROOT, root)
    environment = Environment({"repo": str(Path(__file__).resolve().parents[3])})
    qwen = resolve_model("qwen35-0.8b", environment, root=root)
    assert qwen["candidate"]["max_sequence_length"] == 2048
    assert qwen["candidate"]["bundle"].startswith("qwen35-0.8b-qual-2048/")


def test_the_native_side_falls_back_to_the_family_reference(tmp_path):
    from contextlib import contextmanager
    from unittest.mock import patch

    from trtmc_aiperf_qual.config import Environment

    started = []

    @contextmanager
    def serving_replicas(environment, model, backend, out, *, count, **kwargs):
        started.append((backend, kwargs["precision"], count))
        if backend == "reference":
            raise RuntimeError("Transformers does not recognize this architecture")
        yield {"url": "http://unused", "replicas": count}

    model = {"operation": "transcribe", "absolute": [{"suite": "s"}],
             "reference": {"backend": "reference", "fallback": "script", "perf_precision": "fp16", "precision": "fp32"}}
    with patch.object(absolute, "serving_replicas", serving_replicas), \
            patch.object(absolute, "run_side", return_value={"ok": 1}):
        native = absolute.run_native(Environment({"native_replicas": 4}), model, "python", {"s": []}, tmp_path)
    assert native["backend"] == "script" and native["runs"] == {"s": {"ok": 1}} and "recognize" in native["fallback_from"]
    # The generic adapter runs as several copies; a family script stays a single server.
    assert started[0] == ("reference", "fp16", 4) and started[-1][0] == "script" and native["replicas"] == 1


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
    item = {"suite": "s", "plugin": "p", "endpoint": "chat", "gate": {"max_delta_points": 1.0}}
    problems = [{"task": "t", "gold": "A"}, {"task": "t", "gold": "B"}]
    side = {"records": {"greedy": {0: {"passed": True, "unparsed": False}, 1: {"passed": False, "unparsed": False}}},
            "exit": {"greedy": 0},
            "timings": {"greedy": {0: {"model_call_ms": 10.0, "completion_tokens": 1},
                                   1: {"model_call_ms": 10.0, "completion_tokens": 1}}}}
    model = {"absolute": [item], "candidate": {}}
    native = {"backend": "reference", "precision": "fp16", "runs": {"s": side}, "replicas": 4}
    [entry] = absolute.entries(model, {"s": problems}, {"s": side}, native, None)
    assert entry["native"]["replicas"] == 4 and entry["workload_perf"]["light"] == "white"
    assert "not comparable" in entry["workload_perf"]["note"] and entry["status"] == "pass"


def test_a_long_benchmark_bundle_that_rejects_requests_is_rebuilt_shorter(tmp_path):
    from unittest.mock import patch

    from trtmc_aiperf_qual import bundles, runner

    model = {"catalog_profile": "m", "operation": "generate", "absolute": [{"suite": "mmlu-5shot"}],
             "candidate": {"bundle": "m-qual/m.bundle", "max_sequence_length": 4096,
                           "build": {"name": "m-qual", "max_sequence_length": 4096}}}
    probed = []

    def probe(environment, candidate, suite, out):
        probed.append(candidate["candidate"]["bundle"])
        if candidate["candidate"]["max_sequence_length"] > 2048:
            raise RuntimeError("probe rejected: TensorRT enqueue failed")

    with patch.object(runner, "_probe_candidate", probe), \
            patch.object(bundles, "ensure_bundle", return_value={"status": "built"}), \
            patch.object(runner.absolute, "plan", return_value=["replanned"]):
        shorter, plans = runner.serviceable_candidate({}, model, {"mmlu-5shot": []}, None, "python", tmp_path)
    assert probed == ["m-qual/m.bundle", "m-qual-2048/m.bundle"] and plans == {"mmlu-5shot": ["replanned"]}
    assert shorter["candidate"]["max_sequence_length"] == 2048 and "enqueue failed" in shorter["candidate"]["sequence_fallback"]
    assert model["candidate"]["max_sequence_length"] == 4096  # the configured model is not changed
    catalog = {**model, "candidate": {"bundle": "m/m.bundle", "max_sequence_length": 256, "build": {}}}
    with patch.object(runner, "_probe_candidate", probe), pytest.raises(RuntimeError):
        runner.serviceable_candidate({}, {**catalog, "candidate": {**catalog["candidate"], "max_sequence_length": 4096}},
                                     {}, None, "python", tmp_path)  # no benchmark build to shorten: an error


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


def test_a_benchmark_bundle_that_does_not_build_is_built_shorter(tmp_path, monkeypatch):
    import sys

    from trtmc_aiperf_qual import campaign
    from trtmc_aiperf_qual.config import Environment

    built, qualified = [], []

    def ensure_bundle(environment, model, python, out):
        built.append(model["candidate"]["bundle"])
        if model["candidate"]["max_sequence_length"] > 2048:
            return {"status": "failed", "reason": "OLMo2 batched-prefill engine build failed"}
        return {"status": "built", "bundle": model["candidate"]["bundle"]}

    monkeypatch.setattr(campaign, "reference_python", lambda environment, model: sys.executable)
    monkeypatch.setattr(campaign, "prefetch", lambda environment, model: None)
    monkeypatch.setattr(campaign.bundles, "ensure_bundle", ensure_bundle)
    monkeypatch.setattr(campaign, "qualify", lambda model, environment, out: qualified.append(model) or {
        "verdict": {"category": "pass"}})
    model = {"model": "m", "catalog_profile": "m", "task": "text_generation", "absolute": [{"suite": "mmlu-5shot"}],
             "reference": {}, "candidate": {"bundle": "m-qual/m.bundle", "max_sequence_length": 4096,
                                            "build": {"name": "m-qual", "max_sequence_length": 4096}}}
    record = campaign.run_one(Environment({}), model, tmp_path / "m")
    assert record["category"] == "pass" and built == ["m-qual/m.bundle", "m-qual-2048/m.bundle"]
    assert qualified[0]["candidate"]["max_sequence_length"] == 2048
    assert "did not build" in qualified[0]["candidate"]["sequence_fallback"]


def test_a_native_model_with_no_right_answer_is_not_comparable():
    metrics = {"trtmc_accuracy": 0.0, "native_accuracy": 0.0, "delta_points": 0.0}
    status, reasons = absolute.status(metrics, {"max_delta_points": 1.0}, expected=10, paired=10)
    assert status == "not-comparable" and "does not fit" in reasons[0]
    chance = {"trtmc_accuracy": 0.53, "native_accuracy": 0.53, "delta_points": 0.0}  # gpt-oss on 5-shot MMLU
    assert absolute.status(chance, {"max_delta_points": 1.0, "min_native": 12.5}, expected=10, paired=10)[0] == \
        "not-comparable"
    assert absolute.status({**chance, "native_accuracy": 44.0, "trtmc_accuracy": 44.0},
                           {"max_delta_points": 1.0, "min_native": 12.5}, expected=10, paired=10)[0] == "pass"
    wer = {"trtmc_score": 0.0, "native_score": 0.0, "delta_points": 0.0}  # a corpus metric: 0 is a score
    assert absolute.status(wer, {"max_delta_points": 0.2}, expected=10, paired=10)[0] == "pass"


def test_unmerged_raw_records_still_show_failed_requests(tmp_path):
    import json

    from trtmc_aiperf_qual.aiperf_runner import AiperfRun

    (tmp_path / "raw_records").mkdir()
    record = {"metadata": {"benchmark_phase": "profiling", "session_num": 0}, "status": 400,
              "error": {"message": "multi-message chat requires a chat-template renderer"}}
    (tmp_path / "raw_records" / "raw_records_processor_a.jsonl").write_text(json.dumps(record) + "\n")
    records = AiperfRun(tmp_path, 1, []).raw_records()  # AIPerf merged no export: every request failed
    assert len(records) == 1 and absolute.unanswered(records[0])


def test_a_family_script_native_answers_at_most_the_script_limit(tmp_path):
    from contextlib import contextmanager
    from unittest.mock import patch

    from trtmc_aiperf_qual.config import Environment

    @contextmanager
    def serving_replicas(environment, model, backend, out, *, count, **kwargs):
        if backend == "reference":
            raise RuntimeError("NemotronHForCausalLM.__init__() got an unexpected keyword argument 'dtype'")
        yield {"url": "http://unused", "replicas": count}

    mmlu = [{"task": f"subject{index % 57}", "gold": "A"} for index in range(1140)]
    speech = [{"task": "librispeech", "gold": "x"} for _ in range(2620)]
    small = [{"task": "s", "gold": "y"} for _ in range(100)]
    model = {"operation": "generate", "absolute": [{"suite": "mmlu-5shot", "per_task": 20},
                                                  {"suite": "librispeech", "metric": "wer"},
                                                  {"suite": "small", "metric": "wer"}],
             "reference": {"backend": "reference", "fallback": "script", "perf_precision": "fp16", "precision": "fp32"}}
    asked = {}

    def run_side(environment, service, model, item, problems, out):
        asked[item["suite"]] = (dict(item), len(problems))
        return {"ok": 1}

    replanned = lambda environment, model, item: mmlu[: 57 * item["per_task"]]  # noqa: E731
    with patch.object(absolute, "serving_replicas", serving_replicas), patch.object(absolute, "run_side", run_side), \
            patch.object(absolute, "plan", replanned):
        native = absolute.run_native(Environment({"native_replicas": 4}), model, "python",
                                     {"mmlu-5shot": mmlu, "librispeech": speech, "small": small}, tmp_path)
    assert native["backend"] == "script"
    assert asked["mmlu-5shot"] == ({"suite": "mmlu-5shot", "per_task": 5}, 285)  # every subject, 5 each
    assert asked["librispeech"][1] == 300 and asked["small"] == ({"suite": "small", "metric": "wer"}, 100)
    # The TRTMC side answers the same benchmarks and problems.
    assert [len(native["plans"][name]) for name in ("mmlu-5shot", "librispeech", "small")] == [285, 300, 100]
    assert native["absolute"][0]["per_task"] == 5
