# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from trtmc_aiperf_plugins.accuracy import compare_token_exact, compare_top1, compare_vector
from trtmc_aiperf_qual import judge
from trtmc_aiperf_qual.config import ConfigError
from trtmc_aiperf_qual.services import platform_id
from trtmc_aiperf_qual.suites import Suite, request_sha, select

RECORDS = [{"id": str(i), "task": "a" if i < 5 else "b"} for i in range(10)]


@pytest.mark.parametrize("selection, expected", [
    ({"method": "first", "count": 3}, ["0", "1", "2"]),
    ({"method": "stride", "count": 3}, ["0", "3", "6"]),
    ({"method": "indices", "indices": [9, 2]}, ["9", "2"]),
    ({"method": "first", "count": 2, "per_task": True}, ["0", "1", "5", "6"]),
])
def test_selection_is_deterministic(selection, expected):
    assert [r["id"] for r in select(RECORDS, selection)] == expected


def test_selection_rejects_out_of_range_indices():
    with pytest.raises(ConfigError):
        select(RECORDS, {"method": "indices", "indices": [10]})


def _suite(prompts):
    samples = [{"sample_id": str(i), "task": "t", "request": {"prompt": p}, "request_sha": request_sha({"prompt": p})}
               for i, p in enumerate(prompts)]
    return Suite("s", "key-" + "".join(prompts), samples, {})


GB300 = {"gpu_arch": "sm100", "packages": {"torch": "2.12.0"}, "cuda": "13.0"}


def test_platform_id_is_readable_and_stable():
    assert platform_id(GB300).startswith("sm100-") and platform_id(GB300) == platform_id(dict(GB300))
    assert platform_id(GB300) != platform_id({**GB300, "cuda": "13.1"})


def test_token_exact_reports_first_divergence():
    ok, reason, _, _ = compare_token_exact({"token_ids": [1, 2, 3], "text": "x"}, {"token_ids": [1, 9, 3], "text": "y"})
    assert not ok and "token 1" in reason
    assert compare_token_exact({"token_ids": [1], "text": ""}, {"token_ids": [1], "text": ""})[0]


def test_top1_and_vector_comparators():
    assert compare_top1({"scores": [0.1, 0.9]}, {"scores": [0.2, 0.8]})[0]
    assert not compare_top1({"scores": [0.9, 0.1]}, {"scores": [0.2, 0.8]})[0]
    assert compare_vector({"values": [1, 0]}, {"values": [1, 0.01]})[0]
    assert not compare_vector({"values": [1, 0]}, {"values": [0, 1]})[0]


@pytest.mark.parametrize("candidate, reference, expected", [(90, 100, "green"), (97, 100, "yellow"), (110, 100, "red")])
def test_light_uses_perf_matrix_rule(candidate, reference, expected):
    assert judge.light(candidate, reference, 5) == expected


def test_perf_white_when_unstable_or_outputs_differ():
    stable = {"p50_ms": 10.0, "ci_percent": 1.0}
    assert judge.judge_performance(stable, stable, margin_percent=5, max_ci_percent=5, outputs_match=True,
                                   output_reason="")["light"] == "yellow"
    unstable = judge.judge_performance({"p50_ms": 10.0, "ci_percent": 9.0}, stable, margin_percent=5,
                                       max_ci_percent=5, outputs_match=True, output_reason="")
    assert unstable["light"] == "white" and "within the interval" in unstable["reasons"][0]
    decisive = judge.judge_performance(stable, {"p50_ms": 40.0, "ci_percent": 9.0}, margin_percent=5,
                                       max_ci_percent=5, outputs_match=True, output_reason="")
    assert decisive["light"] == "green" and "unchanged" in decisive["notes"][0]
    assert judge.judge_performance(stable, stable, margin_percent=5, max_ci_percent=5, outputs_match=False,
                                   output_reason="differs")["light"] == "white"


def test_across_runs_confidence_interval():
    stats = judge.across_runs([10.0, 10.2, 9.8])
    assert stats["runs"] == 3 and abs(stats["p50_ms"] - 10.0) < 1e-9
    assert 4.9 < stats["ci_percent"] < 5.0  # t(2)=4.303 * 0.2/sqrt(3) / 10
    assert judge.across_runs([None])["p50_ms"] is None
    assert judge.across_runs([5.0])["ci_percent"] is None


def test_best_aggregation_uses_fastest_run_and_does_not_gate_stability():
    best = judge.across_runs([30.7, 28.4, 30.5], "best")
    assert best["p50_ms"] == 28.4 and best["ci_percent"] > 5
    candidate = judge.across_runs([5.74, 5.73, 5.74])
    verdict = judge.judge_performance(candidate, best, margin_percent=5, max_ci_percent=5, outputs_match=True,
                                      output_reason="")
    assert verdict["light"] == "green" and not verdict["reasons"]
    with pytest.raises(ValueError):
        judge.across_runs([1.0], "median")


def test_answer_line_compares_only_the_answer():
    from trtmc_aiperf_plugins.accuracy import compare_answer_line

    assert compare_answer_line({"text": " D\n\nQuestion: x"}, {"text": " D\n\nQuestion: y"})[0]
    assert not compare_answer_line({"text": " C\n"}, {"text": " D\n"})[0]


def test_wer_and_edit_distance_comparators():
    from trtmc_aiperf_plugins.accuracy import compare_edit_distance, compare_wer, word_error_rate

    assert word_error_rate("the cat sat", "The cat, sat.") == 0.0
    assert abs(word_error_rate("the cat", "the cat sat") - 1 / 3) < 1e-9
    assert compare_wer({"text": "a b c d e f g h i j"}, {"text": "a b c d e f g h i x"}, max_wer=0.1)[0]
    assert not compare_wer({"text": "a b"}, {"text": "a c"}, max_wer=0.1)[0]
    assert compare_edit_distance({"text": "Red car"}, {"text": "red  car"})[0]
    assert not compare_edit_distance({"text": "blue"}, {"text": "red"}, max_distance=0.15)[0]


def test_box_parity_matches_same_class_by_iou():
    from trtmc_aiperf_plugins.accuracy import compare_boxes

    reference = {"boxes": [0, 0, 10, 10, 20, 20, 30, 30], "class_ids": [1, 2]}
    assert compare_boxes({"boxes": [1, 1, 10, 10, 20, 20, 30, 31], "class_ids": [1, 2]}, reference)[0]
    assert not compare_boxes({"boxes": [0, 0, 10, 10, 20, 20, 30, 30], "class_ids": [1, 3]}, reference)[0]
    assert not compare_boxes({"boxes": [0, 0, 10, 10], "class_ids": [1]}, reference)[0]  # recall 0.5
    assert compare_boxes({"boxes": [], "class_ids": []}, {"boxes": [], "class_ids": []})[0]


def test_mask_parity_semantic_and_binary():
    from trtmc_aiperf_plugins.accuracy import compare_mask

    semantic = {"height": 2, "width": 50, "mask": [1] * 50 + [2] * 50}
    near = {"height": 2, "width": 50, "mask": [1] * 50 + [2] * 49 + [1]}
    ok, reason, _, _ = compare_mask(near, semantic, min_pixel_accuracy=0.99, min_mean_iou=0.94)
    assert ok and "pixel accuracy 0.9900" in reason
    binary = {"height": 1, "width": 4, "num_masks": 2, "masks": [1, 1, 0, 0, 0, 1, 1, 0]}
    assert compare_mask(binary, binary, min_mask_iou=0.7)[0]
    shifted = {**binary, "masks": [1, 1, 0, 0, 0, 0, 1, 1]}
    assert not compare_mask(shifted, binary, min_mask_iou=0.7)[0]
    assert not compare_mask({**binary, "height": 2}, binary)[0]


def test_json_manifest_jsonl_explode_and_file_fields(tmp_path):
    import hashlib

    from trtmc_aiperf_qual.config import Environment
    from trtmc_aiperf_qual.suites import _json_manifest_records

    root = tmp_path / "data" / "set"
    root.mkdir(parents=True)
    manifest = root / "m.jsonl"
    manifest.write_text(json.dumps({"a": "x", "b": "y", "img": "i.png"}) + "\n")
    env = Environment({"data_root": str(tmp_path / "data")})
    source = {"path": "set/m.jsonl", "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "file_fields": ["img"], "explode": ["a", "b"]}
    records = _json_manifest_records(source, env)
    assert [(r["id"], r["text"]) for r in records] == [("0:a", "x"), ("0:b", "y")]
    assert records[0]["img"] == str(root / "i.png")
    with pytest.raises(ConfigError):
        _json_manifest_records({**source, "sha256": "0" * 64}, env)


REPOSITORY = __import__("pathlib").Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("accuracy, lights, category", [
    ([{"status": "pass"}], {"eager": "green", "compile": "green"}, "pass"),
    ([{"status": "pass"}], {"eager": "green", "compile": "red"}, "perf-issue"),
    ([{"status": "fail", "isolated_check": {"status": "pass"}}], {"eager": "green", "compile": "green"},
     "acc-session-state"),
    ([{"status": "fail"}], {"eager": "green", "compile": "green"}, "acc-issue"),
    ([], {"eager": "green", "compile": "green"}, "error"),
    ([{"status": "pass"}], {"eager": "white", "compile": "green"}, "perf-inconclusive"),
    ([{"status": "pass"}], {"eager": "green", "compile": "n/a"}, "pass"),
    ([{"status": "pass"}], {"eager": "n/a", "compile": "n/a"}, "error"),
    ([{"status": "not-comparable"}], {"eager": "green", "compile": "green"}, "not-comparable"),
    ([{"status": "inconclusive"}], {"eager": "green", "compile": "green"}, "acc-inconclusive"),
])
def test_verdict_categories(accuracy, lights, category):
    result = {"accuracy": [{"suite": "s", **item} for item in accuracy],
              "performance_l1": [{"reference_mode": mode, "light": light} for mode, light in lights.items()]}
    assert judge.verdict(result, expected_suites=["s"], expected_modes=2)["category"] == category


def test_numeric_parity_matches_shapes_exactly_and_vectors_by_cosine():
    from trtmc_aiperf_plugins.accuracy import compare_numeric

    reference = {"shape": [1, 3], "values": [1.0, 2.0, 3.0], "runtime_ms": 5.0, "note": "x"}
    assert compare_numeric({"shape": [1, 3], "values": [1.0, 2.0, 3.001], "runtime_ms": 99.0}, reference)[0]
    assert not compare_numeric({"shape": [3, 1], "values": [1.0, 2.0, 3.0]}, reference)[0]
    assert not compare_numeric({"shape": [1, 3], "values": [3.0, 2.0, -1.0]}, reference)[0]
    assert not compare_numeric({"valid_pixels": 188255}, {"valid_pixels": 188139})[0]
    assert compare_numeric({"valid_pixels": 188255}, {"valid_pixels": 188139}, count_rtol=0.01)[0]
    assert not compare_numeric({"shape": [1, 3]}, {"shape": [1, 4]}, count_rtol=0.5)[0]
    with pytest.raises(ValueError):
        compare_numeric({"text": "a"}, {"text": "b"})


def test_scores_image_and_audio_parity():
    from trtmc_aiperf_plugins.accuracy import compare_audio, compare_image, compare_scores
    from trtmc_perf_serving.digests import audio_digest, image_digest
    import numpy as np

    assert compare_scores({"scores": [0.9, 0.1, 0.5]}, {"scores": [0.8, 0.2, 0.4]})[0]
    assert not compare_scores({"scores": [0.1, 0.9]}, {"scores": [0.9, 0.1]})[0]
    frame = np.tile(np.linspace(0, 255, 128, dtype=np.uint8)[None, :, None], (96, 1, 3))
    same = image_digest([frame])
    assert compare_image({"media_digest": same}, {"media_digest": image_digest([frame.astype(np.float32) / 255.0])})[0]
    assert not compare_image({"media_digest": image_digest([frame[:, :64]])}, {"media_digest": same})[0]
    tone = np.sin(np.arange(16000) / 16000 * 2 * np.pi * 440).astype(np.float32)
    assert compare_audio({"audio_digest": audio_digest(tone, 16000)}, {"audio_digest": audio_digest(tone * 0.9, 16000)})[0]
    assert not compare_audio({"audio_digest": audio_digest(tone[:4000], 16000)}, {"audio_digest": audio_digest(tone, 16000)})[0]


def test_models_are_derived_from_the_catalog_the_family_cases_and_task_defaults():
    from trtmc_aiperf_qual.config import Environment
    from trtmc_aiperf_qual.models import resolve_model

    environment = Environment({"repo": str(REPOSITORY)})
    qwen = resolve_model("qwen3-0.6b-fp16", environment)
    # Gold-scored MMLU replaces the parity cases; its prompts need a 4096-token bundle.
    assert qwen["accuracy_source"] == "absolute" and not qwen["family_accuracy"]
    assert [(item["suite"], item["endpoint"]) for item in qwen["absolute"]] == [("mmlu-5shot", "chat")]
    assert qwen["candidate"]["bundle"] == "qwen3-0.6b-fp16-qual/qwen3-0.6b-fp16.bundle"
    assert qwen["candidate"]["build"]["max_sequence_length"] == 4096
    assert qwen["performance"]["l1"]["suite"]["source"] == {"kind": "qualification_perf", "profile": "qwen3-0.6b-fp16"}
    small = resolve_model("falcon-rw-1b", environment)  # MMLU is near chance: LAMBADA on the catalog length
    assert [item["suite"] for item in small["absolute"]] == ["lambada"] and small["candidate"]["max_sequence_length"] == 256
    detr = resolve_model("detr-resnet-50", environment)
    assert detr["candidate"]["build"]["image_height"] == 1333 and not detr["family_accuracy"]
    coco, = detr["absolute"]  # COCO mAP in DETR's own class numbering at its family score threshold
    assert coco["metric_params"] == {"label_space": "coco-category-id"}
    assert coco["suite_definition"]["request"] == {"score_threshold": 0.5}
    assert (detr["reference"]["backend"], detr["reference"]["fallback"]) == ("reference", "script")
    stories = resolve_model("mixtral-stories-15m", environment)
    assert [item["suite"] for item in stories["absolute"]] == ["tinystories"]
    import shutil
    import tempfile

    from trtmc_aiperf_qual.config import CONFIG_ROOT

    with tempfile.TemporaryDirectory() as directory:  # `absolute: []` and no family case: no accuracy scheme
        root = Path(directory) / "config"
        shutil.copytree(CONFIG_ROOT, root)
        (root / "models/mixtral-stories-15m.yaml").write_text("absolute: []\n")
        with pytest.raises(ConfigError, match="no accuracy scheme"):
            resolve_model("mixtral-stories-15m", environment, root=root)
        (root / "models/mixtral-stories-15m.yaml").write_text("accuracy_source: none\n")
        with pytest.raises(ConfigError, match="accuracy_note"):
            resolve_model("mixtral-stories-15m", environment, root=root)
    random = resolve_model("qwen3-moe-tiny-random", environment)  # random weights: Perf only
    assert random["accuracy_source"] == "none" and not random["absolute"] and "random" in random["accuracy_note"]
    assert random["performance"]["l1"]["suite"]["source"]["kind"] in ("catalog_testcase", "qualification_perf")
    edit = resolve_model("qwen-image-edit-2511", environment)["reference"]  # Diffusers edit, family fallback
    assert (edit["backend"], edit["fallback"]) == ("reference", "script")


def test_top1_near_tie_tolerance_is_opt_in():
    from trtmc_aiperf_plugins.accuracy import compare_top1

    candidate, reference = {"scores": [5.0, 5.001, 1.0]}, {"scores": [5.001, 5.0, 1.0]}
    assert not compare_top1(candidate, reference)[0]
    assert compare_top1(candidate, reference, tie_cosine=0.999)[0]
    assert not compare_top1({"scores": [0.0, 9.0, 0.0]}, {"scores": [9.0, 0.0, 0.0]}, tie_cosine=0.999)[0]
    assert compare_top1({"top_class": 2, "top_score": 1.0}, {"scores": [0.0, 1.0, 3.0]})[0]


def test_limit_suite_keeps_the_first_samples_under_a_new_key():
    from trtmc_aiperf_qual.suites import limit_suite

    suite = _suite([str(i) for i in range(20)])
    limited = limit_suite(suite, 10)
    assert [s["sample_id"] for s in limited.samples] == [str(i) for i in range(10)]
    assert limited.key != suite.key and limited.manifest["limited_to"] == 10
    assert limit_suite(suite, 50) is suite


def test_box_parity_accepts_nested_boxes():
    from trtmc_aiperf_plugins.accuracy import compare_boxes

    flat = {"boxes": [0.0, 0.0, 10.0, 10.0], "class_ids": [3], "scores": [0.9]}
    nested = {"boxes": [[0.2, 0.0, 10.0, 10.1]], "class_ids": [3], "scores": [0.8]}
    assert compare_boxes(flat, nested)[0]


def test_token_prefix_tolerance_is_opt_in():
    from trtmc_aiperf_plugins.accuracy import compare_token_exact

    candidate, reference = {"token_ids": list(range(20)) + [1]}, {"token_ids": list(range(20)) + [2]}
    assert not compare_token_exact(candidate, reference)[0]
    assert compare_token_exact(candidate, reference, min_prefix=8)[0]
    assert not compare_token_exact({"token_ids": [9, 1, 2]}, {"token_ids": [0, 1, 2]}, min_prefix=8)[0]


def test_equal_text_with_different_token_ids_is_opt_in():
    from trtmc_aiperf_plugins.accuracy import compare_token_exact

    candidate = {"token_ids": [1576, 10150], "text": "The largest"}
    reference = {"token_ids": [450, 10150], "text": "The largest"}
    assert not compare_token_exact(candidate, reference, min_prefix=8)[0]
    assert compare_token_exact(candidate, reference, min_prefix=8, accept_equal_text=True)[0]


def test_answer_line_accepts_identical_tokens_with_different_rendering():
    from trtmc_aiperf_plugins.accuracy import compare_answer_line

    candidate = {"text": "<extra_id_0> B", "token_ids": [32099, 272]}
    reference = {"text": "B", "token_ids": [32099, 272]}
    ok, reason, _, _ = compare_answer_line(candidate, reference)
    assert ok and "rendering" in reason
    assert not compare_answer_line({"text": "C", "token_ids": [1]}, {"text": "B", "token_ids": [2]})[0]


def test_perf_output_check_falls_back_to_the_eager_reference():
    from trtmc_aiperf_qual.runner import output_check

    l1 = {"output_grader": "parity_token_exact"}
    candidate = {"token_ids": [1, 2, 3]}
    references = {"eager": {"token_ids": [1, 2, 3]}, "compile": {"token_ids": [1, 9, 9]}}
    match, reason = output_check(l1, candidate, references, "compile")
    assert match and "eager reference" in reason
    assert not output_check(l1, candidate, {"eager": {"token_ids": [7]}, "compile": {"token_ids": [8]}}, "compile")[0]
    assert not output_check(l1, candidate, {"eager": {"token_ids": [7]}}, "eager")[0]


def test_perf_only_models_and_informational_entries():
    perf = [{"reference_mode": "eager", "light": "green"}]
    perf_only = judge.verdict({"accuracy_source": "none", "accuracy": [], "performance_l1": perf},
                              expected_suites=[], expected_modes=1)
    assert (perf_only["acc"], perf_only["category"]) == ("n/a", "pass")
    # Informational entries (pixel parity, a retired check) are reported, never judged.
    informational = judge.verdict({"accuracy": [{"suite": "geneval", "status": "pass"},
                                                {"suite": "replay-parity", "status": "fail", "informational": True}],
                                   "performance_l1": perf}, expected_suites=["geneval"], expected_modes=1)
    assert (informational["acc"], informational["category"]) == ("pass", "pass")


def test_errors_in_a_suite_make_the_accuracy_an_error():
    verdict = judge.verdict({"accuracy": [{"suite": "a", "status": "pass"}, {"suite": "b", "status": "error"}],
                             "performance_l1": [{"reference_mode": "eager", "light": "green"}]},
                            expected_suites=["a", "b"], expected_modes=1)
    assert (verdict["acc"], verdict["category"]) == ("error", "error")


def test_wer_ignores_markup_tags():
    from trtmc_aiperf_plugins.accuracy import word_error_rate

    assert word_error_rate("Concord returned to its place.", "Concord returned to its place. <en-US>") == 0.0


def test_sampled_perf_requests_compare_generated_length():
    from trtmc_aiperf_qual.runner import output_check, sampled_request

    assert sampled_request({"temperature": 0.7, "top_k": 50}) and not sampled_request({"temperature": 1.0, "top_k": 1})
    l1 = {"output_grader": "parity_token_exact"}
    refs = {"eager": {"token_ids": [5, 6, 7]}}
    assert output_check(l1, {"token_ids": [1, 2, 3]}, refs, "eager", sampled=True)[0]
    assert not output_check(l1, {"token_ids": [1, 2]}, refs, "eager", sampled=True)[0]


def test_family_cases_with_top_k_one_are_greedy_and_judged_strictly():
    from types import SimpleNamespace

    from trtmc_aiperf_qual import family

    def case(request):
        return SimpleNamespace(values={"request": request})
    assert not family.sampled(case({"temperature": 1.0, "top_k": 1}))  # Qwen2.5-VL's family request
    assert not family.sampled(case({"do_sample": True, "top_k": 1}))
    assert family.sampled(case({"do_sample": True})) and family.sampled(case({"temperature": 0.7}))


def _environment(tmp_path=None):
    from trtmc_aiperf_qual.config import Environment

    return Environment({"repo": str(REPOSITORY), **({"data_root": str(tmp_path)} if tmp_path else {})})


def _manifest(path, text):
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def test_tsv_manifests_and_nested_file_fields(tmp_path):
    from trtmc_aiperf_qual.suites import build_suite

    digest = _manifest(tmp_path / "parti/p.tsv", "Prompt\tCategory\na red cube\tBasic\na blue ball\tBasic\n")
    suite = build_suite({"suite": "p", "version": 1, "selection": {"method": "first", "count": 2},
                         "source": {"kind": "json_manifest", "path": "parti/p.tsv", "sha256": digest},
                         "fields": {"Prompt": "prompt"}}, _environment(tmp_path))
    assert [sample["request"]["prompt"] for sample in suite.samples] == ["a red cube", "a blue ball"]

    (tmp_path / "ocr/images").mkdir(parents=True)
    (tmp_path / "ocr/images/a.jpg").write_bytes(b"jpg")
    record = {"id": "a", "question": "What text?", "media": [{"type": "image", "path": "images/a.jpg"}]}
    digest = _manifest(tmp_path / "ocr/d.json", json.dumps({"samples": [record]}))
    suite = build_suite({"suite": "o", "version": 1, "selection": {"method": "first", "count": 1},
                         "source": {"kind": "json_manifest", "path": "ocr/d.json", "sha256": digest,
                                    "records": "samples", "file_fields": ["media.0.path"]},
                         "fields": {"media.0.path": "image_path", "question": "prompt"}}, _environment(tmp_path))
    request = suite.samples[0]["request"]
    assert request["prompt"] == "What text?" and request["image_path"]["$file"]["suffix"] == ".jpg"


def test_etth1_windows_follow_the_declared_window(tmp_path):
    from trtmc_aiperf_qual.suites import build_suite

    rows = "\n".join(f"2016-07-01 {i:05d},{i}.0,{-i}.0" for i in range(400))
    digest = _manifest(tmp_path / "ETTh1/ETTh1.csv", "date,HUFL,OT\n" + rows + "\n")
    window = {"columns": ["HUFL", "OT"], "context_length": 8, "stride": 4, "test_target_start": 300,
              "test_end": 380, "frequency": 0}
    definition = {"suite": "etth1", "version": 1, "selection": {"method": "first", "count": 3},
                  "source": {"kind": "etth1_windows", "path": "ETTh1/ETTh1.csv", "sha256": digest,
                             "seed": 20260715, "window": window}}
    first = build_suite(definition, _environment(tmp_path))
    assert first.samples == build_suite(definition, _environment(tmp_path)).samples  # seeded
    request = first.samples[0]["request"]
    assert len(request["past_values"]) == 16 and request["observed_mask"] == [1.0] * 16 and request["frequency"] == 0
    start = int(request["past_values"][0])
    assert request["past_values"][:4] == [start, -start, start + 1, -(start + 1)]  # row-major [time, channel]
    assert 300 - 8 <= start <= 380 - 8


def test_suites_can_take_the_catalog_request_as_their_base(tmp_path, monkeypatch):
    from trtmc_aiperf_qual import suites

    monkeypatch.setattr(suites, "_catalog_testcase_records", lambda source, environment: [
        {"id": "catalog", "request": {"prompt": "a cat", "num_steps": 28, "seed": 42}}])
    suite = suites.build_suite({"suite": "p", "version": 1, "base_profile": "flux-2-dev",
                                "selection": {"method": "first", "count": 2},
                                "source": {"kind": "inline", "records": [{"id": "1", "p": "a red cube"},
                                                                         {"id": "2", "p": "a blue ball"}]},
                                "fields": {"p": "prompt"}}, _environment())
    assert [sample["request"] for sample in suite.samples] == [
        {"prompt": "a red cube", "num_steps": 28, "seed": 42}, {"prompt": "a blue ball", "num_steps": 28, "seed": 42}]


def test_the_etth1_benchmark_follows_the_family_window():
    from trtmc_aiperf_qual.models import resolve_model

    timesfm = resolve_model("timesfm-2.0-500m-official", _environment())
    # Rolling ETTh1 windows in the family case's columns, context, and horizon, every hour.
    window = timesfm["absolute"][0]["suite_definition"]["source"]["window"]
    assert (window["context_length"], window["prediction_length"], window["stride"]) == (2048, 128, 1)


def test_quantized_candidates_get_the_benchmarks_quantization_gate():
    from trtmc_aiperf_qual.models import resolve_model

    fp8 = resolve_model("qwen3-0.6b-fp8", _environment())
    assert fp8["candidate"]["quantization"] == "fp8"
    assert [(item["suite"], item["gate"]) for item in fp8["absolute"]] == [("mmlu-5shot", {"max_delta_points": 2.0})]


def test_timing_reference_precision_can_differ_from_the_candidate_precision():
    from trtmc_aiperf_qual.models import resolve_model
    from trtmc_aiperf_qual.runner import timing_precisions

    z_image = resolve_model("z-image-turbo", _environment())["reference"]
    assert z_image["perf_precision"] == "fp16" and timing_precisions(z_image) == ["bf16"]
    assert timing_precisions({"perf_precision": "fp16", "precision": "fp32"}) == ["fp16", "fp32"]
    assert timing_precisions({"perf_precision": "fp16", "precision": "fp32", "declared_precision": "bf16"}) == [
        "fp16", "bf16", "fp32"]


def test_always_sampling_tts_models_compare_duration_in_perf():
    from trtmc_aiperf_qual.models import resolve_model

    for profile in ("bark-small", "bark-large", "magpie-tts-357m"):
        model = resolve_model(profile, _environment())
        assert model["family_accuracy"]  # the family's own case judges Acc
        assert model["performance"]["l1"]["output_grader_params"]["max_log_spectral_distance"] > 100


def test_a_missing_mandatory_result_is_an_error_even_when_other_checks_add_rows():
    from trtmc_aiperf_qual.runner import expected_suites, informational_suites, missing_results

    model = {"family": "flux", "family_accuracy": ["generated-media-parity"],
             "supplementary": [{"check": "replay_parity", "latent_replay_families": ["flux"]},
                               {"check": "geneval", "only_families": ["flux"]}]}
    assert expected_suites(model) == {"generated-media-parity": "family_accuracy", "replay-parity": "replay_parity",
                                      "geneval": "geneval"}
    produced = [{"suite": "geneval", "status": "pass"}, {"suite": "replay-parity", "status": "pass"}]
    errors = {"family_accuracy": "RuntimeError: startup failed"}
    missing = missing_results(model, produced, errors)
    assert [(m["suite"], m["status"]) for m in missing] == [("generated-media-parity", "error")]
    assert "startup failed" in missing[0]["error"]
    lights = [{"reference_mode": "eager", "light": "green"}]
    assert judge.verdict({"accuracy": produced, "performance_l1": lights}, expected_suites=list(expected_suites(model)),
                         expected_modes=1)["category"] == "error"
    assert judge.verdict({"accuracy": produced + missing, "performance_l1": lights},
                         expected_suites=list(expected_suites(model)), expected_modes=1)["acc"] == "error"
    other = {**model, "family": "minimax"}  # no caller latents (and not a GenEval family): neither expected
    assert set(expected_suites(other)) == {"generated-media-parity"}
    # Generated media: the family's media parity and the pixel parity are reported only; GenEval decides.
    informational = {**model, "family_informational": True,
                     "supplementary": [{**model["supplementary"][0], "informational": True}, model["supplementary"][1]]}
    assert expected_suites(informational) == {"geneval": "geneval"}
    assert informational_suites(informational) == {"generated-media-parity", "replay-parity"}


def test_incomplete_perf_runs_and_missing_exports_are_errors_not_lights():
    from types import SimpleNamespace

    from trtmc_aiperf_qual.runner import run_completeness

    ok = {"status": 200, "metadata": {}}
    partial = SimpleNamespace(raw_records=lambda: [ok] + [{"status": 500, "metadata": {}}] * 19)
    assert run_completeness(partial, 20) == "1 of 20 requests succeeded (failed with status 500)"
    assert run_completeness(SimpleNamespace(raw_records=lambda: [ok] * 20), 20) is None
    assert "19 of 20" in run_completeness(SimpleNamespace(raw_records=lambda: [ok] * 19), 20)
    fast = {"p50_ms": 1.0, "ci_percent": 0.0, "aggregation": "mean"}
    slow = {"p50_ms": 2.0, "ci_percent": 0.0, "aggregation": "mean"}
    common = dict(margin_percent=5, max_ci_percent=5, outputs_match=True, output_reason="identical")
    broken = judge.judge_performance({**fast, "incomplete": "run_01: 1 of 20 requests succeeded"}, slow, **common)
    assert broken["light"] == "error" and "1 of 20" in broken["reasons"][0]
    assert judge.judge_performance({"p50_ms": None}, slow, **common)["light"] == "error"
    late = judge.judge_performance({**fast, "exit_note": "AIPerf exited 1 after all requests succeeded"}, slow, **common)
    assert late["light"] == "green" and "exited 1" in late["notes"][0]
    result = {"accuracy": [{"suite": "s", "status": "pass"}],
              "performance_l1": [{"reference_mode": "eager", "light": broken["light"]}]}
    assert judge.verdict(result, expected_suites=["s"], expected_modes=1)["category"] == "error"


def test_one_pinned_checkpoint_serves_the_bundle_and_the_reference():
    from trtmc_aiperf_qual.config import Environment
    from trtmc_aiperf_qual.models import resolve_model

    environment = Environment({"repo": str(REPOSITORY)})
    albert = resolve_model("albert-base", environment)  # the catalog does not pin it; the family does
    revision = albert["candidate"]["revision"]
    assert revision and revision.startswith("8e2f239c5f8a")
    assert albert["candidate"]["build"]["hf_revision"] == revision
    assert albert["candidate"]["bundle"].startswith("albert-base-qual/")
    assert albert["reference"]["revision"] == revision and "model" not in albert["reference"]
    k2 = resolve_model("k2-horizon-7b-uno", environment)  # the family's native reference is the base model
    assert (k2["reference"]["model"], k2["reference"]["revision"]) == (
        "IFM/K2-Horizon-7B", "586b03f0fd1fbbf2f13eeafc33749e95ae34dd10")


def test_rejudging_reports_results_the_configuration_no_longer_asks_for(tmp_path, monkeypatch):
    import json

    from trtmc_aiperf_qual import cli

    model = {"catalog_profile": "demo", "task": "text_generation", "family_accuracy": [], "supplementary": [],
             "accuracy_source": "absolute", "absolute": [{"suite": "mmlu-5shot", "gate": {"max_delta_points": 1.0}}],
             "performance": {"l1": {"reference_modes": ["eager"]}}}
    perf = {"reference_mode": "eager", **judge.judge_performance(
        {"p50_ms": 1.0, "ci_percent": 0.0}, {"p50_ms": 2.0, "ci_percent": 0.0}, margin_percent=5, max_ci_percent=5,
        outputs_match=True, output_reason="identical")}
    accuracy = [{"suite": "mmlu-5shot", "source": "absolute", "status": "pass"},
                {"suite": "clip-alignment", "source": "task", "status": "fail"}]  # a retired check
    (tmp_path / "model.json").write_text(json.dumps(model))
    (tmp_path / "report.json").write_text(json.dumps({"model": "demo", "task": "text_generation", "accuracy": accuracy,
                                                      "performance_l1": [perf], "provenance": {}}))
    monkeypatch.setattr(cli, "recheck_output", lambda *args: None)
    cli.rejudge_reports([tmp_path])
    after = json.loads((tmp_path / "report.json").read_text())
    retired = next(item for item in after["accuracy"] if item["suite"] == "clip-alignment")
    assert retired["informational"] and after["verdict"]["category"] == "pass"


def test_a_new_run_sets_the_previous_directory_aside_so_its_pass_cannot_stand(tmp_path, monkeypatch):
    import json
    import sys

    from trtmc_aiperf_qual import campaign
    from trtmc_aiperf_qual.config import Environment

    existing = tmp_path / "results" / "demo"
    existing.mkdir(parents=True)
    (existing / "report.json").write_text(json.dumps({"task": "t", "started": 1, "verdict": {"category": "pass"}}))
    monkeypatch.setattr(campaign, "reference_python", lambda environment, model: sys.executable)
    monkeypatch.setattr(campaign, "prefetch", lambda environment, model: None)
    monkeypatch.setattr(campaign.bundles, "ensure_bundle",
                        lambda *args: {"status": "failed", "reason": "new build failed"})
    model = {"model": "demo", "catalog_profile": "demo", "task": "t", "reference": {}, "candidate": {}}
    assert campaign.run_one(Environment({}), model, existing)["category"] == "build-failed"
    assert campaign.collect([tmp_path / "results"])[0]["demo"]["category"] == "build-failed"
    assert campaign._finished(existing) == "build-failed"
    kept = [path for path in (tmp_path / "results").iterdir() if campaign.KEPT_ASIDE.search(path.name)]
    assert len(kept) == 1 and (kept[0] / "report.json").is_file()  # the old evidence is kept, not shown


def test_a_single_request_family_media_case_counts_its_input():
    from types import SimpleNamespace

    from trtmc_aiperf_qual import family

    case = SimpleNamespace(name="generated-media-parity", benchmark="task_output_parity",
                           values={"request": {"prompt": "cat"}})
    result = {"status": "passed", "gate": {"min_psnr": 5.0},
              "metrics": {"compared_frames": 3.0, "media_count": 320.0, "min_psnr": 11.4}}
    entry = family.item(case, result, "result.json")
    assert (entry["samples"], entry["executed_inputs"], entry["passed"]) == (1, 1, None)
    assert entry["metrics"]["media_count"] == 320.0


def test_failing_family_cases_settle_by_their_native_control(tmp_path, monkeypatch):
    import json

    from trtmc_aiperf_qual import family

    controls = []

    def run(environment, model, out, python, *, only, control):
        controls.append((sorted(only)[0], control))
        failed = {"shared": ["a", "b"], "native-ok": []}[sorted(only)[0]]
        return [{"status": "fail" if failed else "pass", "failed_samples": failed, "passed": 8, "samples": 10}]

    monkeypatch.setattr(family, "run", run)
    model = {"candidate": {"precision": "fp16"}, "reference": {"backend": "reference", "fallback": "script"}}
    entries = [{"suite": "sampled", "status": "fail", "sampled": True, "reference_precision": "fp32"},
               {"suite": "shared", "status": "fail", "failed_samples": ["a"], "reference_precision": "fp32"},
               {"suite": "native-ok", "status": "fail", "failed_samples": ["a"], "reference_precision": "fp32"},
               {"suite": "same", "status": "fail", "failed_samples": ["a"], "reference_precision": "fp16"}]
    family.attribute(None, model, tmp_path, "/py", entries)
    assert [entry["status"] for entry in entries] == ["inconclusive", "inconclusive", "fail", "fail"]
    assert entries[1]["precision_sensitive"] and entries[1]["native_control"]["precision"] == "fp16"
    assert entries[2]["native_control"]["status"] == "pass"
    assert controls == [("shared", ("reference", "fp16")), ("native-ok", ("reference", "fp16"))]
    quantized = [{"suite": "shared", "status": "fail", "reference_precision": "bf16"}]
    family.attribute(None, {**model, "candidate": {"precision": "fp8"}}, tmp_path, "/py", quantized)
    assert quantized[0]["status"] == "fail" and "native_control" not in quantized[0]
    evidence = tmp_path / "result.json"
    evidence.write_text(json.dumps({"status": "failed", "samples": [{"sample_id": "a", "passed": False}]}))
    kept = family.refresh({**entries[1], "source": "family", "evidence": str(evidence)})
    assert kept["status"] == "inconclusive"  # rejudge keeps the control's settlement


def test_box_parity_holds_matched_boxes_to_their_confidence():
    from trtmc_aiperf_plugins.accuracy import compare_boxes

    boxes = {"boxes": [[0, 0, 10, 10], [20, 20, 30, 30]], "class_ids": [1, 2]}
    assert compare_boxes({**boxes, "scores": [0.99, 0.51]}, {**boxes, "scores": [0.98, 0.52]})[0]
    swapped = compare_boxes({**boxes, "scores": [0.99, 0.51]}, {**boxes, "scores": [0.51, 0.99]})
    assert not swapped[0] and "max score difference 0.480" in swapped[1]
    assert compare_boxes(boxes, boxes)[0]  # observations without scores: boxes and classes only


def test_stratified_selection_takes_every_class_first_in_record_order():
    from trtmc_aiperf_qual.suites import select

    records = [{"id": f"{label}-{index}", "label": label} for label in range(10) for index in range(3)]
    picked = select(records, {"method": "stratified", "field": "label", "count": 10})
    assert [record["id"] for record in picked] == [f"{label}-0" for label in range(10)]
