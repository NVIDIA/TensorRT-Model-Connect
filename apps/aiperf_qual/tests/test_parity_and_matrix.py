# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

from trtmc_aiperf_qual import absolute, gold_metrics
from trtmc_aiperf_qual.config import Environment

REPOSITORY = Path(__file__).resolve().parents[3]


def side(observations):
    return {"observations": {"greedy": dict(enumerate(observations))}, "exit": {"greedy": 0}, "timings": {"greedy": {}}}


def test_vector_parity_needs_direction_and_magnitude():
    gate = {"min_cosine": 0.999, "max_relative_l2": 0.02}
    native = {"values": [1.0, 2.0, 3.0]}
    assert gold_metrics.vector_parity({"values": [1.0, 2.0, 3.001]}, native, gate)[0]
    assert not gold_metrics.vector_parity({"values": [2.0, 4.0, 6.0]}, native, gate)[0]  # same direction, twice as long
    assert not gold_metrics.vector_parity({"values": [1.0, 2.0]}, native, gate)[0]
    assert not gold_metrics.vector_parity({"values": [1.0, float("nan"), 3.0]}, native, gate)[0]


def test_parity_entries_pass_only_when_every_output_matches():
    item = {"suite": "encoder-parity", "metric": "vector_parity", "gate": {"min_cosine": 0.999, "max_relative_l2": 0.02}}
    problems = [{"sample_id": "a"}, {"sample_id": "b"}]
    native = side([{"values": [1.0, 0.0]}, {"values": [0.0, 1.0]}])
    assert absolute.judge(item, problems, native, native)["status"] == "pass"
    off = side([{"values": [1.0, 0.0]}, {"values": [1.0, 0.0]}])
    entry = absolute.judge(item, problems, off, native)
    assert entry["status"] == "fail" and entry["passed"] == 1 and entry["failures"][0]["sample_id"] == "b"
    missed = absolute.judge(item, problems, side([{"values": [1.0, 0.0]}]), native)  # TRTMC's miss: outside
    assert missed["status"] == "fail" and missed["passed"] == 1 and missed["failures"][0]["explanation"] == absolute.NO_ANSWER
    assert absolute.judge(item, problems, native, side([{"values": [1.0, 0.0]}]))["status"] == "error"  # native miss
    not_finite = side([{"values": [1.0, 0.0]}, {"values": [float("nan"), 1.0]}])
    assert absolute.judge(item, problems, side([{"values": [1.0, 0.0]}]), not_finite)["status"] == "error"
    assert absolute.judge(item, problems, side([{"values": [1.0, 0.0]}, {"values": "x"}]), native)["status"] == "fail"
    unusable = absolute.judge(item, problems, side([{"values": [1.0, 0.0]}, {}]), native)  # TRTMC's unusable output
    assert unusable["status"] == "fail" and unusable["passed"] == 1
    assert absolute.judge(item, problems, side([{"values": [1.0, 0.0]}]), side([{"values": [1.0, 0.0]}, {}]))[
        "status"] == "error"  # an unusable native output is missing evidence, whatever TRTMC answered


def test_action_and_disparity_parity(tmp_path):
    actions = {"actions": [0.0, 1.0, 2.0, 3.0]}
    assert gold_metrics.action_parity({"actions": [0.0, 1.0, 2.0, 3.002]}, actions, {"max_error_of_range": 1e-3})[0]
    assert not gold_metrics.action_parity({"actions": [0.0, 1.0, 2.0, 3.1]}, actions, {"max_error_of_range": 1e-3})[0]
    native, close, far = (tmp_path / name for name in ("native.f32", "close.f32", "far.f32"))
    base = np.arange(12, dtype="<f4").reshape(3, 4)
    base.tofile(native)
    (base + 0.05).tofile(close)
    (base + 1.0).tofile(far)
    shape = {"height": 3, "width": 4}
    reference = {**shape, "disparity_artifact": str(native)}
    assert gold_metrics.disparity_parity({**shape, "disparity_artifact": str(close)}, reference, {"max_mean_epe": 0.1})[0]
    assert not gold_metrics.disparity_parity({**shape, "disparity_artifact": str(far)}, reference, {"max_mean_epe": 0.1})[0]
    assert absolute.keeps_artifacts({"absolute": [{"metric": "disparity_parity"}]})


def test_geometry_parity_compares_depth_and_valid_masks(tmp_path):
    depth = np.full((2, 2), 2.0, dtype="<f4")
    mask = np.ones((2, 2), dtype=np.uint8)
    def write(name, values, valid):
        values.astype("<f4").tofile(tmp_path / f"{name}.depth.f32")
        valid.astype(np.uint8).tofile(tmp_path / f"{name}.mask.u8")
        return {"height": 2, "width": 2, "depth_artifact": str(tmp_path / f"{name}.depth.f32"),
                "valid_mask_artifact": str(tmp_path / f"{name}.mask.u8")}
    native = write("native", depth, mask)
    gate = {"min_mask_iou": 0.99, "max_median_relative_depth": 0.01}
    assert gold_metrics.geometry_parity(write("same", depth * 1.001, mask), native, gate)[0]
    assert not gold_metrics.geometry_parity(write("deep", depth * 1.1, mask), native, gate)[0]
    assert not gold_metrics.geometry_parity(write("holes", depth, np.array([[1, 0], [0, 1]])), native, gate)[0]


def test_a_size_difference_fails_only_when_both_sides_artifacts_are_readable(tmp_path):
    """Missing evidence on either side is None (an error), even when the sizes differ; a readable pair of
    different sizes is a numerical failure (False)."""
    def geometry(name, height, width, files=True):
        side = {"height": height, "width": width, "depth_artifact": str(tmp_path / f"{name}.depth.f32"),
                "valid_mask_artifact": str(tmp_path / f"{name}.mask.u8")}
        if files:
            np.ones((height, width), dtype="<f4").tofile(side["depth_artifact"])
            np.ones((height, width), dtype=np.uint8).tofile(side["valid_mask_artifact"])
        return side
    gate = {"min_mask_iou": 0.99, "max_median_relative_depth": 0.01}
    native = geometry("native", 2, 2)
    assert gold_metrics.geometry_parity(geometry("missing", 3, 2, files=False), native, gate)[0] is None
    assert gold_metrics.geometry_parity(native, geometry("absent", 3, 2, files=False), gate)[0] is None
    assert gold_metrics.geometry_parity(geometry("taller", 3, 2), native, gate)[0] is False

    def disparity(name, height, width, files=True):
        side = {"height": height, "width": width, "disparity_artifact": str(tmp_path / f"{name}.disp.f32")}
        if files:
            np.zeros((height, width), dtype="<f4").tofile(side["disparity_artifact"])
        return side
    reference = disparity("native", 3, 4)
    assert gold_metrics.disparity_parity(disparity("missing", 2, 4, files=False), reference, {"max_mean_epe": 0.1})[0] is None
    assert gold_metrics.disparity_parity(reference, disparity("absent", 2, 4, files=False), {"max_mean_epe": 0.1})[0] is None
    assert gold_metrics.disparity_parity(disparity("shorter", 2, 4), reference, {"max_mean_epe": 0.1})[0] is False


def test_sampled_text_requests_are_timed_greedy():
    from trtmc_aiperf_qual.runner import timed_request

    model = {"operation": "generate"}
    assert timed_request(model, {"prompt": "p", "temperature": 0.7, "top_k": 50})["temperature"] == 0.0
    assert timed_request(model, {"prompt": "p", "temperature": 0.0}) == {"prompt": "p", "temperature": 0.0}
    assert timed_request({"operation": "generate_audio"}, {"temperature": 0.7}) == {"temperature": 0.7}  # Bark samples


def test_the_matrix_marks_profiles_without_a_native_path(tmp_path):
    from trtmc_aiperf_qual import matrix

    environment = Environment({"repo": str(REPOSITORY)})
    rows = {name: matrix.row(environment, name) for name in ("qwen3-0.6b-fp16", "sana-wm-bidirectional")}
    assert rows["qwen3-0.6b-fp16"]["executable"] and rows["qwen3-0.6b-fp16"]["native_path"] == "generic generate"
    assert rows["sana-wm-bidirectional"]["native_path"].startswith("families/sana_wm/")
    output = tmp_path / "matrix.csv"
    assert matrix.write_matrix(environment, ["qwen3-0.6b-fp16"], output) == 0
    assert output.read_text().startswith("profile,task,operation")


def test_the_near_capacity_prompt_shrinks_until_trtmc_accepts_it(monkeypatch):
    import transformers

    from trtmc_aiperf_qual import absolute, runner

    monkeypatch.setattr(absolute, "tokenizer_source", lambda model: ("tok", None))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "_passage", lambda environment, tokenizer, count: "w " * count)
    monkeypatch.setattr(runner, "rendered_tokens", lambda tokenizer, request: len(request["prompt"].split()) + 1)

    def probe(service, operation, request):  # TRTMC counts 3 tokens more than the Hugging Face tokenizer
        if len(request["prompt"].split()) + 1 + 3 > 224:
            raise RuntimeError('probe rejected: {"error":{"message":"prompt exceeds the prefill profile",'
                               '"code":"backend_rejected_request"}}')

    monkeypatch.setattr(runner, "probe", probe)
    model = {"operation": "generate", "reference": {}}
    greedy = {"prompt": "x", "temperature": 1.0, "top_k": 1}  # a pinned greedy contract (Qwen3-Omni) is kept
    long, tokens = runner.near_capacity_request(None, model, greedy, 224, {"url": "u"})
    assert tokens == 221 and long["max_new_tokens"] == runner.NEAR_CAPACITY_NEW_TOKENS
    assert (long["temperature"], long["top_k"]) == (1.0, 1)

    def broken(service, operation, request):
        raise RuntimeError("probe rejected: CUDA error")

    monkeypatch.setattr(runner, "probe", broken)
    with pytest.raises(RuntimeError, match="CUDA"):
        runner.near_capacity_request(None, model, greedy, 224, {"url": "u"})


def _order_check(tmp_path, monkeypatch, measurements):
    """The real order check and native timing (``_time_reference``) over stubbed servers; ``measurements`` are
    the four timed runs' stats in the order they run: native first, TRTMC second, TRTMC first, native second."""
    from contextlib import contextmanager

    from trtmc_aiperf_qual import runner

    @contextmanager
    def nothing(*args, **kwargs):
        yield {"url": "u", "info": {}}

    suite = type("Suite", (), {"name": "catalog", "samples": [{"request": {"prompt": "p"}}]})()
    measurements = iter(measurements)
    monkeypatch.setattr(runner, "reference_python", lambda environment, model: "python")
    monkeypatch.setattr(runner, "build_suite", lambda definition, environment: "suite")
    monkeypatch.setattr(runner, "serving", nothing)
    monkeypatch.setattr(runner, "gpu_exclusive", nothing)
    monkeypatch.setattr(runner, "probe", lambda *args: None)
    monkeypatch.setattr(runner, "perf_suites", lambda environment, model, perf_suite, service=None: [suite])
    monkeypatch.setattr(runner, "_perf_run", lambda *args: (None, dict(next(measurements))))
    model = {"model": "m", "operation": "generate", "reference": {"perf_precision": "fp16", "precision": "fp32"},
             "performance": {"suite": {}, "max_ci_percent": 5.0, "aggregation": {"eager": "mean"},
                                    "guard_percent": 4, "measurement": {"warmup": 0, "requests": 1, "runs": 3}}}
    return runner.order_check(Environment({}), model, tmp_path)


def timing(p50_ms, **extra):
    return {"p50_ms": p50_ms, "ci_percent": 1.0, "runs": 3, "work": [(("tokens", 32),)], "work_missing": 0,
            "gpu_busy_percent": 0.0, **extra}


def test_the_order_check_reports_each_sides_effect_against_the_perf_guard(tmp_path, monkeypatch):
    result = _order_check(tmp_path, monkeypatch, [timing(10.0), timing(2.02), timing(2.0), timing(10.5)])
    assert result["order_effect"]["native"]["catalog"] == pytest.approx(0.05)  # native slower after TRTMC
    assert result["order_effect"]["trtmc"]["catalog"] == pytest.approx(2.02 / 2.0 - 1)
    assert result["speedup_effect"]["catalog"] == pytest.approx(1.05 * 1.01 - 1)  # both sides compound
    assert result["limit"] == 0.04  # the Perf guard
    assert result["status"] == "above-limit" and result["above_limit"] and (tmp_path / "order.json").is_file()


@pytest.mark.parametrize("native_second, trtmc_second, status", [
    (10.3, 2.0, "within-limit"),   # native +3% within the 4% guard
    (9.62, 2.0, "within-limit"),   # native -3.8%: 1/0.962 = 1.0395
    (9.6, 2.0, "above-limit"),     # native -4%: 1/0.96 = 1.0417 exceeds the guard in the speedup
    (10.25, 1.96, "above-limit"),  # native +2.5%, TRTMC -2%: each within, together 1.0459
])
def test_the_order_check_bounds_the_effect_on_the_speedup(tmp_path, monkeypatch, native_second, trtmc_second, status):
    """A side's ratio counts in its worse direction (a 4% decrease is a 4.17% speedup change) and the two sides'
    ratios compound, so the guard covers what the measured effects can do to a speedup."""
    result = _order_check(tmp_path, monkeypatch, [timing(10.0), timing(trtmc_second), timing(2.0), timing(native_second)])
    assert result["status"] == status


@pytest.mark.parametrize("invalid", [
    timing(10.4, incomplete="2 of 20 requests succeeded"), timing(10.4, work_missing=3),
    timing(10.4, work=[]), timing(10.4, gpu_busy_percent=60.0), timing(10.4, ci_percent=9.0),
    timing(10.4, ci_percent=None), timing(10.4, work=[(("tokens", 31),)]), timing(10.4, gpu_busy_percent=None),
    timing(10.4, gpu_unmeasured_runs=1)])
def test_an_order_check_on_invalid_measurements_is_unresolved(tmp_path, monkeypatch, invalid):
    """A near-equal pair (within the limit) is no evidence when a measurement is incomplete, lacks work
    evidence, ran on a busy GPU, spread too far, or did other work than the side it was compared with."""
    result = _order_check(tmp_path, monkeypatch, [timing(10.0), timing(2.0), timing(2.0), invalid])
    assert result["status"] == "unresolved" and result["problems"]
    assert result["above_limit"] is None and result["order_effect"] is None


def test_an_order_check_whose_native_timing_fails_is_unresolved_and_replaces_the_old_result(tmp_path, monkeypatch):
    """A native run without a model-call time makes ``_time_reference`` raise: the measurement is kept as a
    failure, the other three still run, and a stale within-limit result does not survive."""
    (tmp_path / "order.json").write_text('{"status": "within-limit"}')
    result = _order_check(tmp_path, monkeypatch, [timing(None), timing(2.0), timing(2.0), timing(10.0)])
    assert result["status"] == "unresolved" and result["above_limit"] is None
    assert any("native_first" in problem and "no successful fp16 requests" in problem for problem in result["problems"])
    assert result["p50_ms"]["native_second"]["catalog"] == 10.0
    assert '"unresolved"' in (tmp_path / "order.json").read_text()


@pytest.mark.parametrize("failure, reason", [("start", "Inductor lowering failed"),
                                             ("incomplete", "incomplete under torch.compile")])
def test_a_fixed_request_whose_torch_compile_fails_is_timed_eager_in_its_place(tmp_path, monkeypatch, failure, reason):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from trtmc_aiperf_qual import judge, runner

    started = []

    @contextmanager
    def serving(environment, model, backend, out, *, mode="eager", compile_fallback=None, **kwargs):
        started.append((backend, mode, compile_fallback))
        if backend == "reference" and mode == "compile" and failure == "start":
            raise RuntimeError("Inductor lowering failed")
        yield {"url": "u", "info": {"backend": backend, "timing_scope": "task-call-wall"}}

    run = SimpleNamespace(raw_records=lambda: [])
    monkeypatch.setattr(runner, "serving", serving)
    monkeypatch.setattr(runner, "probe", lambda *args: None)

    def perf_run(*args):  # args[5]: the run's directory, perf-reference-<mode>-<precision>-<request> or perf-candidate-*
        p50 = 10.0 if "reference" in args[5].name else 2.0
        stats = timing(p50, per_run_p50_ms=[p50, p50 * 1.01, p50 * 0.99])
        if args[5].name.startswith("perf-reference-compile"):  # a request timed out while the server compiled
            stats["incomplete"] = "11 of 12 requests succeeded"
        return run, stats

    monkeypatch.setattr(runner, "_perf_run", perf_run)
    monkeypatch.setattr(runner, "output_check", lambda *args: (True, ""))
    suite = SimpleNamespace(name="catalog", samples=[{"request": {"prompt": "p"}}])
    model = {"model": "m", "operation": "generate",
             "reference": {"mode": "compile", "perf_precision": "fp16", "precision": "fp32"}}
    policy = {"reference_modes": ["compile"], "measurement": {}, "aggregation": {"compile": "mean", "eager": "mean"},
              "margin_percent": 5, "max_ci_percent": 5, "guard_percent": 0}
    timed = runner._reference_perf(Environment({}), model, policy, [suite], "python", runner._Phases(tmp_path), tmp_path)
    stats = timed["compile"]["catalog"][1]
    assert stats["mode"] == "eager" and reason in stats["compile_fallback"]
    assert "precision_fallback" not in stats  # the same precision: no fallback baseline
    assert started == [("reference", "compile", None), ("reference", "eager", stats["compile_fallback"])]
    performance = []
    runner._candidate(Environment({}), {**model, "absolute": []}, policy, [suite], timed, [], performance, tmp_path)
    item, = performance
    assert (item["reference_mode"], item["configured_mode"], item["qualifying"]) == ("eager", "compile", True)
    assert judge.qualifying(item) and item["light"] == "green"
    assert any(note.startswith("native torch.compile failed; timed in eager mode") for note in item["notes"])


def test_an_order_check_that_cannot_start_is_unresolved(tmp_path, monkeypatch):
    from trtmc_aiperf_qual import runner

    def no_environment(environment, model):
        raise OSError("no env")

    (tmp_path / "order.json").write_text('{"status": "within-limit"}')
    monkeypatch.setattr(runner, "reference_python", no_environment)
    result = runner.order_check(Environment({}), {"model": "m", "performance": {"max_ci_percent": 5.0}}, tmp_path)
    assert result["status"] == "unresolved" and "no env" in result["problems"][0]
    assert '"unresolved"' in (tmp_path / "order.json").read_text()


def test_the_order_check_command_builds_a_missing_bundle_and_skips_a_failed_build(tmp_path, monkeypatch, capsys):
    import json

    from trtmc_aiperf_qual import bundles, cli, models, runner, services

    built, checked = [], []
    monkeypatch.setattr(cli, "load_environment", lambda path: Environment({}))
    monkeypatch.setattr(models, "resolve_model", lambda name, environment: {"model": name})
    monkeypatch.setattr(services, "reference_python", lambda environment, model: "python")
    monkeypatch.setattr(bundles, "ensure_bundle", lambda environment, model, out, python: built.append(model["model"]) or
                        {"status": "failed" if model["model"] == "broken" else "built", "reason": "no engine"})
    monkeypatch.setattr(runner, "order_check", lambda environment, model, out: checked.append(model["model"]) or
                        {"model": model["model"], "status": "within-limit", "order_effect": {}, "largest": 0.0,
                         "problems": []})
    code = cli.main(["order-check", "--environment", str(tmp_path / "env.yaml"), "--profile", "broken", "--profile", "ok",
                     "--out-root", str(tmp_path / "order")])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert code == 0 and built == ["broken", "ok"] and checked == ["ok"]
    assert lines[0]["status"] == "build-failed" and lines[1]["status"] == "within-limit"
    assert json.loads((tmp_path / "order" / "ok" / "build.json").read_text())["status"] == "built"


def test_cer_counts_characters_without_spaces_or_punctuation():
    """Japanese has no word boundaries: the error rate is over NFKC characters, spaces and punctuation dropped."""
    problems = [{"gold": "インターネットで 敵対的環境コース について"}, {"gold": "ＡＢ、C"}]
    units = gold_metrics.cer_units(problems, {0: {"text": "インターネットで敵対的環境コースについて。"},
                                              1: {"text": "abd"}})
    assert units == [(0.0, 20.0), (1.0, 3.0)]  # full-width letters fold to ASCII; one substitution
    assert gold_metrics.wer(units) == pytest.approx(100.0 / 23.0)
    assert gold_metrics.CORPUS["cer"][2] is False  # lower is better


def test_decision_parity_needs_the_same_choices_within_the_family_tolerance():
    native = {"scores": [{"question_id": "status", "option_ids": ["paid", "overdue"], "probabilities": [0.2, 0.8]},
                         {"question_id": "large", "probabilities": [0.6, 0.4]}]}

    def candidate(status, large):
        return {"scores": [{"question_id": "status", "option_ids": ["paid", "overdue"], "probabilities": status},
                           {"question_id": "large", "option_ids": ["false", "true"], "probabilities": large}]}

    gate = {"atol": 0.015, "rtol": 0.01}
    assert gold_metrics.decision_parity(candidate([0.21, 0.79], [0.59, 0.41]), native, gate)[0] is True
    assert gold_metrics.decision_parity(candidate([0.25, 0.75], [0.6, 0.4]), native, gate)[0] is False
    assert gold_metrics.decision_parity(candidate([0.2, 0.8], [0.49, 0.51]), native, gate)[0] is False
    assert gold_metrics.decision_parity({"scores": []}, native, gate)[0] is None
    missing = {"scores": native["scores"][:1]}
    assert gold_metrics.decision_parity(missing, native, gate)[0] is False
    assert gold_metrics.decision_parity({"scores": []}, {"scores": []}, gate)[0] is True  # no questions
    assert gold_metrics.decision_parity({}, {"scores": []}, gate)[0] is None


def test_open_vocabulary_detections_score_by_their_category_names():
    """coco-names-80: text labels map to contiguous COCO classes; a phrase naming no category drops its box."""
    output = {"boxes": [0, 0, 10, 10, 5, 5, 20, 20, 1, 1, 2, 2], "scores": [0.9, 0.8, 0.7],
              "labels": ["Dining Table", "car truck", "person."]}
    named = gold_metrics._named_detections(output)
    assert named["class_ids"] == [60, 0] and named["scores"] == [0.9, 0.7]
    assert named["boxes"] == [[0, 0, 10, 10], [1, 1, 2, 2]]
    problems = [{"gold": {"bbox": [[0, 0, 10, 10]], "category": [60], "area": [100.0]}}]
    units = gold_metrics.coco_units(problems, {0: output}, {"label_space": "coco-names-80"})
    assert units[0][1]["class_ids"] == [60, 0] and units[0][2] == "category_index"
    assert len(gold_metrics.COCO_NAMES) == len(gold_metrics.COCO_IDS) == 80


def test_dinov2_checker_applies_the_e2e_feature_tolerance_of_the_build_precision():
    from families.dinov2.validation import compare

    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(1, 5, 8))

    def side(values):
        return {"last_hidden_state": values.reshape(-1).tolist(), "last_hidden_state_shape": [1, 5, 8],
                "pooler_output": values[:, 0].reshape(-1).tolist(), "pooler_output_shape": [1, 8]}

    assert compare(side(hidden + 1e-3), side(hidden), {"precision": "fp32"})[0] is True
    assert compare(side(hidden + 1e-2), side(hidden), {"precision": "fp32"})[0] is False
    assert compare(side(hidden + 1e-2), side(hidden), {"precision": "fp16"})[0] is True
    assert compare(side(hidden + 1.5e-2), side(hidden), {"precision": "fp32", "threshold_scale": 2.0})[0] is False
    assert compare(None, side(hidden), {"precision": "fp32"})[0] is None


def test_lossless_images_send_the_decoded_pixels_as_png():
    import base64
    import io

    from PIL import Image

    from trtmc_aiperf_qual.suites import _encoded_media

    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), (200, 30, 90)).save(buffer, format="JPEG")
    jpeg = {"bytes": buffer.getvalue()}
    assert _encoded_media(jpeg, ".png")["$file"]["suffix"] == ".jpg"
    media = _encoded_media(jpeg, ".png", lossless=True)["$file"]
    assert media["suffix"] == ".png"
    with Image.open(io.BytesIO(base64.b64decode(media["b64"]))) as decoded, \
            Image.open(io.BytesIO(jpeg["bytes"])) as original:
        assert decoded.format == "PNG" and decoded.tobytes() == original.convert("RGB").tobytes()


def test_max_audio_seconds_drops_longer_clips_for_both_sides(tmp_path):
    import base64
    import io

    import numpy as np
    import soundfile

    from trtmc_aiperf_qual.absolute import _audio_fitted
    from trtmc_aiperf_qual.config import Environment

    def clip(seconds):
        buffer = io.BytesIO()
        soundfile.write(buffer, np.zeros(int(16000 * seconds), dtype=np.float32), 16000, format="WAV")
        return {"$file": {"suffix": ".wav", "b64": base64.b64encode(buffer.getvalue()).decode()}}

    samples = [{"sample_id": name, "request": {"audio_path": clip(seconds)}}
               for name, seconds in (("short", 12.0), ("limit", 30.0), ("long", 33.9))]
    kept = _audio_fitted(Environment({}), samples, 30.0)
    assert [sample["sample_id"] for sample in kept] == ["short", "limit"]
