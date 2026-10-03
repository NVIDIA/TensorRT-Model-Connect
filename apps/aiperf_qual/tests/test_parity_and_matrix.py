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
    assert absolute.judge(item, problems, side([{"values": [1.0, 0.0]}]), native)["status"] == "error"


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
             "performance": {"l1": {"suite": {}, "max_ci_percent": 5.0, "aggregation": {"eager": "mean"},
                                    "measurement": {"warmup": 0, "requests": 1, "runs": 3}}}}
    return runner.order_check(Environment({}), model, tmp_path)


def timing(p50_ms, **extra):
    return {"p50_ms": p50_ms, "ci_percent": 1.0, "runs": 3, "work": [(("tokens", 32),)], "work_missing": 0,
            "gpu_busy_percent": 0.0, **extra}


def test_the_order_check_reports_each_sides_effect(tmp_path, monkeypatch):
    result = _order_check(tmp_path, monkeypatch, [timing(10.0), timing(2.02), timing(2.0), timing(10.5)])
    assert result["order_effect"]["native"]["catalog"] == pytest.approx(0.05)  # native slower after TRTMC
    assert result["order_effect"]["trtmc"]["catalog"] == pytest.approx(2.02 / 2.0 - 1)
    assert result["status"] == "above-limit" and result["above_limit"] and (tmp_path / "order.json").is_file()


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


def test_an_order_check_that_cannot_start_is_unresolved(tmp_path, monkeypatch):
    from trtmc_aiperf_qual import runner

    def no_environment(environment, model):
        raise OSError("no env")

    (tmp_path / "order.json").write_text('{"status": "within-limit"}')
    monkeypatch.setattr(runner, "reference_python", no_environment)
    result = runner.order_check(Environment({}), {"model": "m", "performance": {"l1": {"max_ci_percent": 5.0}}}, tmp_path)
    assert result["status"] == "unresolved" and "no env" in result["problems"][0]
    assert '"unresolved"' in (tmp_path / "order.json").read_text()
