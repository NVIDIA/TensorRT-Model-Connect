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
