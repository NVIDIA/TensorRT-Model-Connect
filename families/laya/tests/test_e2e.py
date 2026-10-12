# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All released Laya variants through the C++ Task, compared with the original SDK."""

from copy import deepcopy
from functools import cache
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).parent
MANIFESTS = {
    item["name"]: item
    for path in sorted((ROOT / "manifests").glob("*.json"))
    for item in [json.loads(path.read_text())]
}


def _selection(config):
    return {
        name.strip()
        for option in ("--e2e-model", "--e2e-testcase")
        for value in (config.getoption(option, default=[]) or [])
        for name in str(value).split(",")
        if name.strip()
    }


@pytest.fixture(scope="module")
def laya_models(tmp_path_factory):
    @cache
    def reference(checkpoint, variant, maximum):
        from laya import Agent
        from families.laya.cli import VARIANTS

        agent = Agent(
            str(Path(checkpoint) / VARIANTS[variant]), device="cuda", fast=False, compile=False
        )
        assert agent.device.type == "cuda"
        if maximum is not None:
            agent.cfg["max_len"] = maximum
        return agent

    @cache
    def resolve(name):
        from families.laya.cli import build

        manifest = MANIFESTS[name]
        assert manifest["precision"] == "bf16"
        assert manifest["tensor_parallel_size"] == 1
        maximum = manifest.get("max_sequence_length")
        checkpoint = os.environ.get("TRTMC_LAYA_CHECKPOINT")
        if checkpoint is None:
            from huggingface_hub import snapshot_download

            checkpoint = snapshot_download(manifest["hf_id"], revision=manifest["hf_revision"])
        prefix = "TRTMC_" + name.upper().replace("-", "_")
        value = os.environ.get(prefix + "_BUNDLE")
        bundle = Path(value) if value else tmp_path_factory.mktemp(name) / manifest["bundle"]
        if value is None:
            build(
                model=checkpoint,
                output=bundle,
                variant=manifest["variant"],
                max_sequence_length=maximum,
            )
        return bundle, lambda variant: reference(str(checkpoint), variant, maximum)

    return resolve


def compare_response(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare_response(actual[key], expected[key])
    elif isinstance(expected, (float, int)):
        np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.01)
    else:
        assert actual == expected


def compare_native_response(result, reference, internal):
    """Check SDK formatting after the independent reference probability gates.

    Entropy and expected scores propagate probability error differently from
    individual probabilities. Evaluate their SDK formulas on the already
    validated native probabilities, and require exact four-decimal formatting.
    Selected choices, action probabilities, usage, and routing still compare
    directly with the original inference result.
    """
    from laya.common import answer_confidence, confidence_from_probs

    expected = deepcopy(reference)
    assert [score["question_id"] for score in result["scores"]] == list(internal)
    for score in result["scores"]:
        qid = score["question_id"]
        question = internal[qid]
        p = np.asarray(score["probabilities"], dtype=np.float32)
        count = len(p)
        options = (
            list(question["crit"]) if question["t"] == "choice" else [str(i) for i in range(count)]
        )
        if question["t"] == "noul":
            options = ["false", "true"]
        assert score["option_ids"] == options
        actual = result["response"]["answers"][qid]
        answer = expected["answers"][qid]
        compare_response(actual["action"], answer["action"])
        action = actual["action"]["act_probability"]
        assert 0 <= action <= 1 and action == round(action, 4)
        answer["action"] = actual["action"]
        answer["answer_confidence"] = round(answer_confidence(p, count), 4)
        if question["t"] == "noul":
            answer["noul"] = round(float(p[1]), 4)
            answer["confidence"] = round(max(float(p[1]), 1.0 - float(p[1])), 4)
        else:
            answer["probabilities"] = {
                option: round(float(value), 4) for option, value in zip(options, p, strict=True)
            }
            answer["confidence"] = round(confidence_from_probs(p, count), 4)
            if question["t"] == "choice":
                assert options[int(p.argmax())] == answer["choice"]
            else:
                answer["score"] = round(float((np.arange(count) * p).sum()), 4)
    assert result["response"] == expected


@pytest.mark.parametrize(
    ("manifest", "case"),
    [
        pytest.param(manifest, case, id=case["name"])
        for manifest in MANIFESTS.values()
        for case in manifest["testcases"]
    ],
)
def test_e2e(manifest, case, request, tmp_path):
    selected = _selection(request.config)
    if not selected and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("real Laya E2E requires explicit selection")
    if selected and not selected.intersection({"laya", manifest["name"], case["name"]}):
        pytest.skip("Laya case was not selected")
    import torch
    from laya.common import collate_items, temp_bucket

    probe = request.getfixturevalue("laya_task_probe")
    bundle, reference_agent = request.getfixturevalue("laya_models")(manifest["name"])
    fixture = ROOT / case["inputs"]["document_path"]
    record = json.loads(fixture.read_text())
    routing = None
    variant = manifest["variant"]
    if variant == "router":
        from laya import Router

        routing = Router().route(record["state"], record["questions"], model=record.get("model"))
        variant = routing.model
    agent = reference_agent(variant)
    ids = list(record["questions"])
    internal = {qid: agent._to_internal(record["questions"][qid]) for qid in ids}
    items = agent._encode_state(record["state"], ids, internal) if ids else []
    with torch.inference_mode():
        reference = agent.predict(record["state"], record["questions"])
        logits = agent._forward(collate_items([items], agent.tok.pad_token_id))[0] if items else []
    assert agent.device.type == "cuda", "the reference must not silently fall back to CPU"
    if routing is not None:
        reference["routing"] = routing
    native = json.loads(
        subprocess.check_output(
            [str(probe), str(bundle), os.environ["TRTMC_RUNTIME_ROOT"], str(fixture), "2"],
            text=True,
        )
    )
    assert not any(
        any(token in name.lower() for token in ("python", "torch", "c10"))
        for name in native["loaded_libraries"]
    )
    result = native["results"][0]
    if routing is not None:
        assert result["response"]["routing"] == routing
    assert result["scores"] == native["results"][1]["scores"]
    assert result["response"] == native["results"][1]["response"]
    assert result["response"]["usage"] == reference["usage"]
    assert len(result["scores"]) == len(ids)
    for qid, item, raw, actual in zip(ids, items, logits, result["scores"], strict=True):
        assert actual["question_id"] == qid
        count = len(item["markers"])
        scale = agent.temperature_by_options.get(
            temp_bucket(item["qtype"], count), agent.temperature[item["qtype"]]
        )
        expected = raw[:count] / scale
        probabilities = np.exp(expected - expected.max())
        probabilities /= probabilities.sum()
        np.testing.assert_allclose(actual["logits"], expected, atol=0.125, rtol=0.015)
        np.testing.assert_allclose(actual["probabilities"], probabilities, atol=0.002, rtol=0.01)
        assert np.argmax(actual["probabilities"]) == np.argmax(probabilities)
    compare_native_response(result, reference, internal)
    (tmp_path / "native.json").write_text(json.dumps(native, indent=2))
