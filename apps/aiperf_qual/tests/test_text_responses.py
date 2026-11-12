# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from types import SimpleNamespace

import pytest

from trtmc_aiperf_qual.config import ConfigError
from trtmc_aiperf_qual.text_profile import arguments, cases, response_text
from trtmc_aiperf_qual.text_responses import prompt_text, verify, verify_records


def test_responses_command_is_accepted_by_published_client_and_is_stateless():
    app = pytest.importorskip("aiperf.cli").app
    config = {"tokenizer": {"name": "test", "revision": "a" * 40},
        "workload": {"endpoints": ["responses"], "input_tokens": [32], "output_tokens": [16]}}
    case, = cases(config)
    args = arguments("http://localhost", config, case)
    app.parse_args(["profile", "--model", "test", *args], print_error=False, exit_on_error=False)
    assert "store:false" in args and "enable_thinking:false" in args
    config["workload"]["streaming"] = [True]
    with pytest.raises(ConfigError, match="nonstreaming"):
        cases(config)
    config["workload"]["streaming"] = [False]
    config["reference"] = {"profile": "test"}
    with pytest.raises(ConfigError, match="reference server"):
        cases(config)


@pytest.fixture
def capture(tmp_path):
    prompt = "  First\nsecond\u2028third  "
    body = {"id": "resp_unique", "object": "response", "status": "completed", "model": "test",
            "usage": None, "max_output_tokens": 16,
            "output": [{"id": "msg_unique", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "OK", "annotations": []}]}]}
    raw = {"metadata": {"x_request_id": "request-1", "benchmark_phase": "profiling"}, "status": 200,
           "payload": {"model": "test", "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                       "max_output_tokens": 16, "store": False},
           "responses": [{"text": json.dumps(body)}]}
    joined = [{"request_id": "request-1", "server": {"status": 200, "timing_scope": "public_task_call_wall",
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "completion_sha256": hashlib.sha256(b"OK").hexdigest(), "completion_tokens": 2,
        "generation_config": {"max_new_tokens": 16}}}]
    return SimpleNamespace(directory=tmp_path, raw_records=lambda phase: [raw]), joined, raw, body


def test_typed_response_and_native_input_output_caps_are_verified(capture):
    run, joined, raw, _ = capture
    report = verify_records(run, joined)
    assert report["verified"] and report["requests"][0]["response_id"] == "resp_unique"
    assert prompt_text(raw["payload"]) == "  First\nsecond\u2028third  "
    assert response_text(raw) == "OK"


@pytest.mark.parametrize("change", ["text", "role", "usage", "identity", "cap", "status"])
def test_changed_response_or_native_evidence_cannot_pass_and_retains_failure(capture, change):
    run, joined, raw, body = capture
    if change == "text":
        body["output"][0]["content"][0]["text"] = "invented"
    elif change == "role":
        body["output"][0]["role"] = "user"
    elif change == "usage":
        body["usage"] = {"input_tokens": 0, "output_tokens": 2, "total_tokens": 2}
    elif change == "identity":
        joined[0]["server"]["prompt_sha256"] = "bad"
    elif change == "cap":
        raw["payload"]["max_output_tokens"] = 32
    else:
        body["status"] = "failed"
    raw["responses"][0]["text"] = json.dumps(body)
    with pytest.raises(ConfigError):
        verify(run, joined)
    assert json.loads((run.directory / "responses-verification.json").read_text())["verified"] is False


def test_non_mapping_and_unsupported_input_shapes_are_not_interpreted_as_text():
    for payload in [None, [], {"input": []}, {"input": [{"role": "assistant", "content": "old"}]},
                    {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]}]:
        assert prompt_text(payload) is None
    assert response_text({"status": 200, "responses": [{"text": "[]"}]}) is None


def test_malformed_response_request_metadata_retains_a_failed_verification_record(capture):
    run, joined, raw, _ = capture
    raw["metadata"] = None
    with pytest.raises(ConfigError, match="metadata is not an object"):
        verify(run, joined)
    assert json.loads((run.directory / "responses-verification.json").read_text())["verified"] is False
