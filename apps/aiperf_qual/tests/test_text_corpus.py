# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from types import SimpleNamespace

import pytest

from trtmc_aiperf_qual.config import ConfigError
from trtmc_aiperf_qual.text_corpus import select_prompts, verify, verify_and_record
from trtmc_aiperf_qual.text_profile import arguments, cases


def conversation(text):
    return {"conversations": [{"from": "human", "value": text}, {"from": "gpt", "value": "answer"}]}


def test_selection_keeps_original_unicode_and_audits_capacity_rejections():
    original = "  first\nsecond\u2028third\u0085fourth  "
    selected, ledger = select_prompts([None, conversation("x" * 100), conversation(original),
                                       conversation("unexamined")], 1, 80, 8, list)
    assert selected[0]["prompt"] == original
    assert selected[0]["source_index"] == 2 and selected[0]["input_tokens"] == len(original)
    assert ledger[0]["status"] == ledger[1]["status"] == "rejected"
    assert "exceeds" in ledger[1]["reason"] and len(ledger) == 3
    assert select_prompts([conversation("x" * 100)], 1, 80, 8, list)[0] == []


@pytest.mark.parametrize("count,context,output", [(0, 20, 1), (True, 20, 1), (1, 0, 1), (1, 20, 20)])
def test_invalid_selection_cannot_silently_change_the_workload(count, context, output):
    with pytest.raises(ConfigError):
        select_prompts([], count, context, output, list)


def test_repeated_source_text_has_an_explicit_rejection_and_distinct_coverage():
    selected, ledger = select_prompts([conversation("same"), conversation("same"),
                                      conversation("different")], 2, 80, 8, list)
    assert [row["source_index"] for row in selected] == [0, 2]
    assert ledger[1]["status"] == "rejected" and "duplicate" in ledger[1]["reason"]


def test_real_corpus_uses_published_loader_without_synthetic_length_rewriting(tmp_path):
    config = {"corpus": {"count": 2}, "tokenizer": {"name": "pinned", "revision": "a" * 40},
              "workload": {"endpoints": ["completions"], "requests": 2, "warmup_requests": 0}}
    case, = cases(config)
    assert case["input_tokens"] is None
    args = arguments("http://localhost", config, case, input_file=tmp_path / "inputs.jsonl")
    assert args[args.index("--custom-dataset-type") + 1] == "single_turn"
    assert "--synthetic-input-tokens-mean" not in args and "--warmup-request-count" not in args
    config["workload"]["requests"] = 1
    with pytest.raises(ConfigError, match="every prompt"):
        cases(config)


@pytest.fixture
def captured_request(tmp_path):
    prompt = "original\u2028text"
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    corpus = {"selected": [{"prompt": prompt, "prompt_sha256": sha, "prompt_bytes": len(prompt.encode()),
                             "input_tokens": 3, "source_index": 7}]}
    raw = {"metadata": {"x_request_id": "request", "benchmark_phase": "profiling"},
           "payload": {"prompt": prompt}, "status": 200,
           "responses": [{"text": json.dumps({"choices": [{"text": "answer"}]})}]}
    server = {"status": 200, "prompt_sha256": sha, "prompt_bytes": len(prompt.encode()),
              "completion_sha256": hashlib.sha256(b"answer").hexdigest()}
    rows = [{"request_id": "request", "server": server}]
    source = tmp_path / "inputs.jsonl"
    source.write_text(json.dumps({"text": prompt}) + "\n")
    corpus.update(input_file=str(source), input_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    run = SimpleNamespace(raw_records=lambda phase: [raw], directory=tmp_path)
    return run, rows, corpus, raw, server


def test_verification_joins_original_text_actual_wire_and_native_output(captured_request):
    run, rows, corpus, raw, server = captured_request
    prompt = corpus["selected"][0]["prompt"]
    assert verify(run, rows, corpus)["requests"][0]["source_index"] == 7
    verify_and_record(run, rows, corpus)
    assert json.loads((run.directory / "corpus-verification.json").read_text())["verified"] is True
    raw["payload"]["prompt"] = "rewritten text"
    with pytest.raises(ConfigError, match="changed text"):
        verify(run, rows, corpus)
    raw["payload"]["prompt"] = prompt
    server["completion_sha256"] = "0" * 64
    with pytest.raises(ConfigError, match="native input/output"):
        verify(run, rows, corpus)


@pytest.mark.parametrize("payload", [None, False, [], ["text"], "text", {"prompt": []}])
def test_malformed_export_preserves_a_failed_verification_record(captured_request, payload):
    run, rows, corpus, raw, _ = captured_request
    raw["payload"] = payload
    with pytest.raises(ConfigError, match="changed text"):
        verify_and_record(run, rows, corpus)
    verdict = json.loads((run.directory / "corpus-verification.json").read_text())
    assert verdict["verified"] is False and verdict["request_records"] == "joined.jsonl"


@pytest.mark.parametrize("missing", [False, True])
def test_changed_or_missing_corpus_preserves_a_failed_verification_record(captured_request, missing):
    run, rows, corpus, _, _ = captured_request
    path = run.directory / "inputs.jsonl"
    if missing:
        path.unlink()
    else:
        path.write_text("changed\n")
    with pytest.raises(ConfigError, match="input"):
        verify_and_record(run, rows, corpus)
    verdict = json.loads((run.directory / "corpus-verification.json").read_text())
    assert verdict["verified"] is False and "input" in verdict["error"]
