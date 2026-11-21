# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from trtmc_aiperf_qual.config import ConfigError
from trtmc_aiperf_qual.text_specbench import prepare, select
from trtmc_aiperf_qual.text_corpus import verify_and_record
from trtmc_aiperf_qual.text_profile import cases


def test_specbench_cap_sweeps_are_rejected_before_the_native_work_can_be_misidentified():
    config = {"corpus": {"format": "spec_bench", "count": 1},
              "workload": {"endpoints": ["completions"], "output_tokens": [16, 32],
                           "requests": 1, "warmup_requests": 0}}
    with pytest.raises(ConfigError, match="one output-token cap"):
        cases(config)


def test_selection_preserves_original_first_turn_and_source_identity():
    prompt = "  Hello\u2028world  "
    records = [None, {"question_id": "large", "turns": ["x" * 100]},
        {"question_id": "first", "category": "writing", "turns": [prompt, "later"]},
        {"question_id": "duplicate", "turns": [prompt]}, {"question_id": "last", "turns": ["short?"]}]
    selected, ledger = select(records, 2, 40, 4, list)
    assert [row["source_index"] for row in selected] == [2, 4]
    assert selected[0]["prompt"] == prompt and selected[0]["question_id"] == "first"
    assert selected[0]["category"] == "writing"
    assert [row["status"] for row in ledger] == ["rejected", "rejected", "selected", "rejected", "selected"]
    assert ledger[1]["input_tokens"] == 100 and ledger[3]["reason"] == "duplicate selected prompt text"


def test_preparation_pins_source_keeps_original_strings_and_labels_ordinary_mode(tmp_path, monkeypatch):
    prompt = "  Hello\u2028world  "
    data = (json.dumps({"question_id": 7, "category": "writing", "turns": [prompt]}, ensure_ascii=False) + "\n").encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(data))
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)))
    source = {"revision": "a" * 40, "url": "https://example.com/" + "a" * 40 + "/source.jsonl",
              "sha256": hashlib.sha256(data).hexdigest()}
    config = {"checkpoint": {"name": "test", "revision": "b" * 40, "precision": "fp16"},
              "tokenizer": {"name": "test", "revision": "b" * 40},
              "corpus": {"format": "spec_bench", "count": 1, "source": source,
                         "runtime_audit": {"source_file": "AGENTS.md", "conclusion": "controlled fixture audit"}}}
    report = prepare(config, 40, 4, tmp_path, repo=Path(__file__).resolve().parents[3])
    assert report["source_sha256"] == source["sha256"]
    assert json.loads((tmp_path / "inputs.jsonl").read_text(encoding="utf-8"))["text"] == prompt
    assert report["selected"][0]["question_id"] == 7
    assert report["runtime_mode"] == "ordinary_native_generation"
    assert report["speculative_execution"]["speculative_execution_qualified"] is False
    assert report["speculative_execution"]["draft_model"] is None
    assert report["speculative_execution"]["acceptance_metrics"] is None
    assert report["runtime_audit"]["source_sha256"]
    source["sha256"] = "wrong"
    with pytest.raises(ConfigError, match="checksum"):
        prepare(config, 40, 4, tmp_path)
    config["corpus"]["runtime_mode"] = "speculative"
    with pytest.raises(ConfigError, match="ordinary generation only"):
        prepare(config, 40, 4, tmp_path)


def test_corpus_name_does_not_hide_changed_native_generation_work(tmp_path):
    selected, _ = select([{"question_id": 1, "turns": ["hello"]}], 1, 40, 4, list)
    path = tmp_path / "inputs.jsonl"
    path.write_text('{"text":"hello"}\n')
    corpus = {"format": "spec_bench", "selected": selected, "reserved_output_tokens": 4,
        "runtime_mode": "ordinary_native_generation", "runtime_audit": {},
        "speculative_execution": {"speculative_execution_qualified": False, "acceptance_metrics": None},
        "input_file": str(path), "input_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    raw = {"metadata": {"x_request_id": "one", "benchmark_phase": "profiling"}, "status": 200,
        "payload": {"prompt": "hello"}, "responses": [{"text": '{"choices":[{"text":"OK"}]}'}]}
    server = {"status": 200, "prompt_sha256": hashlib.sha256(b"hello").hexdigest(), "prompt_bytes": 5,
        "completion_sha256": hashlib.sha256(b"OK").hexdigest(), "completion_tokens": 2,
        "generation_config": {"max_new_tokens": 4}, "timing_scope": "public_task_call_wall"}
    run = SimpleNamespace(directory=tmp_path, raw_records=lambda phase: [raw])
    verify_and_record(run, [{"request_id": "one", "server": server}], corpus)
    report = json.loads((tmp_path / "corpus-verification.json").read_text())
    assert report["verified"] and report["runtime_mode"] == "ordinary_native_generation"
    assert report["speculative_execution"]["acceptance_metrics"] is None
    server["generation_config"]["max_new_tokens"] = 32
    with pytest.raises(ConfigError, match="changed generation work"):
        verify_and_record(run, [{"request_id": "one", "server": server}], corpus)
    assert json.loads((tmp_path / "corpus-verification.json").read_text())["verified"] is False
