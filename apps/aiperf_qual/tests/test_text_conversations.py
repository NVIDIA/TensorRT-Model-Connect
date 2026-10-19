# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from trtmc_aiperf_qual.config import ConfigError
from trtmc_aiperf_qual.text_conversations import fingerprint, select, template_tokens, verify_records
from trtmc_aiperf_qual.text_profile import arguments, cases


def conversation(first, second):
    return {"conversations": [{"from": "human", "value": first}, {"from": "gpt", "value": "source answer"},
                              {"from": "human", "value": second}]}


def count_messages(messages):
    return sum(len(message["content"]) for message in messages) + 10


def test_context_budget_counts_token_ids_instead_of_tokenizer_container_fields():
    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            assert options == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return "rendered original history"

        def encode(self, text, *, add_special_tokens):
            assert text == "rendered original history" and add_special_tokens is False
            return [100, 200, 300, 400]

    assert template_tokens(Tokenizer(), []) == 4


def test_original_user_turns_survive_selection_and_rejections_are_visible():
    original = "  first\n\u2028second  "
    row = conversation(original, "Follow up?")
    selected, ledger = select([None, conversation("x" * 200, "Q"), row, row,
                               conversation("another", "follow up")], 2, 100, 8, count_messages)
    assert [entry["source_index"] for entry in selected] == [2, 4]
    assert selected[0]["user_turns"] == [original, "Follow up?"]
    assert [entry["status"] for entry in ledger] == ["rejected", "rejected", "selected", "rejected", "selected"]


def test_multi_turn_uses_the_existing_loader_and_counts_both_requests(tmp_path):
    config = {"corpus": {"format": "sharegpt_multi_turn", "count": 2},
              "tokenizer": {"name": "pinned", "revision": "a" * 40},
              "workload": {"endpoints": ["chat"], "requests": 4, "warmup_requests": 2}}
    case, = cases(config)
    args = arguments("http://localhost", config, case, input_file=tmp_path / "inputs.jsonl")
    assert args[args.index("--custom-dataset-type") + 1] == "multi_turn"
    assert "--synthetic-input-tokens-mean" not in args
    config["workload"]["requests"] = 3
    with pytest.raises(ConfigError, match="every prompt"):
        cases(config)


@pytest.fixture
def capture():
    texts = ["Remember ORCHID.", "What word?"]
    original = [{"role": "user", "content": texts[0]}]
    history = [*original, {"role": "assistant", "content": "OK"}, {"role": "user", "content": texts[1]}]
    raw, joined = [], []
    for turn, (messages, text) in enumerate([(original, "OK"), (history, "ORCHID")]):
        metadata = {"x_request_id": str(turn), "conversation_id": "sharegpt-0", "session_num": turn,
                    "x_correlation_id": "conversation", "turn_index": turn, "benchmark_phase": "profiling",
                    "request_start_ns": 1_000_000_000 + turn * 150_000_000,
                    "request_end_ns": 1_100_000_000 + turn * 150_000_000}
        raw.append({"metadata": metadata, "payload": {"messages": messages}, "status": 200,
                    "responses": [{"text": json.dumps({"choices": [{"message": {"content": text}}]})}]})
        joined.append({"request_id": str(turn), "server": {"status": 200,
            "native_operation": "converse" if turn else "generate",
            "prompt_sha256": hashlib.sha256(texts[0].encode()).hexdigest(),
            "conversation_sha256": fingerprint(messages), "completion_sha256": hashlib.sha256(text.encode()).hexdigest()}})
    corpus = {"selected": [{"session_id": "sharegpt-0", "user_turns": texts}],
              "context_limit": 256, "output_tokens": 8, "think_time_ms": 50}
    return SimpleNamespace(raw_records=lambda phase: raw), joined, corpus, raw


def test_live_assistant_history_and_per_turn_identity_are_verified(capture):
    run, joined, corpus, _ = capture
    result = verify_records(run, joined, corpus, count_messages)
    assert result["verified"] and result["measured_conversations"] == 1
    assert [row["turn_index"] for row in result["requests"]] == [0, 1]
    assert result["requests"][1]["observed_gap_ms"] == 50


@pytest.mark.parametrize("phase", ["profiling", "warmup"])
def test_incomplete_replayed_profiling_group_is_rejected_even_after_full_coverage(capture, phase):
    run, joined, corpus, raw = capture
    replay = copy.deepcopy(raw[0])
    replay["metadata"].update(x_request_id="replay-first-only", x_correlation_id="another-instance",
        session_num=2, benchmark_phase=phase, request_start_ns=1_400_000_000, request_end_ns=1_500_000_000)
    raw.append(replay)
    joined.append({"request_id": "replay-first-only", "server": copy.deepcopy(joined[0]["server"])})
    if phase == "profiling":
        with pytest.raises(ConfigError, match="both turns"):
            verify_records(run, joined, corpus, count_messages)
    else:
        # Warmup request limits may intentionally end before a conversation does.
        assert verify_records(run, joined, corpus, count_messages)["verified"]


@pytest.mark.parametrize("change", ["drop_history", "role", "original_text", "assistant_text", "native_history", "capacity"])
def test_missing_or_changed_history_cannot_be_reported_as_qualified(capture, change):
    run, joined, corpus, raw = capture
    if change == "drop_history":
        raw[1]["payload"]["messages"] = raw[1]["payload"]["messages"][-1:]
    elif change == "role":
        raw[1]["payload"]["messages"][1]["role"] = "user"
    elif change == "original_text":
        raw[1]["payload"]["messages"][2]["content"] = "rewritten"
    elif change == "assistant_text":
        raw[1]["payload"]["messages"][1]["content"] = "source answer"
    elif change == "native_history":
        joined[1]["server"]["conversation_sha256"] = "0" * 64
    else:
        corpus["context_limit"] = 16
    with pytest.raises(ConfigError):
        verify_records(run, joined, corpus, count_messages)
