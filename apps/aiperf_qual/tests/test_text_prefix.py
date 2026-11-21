# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from types import SimpleNamespace

import pytest

from trtmc_aiperf_qual.config import ConfigError
from trtmc_aiperf_qual.text_prefix import construct, verify
from trtmc_aiperf_qual.text_profile import cases


@pytest.mark.parametrize("axis,values", [
    ("output_tokens", [16, 32]), ("concurrency", [1, 2]), ("request_rate", [None, 10]),
])
def test_controlled_prefix_sweeps_cannot_reuse_a_process_between_cases(axis, values):
    workload = {"endpoints": ["completions"], "output_tokens": [16], "concurrency": [1],
                "requests": 2, "warmup_requests": 0, axis: values}
    with pytest.raises(ConfigError, match="one workload case per fresh server"):
        cases({"corpus": {"format": "controlled_prefix", "count": 2}, "workload": workload})


def test_overlap_counts_complete_prompt_tokens_at_the_suffix_boundary():
    tokens = {"ab": [1, 9], "abc": [1, 2], "abd": [1, 3]}
    selected, ledger, layout = construct({"prefix": "ab", "suffixes": ["c", "d"], "count": 2},
                                         8, 2, tokens.__getitem__)
    assert layout["configured_prefix_tokens"] == 2
    assert layout["common_prefix_tokens"] == 1
    assert [row["shared_fraction"] for row in selected] == [0.5, 0.5]
    assert [row["prompt"] for row in selected] == ["abc", "abd"]
    assert [row["status"] for row in ledger] == ["selected", "selected"]


def test_over_limit_prefix_is_recorded_without_shortening():
    selected, ledger, _ = construct({"prefix": "prefix", "suffixes": ["short", "longer suffix"], "count": 2},
                                     15, 2, list)
    assert [row["prompt"] for row in selected] == ["prefixshort"]
    assert ledger[1]["status"] == "rejected" and ledger[1]["input_tokens"] == 19


def test_duplicate_suffixes_are_not_a_controlled_distinct_corpus():
    with pytest.raises(ConfigError, match="distinct"):
        construct({"prefix": "prefix", "suffixes": ["same", "same"], "count": 2}, 256, 16, list)


def test_fresh_warm_policy_is_verified_in_addition_to_exact_prompt_identity(tmp_path):
    selected, _, layout = construct({"prefix": "ab", "suffixes": ["c", "d"], "count": 2}, 256, 16, list)
    raw, joined = [], []
    for index, (phase, prompt) in enumerate([("warmup", "abc"), ("profiling", "abc"), ("profiling", "abd")]):
        raw.append({"metadata": {"x_request_id": str(index), "benchmark_phase": phase},
            "payload": {"prompt": prompt}, "status": 200,
            "responses": [{"text": json.dumps({"choices": [{"text": "OK"}]})}]})
        joined.append({"request_id": str(index), "server": {"status": 200,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_bytes": len(prompt),
            "completion_sha256": hashlib.sha256(b"OK").hexdigest()}})
    corpus = {"selected": selected, "layout": layout,
        "execution_policy": {"warmup_requests": 1, "measured_requests": 2},
        "cache_audit": {"conclusion": "no cross-request reuse"}, "cache_hit_telemetry": "unavailable"}
    run = SimpleNamespace(directory=tmp_path, raw_records=lambda phase: raw)
    report = verify(run, joined, corpus)
    assert report["verified"] and report["measured_unique_prompts"] == 2
    assert report["cache_hit_telemetry"] == "unavailable"
    raw[0]["metadata"]["benchmark_phase"] = "profiling"
    with pytest.raises(ConfigError, match="fresh/warm"):
        verify(run, joined, corpus)
