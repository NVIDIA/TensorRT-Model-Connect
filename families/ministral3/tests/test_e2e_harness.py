# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the E2E harness: native call plan and the parity oracle."""

from pathlib import Path

import numpy as np
import pytest

from families.ministral3.tests import test_e2e

CASE = {"queries": ["q1", "q2"], "documents": ["d1", "d2"]}


def case_of(rows):
    """A case whose texts have ``rows`` bytes, one byte per synthetic token."""
    return {"queries": ["q" * rows, "r" * rows], "documents": ["d" * rows, "e" * rows]}


def test_every_manifest_case_is_well_formed():
    premerge = [name for name, (_, _, case) in test_e2e.CASES.items() if case.get("premerge")]
    assert premerge == ["nemotron-3-embed-1b-bf16"]
    for _, manifest, case in test_e2e.CASES.values():
        assert manifest["precision"] in {"bf16", "fp32"}
        assert case["queries"] and case["documents"]


def test_native_runs_cli_consumers_and_feature_tasks(tmp_path, monkeypatch):
    for language in ("c", "cpp"):
        (tmp_path / f"test_ministral3_sdk_{language}").touch()
    calls = []

    def run(program, arguments, runtime_root, timeout):
        calls.append((Path(program).name, arguments))
        if "text_to_token_features" in arguments:
            return {
                "shape": [1, 2],
                "values": [1.0, 2.0],
                "tokens": [{"token_id": 7, "byte_offsets": [0, 2]}],
            }
        if "text_to_pooled_features" in arguments:
            return {"pooling": "mean", "normalization": "none", "values": [1.0, 2.0]}
        return {
            "pooling": "mean",
            "normalization": "l2",
            "values": [0.6, 0.8],
            "task": "text_to_embedding",
        }

    monkeypatch.setattr(test_e2e, "_run_json", run)
    monkeypatch.setattr(test_e2e, "record_evidence", lambda *args: None)
    native = test_e2e._native(Path("trtmc"), tmp_path, tmp_path, tmp_path / "b.bundle", CASE)
    roles = [
        arguments[arguments.index("--role") + 1] for _, arguments in calls if "--role" in arguments
    ]
    assert roles == ["query", "query", "document", "document"]
    assert [name for name, _ in calls[4:6]] == ["test_ministral3_sdk_c", "test_ministral3_sdk_cpp"]
    assert calls[4][1][2:] == ["query", "q1"]
    token_texts = [
        arguments[arguments.index("--text") + 1]
        for _, arguments in calls
        if "text_to_token_features" in arguments
    ]
    assert token_texts == ["q1", "q2", "d1", "d2"]
    assert "text_to_pooled_features" in calls[-1][1] and "d1" in calls[-1][1]
    assert native["embeddings"].shape == (4, 2) and len(native["tokens"]) == 4


def parity_inputs(rng, noise, texts=("q1", "q2", "d1", "d2"), rows=2):
    def unit(rows):
        values = rng.standard_normal((rows, 8))
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    def disturb(values):
        return values + noise * rng.standard_normal(values.shape)

    truth_embeddings = unit(len(texts))
    truth_tokens = [
        {
            "token_ids": [1] * rows,
            "byte_offsets": [[index, index + 1] for index in range(rows)],
            "hidden": rng.standard_normal((rows, 8)),
        }
        for _ in texts
    ]
    native_embeddings = disturb(truth_embeddings)
    native = {
        "embeddings": native_embeddings / np.linalg.norm(native_embeddings, axis=1, keepdims=True),
        "tokens": [dict(item, hidden=disturb(item["hidden"])) for item in truth_tokens],
        "pooled": truth_tokens[2]["hidden"].mean(axis=0),
    }
    return native, {"embeddings": truth_embeddings, "tokens": truth_tokens}


def test_fp32_oracle_accepts_rounding_and_rejects_drift():
    native, truth = parity_inputs(np.random.default_rng(0), 1e-8)
    summary = test_e2e._assert_parity(native, truth, None, CASE, test_e2e.DEFAULT_THRESHOLDS)
    assert summary["embedding_cosine_min"] > 0.999999 and summary["tokens"] == 8
    native, truth = parity_inputs(np.random.default_rng(0), 1e-2)
    with pytest.raises(AssertionError):
        test_e2e._assert_parity(native, truth, None, CASE, test_e2e.DEFAULT_THRESHOLDS)


def bf16_inputs():
    rng = np.random.default_rng(1)
    native, truth = parity_inputs(rng, 1e-3, rows=30)
    official, _ = parity_inputs(np.random.default_rng(1), 1e-3, rows=30)
    official["tokens"] = [
        dict(item, hidden=item["hidden"] + 2e-3 * rng.standard_normal(item["hidden"].shape))
        for item in truth["tokens"]
    ]
    official["embeddings"] = native["embeddings"].copy()
    return native, truth, official


def test_bf16_oracle_is_relative_to_the_official_drift():
    native, truth, official = bf16_inputs()
    case = case_of(30)
    summary = test_e2e._assert_parity(native, truth, official, case, test_e2e.DEFAULT_THRESHOLDS)
    assert set(summary["drift_statistics"]) == {"embedding", "score", "token", "token_relative"}
    assert summary["pooled_drift"]["native"] <= summary["pooled_drift"]["official_bf16"] * 3
    worse, _ = parity_inputs(np.random.default_rng(2), 1e-1, rows=30)
    worse["tokens"], worse["pooled"] = native["tokens"], native["pooled"]
    with pytest.raises(AssertionError):
        test_e2e._assert_parity(worse, truth, official, case, test_e2e.DEFAULT_THRESHOLDS)
    few, few_truth = parity_inputs(np.random.default_rng(1), 1e-3)
    with pytest.raises(AssertionError, match="100 tokens"):
        test_e2e._assert_parity(few, few_truth, few, CASE, test_e2e.DEFAULT_THRESHOLDS)


def test_bf16_long_documents_are_checked_one_by_one():
    """Only the per-document check can reject a drifted long document here."""
    case = dict(case_of(30), repeated_documents=[{"text": "long", "count": 2}])
    rng = np.random.default_rng(4)
    _, truth = parity_inputs(rng, 0.0, texts=("q1", "q2", "d1", "d2", "long"), rows=30)
    truth["tokens"] = truth["tokens"][:4]

    def noisy(values, scales):
        result = values + np.asarray(scales)[:, None] * rng.standard_normal(values.shape)
        return result / np.linalg.norm(result, axis=1, keepdims=True)

    def tokens(scale):
        return [
            dict(item, hidden=item["hidden"] + scale * rng.standard_normal(item["hidden"].shape))
            for item in truth["tokens"]
        ]

    official = {
        "embeddings": noisy(truth["embeddings"], [1e-1] * 4 + [1e-5]),
        "tokens": tokens(1e-2),
    }

    def native(long_noise):
        values = {
            "embeddings": noisy(truth["embeddings"], [1e-5] * 4 + [long_noise]),
            "tokens": tokens(1e-4),
        }
        values["pooled"] = values["tokens"][2]["hidden"].mean(axis=0)
        return values

    summary = test_e2e._assert_parity(
        native(1e-5), truth, official, case, test_e2e.DEFAULT_THRESHOLDS
    )
    assert len(summary["long_documents"]) == 1
    with pytest.raises(AssertionError, match="long document"):
        test_e2e._assert_parity(native(1e-2), truth, official, case, test_e2e.DEFAULT_THRESHOLDS)


def test_token_ids_and_offsets_must_match():
    native, truth = parity_inputs(np.random.default_rng(3), 0.0)
    test_e2e._assert_parity(native, truth, None, CASE, test_e2e.DEFAULT_THRESHOLDS)
    changed_ids = dict(native, tokens=list(native["tokens"]))
    changed_ids["tokens"][1] = dict(native["tokens"][1], token_ids=[1, 3])
    with pytest.raises(AssertionError, match="token IDs"):
        test_e2e._assert_parity(changed_ids, truth, None, CASE, test_e2e.DEFAULT_THRESHOLDS)
    changed_offsets = dict(native, tokens=list(native["tokens"]))
    changed_offsets["tokens"][1] = dict(native["tokens"][1], byte_offsets=[[0, 1]])
    with pytest.raises(AssertionError, match="byte offsets"):
        test_e2e._assert_parity(changed_offsets, truth, None, CASE, test_e2e.DEFAULT_THRESHOLDS)


def test_offsets_must_be_contiguous_not_only_cover_the_bytes():
    """For ``aa`` the spans [0, 1] and [0, 1] concatenate to the input but do not tile it."""
    native, truth = parity_inputs(np.random.default_rng(3), 0.0)
    case = dict(CASE, queries=["aa", "q2"])
    test_e2e._assert_parity(native, truth, None, case, test_e2e.DEFAULT_THRESHOLDS)
    for offsets in ([[0, 1], [0, 1]], [[0, 1], [1, 1], [1, 2]], [[0, 2], [2, 2]]):
        repeated = dict(native, tokens=list(native["tokens"]))
        repeated["tokens"][0] = dict(native["tokens"][0], byte_offsets=offsets)
        with pytest.raises(AssertionError, match="byte offsets"):
            test_e2e._assert_parity(repeated, truth, None, case, test_e2e.DEFAULT_THRESHOLDS)


def scaled(values, factor):
    result = dict(
        values, tokens=[dict(item, hidden=item["hidden"] * factor) for item in values["tokens"]]
    )
    return result


def test_fp32_raw_features_are_checked_for_length_not_only_direction():
    """Features scaled by 1 % keep every cosine at 1 and must still fail."""
    native, truth = parity_inputs(np.random.default_rng(5), 0.0)
    test_e2e._assert_parity(native, truth, None, CASE, test_e2e.DEFAULT_THRESHOLDS)
    with pytest.raises(AssertionError, match="token-feature relative error"):
        test_e2e._assert_parity(
            scaled(native, 1.01), truth, None, CASE, test_e2e.DEFAULT_THRESHOLDS
        )
    longer = dict(native, pooled=native["pooled"] * 1.01)
    with pytest.raises(AssertionError, match="pooled-feature relative error exceeds its bound"):
        test_e2e._assert_parity(longer, truth, None, CASE, test_e2e.DEFAULT_THRESHOLDS)


def test_bf16_raw_features_are_checked_for_length_not_only_direction():
    native, truth, official = bf16_inputs()
    case = case_of(30)
    test_e2e._assert_parity(native, truth, official, case, test_e2e.DEFAULT_THRESHOLDS)
    with pytest.raises(AssertionError, match="token_relative mean drift"):
        test_e2e._assert_parity(
            scaled(native, 1.01), truth, official, case, test_e2e.DEFAULT_THRESHOLDS
        )
    longer = dict(native, pooled=native["pooled"] * 1.01)
    with pytest.raises(AssertionError, match="pooled-feature relative error"):
        test_e2e._assert_parity(longer, truth, official, case, test_e2e.DEFAULT_THRESHOLDS)
