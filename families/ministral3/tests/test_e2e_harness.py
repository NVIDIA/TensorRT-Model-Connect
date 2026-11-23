# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the E2E harness: native call plan and the parity oracle."""

import math
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


EMBEDDINGS = {"q1": [0.6, 0.8], "q2": [0.8, 0.6], "d1": [1.0, 0.0], "d2": [0.0, 1.0]}


def native_calls(tmp_path, monkeypatch):
    """Run ``_native`` on CASE against a stub runtime; return the calls and the result."""
    for language in ("c", "cpp"):
        (tmp_path / f"test_ministral3_sdk_{language}").touch()
    calls = []

    def run(program, arguments, runtime_root, timeout):
        calls.append((Path(program).name, arguments))
        text = arguments[arguments.index("--text") + 1] if "--text" in arguments else arguments[-1]
        if "text_to_token_features" in arguments:
            return {
                "shape": [1, 2],
                "values": [1.0, 2.0],
                "tokens": [{"token_id": 7, "byte_offsets": [0, 2]}],
            }
        if "text_to_pooled_features" in arguments:
            return {"pooling": "mean", "normalization": "none", "values": [1.0, 2.0]}
        # Each text has its own embedding, so a swapped text changes the result.
        return {
            "pooling": "mean",
            "normalization": "l2",
            "values": EMBEDDINGS[text],
            "task": "text_to_embedding",
        }

    monkeypatch.setattr(test_e2e, "_run_json", run)
    monkeypatch.setattr(test_e2e, "record_evidence", lambda *args: None)
    native = test_e2e._native(Path("trtmc"), tmp_path, tmp_path, tmp_path / "b.bundle", CASE)
    return calls, native


def test_native_runs_cli_consumers_and_feature_tasks(tmp_path, monkeypatch):
    calls, native = native_calls(tmp_path, monkeypatch)
    embedded = [
        (arguments[arguments.index("--role") + 1], arguments[arguments.index("--text") + 1])
        for _, arguments in calls
        if arguments[0] == "embed"
    ]
    assert embedded == [("query", "q1"), ("query", "q2"), ("document", "d1"), ("document", "d2")]
    consumer_arguments = [str(tmp_path / "b.bundle"), str(tmp_path), "query", "q1"]
    assert calls[4:6] == [
        ("test_ministral3_sdk_c", consumer_arguments),
        ("test_ministral3_sdk_cpp", consumer_arguments),
    ]
    token_texts = [
        arguments[arguments.index("--text") + 1]
        for _, arguments in calls
        if "text_to_token_features" in arguments
    ]
    assert token_texts == ["q1", "q2", "d1", "d2"]
    assert "text_to_pooled_features" in calls[-1][1] and "d1" in calls[-1][1]
    np.testing.assert_allclose(
        native["embeddings"], [EMBEDDINGS[text] for text in ("q1", "q2", "d1", "d2")], rtol=1e-7
    )
    assert len(native["tokens"]) == 4


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


def valid_row(row):
    """A row the E2E report accepts: plain finite numbers and a verdict that matches them."""
    assert set(row) == {"name", "label", "scope", "actual", "operator", "expected", "passed"}
    assert row["scope"] in {"contract", "independent_reference"}
    assert row["operator"] in {"==", ">=", "<="} and row["label"]
    assert type(row["actual"]) in (int, float) and type(row["expected"]) in (int, float)
    assert math.isfinite(row["actual"]) and math.isfinite(row["expected"])
    assert type(row["passed"]) is bool
    evaluated = {
        "==": row["actual"] == row["expected"],
        ">=": row["actual"] >= row["expected"],
        "<=": row["actual"] <= row["expected"],
    }[row["operator"]]
    assert evaluated is row["passed"]


def test_every_check_is_recorded_before_it_is_asserted():
    """The report reads these rows; a failing check is recorded as failed before it raises."""
    native, truth = parity_inputs(np.random.default_rng(5), 0.0)
    recorded = []
    test_e2e._assert_parity(
        native, truth, None, CASE, test_e2e.DEFAULT_THRESHOLDS, record=recorded.append
    )
    (rows,) = recorded
    names = [row["name"] for row in rows]
    assert len(set(names)) == len(names)
    assert {"token_ids_differing_texts", "embedding_cosine_min", "pooled_feature_cosine"} <= set(
        names
    )
    for row in rows:
        valid_row(row)
    assert all(row["passed"] for row in rows)
    recorded.clear()
    with pytest.raises(AssertionError, match="token-feature relative error"):
        test_e2e._assert_parity(
            scaled(native, 1.01),
            truth,
            None,
            CASE,
            test_e2e.DEFAULT_THRESHOLDS,
            record=recorded.append,
        )
    (rows,) = recorded
    for row in rows:
        valid_row(row)
    assert [row["name"] for row in rows if not row["passed"]] == [
        "token_feature_relative_error_max"
    ]


def test_bf16_checks_are_recorded_relative_to_the_official_run():
    native, truth, official = bf16_inputs()
    recorded = []
    test_e2e._assert_parity(
        native, truth, official, case_of(30), test_e2e.DEFAULT_THRESHOLDS, record=recorded.append
    )
    (rows,) = recorded
    names = {row["name"] for row in rows}
    assert {
        "bf16_token_count",
        "bf16_embedding_mean",
        "bf16_token_relative_max",
        "bf16_pooled_feature_relative_error",
    } <= names
    for row in rows:
        valid_row(row)
    assert all(row["passed"] for row in rows)


def test_token_feature_shapes_are_checked_before_their_drift():
    """A native row count that would broadcast, or one that cannot, fails on its own check."""
    native, truth = parity_inputs(np.random.default_rng(6), 0.0, rows=3)
    case = case_of(3)
    test_e2e._assert_parity(native, truth, None, case, test_e2e.DEFAULT_THRESHOLDS)
    row = truth["tokens"][0]["hidden"][:1]
    repeated = dict(truth, tokens=list(truth["tokens"]))
    repeated["tokens"][0] = dict(truth["tokens"][0], hidden=np.repeat(row, 3, axis=0))
    one_row = dict(native, tokens=list(native["tokens"]))
    one_row["tokens"][0] = dict(native["tokens"][0], hidden=row)
    with pytest.raises(AssertionError, match="token-feature shape"):
        test_e2e._assert_parity(one_row, repeated, None, case, test_e2e.DEFAULT_THRESHOLDS)
    two_rows = dict(native, tokens=list(native["tokens"]))
    two_rows["tokens"][0] = dict(native["tokens"][0], hidden=native["tokens"][0]["hidden"][:2])
    with pytest.raises(AssertionError, match="token-feature shape"):
        test_e2e._assert_parity(two_rows, truth, None, case, test_e2e.DEFAULT_THRESHOLDS)


def test_native_embeddings_must_be_unit_norm():
    """Queries x2 and documents x0.5 keep every cosine, score and ranking; only the norm shows."""
    weights = np.array([[2.0], [2.0], [0.5], [0.5]])
    native, truth = parity_inputs(np.random.default_rng(7), 0.0)
    test_e2e._assert_parity(native, truth, None, CASE, test_e2e.DEFAULT_THRESHOLDS)
    stretched = dict(native, embeddings=native["embeddings"] * weights)
    with pytest.raises(AssertionError, match="not L2-normalized"):
        test_e2e._assert_parity(stretched, truth, None, CASE, test_e2e.DEFAULT_THRESHOLDS)
    native, truth, official = bf16_inputs()
    stretched = dict(native, embeddings=native["embeddings"] * weights)
    with pytest.raises(AssertionError, match="not L2-normalized"):
        test_e2e._assert_parity(
            stretched, truth, official, case_of(30), test_e2e.DEFAULT_THRESHOLDS
        )


def ranking_inputs(truth_scores, native_scores):
    """One query whose documents have the given FP32-truth and native scores."""
    count = len(truth_scores)

    def embeddings(scores):
        rows = [np.eye(count + 1)[0]]
        for index, score in enumerate(scores):
            row = np.zeros(count + 1)
            row[0], row[index + 1] = score, math.sqrt(1 - score * score)
            rows.append(row)
        return np.stack(rows)

    rng = np.random.default_rng(9)
    tokens = [
        {"token_ids": [1], "byte_offsets": [[0, 1]], "hidden": rng.standard_normal((1, 8))}
        for _ in range(count + 1)
    ]
    case = {"queries": ["q"], "documents": [chr(ord("a") + index) for index in range(count)]}
    truth = {"embeddings": embeddings(truth_scores), "tokens": tokens}
    native = dict(truth, embeddings=embeddings(native_scores), pooled=tokens[1]["hidden"].mean(0))
    return native, truth, case


def test_retrieval_order_below_the_top_document_is_checked():
    """FP32: documents further apart in the truth than the score bound keep their order."""
    native, truth, case = ranking_inputs(
        [0.8, 0.4 + 1.5e-5, 0.4], [0.8, 0.4 + 0.7e-5, 0.4 + 0.8e-5]
    )
    with pytest.raises(AssertionError, match="truth's order"):
        test_e2e._assert_parity(native, truth, None, case, test_e2e.DEFAULT_THRESHOLDS)
    # Closer than the bound in the truth, two documents are a tie and may swap.
    native, truth, case = ranking_inputs(
        [0.8, 0.4 + 0.5e-5, 0.4], [0.8, 0.4 + 0.1e-5, 0.4 + 0.4e-5]
    )
    summary = test_e2e._assert_parity(native, truth, None, case, test_e2e.DEFAULT_THRESHOLDS)
    assert summary["retrieval_order_tolerance"] == test_e2e.DEFAULT_THRESHOLDS["score_abs_diff"]


def test_undefined_metrics_are_recorded_as_failed_contracts():
    """A zero native embedding has no cosine; its row stays a finite, failed contract check."""
    native, truth = parity_inputs(np.random.default_rng(8), 0.0)
    native["embeddings"] = native["embeddings"].copy()
    native["embeddings"][0] = 0.0
    recorded = []
    with pytest.raises(AssertionError):
        test_e2e._assert_parity(
            native, truth, None, CASE, test_e2e.DEFAULT_THRESHOLDS, record=recorded.append
        )
    (rows,) = recorded
    for row in rows:
        valid_row(row)
    assert "embedding_cosine_min_defined" in {row["name"] for row in rows if not row["passed"]}


def test_an_exact_bf16_match_passes_despite_cosine_round_off():
    """The computed cosine of [1, 1, 1] with itself is 1 + 2e-16; an exact match drifts 0."""
    rows = 30
    ones = np.ones((rows, 3))
    tokens = [
        {
            "token_ids": [1] * rows,
            "byte_offsets": [[index, index + 1] for index in range(rows)],
            "hidden": ones,
        }
        for _ in range(4)
    ]
    exact = {"embeddings": np.eye(3)[[0, 1, 2, 0]], "tokens": tokens, "pooled": ones.mean(axis=0)}
    summary = test_e2e._assert_parity(
        exact, exact, exact, case_of(rows), test_e2e.DEFAULT_THRESHOLDS
    )
    assert summary["drift_statistics"]["token"]["native_max"] == 0.0


def test_reference_restores_the_process_wide_tf32_settings(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    def settings():
        return (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.get_float32_matmul_precision(),
        )

    saved = settings()[:2]
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        before = settings()
        # The checkpoint directory is empty, so the reference fails after disabling
        # TF32. A case runs it twice in one process (FP32 truth, official BF16 run).
        for precision in ("fp32", "bf16"):
            with pytest.raises(FileNotFoundError):
                test_e2e._official_reference(tmp_path, CASE, precision)
            assert settings() == before
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved
