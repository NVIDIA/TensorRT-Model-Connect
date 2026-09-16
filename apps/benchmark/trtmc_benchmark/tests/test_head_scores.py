# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from trtmc_benchmark.dataframe import head_score_rows, format_head_scores_frame
from trtmc_benchmark.metrics import reduce_metrics
from trtmc_benchmark.task_adapters import resolve_task_case
from trtmc_benchmark.types import BenchmarkError


def observation():
    return {
        "values": [-2.0, 3.0, 6.0, -8.0],
        "shape": [1, 2, 2],
        "score_kind": "logit",
        "pooling": "none",
        "normalization": "none",
        "head_score_tensors": 1,
        "head_score_values": 4,
        "runtime_e2e_wall_ms": 10.0,
    }


def test_head_score_adapter_preserves_text_ids_and_explicit_config(tmp_path):
    text = resolve_task_case("text_to_head_scores", {"prompt": "raw text"}, tmp_path)
    assert text.operation == "head_scores" and text.request == {"prompt": "raw text"}
    tokens = resolve_task_case(
        "text_to_head_scores",
        {"inputs": {"token_ids": [1, 2]}, "config": {"representation": "first"}},
        tmp_path,
    )
    assert tokens.request == {"token_ids": [1, 2], "config": {"representation": "first"}}
    assert tokens.measurement.warmup == 50 and tokens.measurement.iterations == 500


@pytest.mark.parametrize(
    "case",
    [
        {"token_ids": [True]},
        {"token_ids": [1.5]},
        {"token_ids": [1 << 31]},
        {"token_ids": [1], "prompt": "x"},
        {"prompt": "x", "role": "query"},
    ],
)
def test_head_score_adapter_rejects_ambiguous_or_wrong_kind_inputs(tmp_path, case):
    with pytest.raises(BenchmarkError):
        resolve_task_case("text_to_head_scores", case, tmp_path)


def test_head_score_metrics_are_not_embeddings_or_tokens():
    metrics = reduce_metrics("head_scores", [observation(), observation()])
    assert metrics["head_score_tensors_per_s"] == 100
    assert metrics["head_score_values_per_s"] == 400
    assert "embedding_vectors_per_s" not in metrics and "output_tokens_per_s" not in metrics


def test_head_score_table_keeps_real_rank_and_every_signed_value():
    source = observation()
    original = deepcopy(source)
    rows = head_score_rows(source)
    assert [(row["axis_0"], row["axis_1"], row["axis_2"]) for row in rows] == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
    ]
    assert [row["value"] for row in rows] == source["values"]
    assert all(row["score_kind"] == "logit" and row["pooling"] == "none" for row in rows)
    assert source == original


@pytest.mark.parametrize(
    "changes",
    [
        {"shape": []},
        {"shape": [0, 2]},
        {"shape": [True, 4]},
        {"shape": [3, 2]},
        {"score_kind": "hidden"},
        {"pooling": ""},
        {"normalization": None},
    ],
)
def test_head_score_table_rejects_malformed_shape_or_metadata(changes):
    with pytest.raises(ValueError):
        head_score_rows({**observation(), **changes})


def test_head_score_dataframe_preserves_values_shape_and_metadata():
    pytest.importorskip("pandas")
    frame = format_head_scores_frame(observation())
    assert frame.attrs["shape"] == [1, 2, 2]
    assert frame["value"].tolist() == [-2.0, 3.0, 6.0, -8.0]
    assert frame["score_kind"].tolist() == ["logit"] * 4
    assert frame["axis_2"].tolist() == [0, 1, 0, 1]
