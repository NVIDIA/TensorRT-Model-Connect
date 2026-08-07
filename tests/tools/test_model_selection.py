# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from tools import model_selection


def test_model_ci_matrix_order_is_preserved_and_duplicates_are_removed(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "affected_models": ["ignored-order"],
                "matrix": {
                    "include": [
                        {"model": "model-b"},
                        {"model": "model-a"},
                        {"model": "model-b"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    assert model_selection.load_model_selection(selection) == (
        "model-b",
        "model-a",
    )


def test_minimal_models_selection_is_supported(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"models": [" model-a ", "model-b"]}),
        encoding="utf-8",
    )

    assert model_selection.load_model_selection(selection) == (
        "model-a",
        "model-b",
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "JSON object"),
        ({}, "must contain"),
        ({"models": []}, "contains no models"),
        ({"models": [""]}, "must not be empty"),
        ({"matrix": {"include": [{}]}}, "must contain a string model"),
    ],
)
def test_invalid_selections_fail_closed(tmp_path, payload, message):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(model_selection.ModelSelectionError, match=message):
        model_selection.load_model_selection(selection)
