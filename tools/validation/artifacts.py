# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation artifact contracts shared by reference and comparison runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def int_list(value: Any) -> list[int] | None:
    """Return an integer list when ``value`` satisfies the token-id contract."""
    if not isinstance(value, list):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def generated_token_ids(row: dict[str, Any]) -> list[int] | None:
    """Read generated token IDs from either supported prediction field."""
    generated = int_list(row.get("generated_token_ids"))
    if generated is not None:
        return generated
    return int_list(row.get("token_ids"))


def predictions_file_valid(
    predictions_path: Path,
    answers_path: Path,
    *,
    require_token_ids: bool = False,
) -> bool:
    """Check the durable prediction/answer cardinality and token contracts."""
    if not predictions_path.is_file() or not answers_path.is_file():
        return False
    try:
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        answers = json.loads(answers_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    responses = predictions.get("responses")
    requests = answers.get("requests")
    if (
        not isinstance(responses, list)
        or not isinstance(requests, list)
        or len(responses) != len(requests)
    ):
        return False
    if require_token_ids:
        return all(
            isinstance(row, dict) and generated_token_ids(row) is not None for row in responses
        )
    return True
