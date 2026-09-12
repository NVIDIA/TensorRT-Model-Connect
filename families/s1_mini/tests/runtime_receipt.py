# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen native-KV runtime receipt checks used by family E2E."""

from __future__ import annotations

import re

_PREFILL = re.compile(r"^\[trtmc\.prefill\] tokens=(\d+) launches=(\d+) max_chunk=(\d+)$")
_RUNTIME_ERROR = re.compile(
    r"\[trt\]\s+ERROR:|IExecutionContext::enqueueV3:\s+Error Code|"
    r"Internal Error:|Cuda Runtime|illegal memory access",
    re.IGNORECASE,
)


def prefill_observations(stderr: str) -> tuple[tuple[int, int, int], ...]:
    values = []
    for line in stderr.splitlines():
        match = _PREFILL.fullmatch(line.strip())
        if match:
            values.append(tuple(int(value) for value in match.groups()))
    return tuple(values)


def assert_native_kv_receipt(payload: dict, case: dict, prompt_tokens: int) -> None:
    stderr = str(payload["runtime_stderr"])
    expected_rows = int(case["expected_kv_cache_rows"])
    expected_launches = int(case["expected_prefill_chunks"])
    expected_limit = int(case["expected_prefill_chunk_limit"])
    expected_prompt = case.get("expected_prompt_tokens")
    if expected_prompt is not None:
        expected_prompt = int(expected_prompt)
        assert prompt_tokens == expected_prompt
        assert expected_launches == (expected_prompt + expected_limit - 1) // expected_limit
    observations = prefill_observations(stderr)
    observed_tokens = sum(value[0] for value in observations)
    observed_launches = sum(value[1] for value in observations)
    observed_max_chunk = max((value[2] for value in observations), default=0)
    if expected_prompt is not None:
        assert observed_tokens == expected_prompt
    assert observed_launches == expected_launches
    assert 0 < observed_max_chunk <= expected_limit
    assert f"KV cache rows={expected_rows} (bundle max={expected_rows}" in stderr
    assert not _RUNTIME_ERROR.search(stderr)
    assert len(payload["token_ids"]) == int(case["max_new_tokens"])
    assert float(payload["decode_ms"]) > 0.0
