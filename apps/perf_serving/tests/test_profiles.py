# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from trtmc_benchmark.types import BenchmarkError
from trtmc_perf_serving.profiles import resolve_profile

FAMILIES = Path(__file__).resolve().parents[3] / "families"


def test_a_server_of_whole_requests_serves_a_testcase_that_is_no_request() -> None:
    # blip's first testcase is an unconditional caption (an empty prompt) that only its family test sends: a
    # server whose clients send whole requests starts with an empty base request; a payload still needs one.
    profile = resolve_profile("blip-image-captioning-base", manifest_root=FAMILIES, request_optional=True)
    assert profile.operation == "generate" and profile.base_request == {}
    with pytest.raises(BenchmarkError, match="non-empty prompt"):
        resolve_profile("blip-image-captioning-base", manifest_root=FAMILIES)
    assert resolve_profile("distilgpt2", manifest_root=FAMILIES, request_optional=True).base_request["prompt"]
