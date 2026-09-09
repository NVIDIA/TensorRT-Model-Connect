# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selection contracts for family-owned LFM2 E2E cases."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from .test_e2e import _CASES, _require_selected


@dataclass
class _Config:
    options: dict[str, object] = field(default_factory=dict)

    def getoption(self, name: str, default=None):
        return self.options.get(name, default)


def test_exact_testcase_selector_does_not_select_sibling_from_same_manifest() -> None:
    manifest, _ = _CASES["lfm2-350m-chat"]
    config = _Config(
        {
            "--e2e-model": [],
            "--e2e-testcase": ["lfm2-350m-fp16"],
            "--e2e-models-file": None,
        }
    )

    with pytest.raises(pytest.skip.Exception, match="lfm2-350m-chat was not selected"):
        _require_selected("lfm2-350m-chat", manifest, config)
