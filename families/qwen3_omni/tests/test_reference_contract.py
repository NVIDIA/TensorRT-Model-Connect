# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from families.qwen3_omni.tests.test_e2e import _reference_eos_token_id


def test_reference_eos_comes_from_the_root_omni_config() -> None:
    config = SimpleNamespace(im_end_token_id=151645, thinker_config=SimpleNamespace())
    assert _reference_eos_token_id(config) == 151645


@pytest.mark.parametrize("value", [True, "151645", None])
def test_reference_eos_rejects_non_integer_values(value) -> None:
    with pytest.raises(AssertionError):
        _reference_eos_token_id(SimpleNamespace(im_end_token_id=value))
