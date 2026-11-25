# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from families.hstu_research.examples import history_cache
from families.hstu_research.triton import build_image_plugin


@pytest.mark.parametrize("module", [history_cache, build_image_plugin])
def test_documented_example_and_deployment_help(module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [module.__name__, "--help"])
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 0
    assert "usage:" in capsys.readouterr().out
