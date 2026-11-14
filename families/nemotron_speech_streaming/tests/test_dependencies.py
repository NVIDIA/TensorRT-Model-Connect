# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_official_reference_dependencies_import():
    pytest.importorskip("nemo", exc_type=ModuleNotFoundError)
    # Import the real NeMo -> datasets -> PyArrow chain before an expensive build.
    # ImportError from an installed dependency must fail rather than become a skip.
    from nemo.collections.asr.models import ASRModel

    assert callable(ASRModel.from_pretrained)
