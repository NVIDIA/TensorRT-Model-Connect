# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate build selection before loading checkpoints or TensorRT."""

from pathlib import Path

import pytest

from families.lfm2_audio.model import BuildRequest


def test_standard_tensorrt_request():
    request = BuildRequest(Path("unused"))
    assert request.backend == "trt" and request.precision == "bf16"


@pytest.mark.parametrize("backend", ["trt_rtx", "unknown"])
def test_reject_backend_that_cannot_build_these_plans(backend):
    with pytest.raises(ValueError, match="only standard TensorRT"):
        BuildRequest(Path("unused"), backend=backend)
