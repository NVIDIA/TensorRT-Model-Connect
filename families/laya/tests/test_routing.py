# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import pytest


@pytest.mark.trt
def test_released_routing_metadata(tmp_path, request):
    pytest.importorskip("laya")
    probe = request.getfixturevalue("laya_routing_probe")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "families.laya.tests.compare_routing",
            "--probe",
            str(probe),
            "--output",
            str(tmp_path),
        ],
        check=True,
    )
