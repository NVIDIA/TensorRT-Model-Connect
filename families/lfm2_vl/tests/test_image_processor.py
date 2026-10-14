# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native pixel tensors must be bit-exact to the independent processor oracle."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORACLE = json.loads((HERE / "fixtures/preprocessing-reference.json").read_text())


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    output = tmp_path_factory.mktemp("image-probe") / "probe"
    subprocess.run(
        [
            shutil.which("c++"),
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT),
            "-I",
            str(ROOT / "core/runtime/include"),
            str(HERE / "cpp/image_preprocess_probe.cpp"),
            str(HERE.parent / "runtime/image_processor.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda case: case["name"])
def test_reference_pixels_shapes_and_placeholders(probe, tmp_path, case):
    prefix = tmp_path / "tile-"
    result = subprocess.run(
        [str(probe), *map(str, case["arguments"]), str(prefix)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert hashlib.sha256(result[0].encode()).hexdigest() == case["placeholders_sha256"]
    assert [list(map(int, line.split())) for line in result[1:]] == [
        tile["shape"] for tile in case["tiles"]
    ]
    for i, tile in enumerate(case["tiles"]):
        assert hashlib.sha256(Path(str(prefix) + str(i)).read_bytes()).hexdigest() == tile["sha256"]
        assert (
            hashlib.sha256(Path(str(prefix) + str(i) + ".padded").read_bytes()).hexdigest()
            == tile["padded_sha256"]
        )
        assert (
            hashlib.sha256(Path(str(prefix) + str(i) + ".mask").read_bytes()).hexdigest()
            == tile["mask_sha256"]
        )


@pytest.mark.parametrize(
    "arguments",
    [
        [0, 256, 64, 256, 10, 1],
        [256, 256, 0, 256, 10, 1],
        [256, 256, 256, 64, 10, 1],
        [256, 256, 64, 2048, 10, 1],
        [256, 256, 64, 256, 11, 1],
        [256, 256, 64, 256, 1, 1],
    ],
)
def test_invalid_image_budget_fails(probe, tmp_path, arguments):
    result = subprocess.run(
        [str(probe), *map(str, arguments), str(tmp_path / "tile-")], capture_output=True, text=True
    )
    assert result.returncode != 0
