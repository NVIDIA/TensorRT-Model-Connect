# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile native host logic and compare with frozen original-reference outputs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).parent
REPO = ROOT.parents[2]
GOLDEN = json.loads((ROOT / "fixtures/host-golden.json").read_text())


@pytest.fixture(scope="module")
def host_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native host tests require a C++ compiler")
    includes = []
    try:
        import nlohmann_json
        includes.append("-I" + str(nlohmann_json.get_include()))
    except ImportError:
        if not Path("/usr/include/nlohmann/json.hpp").exists():
            pytest.skip("native host tests require nlohmann-json headers")
    binary = tmp_path_factory.mktemp("d3-native") / "host_probe"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic",
                    "-I" + str(REPO), "-I" + str(REPO / "core/runtime/include"), *includes,
                    str(ROOT / "host_probe.cpp"), str(ROOT.parent / "runtime/record.cpp"),
                    str(ROOT.parent / "runtime/media.cpp"), str(ROOT.parent / "runtime/batch.cpp"), str(ROOT.parent / "runtime/bpe_tokenizer.cpp"),
                    "-o", str(binary)], check=True)
    return binary


def compare(a, e):
    if isinstance(e, dict):
        assert a.keys() == e.keys()
        for k in e:
            compare(a[k], e[k])
    elif isinstance(e, list):
        assert len(a) == len(e)
        for x, y in zip(a, e, strict=True):
            compare(x, y)
    elif isinstance(e, float):
        assert a == pytest.approx(e, abs=1e-7)
    else:
        assert a == e


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=[str(i) for i in range(len(GOLDEN["cases"]))])
def test_original_prompt_and_answers(host_probe, tmp_path, case):
    path = tmp_path / "record.json"
    path.write_text(json.dumps(case["record"], ensure_ascii=False))
    command = [str(host_probe), str(path)]
    tokenizer = os.environ.get("TRTMC_D3_TOKENIZER")
    if tokenizer:
        command.append(tokenizer)
    actual = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    expected = json.loads(json.dumps(case["expected"]))
    if not tokenizer:
        for row in expected:
            del row["ids"]
    compare(actual, expected)


@pytest.mark.parametrize("record", [
    {"questions": {}},
    {"questions": {"a": {"type": "box"}}},
    {"questions": {"a": {"type": "score", "criteria": ["one"]}}},
    {"questions": {"a": {"type": "score", "criteria": ["one", "one"]}}},
    {"questions": {"a": {"type": "noul", "criteria": {"maybe": "yes"}}}},
    {"state": "<|image_pad|>", "questions": {"a": {"type": "noul"}}},
])
def test_reject_unsupported_inputs(host_probe, tmp_path, record):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(record))
    result = subprocess.run([str(host_probe), str(path)], capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stderr.strip()


def test_original_image_processor(host_probe):
    import numpy as np
    expected = np.load(ROOT / "fixtures/image-reference.npz")
    actual = json.loads(subprocess.run([str(host_probe), "--media"], check=True,
                                      capture_output=True, text=True).stdout)
    np.testing.assert_array_equal(np.asarray(actual["patches"], dtype=np.uint16).reshape(256, 1536),
                                  expected["patches"])
    np.testing.assert_array_equal(np.asarray(actual["indices"]).reshape(4, 256), expected["indices"])
    np.testing.assert_allclose(np.asarray(actual["weights"]).reshape(4, 256), expected["weights"],
                               atol=1e-6, rtol=1e-6)


def test_batch_padding_and_image_row_layout(host_probe):
    actual = json.loads(subprocess.run([str(host_probe), "--batch"], check=True,
                                      capture_output=True, text=True).stdout)
    assert actual["ids"] == [11, 12, 0, 21, 22, 23]
    assert actual["lengths"] == [2, 3]
    assert actual["visible"] == [True, True, False, True, True, True]
    assert actual["angles"] == [1] * 64 + [0] * 32 + [2] * 96
    assert actual["visual_indices"] == [0, 1, 10, 11, 2, 3, 12, 13, 4, 5, 14, 15, 6, 7, 16, 17]
    assert actual["visual_factors"] == [1, 1, 2, 2] * 4
    assert actual["patches"] == [31, 32, 41, 42]
    assert actual["visual_angles"] == [1, 2, 3, 4]
