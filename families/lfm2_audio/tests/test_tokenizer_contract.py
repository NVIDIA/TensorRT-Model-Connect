# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual loader factory and native tokenizer without a GPU."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("tokenizer contract test requires a C++17 compiler")
    directory = tmp_path_factory.mktemp("audio-tokenizer-contract")
    source = (FAMILY / "runtime/plugin_helpers.cpp").read_text()
    begin, end = "std::shared_ptr<ITokenizer> create_tokenizer(", "DType state_dtype("
    assert source.count(begin) == source.count(end) == 1
    (directory / "tokenizer_factory.inc").write_text(
        source[source.index(begin) : source.index(end)]
    )
    output = directory / "tokenizer_contract"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(ROOT),
            "-I",
            str(directory),
            str(FAMILY / "tests/cpp/tokenizer_contract.cpp"),
            *[
                str(FAMILY / "runtime" / name)
                for name in ("bpe_tokenizer.cpp", "pretokenizer.cpp", "byte_level_decoder.cpp")
            ],
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


@pytest.mark.parametrize("variant", ["pinned", "whitespace", "prefix-space"])
def test_loader_tokenizer_contract(probe, tmp_path, variant):
    config = json.loads((FAMILY / "tests/fixtures/tokenizer_contract.json").read_text())
    if variant == "whitespace":
        config["pre_tokenizer"] = {"type": "Whitespace"}
    elif variant == "prefix-space":
        config["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] = True
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(config))
    result = subprocess.run(
        [str(probe), str(path), "accept" if variant == "pinned" else "reject"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
