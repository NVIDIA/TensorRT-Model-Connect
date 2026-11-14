# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute native contracts on CPU; pipeline compilation needs CUDA headers only."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize(("name", "sources"), [
    ("audio_input", ("resampler.cpp",)),
    ("audio_helpers", ("audio_helpers.cpp", "resampler.cpp")),
    ("decode_policy", ()),
    ("pipeline", ("pipeline.cpp", "audio_helpers.cpp", "resampler.cpp")),
    ("tokenizer", ("bpe_tokenizer.cpp",)),
])
def test_native_cpu_contract(tmp_path, name, sources):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("native CPU tests require a C++17 compiler (set CXX)")
    family = Path(__file__).resolve().parents[1]
    root = family.parents[1]
    output = tmp_path / name
    includes = []
    if name == "pipeline":
        cuda = Path(os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda")))
        if not (cuda / "include/cuda_runtime_api.h").is_file():
            pytest.skip("pipeline contract requires CUDA headers (set CUDA_HOME); no GPU is used")
        includes = ["-I", str(cuda / "include")]
    if name == "tokenizer":
        json_include = Path(os.environ.get("NLOHMANN_JSON_INCLUDE_DIR", "/usr/include"))
        if not (json_include / "nlohmann/json.hpp").is_file():
            pytest.skip("tokenizer contract requires nlohmann/json.hpp (set NLOHMANN_JSON_INCLUDE_DIR)")
        includes += ["-I", str(json_include)]
    subprocess.run([
        compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
        "-I", str(root), "-I", str(root / "core/runtime/include"),
        *includes,
        str(family / "tests/cpp" / f"test_{name}.cpp"),
        *(str(family / "runtime" / source) for source in sources),
        "-o", str(output),
    ], check=True, capture_output=True, text=True, timeout=60)
    subprocess.run([str(output)], check=True, capture_output=True, text=True, timeout=60)
