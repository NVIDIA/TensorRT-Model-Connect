# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions against probabilities captured from the pinned reference."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = json.loads((Path(__file__).parent / "fixtures/sampling.json").read_text())


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("CPU sampling test requires a C++17 compiler")
    output = tmp_path_factory.mktemp("lfm2-audio-sampling") / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(ROOT),
            str(Path(__file__).parent / "cpp/sampling_probe.cpp"),
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


def command(mode, case, uniform=0.5):
    return (
        " ".join(
            map(
                str,
                [
                    mode,
                    case["temperature"],
                    case["top_k"],
                    len(case["logits"]),
                    *case["logits"],
                    uniform,
                ],
            )
        )
        + "\n"
    )


def run(probe, mode, case, uniform=0.5):
    return subprocess.check_output(
        [str(probe)], input=command(mode, case, uniform), text=True
    ).strip()


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_reference_sampling(probe, case):
    case = dict(case)
    for key in ("logits", "probabilities"):
        if key + "_runs" in case:
            case[key] = [value for value, count in case[key + "_runs"] for _ in range(count)]
    if "greedy_token" in case:
        assert int(run(probe, "sample", case)) == case["greedy_token"]
        return
    actual = list(map(float, run(probe, "probabilities", case).split()))
    # JSON text roundtrip uses nine significant digits for float32; BF16 values
    # have exact float32 representations. Compare the recovered bits exactly.
    import struct

    def bits(values):
        return struct.pack(f"{len(values)}f", *values)

    assert bits(actual) == bits(case["probabilities"])
    draws = subprocess.check_output(
        [str(probe)],
        input="".join(command("uniform", case, uniform) for uniform in FIXTURE["uniforms"]),
        text=True,
    )
    assert list(map(int, draws.split())) == case["tokens"]


@pytest.mark.parametrize(
    "mode,logits,temperature,top_k,uniform",
    [
        ("sample", [], 0.0, 1, 0.5),
        ("probabilities", [], 1.0, 0, 0.5),
        ("probabilities", [0.0, 1.0], 0.0, 0, 0.5),
        ("probabilities", [0.0, 1.0], 1.0, -1, 0.5),
        ("probabilities", [0.0, 1.0], 1.0, 3, 0.5),
        ("weights", [0.0, 0.0], 1.0, 0, 0.5),
        ("weights", [0.5, -0.5], 1.0, 0, 0.5),
        ("weights", [0.5, 0.5], 1.0, 0, -0.1),
        ("weights", [0.5, 0.5], 1.0, 0, 1.0),
    ],
)
def test_rejects_invalid_distribution(probe, mode, logits, temperature, top_k, uniform):
    case = {"logits": logits, "temperature": temperature, "top_k": top_k}
    assert run(probe, mode, case, uniform) == "invalid_argument"
