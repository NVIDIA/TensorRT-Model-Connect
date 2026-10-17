# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the production CPU sampler to pinned Transformers probability fixtures.

The sampled output contract uses a shared uniform draw with identical candidate
ordering. A shared seed is deliberately not treated as identical randomness:
native xorshift and torch.multinomial use different generators and algorithms.
Regenerate fixtures with the adjacent lfm25_sampling_reference.py script.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest


TEST_DIR = Path(__file__).resolve().parent
ROOT = TEST_DIR.parents[2]
FAMILY = TEST_DIR.parent.name
FIXTURES = json.loads((TEST_DIR / "fixtures/lfm25-sampling.json").read_text())


def encode_probe_case(case: dict) -> str:
    params = case["params"]
    fields = [len(case["logits"]), len(case["history"]), len(case["uniforms"])]
    fields += [params[key] for key in
               ("temperature", "top_k", "top_p", "min_p", "repetition_penalty", "seed")]
    fields += case["logits"] + case["history"] + case["uniforms"]
    return " ".join(map(str, fields)) + "\n"


@pytest.fixture(scope="module")
def native_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    assert compiler, "the CPU sampler contract requires a C++ compiler"
    output = tmp_path_factory.mktemp("sampling") / "probe"
    subprocess.run([
        compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        "-I", str(ROOT), "-I", str(ROOT / "core/runtime/include"),
        str(TEST_DIR / "cpp/lfm25_sampler_probe.cpp"),
        str(TEST_DIR.parent / "runtime/sampler.cpp"), "-o", str(output),
    ], check=True, capture_output=True, text=True)
    return output


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda case: case["name"])
def test_hf_processed_probability_distribution_and_shared_uniform(native_probe, case):
    completed = subprocess.run([str(native_probe)], input=encode_probe_case(case),
                               capture_output=True, text=True, check=True)
    actual = json.loads(completed.stdout)
    expected = case["probabilities"]
    assert len(actual["probabilities"]) == len(expected)
    # A removed candidate must stay removed. Approximate probability agreement
    # alone could hide a top-k boundary or filter-order defect at tiny mass.
    assert [p > 0 for p in actual["probabilities"]] == [p > 0 for p in expected]
    assert actual["probabilities"] == pytest.approx(expected, abs=2e-7, rel=2e-6)
    assert math.fsum(actual["probabilities"]) == pytest.approx(1.0, abs=2e-7)
    assert actual["coupled_ids"] == case["coupled_ids"]


def test_fixture_provenance_and_model_card_policy():
    assert FIXTURES["transformers"] == "5.2.0"
    assert FIXTURES["processor_order"] == [
        "RepetitionPenaltyLogitsProcessor", "TemperatureLogitsWarper",
        "TopKLogitsWarper", "TopPLogitsWarper", "MinPLogitsWarper",
    ]
    card = next(case for case in FIXTURES["cases"] if case["name"] == "model-card")
    assert card["params"]["temperature"] == 0.1
    assert card["params"]["top_k"] == 50
    assert card["params"]["repetition_penalty"] == 1.1
