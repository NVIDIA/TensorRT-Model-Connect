# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The family's OCRBench benchmark bundles hold the prompt and the whole answer.

A full Qwen-VL KV cache slides and drops the oldest rows, the image among them, so each profile reserves
its largest image, its chat template and the answer on top of the counted question tokens.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

AIPERF = Path(__file__).resolve().parent / "aiperf"
MAX_PIXELS = 200704  # the image budget both sides use (the reference's processor_kwargs)
MERGE = 2
ANSWER_TOKENS = 128  # OCRBench's max_new_tokens
# profile: (vision patch size, chat-template tokens around the image and question)
PROFILES = {
    "qwen3-vl-2b": (16, 10),
    "qwen3-vl-8b": (16, 10),
    "qwen25vl-3b": (14, 21),  # the default system turn included
    "qwen25vl-7b-fp8": (14, 21),
    "qwen25vl-7b-nvfp4": (14, 21),
}


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_benchmark_bundle_reserves_the_image_template_and_answer(profile: str) -> None:
    patch, template = PROFILES[profile]
    image_tokens = MAX_PIXELS // (patch * MERGE) ** 2
    candidate = yaml.safe_load((AIPERF / f"{profile}.yaml").read_text())["candidate"]
    assert candidate["benchmark_sequence_length"] == {"extra_tokens": image_tokens + template + ANSWER_TOKENS}
    # Its own bundle, so the catalog's smaller-cache bundle is never served for the benchmark.
    assert candidate["build"] == {"name": f"{profile}-benchmark"}
