# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate CPU sampler fixtures using the actual Transformers processors.

Run directly in an environment with torch and transformers==5.2.0. This script
does not load model weights, initialize CUDA, or emulate the reference math.
It also exposes reference_distribution for same-history model-logit probes.
"""

from __future__ import annotations

import json
from pathlib import Path


def reference_distribution(logits: list[float], history: list[int], params: dict):
    import torch
    from transformers.generation.logits_process import (
        MinPLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    scores = torch.tensor([logits], dtype=torch.float32)
    input_ids = torch.tensor([history], dtype=torch.long)
    processors = [
        RepetitionPenaltyLogitsProcessor(params["repetition_penalty"]),
        TemperatureLogitsWarper(params["temperature"]),
    ]
    if params["top_k"] > 0:
        processors.append(TopKLogitsWarper(params["top_k"]))
    if params["top_p"] < 1:
        processors.append(TopPLogitsWarper(params["top_p"]))
    if params["min_p"] > 0:
        processors.append(MinPLogitsWarper(params["min_p"]))
    for processor in processors:
        scores = processor(input_ids, scores)
    return scores[0].tolist(), scores.softmax(-1)[0].tolist()


def couple(probabilities: list[float], scores: list[float], uniform: float) -> int:
    """Inverse-CDF coupling, descending score and token ID tie-break."""
    import torch

    indices = sorted(
        (i for i, probability in enumerate(probabilities) if probability > 0),
        key=lambda i: (-scores[i], i),
    )
    cumulative = torch.tensor([probabilities[i] for i in indices], dtype=torch.float32).cumsum(0)
    for i, value in zip(indices, cumulative.tolist()):
        if uniform < value:
            return i
    return indices[-1]


def fixture_cases():
    import math

    defaults = dict(temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0, seed=42)
    specs = [
        ("top-k-boundary-tie", [2.0, 1.0, 1.0, -1.0], [], dict(top_k=2)),
        ("top-k-all-tied", [1.0] * 7, [], dict(top_k=3)),
        ("top-k-one-tie", [3.0, 3.0, 0.0], [], dict(top_k=1)),
        (
            "repetition-sign-and-unique-history",
            [0.5, 4.0, -1.0, 3.5],
            [1, 2, 1],
            dict(repetition_penalty=2.0),
        ),
        (
            "top-p-before-min-p",
            [math.log(p) for p in [0.5, 0.25, 0.15, 0.1]],
            [],
            dict(top_p=0.8, min_p=0.3),
        ),
        ("top-p-only", [2.0, 1.0, 0.0, -1.0], [], dict(top_p=0.8)),
        ("top-p-tied-tail", [1.0] * 4, [], dict(top_p=0.5)),
        ("min-p-only", [2.0, 1.0, 0.0, -1.0], [], dict(min_p=0.3)),
        (
            "all-processors",
            [1.75, -1.25, 3.5, 2.5, 1.0, -2.0, 3.0],
            [0, 2, 1, 2],
            dict(temperature=0.7, top_k=5, top_p=0.94, min_p=0.1, repetition_penalty=1.1),
        ),
        ("temperature-fp32-order", [16.125, 16.0, 15.875, 14.0], [], dict(temperature=0.1)),
        (
            "model-card",
            [(i % 11) / 64 + i / 512 for i in range(96)],
            [1, 1, 93, 89, 3],
            dict(temperature=0.1, top_k=50, min_p=0.15, repetition_penalty=1.05),
        ),
    ]
    for name, logits, history, updates in specs:
        params = defaults | updates
        # Include interval interiors for every retained token, so a candidate
        # with small mass is exercised even if fixed random draws miss it.
        scores, probabilities = reference_distribution(logits, history, params)
        indices = sorted(
            (i for i, p in enumerate(probabilities) if p > 0), key=lambda i: (-scores[i], i)
        )
        uniforms = [0.0, 0.125, 0.375, 0.625, 0.875, 0.999]
        cumulative = 0.0
        for i in indices:
            uniforms.append(min(cumulative + probabilities[i] / 2, 0.9999999))
            cumulative += probabilities[i]
        yield dict(
            name=name,
            logits=logits,
            history=history,
            params=params,
            uniforms=uniforms,
            probabilities=probabilities,
            coupled_ids=[couple(probabilities, scores, u) for u in uniforms],
        )


if __name__ == "__main__":
    import torch
    import transformers

    assert transformers.__version__ == "5.2.0", "revalidate changes to the reference processors"
    output = Path(__file__).with_name("fixtures") / "lfm25_vl-sampling.json"
    output.write_text(
        json.dumps(
            {
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "processor_order": [
                    "RepetitionPenaltyLogitsProcessor",
                    "TemperatureLogitsWarper",
                    "TopKLogitsWarper",
                    "TopPLogitsWarper",
                    "MinPLogitsWarper",
                ],
                "cases": list(fixture_cases()),
            },
            indent=2,
        )
        + "\n"
    )
