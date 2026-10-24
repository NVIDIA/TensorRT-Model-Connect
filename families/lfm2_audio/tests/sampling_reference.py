# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate CPU sampling fixtures by executing the pinned upstream methods.

Only torch.multinomial is intercepted: the reference itself performs BF16
temperature scaling, top-k masking and softmax. Shared uniforms verify the
resulting categorical distribution; native seeds are not Torch seeds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

REFERENCE_REVISION = "19e65845923a7f136442c95137884ec61eb386aa"
UNIFORMS = [0.0, 0.03125, 0.125, 0.25, 0.5, 0.75, 0.9375, 0.9999999999999999]


def categorical(probs, uniform):
    # BF16 softmax sums need not equal one; multinomial treats them as weights.
    weights = probs.double()
    target = uniform * weights.sum()
    return int(torch.searchsorted(weights.cumsum(0), target, right=True))


def capture(model, logits, temperature, top_k, audio):
    distributions = []

    def multinomial(probs, count):
        assert count == 1
        distributions.append(probs.clone())
        return torch.tensor([categorical(probs, 0.5)])

    # Use distinct codebook logits so the audio test covers all eight decisions.
    class Embedding:
        def __init__(self, index):
            self.index = index

        def get_logits(self, _hidden):
            return logits.roll(self.index).clone()

        def __call__(self, _token):
            return torch.zeros(1, dtype=torch.bfloat16)

    with patch("torch.multinomial", multinomial):
        if audio:
            fake = SimpleNamespace(
                codebooks=8,
                depthformer_dim=1,
                depth_linear=lambda _embedding: torch.zeros(8, dtype=torch.bfloat16),
                depthformer=SimpleNamespace(forward_cached=lambda _input, _cache: (_input, None)),
                depth_embeddings=[Embedding(i) for i in range(8)],
            )
            tokens = model._sample_audio_frame(
                fake, torch.zeros(1), temperature=temperature, top_k=top_k
            )
        else:
            tokens = model._sample_text_token(
                None, logits.clone(), temperature=temperature, top_k=top_k
            )
    return distributions, tokens.tolist()


def generate(reference):
    from liquid_audio.model.lfm2_audio import LFM2AudioModel

    revision = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == REFERENCE_REVISION, revision
    cases = []
    definitions = [
        ("greedy_zero", [0.0, 3.0, 3.0, -2.0], 0.0, None),
        ("greedy_negative", [0.0, 3.0, 3.0, -2.0], -0.7, None),
        ("greedy_top_one", [0.0, 3.0, 3.0, -2.0], 0.7, 1),
        ("threshold_ties", [-4.0, 1.0, 1.0, 1.0, 3.0, -5.0], 0.7, 2),
        ("bf16_temperature", [-1.4375, -0.375, 0.03125, 0.625, 1.0625, 2.140625], 0.7, None),
        ("bf16_softmax", [-8.0, -2.125, -0.375, 0.0, 0.3125, 0.75, 1.5], 1.3, 5),
        ("uniform_ties", [0.0] * 17, 1.0, 4),
        ("underflow_tail", [-200.0, 1.0, -100.0, 0.5, -0.125], 0.7, None),
        ("all_top_k", [3.0, -2.0, 0.125, 2.0], 1.7, 4),
        # Full-size logits exercise the reduction and rounded probability sum.
        ("text_vocab", [float(i - 16) / 8 for i in range(32) for _ in range(2048)], 0.7, None),
        ("audio_vocab", [0.0] * 2049, 1.0, 64),
    ]
    for name, values, temperature, top_k in definitions:
        logits = torch.tensor(values, dtype=torch.bfloat16)
        for audio in (False, True):
            if (name == "text_vocab" and audio) or (name == "audio_vocab" and not audio):
                continue
            distributions, greedy_tokens = capture(
                LFM2AudioModel, logits, temperature, top_k, audio
            )
            for codebook in range(8 if audio else 1):
                case = {
                    "name": f"{'audio' if audio else 'text'}_{name}_{codebook}",
                    "logits": logits.roll(codebook).float().tolist(),
                    "temperature": temperature,
                    "top_k": top_k or 0,
                }
                if distributions:
                    probs = distributions[codebook]
                    case["probabilities"] = probs.float().tolist()
                    case["tokens"] = [categorical(probs, u) for u in UNIFORMS]
                else:
                    case["greedy_token"] = greedy_tokens[codebook]
                for key in ("logits", "probabilities"):
                    if key in case and len(case[key]) > 128:
                        case[key + "_runs"] = [
                            [value, len(list(group))] for value, group in groupby(case.pop(key))
                        ]
                cases.append(case)
    return {
        "reference_repository": "https://github.com/Liquid4All/liquid-audio",
        "reference_revision": revision,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "uniforms": UNIFORMS,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(generate(args.reference), indent=2) + "\n")


if __name__ == "__main__":
    main()
