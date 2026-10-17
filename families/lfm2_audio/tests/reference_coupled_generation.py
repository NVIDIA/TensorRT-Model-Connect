# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Couple original categorical choices to recorded native uniform variates.

The original generation, logits, filtering, and BF16 softmax are unchanged.
Only torch.multinomial's random choice is replaced by the same inverse CDF
used by the native sampler. Torch seeds and native seeds are never equated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from liquid_audio import LFM2AudioModel

from families.lfm2_audio.tests.reference_generation import main as reference_main


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "cases", "output", "uniforms"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    payload = Path(args.uniforms).read_bytes()
    draws = json.loads(payload)
    assert draws["algorithm"] == "mt19937-uniform-real-double"
    assert len(draws["uniforms"]) == draws["count"]
    assert all(0 <= value < 1 for value in draws["uniforms"])
    cases = json.loads(Path(args.cases).read_text())
    calls = []
    for case in cases:
        assert case.get("seed", 0) == draws["seed"]
        assert case.get("text_temperature", 0.0) <= 0
        assert case.get("audio_temperature", 0.0) > 0
        assert case.get("audio_top_k", 0) != 1
        turns = case["turns"] if case["mode"] == "chat" else [case]
        calls.extend((case["name"], index) for index in range(len(turns)))
    call_index = 0
    receipts = []
    directory = Path(args.output).parent

    def wrap(original):
        def generate(model, **kwargs):
            nonlocal call_index
            name, turn = calls[call_index]
            call_index += 1
            used = 0
            audio_frames = 0
            prefix = directory / f"{name}-{turn}"
            with (
                Path(str(prefix) + ".original-draws.jsonl").open("w") as probabilities,
                Path(str(prefix) + ".original-events.jsonl").open("w") as events,
            ):

                def sample(weights, count, replacement=False, *, generator=None, out=None):
                    nonlocal used
                    assert count == 1 and not replacement and generator is None and out is None
                    assert weights.ndim == 1 and weights.numel() == 2049
                    values = weights.double().cpu().tolist()
                    assert all(value >= 0 and value < float("inf") for value in values)
                    uniform = draws["uniforms"][used]
                    total = 0.0
                    for value in values:
                        total += value
                    assert total > 0
                    target = uniform * total
                    cumulative = 0.0
                    choice = max(i for i, value in enumerate(values) if value > 0)
                    for i, value in enumerate(values):
                        cumulative += value
                        if target < cumulative:
                            choice = i
                            break
                    probabilities.write(
                        json.dumps(
                            {
                                "draw": used,
                                "uniform": uniform,
                                "selected": choice,
                                "dtype": str(weights.dtype),
                                "nonzero_probabilities": [
                                    [i, value] for i, value in enumerate(values) if value
                                ],
                            }
                        )
                        + "\n"
                    )
                    used += 1
                    return torch.tensor([choice], dtype=torch.long, device=weights.device)

                with patch("torch.multinomial", sample):
                    for token in original(model, **kwargs):
                        audio_frames += int(token.numel() == 8)
                        events.write(json.dumps({"ids": token.cpu().tolist()}) + "\n")
                        yield token
            assert used == audio_frames * 8
            receipts.append(
                {
                    "name": name,
                    "turn": turn,
                    "uniforms_used": used,
                    "audio_frames": audio_frames,
                    "reset_draws_at_turn_start": True,
                }
            )
            Path(str(args.output) + ".coupling.json").write_text(
                json.dumps(
                    {
                        "uniforms": draws["uniforms"],
                        "native_seed": draws["seed"],
                        "calls": receipts,
                    },
                    indent=2,
                )
                + "\n"
            )

        return generate

    sequential = LFM2AudioModel.generate_sequential
    interleaved = LFM2AudioModel.generate_interleaved
    argv = sys.argv
    try:
        LFM2AudioModel.generate_sequential = wrap(sequential)
        LFM2AudioModel.generate_interleaved = wrap(interleaved)
        sys.argv = [
            argv[0],
            "--checkpoint",
            args.checkpoint,
            "--cases",
            args.cases,
            "--output",
            args.output,
        ]
        with torch.inference_mode():
            reference_main()
        assert call_index == len(calls)
    finally:
        LFM2AudioModel.generate_sequential = sequential
        LFM2AudioModel.generate_interleaved = interleaved
        sys.argv = argv


if __name__ == "__main__":
    main()
