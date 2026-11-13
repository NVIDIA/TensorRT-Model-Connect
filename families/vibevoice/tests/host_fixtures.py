# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expected values for the native host-side tests (tests/cpp/test_host.cpp).

The DPM-Solver cases come from diffusers' DPMSolverMultistepScheduler with the
reference's settings, in float32 and with bfloat16 samples; the noise case from
the Python mirror of the family's noise stream; the prompt cases from the
reference chat template. ``tests/data/host_fixtures.json`` holds the output and
``test_host_fixtures.py`` checks that it is current.
"""

from __future__ import annotations

import numpy as np

from families.vibevoice.model import train_sigmas
from families.vibevoice.tests import reference

LATENT = 8
SCRIPTS = {
    "plain": "Hello, nice to meet you. How are you?",
    "one_based": "Speaker 1: Hi there!\nSpeaker 2: Hello, how are you doing today?\nspeaker 1 :Great, thanks.",
    "zero_based": "Speaker 0: 你好，欢迎收听今天的节目。\n\nSpeaker 1: Thank you.",
}


def _scheduler_case(steps: int, dtype_name: str, seed: int) -> dict:
    import torch
    from diffusers import DPMSolverMultistepScheduler

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
    # The reference samples on the GPU while the scheduler keeps its sigmas on the
    # CPU. A CPU float32 scalar meets a CUDA bfloat16 tensor as a float32 factor
    # (on the CPU it would be rounded to bfloat16 first), so the bfloat16 cases
    # are generated where the reference runs.
    device = "cuda" if dtype_name == "bf16" else "cpu"
    scheduler = DPMSolverMultistepScheduler(
        beta_schedule="squaredcos_cap_v2", prediction_type="v_prediction"
    )
    scheduler.set_timesteps(num_inference_steps=steps)
    generator = torch.Generator().manual_seed(seed)
    sample = torch.randn(1, LATENT, generator=generator).to(device=device, dtype=dtype)
    initial = sample.float().cpu().numpy().reshape(-1).tolist()
    velocities = []
    outputs = []
    for timestep in scheduler.timesteps:
        velocity = torch.randn(1, LATENT, generator=generator).to(device=device, dtype=dtype)
        velocities.append(velocity.float().cpu().numpy().reshape(-1).tolist())
        sample = scheduler.step(velocity, timestep, sample).prev_sample
        outputs.append(sample.float().cpu().numpy().reshape(-1).tolist())
    return {
        "steps": steps,
        "precision": dtype_name,
        "initial": initial,
        "velocities": velocities,
        "outputs": outputs,
        "timesteps": [int(t) for t in scheduler.timesteps],
    }


def build_host_fixtures(tokenizer_dir=None) -> dict:
    stream = reference.NoiseStream(1234)
    fixtures = {
        "train_sigmas": [float(value) for value in train_sigmas(1000)],
        "noise": {"seed": 1234, "values": [float(value) for value in stream.normals(9)]},
        "schedulers": [
            _scheduler_case(10, "fp32", 0),
            _scheduler_case(10, "bf16", 1),
            _scheduler_case(7, "fp32", 2),
            _scheduler_case(25, "bf16", 3),
        ],
        "timesteps": {
            str(steps): [int(t) for t in _linspace(steps)] for steps in (1, 5, 10, 20, 50)
        },
        "scripts": {},
    }
    processor = (
        reference.reference_processor(tokenizer_dir, 3200) if tokenizer_dir is not None else None
    )
    for name, text in SCRIPTS.items():
        turns = _turns(text)
        entry = {"text": text, "turns": turns}
        if processor is not None:
            conversation = [
                {"role": str(speaker), "content": [{"type": "text", "text": body}]}
                for speaker, body in turns
            ]
            entry["prompt"] = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            entry["ids"] = processor.tokenizer(entry["prompt"])["input_ids"]
        fixtures["scripts"][name] = entry
    return fixtures


def _linspace(steps: int) -> np.ndarray:
    return np.linspace(0, 999, steps + 1).round()[::-1][:-1].astype(np.int64)


def _turns(text: str) -> list[list]:
    """The family's script rules (runtime/script.cpp), restated for the fixtures."""
    import re

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    pattern = re.compile(r"^speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)
    matches = [pattern.match(line) for line in lines]
    if not any(matches):
        return [[0, text.strip()]]
    turns = [[int(match.group(1)), match.group(2).strip()] for match in matches]
    if min(speaker for speaker, _ in turns) > 0:
        turns = [[speaker - 1, body] for speaker, body in turns]
    return turns
