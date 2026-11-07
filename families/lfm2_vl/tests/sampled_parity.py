# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-logit sampling proof with identical histories and recorded uniform draws."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from .lfm25_vl_sampling_reference import couple
from .native_probes import build_native_probe
from .reference_source import guard_reference_model


def compare_sampled(model, inputs, media, bundle, runtime, directory, *, steps=64):
    guard_reference_model(model)
    import numpy as np
    import torch
    from transformers.generation.logits_process import (
        MinPLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
    )

    here = Path(__file__).resolve().parent
    root = here.parents[2]
    executable = directory / "sampler-probe"
    subprocess.run(
        [
            shutil.which("c++"),
            "-std=c++17",
            "-O2",
            "-I",
            str(root),
            "-I",
            str(root / "core/runtime/include"),
            str(here / "cpp/lfm25_vl_sampler_probe.cpp"),
            str(here.parent / "runtime/sampler.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    params = dict(
        temperature=0.1, top_k=50, top_p=1.0, min_p=0.15, repetition_penalty=1.05, seed=42
    )
    processors = [
        RepetitionPenaltyLogitsProcessor(1.05),
        TemperatureLogitsWarper(0.1),
        TopKLogitsWarper(50),
        MinPLogitsWarper(0.15),
    ]
    history = inputs.input_ids[0].tolist()
    next_ids = history[:]
    random = np.random.default_rng(42)
    rows = []
    cache = None
    stderr = (directory / "sampling-native.log").open("w")
    logits_probe = build_native_probe("lfm25_vl_logits_probe", runtime, directory)
    probe = subprocess.Popen(
        [str(logits_probe), str(bundle)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr,
        text=True,
        bufsize=1,
    )
    sampler = subprocess.Popen(
        [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
    )
    try:
        assert json.loads(probe.stdout.readline())["ready"]
        for step in range(steps):
            path = directory / "sampling-logits.bin"
            request = {"reset": step == 0, "tokens": next_ids, "output": str(path)}
            if step == 0:
                request["images"] = media
            probe.stdin.write(json.dumps(request) + "\n")
            probe.stdin.flush()
            assert probe.stdout.readline(), "native logits probe terminated"
            native = np.fromfile(path, dtype=np.float32)
            with torch.inference_mode():
                result = (
                    model(**inputs, use_cache=True)
                    if step == 0
                    else model(
                        input_ids=torch.tensor([next_ids], device="cuda"),
                        past_key_values=cache,
                        use_cache=True,
                    )
                )
            cache = result.past_key_values
            logits = result.logits[0, -1].float()
            scores = logits[None].clone()
            for processor in processors:
                scores = processor(torch.tensor([history], device="cuda"), scores)
            probabilities = scores.softmax(-1)[0].cpu().tolist()
            scores = scores[0].cpu().tolist()
            uniform = float(np.float32(random.random()))
            values = [len(native), len(history), 1]
            values += [
                params[key]
                for key in ("temperature", "top_k", "top_p", "min_p", "repetition_penalty", "seed")
            ]
            values += native.tolist() + history + [uniform]
            sampler.stdin.write(" ".join(map(str, values)) + "\n")
            sampler.stdin.flush()
            sampled = json.loads(sampler.stdout.readline())
            expected = couple(probabilities, scores, uniform)
            actual = sampled["coupled_ids"][0]
            reference = logits.cpu().numpy()
            rows.append(
                {
                    "step": step,
                    "uniform": uniform,
                    "reference_id": expected,
                    "native_id": actual,
                    "exact_match": actual == expected,
                    "logit_max_abs": float(np.max(np.abs(native - reference))),
                    "probability_total_variation": float(
                        np.abs(np.asarray(sampled["probabilities"]) - probabilities).sum() / 2
                    ),
                }
            )
            history.append(expected)
            next_ids = [expected]
            if expected == 7:
                break
        report = {
            "parameters": params,
            "coupling": "recorded float32 uniforms; descending score then token ID; original-reference histories",
            "rows": rows,
            "all_exact": all(row["exact_match"] for row in rows),
        }
        (directory / "sampled-parity.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        for process in (probe, sampler):
            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=30)
        stderr.close()
