# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and benchmark a stateless Llama bundle through the public C++ SDK.

Run as ``python -m families.llama.tests.forward_validation --help``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def run_case(runner, bundle, root, request, directory, name, *, warmup=0, iterations=1):
    path = directory / f"{name}.json"
    path.write_text(json.dumps(request))
    output = directory / f"{name}.f32"
    completed = subprocess.run(
        [str(runner), str(bundle), str(root), str(path), str(output), str(warmup), str(iterations)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        print(completed.stderr, flush=True)
        raise subprocess.CalledProcessError(
            completed.returncode, completed.args, output=completed.stdout, stderr=completed.stderr
        )
    summary = json.loads(completed.stdout)
    if isinstance(summary, list):
        values = [
            None
            if item.get("rejected")
            else np.fromfile(item["output_path"], dtype=np.float32).reshape(
                item["rows"], item["columns"]
            )
            for item in summary
        ]
        return values, summary
    values = np.fromfile(output, dtype=np.float32).reshape(summary["rows"], summary["columns"])
    return values, summary


def reference(model, request):
    ids = torch.tensor([request["token_ids"]], device=model.device)
    length = ids.shape[1]
    blocked = torch.zeros(length, length, dtype=torch.bool, device=model.device)
    if request.get("causal", True):
        blocked |= torch.ones_like(blocked).triu(1)
    if "attention_mask" in request:
        blocked |= ~torch.tensor(request["attention_mask"], device=model.device).bool()[None, :]
    if "blocked_attention" in request:
        blocked |= (
            torch.tensor(request["blocked_attention"], device=model.device)
            .bool()
            .reshape(length, length)
        )
    mask = torch.zeros(length, length, dtype=torch.float32, device=model.device)
    mask.masked_fill_(blocked, float("-inf"))
    positions = torch.tensor(
        [request.get("position_ids", list(range(length)))], device=model.device
    )
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        logits = (
            model(
                ids,
                attention_mask=mask[None, None],
                position_ids=positions,
                use_cache=False,
                is_causal=False,
            )
            .logits[0]
            .float()
            .cpu()
            .numpy()
        )
    rows = request.get("prediction_positions", list(range(length)))
    return logits[rows]


def check_logits(actual, expected, precision):
    # Fixed before execution: compare every returned logit, including near-zero
    # logits, against an independent FP32 source model. No task gate is relaxed.
    atol, rtol = (0.002, 0.002) if precision == "fp32" else (0.1, 0.02)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return {
        "max_abs_error": float(np.max(np.abs(actual - expected))),
        "relative_l2_error": float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)),
        "atol": atol,
        "rtol": rtol,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Local checkpoint snapshot (revision recorded in report)"
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float32,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    # Like the bringup CausalLM runner, exercise token IDs directly. This path
    # neither requires tokenizer assets nor changes token framing.
    ids = np.random.default_rng(42).integers(0, model.config.vocab_size, size=16).tolist()
    ids_tensor = torch.tensor([ids], device=model.device)
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        keyword = (
            model(
                ids_tensor,
                attention_mask=torch.ones_like(ids_tensor),
                use_cache=False,
                is_causal=False,
            )
            .logits[0]
            .float()
            .cpu()
            .numpy()
        )
    bidirectional = reference(model, {"token_ids": ids, "causal": False})
    # Verify the actual bringup keyword path; never infer it from the flag name.
    np.testing.assert_allclose(keyword, bidirectional, atol=1e-5, rtol=1e-5)
    causal = reference(model, {"token_ids": ids, "causal": True})
    assert np.max(np.abs(causal[:-1] - bidirectional[:-1])) > 0.01
    cases = {
        "bidirectional": {"token_ids": ids, "causal": False},
        "causal": {"token_ids": ids, "causal": True},
        "selected_rows": {"token_ids": ids[:7], "causal": False, "prediction_positions": [6, 0, 6]},
        "masked_positions": {
            "token_ids": ids[:7],
            "causal": False,
            "attention_mask": [1, 1, 1, 1, 1, 1, 0],
            "position_ids": [2, 3, 4, 5, 6, 7, 8],
        },
        "blocked_attention": {
            "token_ids": ids[:4],
            "causal": False,
            "blocked_attention": [0, 0, 1, 1] * 4,
        },
        "single_token": {"token_ids": ids[:1]},
    }
    invalid = {
        "empty": {"token_ids": []},
        "bad_token": {"token_ids": [-1]},
        "bad_mask": {"token_ids": ids[:2], "attention_mask": [1]},
        "bad_row": {"token_ids": ids[:2], "prediction_positions": [2]},
        "bad_position": {"token_ids": ids[:1], "position_ids": [-1]},
        "all_blocked": {"token_ids": ids[:2], "attention_mask": [0, 0]},
        "segment_ids": {"token_ids": ids[:1], "segment_ids": [0]},
    }
    # One loaded model exercises changing lengths/masks and recovery after invalid
    # requests. This is a validation sequence, not synthesized inference batching.
    suite = [{"request": request} for request in cases.values()]
    suite.extend({"request": request, "expect_invalid": True} for request in invalid.values())
    suite.append({"request": cases["bidirectional"]})
    actuals, summaries = run_case(
        args.runner, args.bundle, args.runtime_root, suite, args.output_dir, "validation_suite"
    )
    checks = {}
    for (name, request), actual, summary in zip(cases.items(), actuals, summaries):
        checks[name] = {
            **check_logits(actual, reference(model, request), args.precision),
            **summary,
        }
    assert all(item.get("rejected") for item in summaries[len(cases) : -1])
    check_logits(actuals[-1], bidirectional, args.precision)
    # Correctness on the same bundle and primary input passes before timing.
    measured, timing = run_case(
        args.runner,
        args.bundle,
        args.runtime_root,
        cases["bidirectional"],
        args.output_dir,
        "benchmark",
        warmup=args.warmup,
        iterations=args.iterations,
    )
    check_logits(measured, bidirectional, args.precision)
    checks["bidirectional"].update(timing)
    import tensorrt

    report = {
        "checkpoint": str(Path(args.model).resolve()),
        "precision": args.precision,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensorrt": tensorrt.__version__,
        "transformers": __import__("transformers").__version__,
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "repository_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "runtime_root": str(args.runtime_root.resolve()),
        "family_dso_sha256": hashlib.sha256(
            (args.runtime_root / "libtrtmc_model_llama.so").read_bytes()
        ).hexdigest(),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).splitlines()[0],
        "seed": 42,
        "primary_request": cases["bidirectional"],
        "timing_scope": "public_task_call_wall",
        "reference_precision": "fp32",
        "reference_attn_implementation": "sdpa_math",
        "bringup_is_causal_false_verified_bidirectional": True,
        "checks": checks,
        "invalid_requests_rejected": list(invalid),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
