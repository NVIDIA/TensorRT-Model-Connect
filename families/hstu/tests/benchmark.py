# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic family NativeClient.score wall time; includes copies and synchronization.

The generic benchmark CLI does not yet expose structured_ranking. Run the
family correctness cases on the same bundles before interpreting these numbers.
"""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from ..config import Config
from ..checkpoint import load_weights
from ..reference import Reference, example_inputs
from ..triton.native_client import NativeClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Prepared output directory")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True
    ).strip()
    result = dict(
        scope="family_native_client_score_wall",
        gpu=gpu,
        machine=platform.machine(),
        seed=42,
        warmup=args.warmup,
        iterations=args.iterations,
        rows=[],
    )
    for profile in ("default", "short"):
        for precision in ("fp32", "bf16"):
            directory = args.root / f"dlrmv3-{profile}-{precision}"
            config = Config.load(directory)
            oracle = Reference(config, load_weights(directory))
            with NativeClient(directory / "model.bundle", args.runtime_root) as task:
                for batch in (1, 4) if profile == "default" else (4,):
                    inputs = example_inputs(config, batch, seed=42)
                    expected = oracle.score(inputs)["scores"]
                    actual = task.score(inputs)["scores"]
                    gate = json.loads(
                        (
                            Path(__file__).parent
                            / "thresholds"
                            / f"dlrmv3-{precision}-{profile}-b{batch}.json"
                        ).read_text()
                    )["threshold_overrides"]
                    np.testing.assert_allclose(
                        actual, expected, atol=gate["score_atol"], rtol=gate["score_rtol"]
                    )
                    for _ in range(args.warmup):
                        task.score(inputs)
                    samples = []
                    for _ in range(args.iterations):
                        start = time.perf_counter_ns()
                        task.score(inputs)
                        samples.append((time.perf_counter_ns() - start) / 1e6)
                    row = dict(
                        profile=profile,
                        precision=precision,
                        batch=batch,
                        bundle=str(directory / "model.bundle"),
                        bundle_id=task.metadata["bundle_id"],
                        cuda_graphs=task.metadata["cuda_graphs"],
                        bf16_policy=task.metadata.get("bf16_policy", "conservative"),
                        bf16_extra_ops=task.metadata.get("bf16_extra_ops", []),
                        fp32_block_indices=list(range(5))
                        if precision == "fp32"
                        else config.fp32_block_indices,
                        p50_ms=float(np.median(samples)),
                        p95_ms=float(np.percentile(samples, 95)),
                    )
                    result["rows"].append(row)
                    print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
