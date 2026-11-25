# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matched, synchronized Task timings after separate checkpoint qualification."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from families.hstu_research.checkpoint import digest
from families.hstu_research.config import Config
from families.hstu_research.reference import example_inputs
from families.hstu_research.triton.native_client import NativeClient


def measure(clients, inputs, batch, iterations, warmup):
    handles, requests = [], []
    for client in clients:
        history = {s["name"]: inputs[s["name"]] for s in client.metadata["stages"][0]["inputs"]}
        requests.append(
            {s["name"]: inputs[s["name"]] for s in client.metadata["stages"][1]["inputs"]}
        )
        handles.append(client.restore_history(client.export_history(history, ["v1"] * batch)))
    rows = []
    for mode in ("cold", "restored"):

        def call(index):
            if mode == "cold":
                return clients[index].score(inputs)
            return clients[index].score_with_history(
                handles[index], ["v1"] * batch, requests[index]
            )

        outputs = [call(i) for i in range(2)]
        np.testing.assert_allclose(outputs[1]["scores"], outputs[0]["scores"], atol=0.02, rtol=0.02)
        for i in range(2):
            for _ in range(warmup):
                call(i)
        samples = [[], []]
        for repeat in range(3):
            for iteration in range(iterations // 3):
                for i in ((repeat + iteration) % 2, (repeat + iteration + 1) % 2):
                    start = time.perf_counter_ns()
                    call(i)
                    samples[i].append((time.perf_counter_ns() - start) / 1e6)
        fp32, bf16 = [float(np.median(values)) for values in samples]
        rows.append(
            dict(
                batch=batch,
                history=mode,
                fp32_ms=fp32,
                bf16_ms=bf16,
                latency_reduction=1 - bf16 / fp32,
                samples_ms=samples,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()
    if args.iterations < 3 or args.iterations % 3 or args.warmup < 1:
        parser.error("iterations must be a positive multiple of 3 and warmup must be positive")
    configs = [Config.load(path) for path in (args.fp32, args.bf16)]
    assert configs[0].architecture == configs[1].architecture
    assert configs[0].serving == configs[1].serving
    assert configs[0].cuda_graphs == configs[1].cuda_graphs
    weights = [digest(path / "hstu_research_weights.npz") for path in (args.fp32, args.bf16)]
    assert weights[0] == weights[1], "timing requires identical checkpoint weights"
    rows = []
    for batch in args.batches:
        with ExitStack() as stack:
            clients = [
                stack.enter_context(
                    NativeClient(
                        path / "model.bundle",
                        args.runtime_root,
                        hydration=True,
                        kv_cache_size_bytes=134217728,
                        max_entries=8,
                        lease_seconds=0,
                    )
                )
                for path in (args.fp32, args.bf16)
            ]
            assert [client.metadata["precision"] for client in clients] == ["fp32", "bf16"]
            rows.extend(
                measure(
                    clients, example_inputs(configs[0], batch), batch, args.iterations, args.warmup
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                checkpoint_sha256=weights[0],
                configs=[asdict(c) for c in configs],
                warmup=args.warmup,
                iterations=args.iterations,
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps([{k: v for k, v in row.items() if k != "samples_ms"} for row in rows], indent=2)
    )


if __name__ == "__main__":
    main()
