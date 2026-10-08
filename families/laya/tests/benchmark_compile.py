# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time complete original Router calls after validating torch.compile accuracy."""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from families.laya.cli import VARIANTS
from families.laya.tests.test_e2e import MANIFESTS, ROOT, compare_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=tuple(
            name for name, manifest in MANIFESTS.items() if manifest["variant"] in VARIANTS
        ),
        required=True,
    )
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--emulate-precision-casts", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    from laya import Agent, Router
    from laya.common import collate_items, temp_bucket
    import torch._inductor.config as inductor

    inductor.emulate_precision_casts = args.emulate_precision_casts
    manifest = MANIFESTS[args.model]
    variant = manifest["variant"]
    agent = Agent(
        str(args.checkpoint / VARIANTS[variant]), device="cuda", fast=False, compile=False
    )
    assert agent.device.type == "cuda"
    router = Router()
    router.attach(variant, agent)
    eager = agent.model
    compiled = torch.compile(eager, mode="max-autotune")
    args.output.mkdir(parents=True, exist_ok=True)

    def call(record):
        return router.predict(record["state"], record["questions"], model=record.get("model"))

    def arrays(record):
        ids = list(record["questions"])
        internal = {qid: agent._to_internal(record["questions"][qid]) for qid in ids}
        items = agent._encode_state(record["state"], ids, internal)
        logits, actions = agent._forward(collate_items([items], agent.tok.pad_token_id))
        result = []
        for row, item in zip(logits, items, strict=True):
            count = len(item["markers"])
            tau = agent.temperature_by_options.get(
                temp_bucket(item["qtype"], count), agent.temperature[item["qtype"]]
            )
            values = row[:count] / tau
            probabilities = np.exp(values - values.max())
            result.append((values, probabilities / probabilities.sum()))
        return result, actions

    receipts = []
    with torch.inference_mode():
        cases = [case for case in manifest["testcases"] if case["name"] in args.case]
        if {case["name"] for case in cases} != set(args.case):
            parser.error("unknown case for the selected checkpoint")
        for case in cases:
            record = json.loads((ROOT / case["inputs"]["document_path"]).read_text())
            assert (
                router.route(record["state"], record["questions"], model=record.get("model")).model
                == variant
            )
            agent.model = eager
            expected_response = call(record)
            expected, expected_actions = arrays(record)
            agent.model = compiled
            print("Compile and warmup", case["name"], flush=True)
            for _ in range(args.warmup):
                response = call(record)
                torch.cuda.synchronize()
            actual, actual_actions = arrays(record)
            assert agent.device.type == "cuda"
            for (values, probabilities), (ref_values, ref_probabilities) in zip(
                actual, expected, strict=True
            ):
                np.testing.assert_allclose(values, ref_values, atol=0.125, rtol=0.015)
                np.testing.assert_allclose(probabilities, ref_probabilities, atol=0.002, rtol=0.01)
                assert np.argmax(probabilities) == np.argmax(ref_probabilities)
            np.testing.assert_allclose(actual_actions, expected_actions, atol=0.002, rtol=0.01)
            compare_response(response, expected_response)
            samples = []
            for _ in range(args.iterations):
                torch.cuda.synchronize()
                started = time.perf_counter()
                response = call(record)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000)
            receipt = {
                "case": case["name"],
                "scope": "public_task_call_wall",
                "mode": "max-autotune",
                "warmup": args.warmup,
                "iterations": args.iterations,
                "latencies_ms": samples,
                "p50_ms": statistics.median(samples),
                "p95_ms": float(np.percentile(samples, 95)),
                "response": response,
                "eager_response": expected_response,
            }
            receipts.append(receipt)
            print(json.dumps(receipt), flush=True)
            (args.output / "torch-compile.json").write_text(
                json.dumps(
                    {
                        "selected_manifest": {
                            "name": manifest["name"],
                            "hf_id": manifest["hf_id"],
                            "hf_revision": manifest["hf_revision"],
                        },
                        "supplied_artifacts": {
                            "checkpoint_path": str(args.checkpoint.resolve()),
                            "identity_verified": False,
                        },
                        "variant": variant,
                        "torch": torch.__version__,
                        "device": torch.cuda.get_device_name(),
                        "emulate_precision_casts": bool(inductor.emulate_precision_casts),
                        "results": receipts,
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
