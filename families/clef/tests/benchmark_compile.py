# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the original complete request path using torch.compile(max-autotune).

Compilation, model loading, and warmup are excluded. Timed calls include record
encoding, host/device transfers, the backbone/head, and SystemOne formatting,
matching the native public Task boundary. Accuracy is checked before timing.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch


def main():
    root = Path(__file__).resolve().parent
    manifests = {
        value["name"]: value
        for path in (root / "manifests").glob("*.json")
        for value in [json.loads(path.read_text())]
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--model", choices=sorted(manifests), default="clef")
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--dynamic", choices=("auto", "static", "dynamic"), default="auto")
    parser.add_argument(
        "--emulate-precision-casts",
        action="store_true",
        help="Preserve the original BF16 rounding boundaries when validating compiled accuracy",
    )
    parser.add_argument(
        "--aten-masked-scatter",
        action="store_true",
        help="Use ATen for masked_scatter if the compiler's scan kernel fails; record this fallback",
    )
    parser.add_argument(
        "--aten-layer-norm",
        action="store_true",
        help="Retain the original ATen LayerNorm when compiler lowering fails accuracy",
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    if args.emulate_precision_casts:
        import torch._inductor.config as inductor_config

        inductor_config.emulate_precision_casts = True
    aten_fallbacks = []
    if args.aten_masked_scatter or args.aten_layer_norm:
        # Keep the original operation and inputs. Only its compiler lowering
        # changes; the rest of the model still uses max-autotune compilation.
        from torch._inductor.decomposition import decompositions
        from torch._inductor.lowering import make_fallback
        import torch._inductor.config as inductor_config
        import torch._functorch.config as autograd_config

        for enabled, name, operation in (
            (args.aten_masked_scatter, "masked_scatter", torch.ops.aten.masked_scatter.default),
            (args.aten_layer_norm, "native_layer_norm", torch.ops.aten.native_layer_norm.default),
        ):
            if enabled:
                decompositions.pop(operation, None)
                make_fallback(operation, warn=False)
                aten_fallbacks.append(name)
        # Graph cache keys do not include this decomposition-table override.
        # Keep kernel autotune caches, but regenerate the affected graph IR.
        inductor_config.fx_graph_cache = False
        autograd_config.enable_autograd_cache = False
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = manifests[args.model]
    fixtures = (
        sorted(args.fixtures.glob("*.json"))
        if args.fixtures
        else [root / case["inputs"]["document_path"] for case in manifest["testcases"]]
    )
    sys.path.insert(0, str(args.checkpoint))
    from joint_schema_model import (
        encode_record,
        collate_records,
        load_release_model,
        systemone_answer,
    )

    model, processor = load_release_model(args.checkpoint, device="cpu")
    model = model.to("cuda")
    dynamic = {"auto": None, "static": False, "dynamic": True}[args.dynamic]
    compiled = torch.compile(model, mode="max-autotune", dynamic=dynamic)
    device = torch.device("cuda")

    def call(record, function):
        encoded = encode_record(processor.tokenizer, record, processor=processor)
        batch = collate_records([encoded], processor.tokenizer.pad_token_id, device)
        logits = function(batch)[0]
        probabilities = [tensor.float().softmax(-1).tolist() for tensor in logits]
        answers = {
            q.question_id: systemone_answer(
                record["questions"][q.question_id], dict(zip(q.option_ids, p))
            )
            for q, p in zip(encoded.questions, probabilities)
        }
        return probabilities, {
            "model": record.get("model", "clef"),
            "answers": answers,
            "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
        }

    receipts = []
    with torch.inference_mode():
        for fixture in fixtures:
            if args.case and fixture.stem not in args.case:
                continue
            from families.clef.tests.media_fixtures import fixture_record

            record = fixture_record(fixture)
            print("Warmup and compile", fixture.stem, flush=True)
            expected, expected_response = call(record, model)
            for _ in range(args.warmup):
                actual, response = call(record, compiled)
                torch.cuda.synchronize()
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_allclose(a, b, atol=0.002, rtol=0.01)
                assert int(np.argmax(a)) == int(np.argmax(b))
            samples = []
            for _ in range(args.iterations):
                torch.cuda.synchronize()
                started = time.perf_counter()
                _, response = call(record, compiled)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000)
            receipt = {
                "fixture": fixture.stem,
                "mode": "max-autotune",
                "fullgraph": False,
                "dynamic": dynamic,
                "aten_fallbacks": aten_fallbacks,
                "scope": "public_task_call_wall",
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
                        "torch": torch.__version__,
                        "device": torch.cuda.get_device_name(),
                        "checkpoint": manifest["hf_id"],
                        "checkpoint_revision": manifest["hf_revision"],
                        "emulate_precision_casts": bool(
                            torch._inductor.config.emulate_precision_casts
                        ),
                        "results": receipts,
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
