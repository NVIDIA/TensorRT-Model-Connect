# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a complete native router bundle against every released variant."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from families.laya.cli import VARIANTS
from families.laya.tests.test_e2e import MANIFESTS, ROOT, compare_response


def main():
    parser = argparse.ArgumentParser()
    for argument in ("checkpoint", "bundle", "probe", "runtime-root", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), action="append")
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    from laya import Agent, Router
    from laya.common import collate_items, temp_bucket

    args.output.mkdir(parents=True, exist_ok=True)
    selector = Router()
    receipts = []
    for manifest in MANIFESTS.values():
        variant = manifest["variant"]
        if variant not in VARIANTS:
            continue
        if args.variant and variant not in args.variant:
            continue
        agent = Agent(
            str(args.checkpoint / VARIANTS[variant]), device="cuda", fast=False, compile=False
        )
        for case in manifest["testcases"]:
            if args.case and case["name"] not in args.case:
                continue
            record = json.loads((ROOT / case["inputs"]["document_path"]).read_text())
            record["model"] = variant
            modes = ["explicit"]
            if (variant, Path(case["inputs"]["document_path"]).stem) in {
                ("english", "email"),
                ("multilingual", "hindi"),
                ("multilingual", "spanish"),
            }:
                modes.append("automatic")
            for mode in modes:
                if mode == "automatic":
                    record.pop("model")
                decision = selector.route(
                    record["state"], record["questions"], model=record.get("model")
                )
                assert decision.model == variant
                ids = list(record["questions"])
                internal = {qid: agent._to_internal(record["questions"][qid]) for qid in ids}
                items = agent._encode_state(record["state"], ids, internal) if ids else []
                with torch.inference_mode():
                    reference = agent.predict(record["state"], record["questions"])
                    logits = (
                        agent._forward(collate_items([items], agent.tok.pad_token_id))[0]
                        if items
                        else []
                    )
                assert agent.device.type == "cuda"
                path = args.output / (case["name"] + "-" + mode + ".json")
                path.write_text(json.dumps(record, ensure_ascii=False))
                native = json.loads(
                    subprocess.check_output(
                        [
                            str(args.probe),
                            str(args.bundle),
                            str(args.runtime_root),
                            str(path),
                            "2",
                        ],
                        text=True,
                    )
                )
                path.with_suffix(".result.json").write_text(json.dumps(native, indent=2))
                path.with_suffix(".reference.json").write_text(json.dumps(reference, indent=2))
                assert not any(
                    any(token in lib.lower() for token in ("python", "torch", "c10"))
                    for lib in native["loaded_libraries"]
                )
                result = native["results"][0]
                assert result["scores"] == native["results"][1]["scores"]
                assert result["response"]["routing"] == decision
                reference["routing"] = decision
                assert result["response"]["usage"] == reference["usage"]
                assert len(result["scores"]) == len(ids)
                for qid, item, raw, actual in zip(
                    ids, items, logits, result["scores"], strict=True
                ):
                    assert actual["question_id"] == qid
                    count = len(item["markers"])
                    tau = agent.temperature_by_options.get(
                        temp_bucket(item["qtype"], count), agent.temperature[item["qtype"]]
                    )
                    expected = raw[:count] / tau
                    probabilities = np.exp(expected - expected.max())
                    probabilities /= probabilities.sum()
                    np.testing.assert_allclose(actual["logits"], expected, atol=0.125, rtol=0.015)
                    np.testing.assert_allclose(
                        actual["probabilities"], probabilities, atol=0.002, rtol=0.01
                    )
                    assert np.argmax(actual["probabilities"]) == np.argmax(probabilities)
                print(
                    "Reference logit/probability/argmax gates passed:",
                    case["name"],
                    mode,
                    flush=True,
                )
                compare_response(result["response"], reference)
                receipt = {"case": case["name"], "mode": mode, "response": result["response"]}
                print(json.dumps(receipt, ensure_ascii=False), flush=True)
                receipts.append(receipt)
        del agent
        torch.cuda.empty_cache()
    if not receipts:
        raise ValueError("no requested Laya case executed")
    (args.output / "comparison.json").write_text(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
