# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply the current behavior policy to retained paired outputs without inference."""

import argparse
import json
from pathlib import Path

if __package__:
    from .lfm25_semantic_contract import POLICY, evaluate, reference_parts
else:
    from lfm25_semantic_contract import POLICY, evaluate, reference_parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve earlier receipts with a new filename")
    from transformers import AutoTokenizer

    path = args.source / "parity.json"
    if not path.exists():
        path = args.source / "semantic-parity.json"
    source = json.loads(path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    generation = json.loads((args.model_dir / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    results = []
    for original in source["cases"]:
        name = original["case"]
        row = {
            "case": name,
            "prior_contract": original.get("semantic_contract"),
            "semantic_contract": "not_run",
        }
        native_path = args.source / (name + ".native.json")
        request_path = args.source / (name + ".reference-request.json")
        if not native_path.is_file() or "reference_ids" not in original:
            row["reason"] = "paired native/reference output is not available"
            results.append(row)
            continue
        native = json.loads(native_path.read_text())
        request = json.loads(request_path.read_text())
        reference_ids = original["reference_ids"]
        row.update(
            native_ids=native["token_ids"],
            reference_ids=reference_ids,
            exact_ids=native["token_ids"] == reference_ids,
            input_tokens=original["input_tokens"],
        )
        try:
            row["semantic_checks"] = evaluate(
                name,
                native,
                reference_parts(tokenizer, reference_ids, request),
                reference_ids,
                eos,
                request,
                original["input_tokens"],
            )
            row["semantic_contract"] = "passed"
        except (AssertionError, ValueError, KeyError) as error:
            row.update(semantic_contract="failed", semantic_error=str(error))
        results.append(row)
    receipt = {
        "acceptance_policy": POLICY,
        "scope": "Reassessment of retained paired outputs; no new inference was executed.",
        "source_receipt": {
            "directory": args.source.name,
            "file": path.name,
            "bytes": path.stat().st_size,
        },
        "checkpoint": source.get("checkpoint"),
        "revision": source.get("revision"),
        "reference_precision": source.get("reference_precision"),
        "native_metadata": source.get("native_metadata"),
        "previous_external_identity": source.get("native_identity"),
        "reference_execution": source.get("reference_execution"),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    counts = {
        state: sum(row["semantic_contract"] == state for row in results)
        for state in ("passed", "failed", "not_run")
    }
    print(json.dumps(counts))
    if counts["failed"] or counts["not_run"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
