# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare native record encoding with the pinned, unmodified release code."""

import argparse
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkpoint))
    from joint_schema_model import encode_record
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    fixtures = Path(__file__).parent / "fixtures"
    records = [json.loads(p.read_text()) for p in sorted(fixtures.glob("*.json"))]
    records += [
        {
            "state": {
                "数字": 12345.67,
                "message": "你好，世界！ café naïve\n  test\t👋",
                "null": None,
            },
            "questions": {
                "z": {"type": "choice", "criteria": {"z": None, "A": {"b": 2, "a": 1}}},
                "a": {
                    "type": "noul",
                    "instructions": "",
                    "criteria": {"true": "Oui", "false": "Non"},
                },
                "score": {"type": "score", "criteria": ["low", "medium", "high"]},
            },
        }
    ]
    count = 0
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "record.json"
        for record in records:
            path.write_text(json.dumps(record, ensure_ascii=False))
            for state_limit in [None, 0, 7]:
                expected = dataclasses.asdict(
                    encode_record(tokenizer, record, max_state_tokens=state_limit)
                )
                expected = json.loads(json.dumps(expected))
                actual = json.loads(
                    subprocess.check_output(
                        [
                            str(args.probe),
                            str(args.checkpoint / "tokenizer.json"),
                            str(path),
                            "16384",
                            str(-1 if state_limit is None else state_limit),
                        ]
                    )
                )
                for key in ("input_ids", "questions"):
                    if actual[key] != expected[key]:
                        raise AssertionError(
                            f"case {count}: {key}\nactual={actual[key]}\nexpected={expected[key]}"
                        )
                count += 1
    print(json.dumps({"exact_native_encoding_cases": count}))


if __name__ == "__main__":
    main()
