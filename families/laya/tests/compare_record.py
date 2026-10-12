# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require exact released-tokenizer IDs and sequence construction for all variants."""

import argparse
import json
from pathlib import Path
import subprocess

from families.laya.cli import VARIANTS
from families.laya.tokenizer import tokenizer_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from laya import Agent
    from laya.agent import _load_tokenizer

    texts = [
        "",
        "hello",
        " hello",
        "  hello  world",
        "\nhello\tworld\r\n",
        "▁already ▁separated",
        "cafe\u0301",
        "e\u0323\u0301",
        "\u1100\u1161\u11a8",
        "Å Å A\u030a",
        "मैं अपने खाते में लॉग इन नहीं कर पा रहा हूँ",
        "Mein Konto wurde zweimal belastet",
        "账单重复收费",
        "계정이 두 번 청구되었습니다",
        "متى يصل طلبي؟",
        "🦄🙂👩‍💻",
        "'ll I'll don't can't WE'RE 12345 99999999999999",
        "x\u00a0y\u202fz\u3000q",
    ]
    # Canonical decompositions, exclusions, reordering, and byte fallbacks.
    import unicodedata2 as unicode

    texts.extend(
        "x " + unicode.normalize("NFD", chr(code)) + " y"
        for code in range(0x110000)
        if unicode.decomposition(chr(code)) and not unicode.decomposition(chr(code)).startswith("<")
    )
    original = json.loads((Path(__file__).parent / "fixtures/email.json").read_text())
    records = [original, {"state": "empty", "questions": {}}]
    for state in [
        "मैं अपने खाते में लॉग इन नहीं कर पा रहा हूँ",
        {"message": "cafe\u0301", "amount": 1e-7, "paid": False},
        "long state " * 1500,
        [{"role": "user", "content": "old message " * 1000}, {"role": "user", "content": "newest"}],
    ]:
        records.append({"state": state, "questions": original["questions"]})
    records.append(
        {
            "state": "a [MASK] b",
            "questions": {
                "one": {"type": "choice", "instructions": "Pick", "criteria": ["only"]},
                "labels": {
                    "type": "noul",
                    "instructions": {"statement": "cafe\u0301"},
                    "labels": {"false": " no ", "true": " yes "},
                    "criteria": {"TRUE": {"value": 0}, "FALSE": False},
                },
                "many": {
                    "type": "choice",
                    "instructions": "many " * 200,
                    "criteria": {str(i): "description " * 60 for i in range(24)},
                },
            },
        }
    )
    records.append(
        {
            "state": [0.0, -0.0, 1e-4, 1e-5, 1e15, 1e16, -1.2345e15, 1e20],
            "questions": original["questions"],
        }
    )
    records.append(
        {
            "state": "Refund requested.",
            "questions": {
                "null_labels": {
                    "type": "noul",
                    "instructions": "Is a refund requested?",
                    "labels": None,
                },
                "unicode_labels": {
                    "type": "noul",
                    "instructions": "Is a refund requested?",
                    "labels": {"false": "\u3000no\u00a0", "true": "\u2007yes\u202f"},
                },
            },
        }
    )
    records.extend(
        json.loads(path.read_text())
        for path in sorted((Path(__file__).parent / "fixtures").glob("*.json"))
    )
    receipts = []
    for variant, subfolder in VARIANTS.items():
        checkpoint = args.checkpoint / subfolder
        config = json.loads((checkpoint / "rl_agent_config.json").read_text())
        tok = _load_tokenizer(str(checkpoint / "tokenizer"), config)
        for token in ("cls", "sep", "pad", "mask"):
            config[token + "_token_id"] = getattr(tok, token + "_token_id")
        root = args.output / variant
        root.mkdir(parents=True, exist_ok=True)
        (root / "tokenizer.json").write_bytes(
            tokenizer_data(checkpoint / "tokenizer/tokenizer.json")
        )
        (root / "config.json").write_text(json.dumps(config))
        rows = texts + records
        process = subprocess.run(
            [str(args.probe), str(root / "tokenizer.json"), str(root / "config.json")],
            input="\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
        actual = [json.loads(line) for line in process.stdout.splitlines()]
        assert len(actual) == len(rows)
        for text, ids in zip(texts, actual):
            expected = tok(text, add_special_tokens=False)["input_ids"]
            assert ids == expected, (variant, text, ids, expected)
        agent = Agent.__new__(Agent)
        agent.cfg, agent.tok = config, tok
        for record, native in zip(records, actual[len(texts) :]):
            qids = list(record["questions"])
            internal = {qid: agent._to_internal(record["questions"][qid]) for qid in qids}
            expected = agent._encode_state(record["state"], qids, internal)
            assert len(native) == len(expected)
            for qid, got, want in zip(qids, native, expected):
                assert got["id"] == qid
                assert got["tokens"] == want["ids"], (variant, qid, got["tokens"], want["ids"])
                assert got["markers"] == want["markers"], (variant, qid)
                assert got["type"] == want["qtype"], (variant, qid)
        receipt = {"variant": variant, "exact_texts": len(texts), "exact_records": len(records)}
        print(receipt, flush=True)
        receipts.append(receipt)
    (args.output / "comparison.json").write_text(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
