# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare native routing metadata with the released Router without loading weights."""

import argparse
import json
from pathlib import Path
import subprocess

from families.laya.routing import routing_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from laya import Router
    from laya.router import _TYPED_DECISION_WORKFLOWS

    args.output.mkdir(parents=True, exist_ok=True)
    table = args.output / "router.json"
    table.write_text(json.dumps(routing_data(), ensure_ascii=False))
    states = [
        json.loads(path.read_text())["state"]
        for path in (Path(__file__).parent / "fixtures").glob("*.json")
    ] + [
        None,
        1234,
        "",
        "123 @foo.example",
        "Hello",
        "Please refund the payment today.",
        "Mein Konto wurde zweimal belastet",
        "Je veux un remboursement pour les deux frais",
        "Voce pode me mandar a nota fiscal?",
        "La fattura è stata pagata due volte",
        "Я хочу вернуть деньги",
        "The sender is Дмитрий Петрович Савицкий.",
        "Set α to 0.05 and β to 0.1.",
        "这是中文请求",
        "𠀀𠀁𠀂",
        "ᚠᚢᚦᚨᚱᚲ",
        "日本語のメッセージです",
        {"English keys are ignored": "मुझे पैसे वापस चाहिए"},
        ["", "Please refund"],
        ["", "éé"],
        {"nested": {"more": {"text": "No letters 123"}}},
        "English support ticket, 客户要求退还重复支付的款项，please process it today.",
        "Please visit github.com or example.com for the full report.",
    ]
    states += [
        " ".join([word] * 4) for words in routing_data()["stopwords"].values() for word in words
    ]
    records = [{"state": state, "questions": {}} for state in states]
    for field, values in {
        "model": ["english", "typed", "typed_decisions", "ml", " default "],
        "task": ["typed_decisions", "multilingual"],
        "lang": ["en-US", "en_US.UTF-8", "EN", "eng", "ro", "hi", "", "  ", "\u3000en\u00a0"],
        "lang_guess": ["en", "ro", "", "  "],
    }.items():
        for value in values:
            records.append({"state": "मुझे पैसे वापस चाहिए", "questions": {}, field: value})
    for ids in _TYPED_DECISION_WORKFLOWS.values():
        for enabled in [False, True]:
            records.append(
                {
                    "state": "I need a refund",
                    "questions": {key: {} for key in ids},
                    "auto_task_detection": enabled,
                }
            )
    actual = subprocess.run(
        [str(args.probe), str(table)],
        check=True,
        text=True,
        capture_output=True,
        input="\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
    )
    rows = [json.loads(line) for line in actual.stdout.splitlines()]
    assert len(rows) == len(records)
    for record, native in zip(records, rows, strict=True):
        router = Router(
            default=record.get("default", "english"),
            auto_task_detection=record.get("auto_task_detection", False),
        )
        options = {
            key: record[key] for key in ("model", "task", "lang", "lang_guess") if key in record
        }
        reference = router.route(record["state"], record["questions"], **options)
        assert native == reference, (record, native, reference)
    result = {"exact_routes": len(records)}
    print(result, flush=True)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
