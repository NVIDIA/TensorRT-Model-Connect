# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Present original Laya answer objects as labeled structured decisions."""

import json

from tools.e2e_evidence import evidence_enabled, record_evidence


def record_decision_input(record):
    if not evidence_enabled():
        return
    try:
        text = "Structured decision request\nState:\n" + json.dumps(
            record.get("state"), ensure_ascii=False, indent=2, default=str
        )
        text += "\nQuestions:\n" + json.dumps(
            record.get("questions", {}), ensure_ascii=False, indent=2, default=str
        )
        record_evidence("inputs", {"text": text})
    except Exception as error:
        record_evidence("presentation", {"input_unavailable": str(error)})


def _response_text(response, label):
    lines = [f"{label} structured decisions (display summary; not generated text)"]
    for question, answer in response["answers"].items():
        lines.extend((f"Question: {question}", json.dumps(
            answer, ensure_ascii=False, indent=2, default=str
        )))
    if not response["answers"]:
        lines.append("No questions; no decisions requested.")
    if "routing" in response:
        lines.append("Routing: " + json.dumps(response["routing"], ensure_ascii=False, default=str))
    return "\n".join(lines)


def record_decisions(native, reference, logits, *, empty_questions=False):
    """Never substitute native-derived SDK formatting for the original reference."""
    if not evidence_enabled():
        return
    try:
        record_evidence("native", {**native, "text": _response_text(native["response"], "Native")})
        view = {"response": reference, "logits": logits,
                "text": _response_text(reference, "Original Laya SDK reference")}
        if empty_questions:
            view.update(mode="contract_only", oracle="empty-question response contract")
        record_evidence("reference", view)
    except Exception as error:
        record_evidence("presentation", {"decision_summary_unavailable": str(error)})
