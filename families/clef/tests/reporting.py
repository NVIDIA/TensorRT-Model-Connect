# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display Clef decisions without inventing an independently formatted SDK answer."""

import json
import math

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


def record_decisions(native, reference_options, logits, expected_questions):
    """Add display-only summaries; keep the original response, scores and logits."""
    if not evidence_enabled():
        return
    try:
        answers = native["response"]["answers"]
        lines = ["Native structured decisions (display summary; not generated text)"]
        for question, answer in answers.items():
            lines.extend((f"Question: {question}", json.dumps(
                answer, ensure_ascii=False, indent=2, default=str
            )))
        if not answers:
            lines.append("No questions; no decisions requested.")
        record_evidence("native", {**native, "text": "\n".join(lines)})

        lines = [
            "Independent reference option probabilities (display summary)",
            "These are reference probabilities, not a separately formatted SDK answer.",
        ]
        for question, item in reference_options.items():
            options, probabilities = item["option_ids"], item["probabilities"]
            lines.append(f"Question: {question}")
            if len(options) != len(probabilities) or not options:
                lines.append("Complete option probabilities are unavailable.")
                continue
            lines.extend(
                f"  {option}: {float(probability):.7g}"
                for option, probability in zip(options, probabilities, strict=True)
            )
            if not all(math.isfinite(float(value)) for value in probabilities):
                lines.append("Highest-probability option unavailable: non-finite probabilities.")
                continue
            winner = max(range(len(probabilities)), key=probabilities.__getitem__)
            lines.append(f"Highest-probability option: {options[winner]}")
        if not reference_options:
            lines.append("No questions; no option comparison." if expected_questions == 0
                         else "Reference option probabilities have not been recorded.")
        reference = {"logits": logits, "option_probabilities": reference_options,
                     "text": "\n".join(lines)}
        if expected_questions == 0:
            reference.update(mode="contract_only", oracle="empty-question response contract")
        record_evidence("reference", reference)
    except Exception as error:
        record_evidence("presentation", {"decision_summary_unavailable": str(error)})
