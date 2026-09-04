# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K2-Horizon-owned strict greedy-continuation contract."""

from __future__ import annotations

from .comparators.text import TextComparator
from .contracts import (
    CompareResult,
    E2ECase,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


def _metric(passed: bool, note: str) -> MetricResult:
    return MetricResult(
        value=1.0 if passed else 0.0,
        threshold=1.0,
        operator="==",
        passed=passed,
        note=note,
    )


def _tokens(output: StageOutput) -> list[int]:
    return [int(token) for token in (output.data or {}).get("token_ids", [])]


def _contract_config(case: E2ECase) -> dict:
    config = case.metadata.get("contract_config", {})
    return dict(config) if isinstance(config, dict) else {}


def _cpp_command(output: StageOutput) -> list[str]:
    return [str(item) for item in (output.metadata or {}).get("cpp", {}).get("command", [])]


def _prompt_tokens(output: StageOutput) -> list[int]:
    return [int(token) for token in (output.data or {}).get("prompt_token_ids", [])]


def _command_has_pair(command: list[str], flag: str, value: str) -> bool:
    return any(
        command[index] == flag and command[index + 1] == value for index in range(len(command) - 1)
    )


class K2HorizonGreedyContinuationPlugin:
    reference_families = ["k2_horizon_greedy_continuation"]
    user_contract = "continuation_parity"

    def configure_reference(self, case: E2ECase) -> dict:
        return _contract_config(case)

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
    ) -> CompareResult:
        numeric = TextComparator().compare(
            trt=trt_output,
            ref=ref_output,
            threshold=threshold,
            stage=StageSpec(name=trt_output.stage_name),
        )
        trt_tokens = _tokens(trt_output)
        ref_tokens = _tokens(ref_output)
        revision = str((ref_output.data or {}).get("model_revision", ""))
        expected = case.metadata.get("expected_continuation_token_ids")
        expected_tokens = [int(token) for token in expected] if isinstance(expected, list) else []
        checks = {
            "nonempty_continuation": (
                bool(trt_tokens) and bool(ref_tokens),
                f"TRT={trt_tokens}, HF={ref_tokens}",
            ),
            "exact_token_parity": (
                bool(trt_tokens) and trt_tokens == ref_tokens,
                f"TRT={trt_tokens}, HF={ref_tokens}",
            ),
            "pinned_reference_revision": (
                bool(case.hf_revision) and revision == case.hf_revision,
                f"expected={case.hf_revision}, actual={revision}",
            ),
            "expected_golden_continuation": (
                bool(expected_tokens)
                and ref_tokens == expected_tokens
                and trt_tokens == expected_tokens,
                f"expected={expected_tokens}, TRT={trt_tokens}, HF={ref_tokens}",
            ),
        }
        contract_metrics = {name: _metric(passed, note) for name, (passed, note) in checks.items()}
        metrics = {**numeric.metrics, **contract_metrics}
        contract_passed = all(passed for passed, _note in checks.values())
        passed = numeric.passed and contract_passed
        failed = [name for name, (ok, _note) in checks.items() if not ok]
        if numeric.status == StageStatus.ERROR.value:
            status = StageStatus.ERROR.value
        else:
            status = StageStatus.PASSED.value if passed else StageStatus.FAILED.value
        message_parts = [numeric.message]
        if failed:
            message_parts.append("contract failed: " + ", ".join(failed))
        return CompareResult(
            stage_name=trt_output.stage_name,
            status=status,
            metrics=metrics,
            composite_rule=f"({numeric.composite_rule}) AND " + " AND ".join(checks),
            message="; ".join(part for part in message_parts if part),
        )


class K2HorizonHighReasoningChatPlugin(K2HorizonGreedyContinuationPlugin):
    """Verify the pinned publisher prefix and deterministic high-reasoning reply."""

    reference_families = ["k2_horizon_high_reasoning_chat"]
    user_contract = "chat_prefix_parity"

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
    ) -> CompareResult:
        base = super().verify(trt_output, ref_output, case, threshold)
        config = _contract_config(case)
        command = _cpp_command(trt_output)
        production_prompt = _prompt_tokens(trt_output)
        reference_prompt = _prompt_tokens(ref_output)
        debug_prompt = [
            int(token)
            for token in (trt_output.metadata or {})
            .get("debug_runner", {})
            .get("prompt_token_ids", [])
        ]
        expected_prompt_raw = case.metadata.get("expected_prompt_token_ids")
        expected_prompt = (
            [int(token) for token in expected_prompt_raw]
            if isinstance(expected_prompt_raw, list)
            else []
        )
        expected_text = str(case.metadata.get("expected_continuation_text", ""))
        trt_text = trt_output.text or ""
        reference_text = ref_output.text or ""
        generation = (ref_output.data or {}).get("generation", {})

        checks = {
            "base_numeric_and_golden_contract": (
                base.status == StageStatus.PASSED.value,
                base.message,
            ),
            "chat_contract_config_exact": (
                config == {"use_chat_template": True, "reasoning_effort": "high"},
                f"config={config}",
            ),
            "chat_template_flag_forwarded": (
                "--chat-template" in command,
                f"command={command}",
            ),
            "prompt_token_receipt_requested": (
                _command_has_pair(
                    command,
                    "--set",
                    "k2_horizon.emit_prompt_token_ids=true",
                ),
                f"command={command}",
            ),
            "expected_chat_prompt_declared": (
                bool(expected_prompt),
                f"expected={expected_prompt}",
            ),
            "production_prompt_matches_expected": (
                bool(expected_prompt) and production_prompt == expected_prompt,
                f"expected={expected_prompt}, C++={production_prompt}",
            ),
            "reference_prompt_matches_expected": (
                bool(expected_prompt) and reference_prompt == expected_prompt,
                f"expected={expected_prompt}, HF={reference_prompt}",
            ),
            "debug_prompt_matches_expected": (
                bool(expected_prompt) and debug_prompt == expected_prompt,
                f"expected={expected_prompt}, debug={debug_prompt}",
            ),
            "reference_and_debug_prompt_match": (
                bool(reference_prompt) and reference_prompt == debug_prompt,
                f"HF={reference_prompt}, debug={debug_prompt}",
            ),
            "production_reference_and_debug_prompt_match": (
                bool(production_prompt)
                and production_prompt == reference_prompt
                and production_prompt == debug_prompt,
                f"C++={production_prompt}, HF={reference_prompt}, debug={debug_prompt}",
            ),
            "expected_decoded_continuation": (
                bool(expected_text)
                and trt_text == expected_text
                and reference_text == expected_text,
                f"expected={expected_text!r}, TRT={trt_text!r}, HF={reference_text!r}",
            ),
            "reference_reasoning_effort_forwarded": (
                isinstance(generation, dict) and generation.get("reasoning_effort") == "high",
                f"generation={generation}",
            ),
        }
        chat_metrics = {name: _metric(passed, note) for name, (passed, note) in checks.items()}
        metrics = {**base.metrics, **chat_metrics}
        passed = all(passed for passed, _note in checks.values())
        failed = [name for name, (ok, _note) in checks.items() if not ok]
        status = (
            StageStatus.ERROR.value
            if base.status == StageStatus.ERROR.value
            else StageStatus.PASSED.value
            if passed
            else StageStatus.FAILED.value
        )
        message = "Pinned K2-Horizon high-reasoning chat parity"
        if failed:
            message += " failed: " + ", ".join(failed)
        return CompareResult(
            stage_name=trt_output.stage_name,
            status=status,
            metrics=metrics,
            composite_rule=f"({base.composite_rule}) AND " + " AND ".join(checks),
            message=message,
        )


plugin = [
    K2HorizonGreedyContinuationPlugin(),
    K2HorizonHighReasoningChatPlugin(),
]
