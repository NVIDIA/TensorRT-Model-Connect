# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM2-owned continuation, chat-template, and sampling contracts."""

from __future__ import annotations

from .contracts import (
    CompareResult,
    E2ECase,
    MetricResult,
    StageOutput,
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


def _result(stage: str, checks: dict[str, tuple[bool, str]], label: str) -> CompareResult:
    metrics = {name: _metric(passed, note) for name, (passed, note) in checks.items()}
    passed = all(value for value, _note in checks.values())
    failed = [name for name, (value, _note) in checks.items() if not value]
    return CompareResult(
        stage_name=stage,
        status="passed" if passed else "failed",
        metrics=metrics,
        composite_rule=" AND ".join(checks),
        message=label if passed else f"{label} failed: {', '.join(failed)}",
    )


def _tokens(output: StageOutput) -> list[int]:
    return [int(token) for token in (output.data or {}).get("token_ids", [])]


def _cpp_command(output: StageOutput) -> list[str]:
    return [str(item) for item in (output.metadata or {}).get("cpp", {}).get("command", [])]


def _contract_config(case: E2ECase) -> dict:
    config = case.metadata.get("contract_config", {})
    return dict(config) if isinstance(config, dict) else {}


class Lfm2GreedyContinuationPlugin:
    reference_families = ["lfm2_greedy_continuation"]
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
        del threshold
        trt_tokens = _tokens(trt_output)
        ref_tokens = _tokens(ref_output)
        revision = str((ref_output.data or {}).get("model_revision", ""))
        checks = {
            "cpp_returncode_ok": (
                int((trt_output.data or {}).get("cpp_returncode", -1)) == 0,
                f"returncode={(trt_output.data or {}).get('cpp_returncode', -1)}",
            ),
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
        }
        expected = case.metadata.get("expected_continuation_token_ids")
        if isinstance(expected, list):
            expected_tokens = [int(token) for token in expected]
            checks["expected_golden_continuation"] = (
                ref_tokens == expected_tokens and trt_tokens == expected_tokens,
                f"expected={expected_tokens}, TRT={trt_tokens}, HF={ref_tokens}",
            )
        return _result(
            trt_output.stage_name,
            checks,
            "Exact LFM2 cached continuation parity",
        )


class Lfm2ChatTemplatePlugin(Lfm2GreedyContinuationPlugin):
    reference_families = ["lfm2_chat_template"]
    user_contract = "chat_response"

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
    ) -> CompareResult:
        base = super().verify(trt_output, ref_output, case, threshold)
        command = _cpp_command(trt_output)
        ref_prompt = [int(token) for token in (ref_output.data or {}).get("prompt_token_ids", [])]
        debug_prompt = [
            int(token)
            for token in (trt_output.metadata or {})
            .get("debug_runner", {})
            .get("prompt_token_ids", [])
        ]
        expected = case.metadata.get("expected_prompt_token_ids")
        expected_prompt = [int(token) for token in expected] if isinstance(expected, list) else []

        checks = {
            "base_continuation_contract": (
                base.status == "passed",
                base.message,
            ),
            "chat_template_flag_forwarded": (
                "--chat-template" in command,
                "--chat-template present" if "--chat-template" in command else str(command),
            ),
            "reference_chat_prompt_nonempty": (
                bool(ref_prompt),
                f"prompt_token_ids={ref_prompt}",
            ),
        }
        if debug_prompt:
            checks["trt_and_reference_prompt_tokens_match"] = (
                debug_prompt == ref_prompt,
                f"TRT={debug_prompt}, HF={ref_prompt}",
            )
        if expected_prompt:
            checks["expected_chat_prompt_tokens"] = (
                ref_prompt == expected_prompt
                and (not debug_prompt or debug_prompt == expected_prompt),
                f"expected={expected_prompt}, TRT={debug_prompt}, HF={ref_prompt}",
            )
        expected_text = str(case.metadata.get("expected_continuation_text", ""))
        if expected_text:
            trt_text = (trt_output.text or "").strip()
            ref_text = (ref_output.text or "").strip()
            checks["expected_decoded_continuation"] = (
                trt_text == expected_text and ref_text == expected_text,
                f"expected={expected_text!r}, TRT={trt_text!r}, HF={ref_text!r}",
            )
        return _result(
            trt_output.stage_name,
            checks,
            "Pinned LFM2 chat-template parity",
        )


class Lfm2ModelCardSamplingPlugin:
    reference_families = ["lfm2_model_card_sampling"]
    user_contract = "sampling"

    def configure_reference(self, case: E2ECase) -> dict:
        return _contract_config(case)

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
    ) -> CompareResult:
        del threshold
        command = _cpp_command(trt_output)
        generation = (ref_output.data or {}).get("generation", {})
        trt_tokens = _tokens(trt_output)
        ref_tokens = _tokens(ref_output)
        required_flags = {
            "--temperature": str(case.inputs.get("temperature", 0.3)),
            "--min-p": str(case.inputs.get("min_p", 0.15)),
            "--top-k": str(case.inputs.get("top_k", 50)),
            "--seed": str(case.inputs.get("seed", 42)),
            "--repetition-penalty": str(case.inputs.get("repetition_penalty", 1.05)),
        }
        flag_checks = {}
        for flag, value in required_flags.items():
            try:
                index = command.index(flag)
                actual = command[index + 1]
            except (ValueError, IndexError):
                actual = ""
            flag_checks[f"flag_{flag[2:].replace('-', '_')}"] = (
                actual == value,
                f"expected {flag} {value}, command={command}",
            )

        expected_generation = {
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 1.0,
            "min_p": 0.15,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "max_new_tokens": 512,
            "use_chat_template": True,
        }
        generation_ok = all(
            generation.get(name) == expected for name, expected in expected_generation.items()
        )
        checks = {
            "cpp_returncode_ok": (
                int((trt_output.data or {}).get("cpp_returncode", -1)) == 0,
                f"returncode={(trt_output.data or {}).get('cpp_returncode', -1)}",
            ),
            "trt_sample_nonempty": (
                bool(trt_tokens) or bool((trt_output.text or "").strip()),
                f"tokens={trt_tokens}",
            ),
            "reference_sample_nonempty": (
                bool(ref_tokens) or bool((ref_output.text or "").strip()),
                f"tokens={ref_tokens}",
            ),
            "native_byte_level_decode_clean": (
                all(marker not in (trt_output.text or "") for marker in ("Ġ", "Ċ")),
                f"text={trt_output.text!r}",
            ),
            "chat_template_flag_forwarded": (
                "--chat-template" in command,
                str(command),
            ),
            "reference_model_card_parameters": (
                generation_ok,
                f"expected={expected_generation}, actual={generation}",
            ),
            "pinned_reference_revision": (
                (ref_output.data or {}).get("model_revision") == case.hf_revision,
                f"expected={case.hf_revision}, actual={(ref_output.data or {}).get('model_revision')}",
            ),
            **flag_checks,
        }
        return _result(
            trt_output.stage_name,
            checks,
            "LFM2 model-card sampling invocation",
        )


plugin = [
    Lfm2GreedyContinuationPlugin(),
    Lfm2ChatTemplatePlugin(),
    Lfm2ModelCardSamplingPlugin(),
]
