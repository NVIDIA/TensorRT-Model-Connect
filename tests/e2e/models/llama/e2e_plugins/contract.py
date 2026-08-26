# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""llama-owned E2E contract plugins."""
from __future__ import annotations

import re

from tests.e2e_harness.contracts import (
    CompareResult,
    E2ECase,
    MetricResult,
    StageOutput,
    StageStatus,
    ThresholdProfile,
)


_PREFILL_OBSERVATION_RE = re.compile(
    r"^\[trtmc\.prefill\] tokens=(\d+) launches=(\d+) max_chunk=(\d+)$"
)


def _parse_prefill_observations(stderr: str) -> tuple[tuple[int, int, int], ...]:
    """Return all runtime token, launch, and observed-max-chunk counters."""
    observations = []
    for line in stderr.splitlines():
        match = _PREFILL_OBSERVATION_RE.fullmatch(line.strip())
        if match is not None:
            observations.append(tuple(int(value) for value in match.groups()))
    return tuple(observations)


# Model-owned contract helpers. Keep behavior here so contract semantics do not
# drift across model families through shared harness code.
def contract_config(case):
    config = case.metadata.get("contract_config", {})
    return dict(config) if isinstance(config, dict) else {}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip().lower()


def strip_prompt_echo(text: str, prompt: str) -> str:
    if not text or not prompt:
        return text
    idx = text.find(prompt)
    if 0 <= idx <= 2048:
        return text[idx + len(prompt):].lstrip()
    norm_text = normalize_text(text)
    norm_prompt = normalize_text(prompt)
    if norm_prompt and norm_text.startswith(norm_prompt):
        return text[len(prompt):].lstrip() if text.startswith(prompt) else text
    return text


_CHAT_ROLE_PREFIXES = (
    "### response:", "### assistant:", "assistant:",
    "<|assistant|>", "<|im_start|>assistant\n",
)

_CHAT_TURN_MARKERS = (
    "### response:", "### instruction:", "### assistant:",
    "### user:", "<|assistant|>", "<|user|>",
    "<|im_start|>", "<|im_end|>",
)


def strip_chat_markup(text: str) -> str:
    if not text:
        return ""
    out = text.lstrip()
    while True:
        lowered = out.lower()
        matched = False
        for prefix in _CHAT_ROLE_PREFIXES:
            if lowered.startswith(prefix):
                out = out[len(prefix):].lstrip()
                matched = True
                break
        if not matched:
            break
    lowered = out.lower()
    cut = len(out)
    for marker in _CHAT_TURN_MARKERS:
        idx = lowered.find(marker)
        if idx > 0:
            cut = min(cut, idx)
    if cut < len(out):
        out = out[:cut]
    out = re.sub(r"(?:\s*#{2,}\s*)+$", "", out).strip()
    return out


def extract_answer(output, prompt: str = "") -> str:
    raw = output.text or ""
    if prompt:
        raw = strip_prompt_echo(raw, prompt)
    raw = strip_chat_markup(raw)
    return raw.strip()


def levenshtein_ned(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        curr = [i + 1]
        for j, c2 in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if c1 == c2 else 1),
            ))
        prev = curr
    return prev[-1] / max_len


def make_pass(stage_name: str, metrics, rule: str = ""):
    from tests.e2e_harness.contracts import CompareResult
    return CompareResult(
        stage_name=stage_name,
        status="passed",
        metrics=metrics,
        composite_rule=rule,
        message="Contract verified",
    )


def make_fail(stage_name: str, metrics, rule: str = "", message: str = ""):
    from tests.e2e_harness.contracts import CompareResult
    return CompareResult(
        stage_name=stage_name,
        status="failed",
        metrics=metrics,
        composite_rule=rule,
        message=message or "Contract verification failed",
    )


def make_skip(stage_name: str, metrics, rule: str = "", message: str = ""):
    from tests.e2e_harness.contracts import CompareResult
    return CompareResult(
        stage_name=stage_name,
        status="skipped",
        metrics=metrics,
        composite_rule=rule,
        message=message or "Contract validation skipped",
    )


def make_error(stage_name: str, error: str):
    from tests.e2e_harness.contracts import CompareResult
    return CompareResult(
        stage_name=stage_name,
        status="error",
        message=f"Contract verification error: {error}",
    )

class LlamaCausalContinuationPlugin:
    reference_families = ["causal_base_continuation"]
    user_contract = "continuation_parity"

    def configure_reference(self, case):
        return contract_config(case)

    def verify(self, trt_output, ref_output, case, threshold):
        cpp_rc = (trt_output.data or {}).get("cpp_returncode")
        if cpp_rc not in (None, 0):
            metrics = {
                "cpp_returncode_ok": MetricResult(
                    value=0.0,
                    threshold=1.0,
                    operator="==",
                    passed=False,
                    note=f"cpp_returncode={cpp_rc}",
                ),
            }
            detail = (trt_output.data or {}).get("cpp_runtime_error")
            suffix = f": {detail}" if detail else ""
            return CompareResult(
                stage_name="full_generation",
                status=StageStatus.ERROR.value,
                metrics=metrics,
                message=f"TRT C++ run failed (cpp_returncode={cpp_rc}){suffix}",
            )

        prompt = case.inputs.get("prompt", "")
        config = contract_config(case)
        preserve_prompt_echo = bool(config.get("preserve_prompt_echo"))
        reconstruction_check = bool(config.get("seq2seq_reconstruction"))
        if preserve_prompt_echo:
            trt_text = normalize_text(trt_output.text or "")
            ref_text = normalize_text(ref_output.text or "")
        else:
            trt_text = normalize_text(strip_prompt_echo(trt_output.text or "", prompt))
            ref_text = normalize_text(strip_prompt_echo(ref_output.text or "", prompt))

        if not trt_text and not ref_text:
            metrics = {
                "non_empty_continuation": MetricResult(
                    value=0.0,
                    threshold=1.0,
                    operator="==",
                    passed=False,
                    note="empty TRT and reference text do not validate parity",
                ),
            }
            return make_fail(
                "full_generation",
                metrics,
                "non-empty continuation required",
                "Both TRT and reference produced empty continuation",
            )

        ned = levenshtein_ned(trt_text, ref_text)
        ned_threshold = threshold.metrics.get("contract_ned_threshold", 0.25)
        prefix_len = min(50, min(len(trt_text), len(ref_text)))
        prefix_match = (trt_text[:prefix_len] == ref_text[:prefix_len]) if prefix_len > 0 else True

        metrics = {
            "ned": MetricResult(
                value=ned,
                threshold=ned_threshold,
                operator="<=",
                passed=ned <= ned_threshold,
            ),
            "prefix_match": MetricResult(
                value=1.0 if prefix_match else 0.0,
                threshold=1.0,
                operator="==",
                passed=prefix_match,
                note=f"first {prefix_len} chars",
            ),
        }
        if reconstruction_check:
            metrics["non_empty_trt_text"] = MetricResult(
                value=1.0 if trt_text else 0.0,
                threshold=1.0,
                operator="==",
                passed=bool(trt_text),
                note="visible TRT reconstruction text",
            )
            metrics["non_empty_reference_text"] = MetricResult(
                value=1.0 if ref_text else 0.0,
                threshold=1.0,
                operator="==",
                passed=bool(ref_text),
                note="visible HF reconstruction text",
            )

        passed = ned <= ned_threshold
        rule = (
            "seq2seq reconstruction parity against HF reference"
            if reconstruction_check
            else "ned <= threshold (continuation parity)"
        )
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            f"Continuation diverged: NED={ned:.3f}",
        )

class LlamaChatInstructPlugin:
    reference_families = ["chat_instruct_template"]
    user_contract = "chat_response"

    def configure_reference(self, case):
        return contract_config(case)

    def verify(self, trt_output, ref_output, case, threshold):
        prompt = case.inputs.get("prompt", "")
        config = contract_config(case)
        if config.get("enable_thinking") is False:
            raw_trt = trt_output.text or ""
            if "<think>" in raw_trt:
                return make_fail(
                    "full_generation",
                    {
                        "thinking_suppressed": MetricResult(
                            value=0.0,
                            threshold=1.0,
                            operator="==",
                            passed=False,
                            note="no-thinking output must not contain <think>",
                        )
                    },
                    message="TRT emitted a thinking block with thinking disabled",
                )

        trt_answer = normalize_text(extract_answer(trt_output, prompt))
        ref_answer = normalize_text(extract_answer(ref_output, prompt))

        if not trt_answer:
            return make_fail("full_generation", {}, message="TRT produced empty response")

        exact_match = trt_answer == ref_answer
        ned = levenshtein_ned(trt_answer, ref_answer)
        ned_threshold = threshold.metrics.get("contract_ned_threshold", 0.15)
        metrics = {
            "exact_match": MetricResult(
                value=1.0 if exact_match else 0.0,
                threshold=1.0,
                operator="==",
                passed=exact_match,
            ),
            "ned": MetricResult(
                value=ned,
                threshold=ned_threshold,
                operator="<=",
                passed=ned <= ned_threshold,
            ),
        }

        passed = exact_match or ned <= ned_threshold
        rule = "exact_match OR ned <= threshold"
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            f"Chat response diverged: NED={ned:.3f}",
        )


class LlamaNativeKvChunkedPrefillRegressionPlugin:
    reference_families = ["llama_native_kv_chunked_prefill_regression"]
    user_contract = "runtime_invariants"

    def configure_reference(self, case: E2ECase) -> dict:
        del case
        return {}

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
    ):
        del ref_output, threshold
        stage = trt_output.stage_name
        if stage != "full_generation":
            return make_pass(stage, {}, f"{stage} completed")

        data = trt_output.data or {}
        cpp_meta = (trt_output.metadata or {}).get("cpp", {})
        cpp_rc = int(data.get("cpp_returncode", -1))
        token_ids = data.get("token_ids", [])
        expected_generated = int(case.inputs.get("max_new_tokens", 0))
        expected_prompt = int(case.inputs.get("expected_prompt_tokens", -1))
        actual_prompt = int(data.get("prompt_token_count", -1))
        decode_s = float(cpp_meta.get("trt_engine_decode_s", 0.0))
        stderr = str(cpp_meta.get("stderr", ""))
        expected_rows = int(case.metadata.get("expected_kv_cache_rows", -1))
        expected_chunks = int(case.metadata.get("expected_prefill_chunks", -1))
        expected_limit = int(case.metadata.get("expected_prefill_chunk_limit", -1))
        cache_marker = f"KV cache rows={expected_rows} (bundle max={expected_rows}"
        prefill_marker = 'label="prefill_engine_plan:prefill"'
        prefill_observations = _parse_prefill_observations(stderr)
        observed_tokens = sum(item[0] for item in prefill_observations)
        observed_launches = sum(item[1] for item in prefill_observations)
        observed_max_chunk = max((item[2] for item in prefill_observations), default=0)
        chunk_plan_is_consistent = (
            expected_limit > 0
            and expected_chunks
            == (expected_prompt + expected_limit - 1) // expected_limit
        )
        chunked_prefill_observed = chunk_plan_is_consistent and any(
            prefill_marker in line and f"launches={expected_chunks}" in line.split()
            for line in stderr.splitlines()
        )
        chunk_limit_observed = (
            bool(prefill_observations)
            and observed_tokens == expected_prompt
            and observed_launches == expected_chunks
            and 0 < observed_max_chunk <= expected_limit
        )

        checks = {
            "runtime_returncode_ok": (
                cpp_rc == 0,
                f"cpp_returncode={cpp_rc}",
            ),
            "fixed_prompt_token_count": (
                actual_prompt == expected_prompt,
                f"expected={expected_prompt}, actual={actual_prompt}",
            ),
            "full_native_kv_capacity_loaded": (
                cache_marker in stderr,
                cache_marker,
            ),
            "chunked_prefill_executed": (
                chunked_prefill_observed,
                f"{prefill_marker} with launches={expected_chunks}",
            ),
            "prefill_chunk_limit_observed": (
                chunk_limit_observed,
                f"expected tokens={expected_prompt}, launches={expected_chunks}, "
                f"max_chunk<={expected_limit}; observed={prefill_observations}",
            ),
            "requested_tokens_generated": (
                isinstance(token_ids, list) and len(token_ids) == expected_generated,
                f"expected={expected_generated}, actual={len(token_ids) if isinstance(token_ids, list) else 'missing'}",
            ),
            "decode_step_executed": (
                expected_generated >= 2 and decode_s > 0.0,
                f"generated_tokens={expected_generated}, decode_s={decode_s}",
            ),
            "no_runtime_error_signature": (
                not data.get("cpp_runtime_error"),
                str(data.get("cpp_runtime_error") or "none detected"),
            ),
        }
        metrics = {
            name: MetricResult(
                value=1.0 if passed else 0.0,
                threshold=1.0,
                operator="==",
                passed=passed,
                note=note,
            )
            for name, (passed, note) in checks.items()
        }
        rule = " AND ".join(checks)
        if all(passed for passed, _note in checks.values()):
            return make_pass(stage, metrics, rule)
        failed = [name for name, (passed, _note) in checks.items() if not passed]
        return make_fail(
            stage,
            metrics,
            rule,
            "Llama native KV chunked-prefill regression contract failed: "
            + ", ".join(failed),
        )


plugin = [
    LlamaCausalContinuationPlugin(),
    LlamaChatInstructPlugin(),
    LlamaNativeKvChunkedPrefillRegressionPlugin(),
]
