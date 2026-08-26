# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-owned post-trained chat contract plugin."""

from __future__ import annotations

from tests.e2e_harness.contracts import E2ECase, MetricResult, StageOutput, ThresholdProfile
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
    import re
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

class QwenPostTrainedChatPlugin:
    reference_families = ["chat_qwen3_posttrained"]
    user_contract = "chat_response"

    _PRE_FORMATTED_MARKERS = (
        "<|im_start|>",
        "[INST]",
        "<|start|>",
        "<|user|>",
        "<start_of_turn>",
        "<|start_header_id|>",
        "<extra_id_0>",
        "<SPECIAL_10>",
    )

    def configure_reference(self, case):
        prompt = case.inputs.get("prompt", "")
        already_formatted = any(marker in prompt for marker in self._PRE_FORMATTED_MARKERS)
        return {"use_chat_template": not already_formatted, "enable_thinking": False}

    def verify(self, trt_output, ref_output, case, threshold):
        prompt = case.inputs.get("prompt", "")
        contract_config = case.metadata.get("contract_config", {})
        if contract_config.get("enable_thinking") is False:
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

        expected_chunks = int(
            case.metadata.get("expected_prefill_chunks", 0) or 0
        )
        chunking_matches = True
        if expected_chunks > 0:
            stderr = str(
                (trt_output.metadata or {}).get("cpp", {}).get("stderr", "")
            )
            marker = 'label="prefill_engine_plan:prefill"'
            chunking_matches = any(
                marker in line
                and f"launches={expected_chunks}" in line.split()
                for line in stderr.splitlines()
            )
            metrics["prefill_chunks"] = MetricResult(
                value=1.0 if chunking_matches else 0.0,
                threshold=1.0,
                operator="==",
                passed=chunking_matches,
                note=f"{marker} with launches={expected_chunks}",
            )

        expected_rows = int(
            case.metadata.get("expected_kv_cache_rows", 0) or 0
        )
        cache_capacity_matches = True
        if expected_rows > 0:
            stderr = str(
                (trt_output.metadata or {}).get("cpp", {}).get("stderr", "")
            )
            cache_marker = (
                f"KV cache rows={expected_rows} (bundle max={expected_rows}"
            )
            cache_capacity_matches = cache_marker in stderr
            metrics["native_kv_capacity"] = MetricResult(
                value=1.0 if cache_capacity_matches else 0.0,
                threshold=1.0,
                operator="==",
                passed=cache_capacity_matches,
                note=cache_marker,
            )

        text_matches = exact_match or ned <= ned_threshold
        runtime_matches = chunking_matches and cache_capacity_matches
        rule = (
            "(exact_match OR ned <= threshold)"
            " AND optional_native_kv_runtime_contract"
        )
        if text_matches and runtime_matches:
            return make_pass("full_generation", metrics, rule)
        if text_matches:
            return make_fail(
                "full_generation",
                metrics,
                rule,
                "Qwen text matched but native-KV runtime markers diverged",
            )
        return make_fail(
            "full_generation",
            metrics,
            rule,
            f"Qwen chat response diverged: NED={ned:.3f}",
        )

class QwenSamplingPlugin:
    reference_families = ["sampling_top_p"]
    user_contract = "sampling"

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
            metrics = {
                "stage_ok": MetricResult(
                    value=1.0,
                    threshold=1.0,
                    operator="==",
                    passed=True,
                    note=f"{stage} completed",
                )
            }
            return make_pass(stage, metrics, f"{stage} invariant check")

        cpp_rc = int((trt_output.data or {}).get("cpp_returncode", -1))
        command = [
            str(x)
            for x in (trt_output.metadata or {}).get("cpp", {}).get("command", [])
        ]
        if not command:
            command = [str(x) for x in (trt_output.data or {}).get("command", [])]

        prompt = str(case.inputs.get("prompt", ""))
        text = normalize_text(strip_prompt_echo(trt_output.text or "", prompt))
        has_text = bool(text)
        rc_ok = cpp_rc == 0

        required_flags = []
        if float(case.inputs.get("top_p", 1.0)) < 1.0 - 1e-6:
            required_flags.append("--top-p")
        if float(case.inputs.get("temperature", 1.0)) != 1.0:
            required_flags.append("--temperature")
        if int(case.inputs.get("top_k", 1)) != 1:
            required_flags.append("--top-k")
        if int(case.inputs.get("seed", -1)) >= 0:
            required_flags.append("--seed")

        missing_flags = [flag for flag in required_flags if flag not in command]
        flags_ok = not missing_flags

        metrics = {
            "cpp_returncode_ok": MetricResult(
                value=1.0 if rc_ok else 0.0,
                threshold=1.0,
                operator="==",
                passed=rc_ok,
                note=f"cpp_returncode={cpp_rc}",
            ),
            "has_output": MetricResult(
                value=1.0 if has_text else 0.0,
                threshold=1.0,
                operator="==",
                passed=has_text,
                note="TRT produced non-empty sampled text",
            ),
            "sampling_flags_forwarded": MetricResult(
                value=1.0 if flags_ok else 0.0,
                threshold=1.0,
                operator="==",
                passed=flags_ok,
                note=(
                    "missing: " + ", ".join(missing_flags)
                    if missing_flags
                    else "all requested flags present"
                ),
            ),
        }

        passed = rc_ok and has_text and flags_ok
        rule = "cpp_returncode_ok AND has_output AND sampling_flags_forwarded"
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            "Top-p sampling contract failed",
        )


class QwenNativeKvChunkedPrefillRegressionPlugin:
    reference_families = ["qwen_native_kv_chunked_prefill_regression"]
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
        chunk_plan_is_consistent = (
            expected_limit > 0
            and expected_chunks
            == (expected_prompt + expected_limit - 1) // expected_limit
        )
        chunked_prefill_observed = chunk_plan_is_consistent and any(
            prefill_marker in line and f"launches={expected_chunks}" in line
            for line in stderr.splitlines()
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
                f"max_chunk={expected_limit}; {prefill_marker} with "
                f"launches={expected_chunks}",
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
            "Qwen native-KV chunked-prefill regression contract failed: "
            + ", ".join(failed),
        )


plugin = [
    QwenPostTrainedChatPlugin(),
    QwenSamplingPlugin(),
    QwenNativeKvChunkedPrefillRegressionPlugin(),
]
