"""olmo-owned E2E contract plugins."""
from __future__ import annotations

from tests.e2e_harness.contracts import (
    CompareResult,
    MetricResult,
    StageStatus,
)
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

class OlmoCausalContinuationPlugin:
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

plugin = OlmoCausalContinuationPlugin()
