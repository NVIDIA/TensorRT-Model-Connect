"""Contract checks for ELF diffusion text generation."""

from __future__ import annotations

import json
from typing import Any

from tests.e2e_harness.contracts import (
    E2ECase,
    MetricResult,
    StageOutput,
    ThresholdProfile,
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

def _metric_value(output: StageOutput, *names: str) -> float | None:
    for name in names:
        value = output.data.get(name) if output.data else None
        if value is None:
            value = output.metadata.get(name) if output.metadata else None
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None

def _text_from_sample(sample: Any) -> str:
    if isinstance(sample, str):
        return sample
    if isinstance(sample, dict):
        for key in ("generated", "generated_text", "text"):
            value = sample.get(key)
            if isinstance(value, str):
                return value
    return ""

def _token_ids_from_sample(sample: Any) -> list[int]:
    if not isinstance(sample, dict):
        return []
    value = sample.get("token_ids")
    if not isinstance(value, list):
        return []
    try:
        return [int(token) for token in value]
    except (TypeError, ValueError):
        return []

def _samples_from_jsonl(payload: str) -> list[Any]:
    samples: list[Any] = []
    for line in payload.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            samples.append({"generated": line})
    return samples

def _generated_samples(output: StageOutput) -> list[Any]:
    data = output.data or {}
    for key in ("generated_samples", "samples", "generated"):
        value = data.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return _samples_from_jsonl(value)

    value = data.get("generated_jsonl")
    if isinstance(value, str):
        return _samples_from_jsonl(value)

    if output.text:
        return [{"generated": output.text}]
    return []

def _expected_samples(output: StageOutput) -> list[Any]:
    data = output.data or {}
    value = data.get("expected_generated_samples")
    if isinstance(value, list):
        return value
    value = data.get("expected_generated_jsonl")
    if isinstance(value, str):
        return _samples_from_jsonl(value)
    return []

def _has_condition_api(case: E2ECase, output: StageOutput) -> bool:
    resolved = (output.data or {}).get("resolved_inputs", {})
    inputs = {**(case.inputs or {})}
    if isinstance(resolved, dict):
        inputs.update(resolved)
    if inputs.get("prompt") or inputs.get("source_text") or inputs.get("condition_text"):
        return True
    replay_samples = inputs.get("replay_samples")
    if isinstance(replay_samples, list) and replay_samples:
        return all(
            isinstance(sample, dict)
            and bool(
                (sample.get("condition_latents_raw") or sample.get("condition_latents_path"))
                and (sample.get("condition_mask_raw") or sample.get("condition_mask_path"))
            )
            for sample in replay_samples
        )
    condition_latents = inputs.get("condition_latents_raw") or inputs.get("condition_latents_path")
    condition_mask = inputs.get("condition_mask_raw") or inputs.get("condition_mask_path")
    return bool(condition_latents and condition_mask)

def _has_condition(case: E2ECase, output: StageOutput) -> bool:
    if case.reference_family != "elf_conditional_text":
        return True

    return _has_condition_api(case, output)

def _optional_metric(
    metrics: dict[str, MetricResult],
    *,
    name: str,
    value: float | None,
    threshold: float | None,
    operator: str,
) -> bool:
    if value is None or threshold is None:
        return True
    if operator == "<=":
        passed = value <= threshold
    elif operator == ">=":
        passed = value >= threshold
    else:
        raise ValueError(f"Unsupported operator: {operator}")
    metrics[name] = MetricResult(
        value=value,
        threshold=threshold,
        operator=operator,
        passed=passed,
    )
    return passed

class ElfDiffusionTextPlugin:
    reference_families = ["elf_unconditional_text", "elf_conditional_text"]
    user_contract = "diffusion_text_generation"

    def configure_reference(self, case: E2ECase) -> dict:
        return {
            "implementation": "github_elf",
            "generation_mode": case.inputs.get("generation_mode", "unconditional"),
            "output_schema": "jsonl_id_generated_token_ids",
        }

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
    ):
        del ref_output
        stage = trt_output.stage_name
        if stage not in ("full_generation", "decoded_text", "end_to_end"):
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

        samples = _generated_samples(trt_output)
        texts = [normalize_text(_text_from_sample(sample)) for sample in samples]
        expected_samples = _expected_samples(trt_output)
        expected_texts = [
            normalize_text(_text_from_sample(sample)) for sample in expected_samples
        ]
        generated_token_ids = [_token_ids_from_sample(sample) for sample in samples]
        expected_token_ids = [
            _token_ids_from_sample(sample) for sample in expected_samples
        ]
        non_empty = sum(1 for text in texts if text)
        min_samples = int(threshold.metrics.get("contract_min_samples", 1))
        has_samples = len(samples) >= min_samples
        non_empty_ok = non_empty == len(samples) and non_empty >= min_samples
        condition_ok = _has_condition(case, trt_output)
        expected_sample_count = threshold.metrics.get("contract_expected_samples")
        expected_sample_count_ok = True

        metrics = {
            "num_generated_samples": MetricResult(
                value=float(len(samples)),
                threshold=float(min_samples),
                operator=">=",
                passed=has_samples,
            ),
            "non_empty_generated_text": MetricResult(
                value=float(non_empty),
                threshold=float(min_samples),
                operator=">=",
                passed=non_empty_ok,
            ),
            "condition_available": MetricResult(
                value=1.0 if condition_ok else 0.0,
                threshold=1.0,
                operator="==",
                passed=condition_ok,
            ),
        }
        if expected_sample_count is not None:
            expected_count = int(expected_sample_count)
            expected_sample_count_ok = len(samples) == expected_count
            metrics["expected_generated_sample_count"] = MetricResult(
                value=float(len(samples)),
                threshold=float(expected_count),
                operator="==",
                passed=expected_sample_count_ok,
                note="exact sample count required by ELF evaluation contract",
            )

        max_ppl = threshold.metrics.get("contract_max_gen_ppl")
        min_entropy = threshold.metrics.get("contract_min_unigram_entropy")
        min_bleu = threshold.metrics.get("contract_min_bleu")
        min_rouge_l = threshold.metrics.get("contract_min_rouge_l")

        metric_ok = True
        metric_ok &= _optional_metric(
            metrics,
            name="gen_ppl",
            value=_metric_value(trt_output, "gen_ppl", "generation_ppl"),
            threshold=max_ppl,
            operator="<=",
        )
        metric_ok &= _optional_metric(
            metrics,
            name="unigram_entropy",
            value=_metric_value(trt_output, "unigram_entropy", "mean_entropy"),
            threshold=min_entropy,
            operator=">=",
        )
        metric_ok &= _optional_metric(
            metrics,
            name="bleu",
            value=_metric_value(trt_output, "bleu"),
            threshold=min_bleu,
            operator=">=",
        )
        metric_ok &= _optional_metric(
            metrics,
            name="rouge_l",
            value=_metric_value(trt_output, "rouge_l", "rougeL"),
            threshold=min_rouge_l,
            operator=">=",
        )
        if expected_texts:
            compare_count = min(len(texts), len(expected_texts))
            matches = sum(
                1
                for idx in range(compare_count)
                if texts[idx] and texts[idx] == expected_texts[idx]
            )
            expected_count = len(expected_texts)
            match_rate = matches / expected_count if expected_count else 0.0
            threshold_value = threshold.metrics.get(
                "contract_min_upstream_text_match_rate", 1.0
            )
            passed = (
                len(texts) >= expected_count
                and match_rate >= threshold_value
            )
            metrics["upstream_text_match_rate"] = MetricResult(
                value=match_rate,
                threshold=threshold_value,
                operator=">=",
                passed=passed,
                note="exact normalized text match against upstream replay artifact",
            )
            metric_ok &= passed
        expected_token_samples = [
            token_ids for token_ids in expected_token_ids if token_ids
        ]
        if expected_token_samples:
            compare_count = min(len(generated_token_ids), len(expected_token_ids))
            matches = sum(
                1
                for idx in range(compare_count)
                if expected_token_ids[idx] and generated_token_ids[idx] == expected_token_ids[idx]
            )
            expected_count = len(expected_token_samples)
            match_rate = matches / expected_count if expected_count else 0.0
            threshold_value = threshold.metrics.get(
                "contract_min_upstream_token_match_rate", 1.0
            )
            passed = (
                len(generated_token_ids) >= len(expected_token_ids)
                and match_rate >= threshold_value
            )
            metrics["upstream_token_id_match_rate"] = MetricResult(
                value=match_rate,
                threshold=threshold_value,
                operator=">=",
                passed=passed,
                note="exact token-id sequence match against upstream replay artifact",
            )
            metric_ok &= passed

        rule = (
            "num_generated_samples >= contract_min_samples AND "
            "expected_generated_sample_count_if_configured AND "
            "non_empty_generated_text AND condition_available AND optional_metrics"
        )
        if has_samples and expected_sample_count_ok and non_empty_ok and condition_ok and metric_ok:
            return make_pass(stage, metrics, rule)
        return make_fail(stage, metrics, rule, "ELF diffusion text contract failed")

plugin = ElfDiffusionTextPlugin()
