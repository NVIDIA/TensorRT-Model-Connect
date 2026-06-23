"""Nemotron Labs Diffusion-owned model-card parity contract."""

from __future__ import annotations

from tests.e2e_harness.contracts import MetricResult
from tests.e2e_harness.plugins.base import (
    levenshtein_ned,
    make_fail,
    make_pass,
    normalize_text,
)


class NemotronLabsDiffusionPlugin:
    reference_families = ["nemotron_labs_diffusion_model_card"]
    user_contract = "model_card_generation_parity"

    def configure_reference(self, case):
        existing = case.metadata.get("contract_config", {})
        return {
            "use_chat_template": existing.get("use_chat_template", True),
            "enable_thinking": existing.get("enable_thinking", False),
            "token_parity_eos_token_ids": existing.get("token_parity_eos_token_ids", [11]),
            "token_parity_ignore_terminal_token_ids": existing.get(
                "token_parity_ignore_terminal_token_ids", [1010]
            ),
            "forbidden_token_ids": existing.get("forbidden_token_ids", [0]),
        }

    def verify(self, trt_output, ref_output, case, threshold):
        trt_tokens = _token_ids(trt_output)
        ref_tokens = _token_ids(ref_output)
        if not trt_tokens:
            return make_fail("full_generation", {}, message="TRT produced no generated token IDs")
        if not ref_tokens:
            return make_fail("full_generation", {}, message="HF reference produced no token IDs")

        eos_ids = _int_set(
            trt_output.data.get("token_parity_eos_token_ids")
            or case.metadata.get("contract_config", {}).get("token_parity_eos_token_ids")
            or [ref_output.data.get("eos_token_id")]
        )
        ignored_terminal_ids = _int_set(
            trt_output.data.get("token_parity_ignore_terminal_token_ids")
            or case.metadata.get("contract_config", {}).get(
                "token_parity_ignore_terminal_token_ids", []
            )
        )
        forbidden_ids = _int_set(
            trt_output.data.get("forbidden_token_ids")
            or case.metadata.get("contract_config", {}).get("forbidden_token_ids", [])
        )

        raw_exact = trt_tokens == ref_tokens
        canonical_trt = _canonical_terminal_tokens(trt_tokens, eos_ids, ignored_terminal_ids)
        canonical_ref = _canonical_terminal_tokens(ref_tokens, eos_ids, ignored_terminal_ids)
        canonical_exact = canonical_trt == canonical_ref
        raw_agreement = _position_agreement(trt_tokens, ref_tokens)
        canonical_agreement = _position_agreement(canonical_trt, canonical_ref)
        forbidden_hits = sorted(set(trt_tokens) & forbidden_ids)

        trt_text = normalize_text(trt_output.text or "")
        ref_text = normalize_text(ref_output.text or "")
        text_ned = levenshtein_ned(trt_text, ref_text)

        text_threshold = threshold.metrics.get("contract_ned_threshold", 0.05)
        canonical_threshold = threshold.metrics.get("canonical_token_agreement_rate", 1.0)
        metrics = {
            "generated_token_raw_exact": MetricResult(
                value=1.0 if raw_exact else 0.0,
                threshold=None,
                operator="",
                passed=True,
                note="informational before terminal-whitespace normalization",
            ),
            "generated_token_raw_agreement_rate": MetricResult(
                value=raw_agreement,
                threshold=None,
                operator="",
                passed=True,
                note="informational before terminal-whitespace normalization",
            ),
            "generated_token_canonical_exact": MetricResult(
                value=1.0 if canonical_exact else 0.0,
                threshold=1.0,
                operator="==",
                passed=canonical_exact,
                note=(
                    "terminal ignored token ids="
                    f"{sorted(ignored_terminal_ids)} before eos ids={sorted(eos_ids)}"
                ),
            ),
            "generated_token_canonical_agreement_rate": MetricResult(
                value=canonical_agreement,
                threshold=canonical_threshold,
                operator=">=",
                passed=canonical_agreement >= canonical_threshold,
            ),
            "forbidden_token_count": MetricResult(
                value=float(len(forbidden_hits)),
                threshold=0.0,
                operator="==",
                passed=not forbidden_hits,
                note=f"forbidden ids present: {forbidden_hits}" if forbidden_hits else "",
            ),
            "normalized_text_edit_distance": MetricResult(
                value=text_ned,
                threshold=text_threshold,
                operator="<=",
                passed=text_ned <= text_threshold,
            ),
        }

        passed = (
            canonical_exact
            and canonical_agreement >= canonical_threshold
            and not forbidden_hits
            and text_ned <= text_threshold
        )
        rule = (
            "canonical token IDs exact after terminal-whitespace normalization "
            "AND no forbidden token IDs AND text NED <= threshold"
        )
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            "Nemotron Labs Diffusion model-card token parity failed",
        )


def _token_ids(output) -> list[int]:
    value = (output.data or {}).get("token_ids", [])
    if not isinstance(value, list):
        return []
    return [int(token) for token in value]


def _int_set(values) -> set[int]:
    if values is None:
        return set()
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    return {int(value) for value in values if value is not None}


def _canonical_terminal_tokens(
    token_ids: list[int],
    eos_ids: set[int],
    ignored_terminal_ids: set[int],
) -> list[int]:
    out = list(token_ids)
    if not out or not eos_ids or out[-1] not in eos_ids:
        return out
    eos = out.pop()
    while out and out[-1] in ignored_terminal_ids:
        out.pop()
    out.append(eos)
    return out


def _position_agreement(lhs: list[int], rhs: list[int]) -> float:
    denom = max(len(lhs), len(rhs))
    if denom == 0:
        return 1.0
    common = min(len(lhs), len(rhs))
    matches = sum(1 for idx in range(common) if lhs[idx] == rhs[idx])
    return matches / denom


plugin = NemotronLabsDiffusionPlugin()
