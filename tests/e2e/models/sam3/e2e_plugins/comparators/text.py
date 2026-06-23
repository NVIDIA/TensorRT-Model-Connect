"""Text generation comparator — multi-metric comparison with composite gating.

Computes logit-level and text-level metrics between TRT and HF reference
outputs and applies composite gating from the ThresholdProfile. No single
metric gates alone; the pass/fail decision uses a composite rule.

Metrics computed:
    1. logit_cosine_p5 — 5th percentile cosine similarity across steps
    2. logit_rel_l2_p95 — 95th percentile relative L2 norm
    3. stable_top1_match_rate — exact top-1 match where HF margin >= stable_margin
    4. unstable_topk_hit_rate — TRT top-1 in HF top-k where margin < stable_margin
    5. token_agreement_rate — fraction of steps with identical argmax
    6. normalized_text_edit_distance — Levenshtein-normalized on decoded text
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from ..contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)
from ._helpers import cosine_similarity, normalized_edit_distance

logger = logging.getLogger(__name__)

# Default top-k for unstable token checking
_DEFAULT_TOP_K = 5
# Default stable margin threshold
_DEFAULT_STABLE_MARGIN = 0.1

_COMPOSITE_RULE = (
    "(cosine_p5 >= T OR rel_l2_p95 <= T) "
    "AND (agreement >= T OR (stable_top1 >= T AND unstable_topk >= T)) "
    "AND ned <= T"
)

# If the model echoes the prompt, allow a modest amount of non-text preamble
# (warnings/logs) before the prompt appears in stdout.
_PROMPT_SEARCH_MAX_PREFIX_CHARS = 2048
_MIN_PREFIX_FALLBACK_CHARS = 24

# Common multi-turn/chat markers that can appear in decoded text and cause
# cosmetic NED mismatches even when token/logit agreement is strong.
_CHAT_ROLE_PREFIXES = (
    "### response:",
    "### assistant:",
    "assistant:",
    "<|assistant|>",
)
_CHAT_TURN_MARKERS = (
    "### response:",
    "### instruction:",
    "### assistant:",
    "### user:",
    "<|assistant|>",
    "<|user|>",
    "<|im_start|>",
    "<|im_end|>",
)


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 norm: ||a - b|| / max(||b||, eps)."""
    diff_norm = np.linalg.norm(a - b)
    ref_norm = np.linalg.norm(b)
    return float(diff_norm / max(ref_norm, 1e-12))


def _strip_prompt_echo(text: str, prompt: str) -> str:
    """Drop echoed prompt from generated text, tolerating warning preambles.

    Some C++ runs can include tokenizer warnings/log lines before the actual
    generated text. If the prompt appears near the beginning of the output,
    treat everything before/including it as preamble and compare only the
    continuation text.
    """
    if not text or not prompt:
        return text

    idx = text.find(prompt)
    if 0 <= idx <= _PROMPT_SEARCH_MAX_PREFIX_CHARS:
        return text[idx + len(prompt):].lstrip()

    return text


def _strip_prompt_echo_normalized(text: str, prompt: str) -> str:
    """Prompt-echo stripping on normalized text for tokenization-format drift.

    This pass runs after normalization and catches cases where decoded prompt
    formatting differs slightly (e.g., whitespace around punctuation), so raw
    substring matching misses obvious prompt echoes.
    """
    if not text or not prompt:
        return text

    norm_prompt = _normalize_for_ned(prompt)
    if not norm_prompt:
        return text

    if text.startswith(norm_prompt):
        return text[len(norm_prompt):].lstrip()

    # Fallback: compare after removing whitespace to handle punctuation-spacing
    # drift from tokenizer decode (e.g., "dog. once" vs "dog.once").
    compact_text = "".join(ch for ch in text if not ch.isspace())
    compact_prompt = "".join(ch for ch in norm_prompt if not ch.isspace())
    if compact_prompt and compact_text.startswith(compact_prompt):
        remaining = len(compact_prompt)
        i = 0
        while i < len(text) and remaining > 0:
            if not text[i].isspace():
                remaining -= 1
            i += 1
        return text[i:].lstrip()

    # Keep search window small in normalized space to avoid stripping
    # naturally generated prompt repeats that happen later in output.
    search_limit = min(_PROMPT_SEARCH_MAX_PREFIX_CHARS, max(256, len(norm_prompt) * 3))
    idx = text.find(norm_prompt)
    if 0 <= idx <= search_limit:
        return text[idx + len(norm_prompt):].lstrip()
    return text


def _normalize_for_ned(text: str) -> str:
    """Lightweight text normalization before edit-distance comparison."""
    if not text:
        return ""
    # Collapse whitespace and case-fold to reduce cosmetic diffs.
    return " ".join(text.split()).strip().lower()


def _strip_leading_role_prefix(text: str) -> str:
    """Remove leading chat role prefixes (if present)."""
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
            return out


def _truncate_after_first_turn(text: str) -> str:
    """Keep only first assistant turn content and trim trailing markdown stubs."""
    if not text:
        return ""

    lowered = text.lower()
    cut = len(text)
    for marker in _CHAT_TURN_MARKERS:
        idx = lowered.find(marker)
        if idx > 0:
            cut = min(cut, idx)

    out = text[:cut] if cut < len(text) else text
    # Some models emit dangling markdown headers (e.g. "##") at the end.
    out = re.sub(r"(?:\s*#{2,}\s*)+$", "", out).strip()
    return out


def _load_logits(stage_output: StageOutput) -> np.ndarray | None:
    """Load logits from StageOutput. Returns 2-D array [steps, vocab] or None."""
    # Try logits field first (path or array)
    logits = stage_output.logits
    if logits is None:
        logits = stage_output.data.get("logits_path")

    if logits is None:
        return None

    if isinstance(logits, np.ndarray):
        return logits

    if isinstance(logits, str) and Path(logits).is_file():
        return np.load(logits)

    return None


def _check_numerical_health(
    arr: np.ndarray, label: str
) -> list[str]:
    """Check for NaN, Inf, and suspicious range. Returns list of warnings."""
    warnings = []
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count > 0:
        warnings.append(f"{label}: {nan_count} NaN values")
    if inf_count > 0:
        warnings.append(f"{label}: {inf_count} Inf values")
    if arr.size > 0:
        abs_max = float(np.nanmax(np.abs(arr[np.isfinite(arr)]))) if np.any(np.isfinite(arr)) else 0.0
        if abs_max > 1e6:
            warnings.append(f"{label}: large absolute values (max={abs_max:.1e})")
    return warnings


class TextComparator:
    """Multi-metric text generation comparator with composite gating."""

    @property
    def task_strategy(self) -> str:
        return "text_generation_causal"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        metrics: dict[str, MetricResult] = {}

        # full_generation runs provide C++ return code from the CLI path.
        # If C++ generation failed, surface that explicitly instead of
        # allowing debug-runner logits to hide the failure.
        cpp_rc = (trt.data or {}).get("cpp_returncode")
        if cpp_rc not in (None, 0):
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics=metrics,
                message=f"TRT C++ run failed (cpp_returncode={cpp_rc})",
            )

        # Load logits
        trt_logits = _load_logits(trt)
        ref_logits = _load_logits(ref)

        # Shape/schema check — fall back to text-only for seq2seq models
        # where the debug runner doesn't produce logits
        if trt_logits is None or ref_logits is None:
            return self._compare_text_only(trt, ref, threshold, stage, metrics)

        if trt_logits.ndim != 2 or ref_logits.ndim != 2:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics=metrics,
                message=f"Logits must be 2-D [steps, vocab]: TRT={trt_logits.shape}, HF={ref_logits.shape}",
            )

        # Truncate to common step count
        n_steps = min(trt_logits.shape[0], ref_logits.shape[0])
        if n_steps == 0:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics=metrics,
                message="No steps to compare",
            )

        trt_l = trt_logits[:n_steps]
        ref_l = ref_logits[:n_steps]

        # Ensure same vocab dimension
        notes: list[str] = []
        if trt_l.shape[1] != ref_l.shape[1]:
            min_vocab = min(trt_l.shape[1], ref_l.shape[1])
            notes.append(
                f"Vocab size mismatch: TRT={trt_l.shape[1]}, HF={ref_l.shape[1]}; "
                f"truncating to {min_vocab}"
            )
            trt_l = trt_l[:, :min_vocab]
            ref_l = ref_l[:, :min_vocab]

        # Numerical health
        health_warnings = []
        health_warnings.extend(_check_numerical_health(trt_l, "TRT logits"))
        health_warnings.extend(_check_numerical_health(ref_l, "HF logits"))
        notes.extend(health_warnings)

        # Replace NaN/Inf with 0 for metric computation
        trt_clean = np.nan_to_num(trt_l, nan=0.0, posinf=0.0, neginf=0.0)
        ref_clean = np.nan_to_num(ref_l, nan=0.0, posinf=0.0, neginf=0.0)

        thresh = threshold.metrics

        # --- Metric 1: logit_cosine_p5 ---
        cosines = np.array([
            cosine_similarity(trt_clean[i], ref_clean[i])
            for i in range(n_steps)
        ])
        logit_cosine_p5 = float(np.percentile(cosines, 5))
        cosine_thresh = thresh.get("logit_cosine_p5", 0.99)
        metrics["logit_cosine_p5"] = MetricResult(
            value=logit_cosine_p5, threshold=cosine_thresh,
            operator=">=", passed=logit_cosine_p5 >= cosine_thresh,
        )

        # --- Metric 2: logit_rel_l2_p95 ---
        rel_l2s = np.array([
            _relative_l2(trt_clean[i], ref_clean[i])
            for i in range(n_steps)
        ])
        logit_rel_l2_p95 = float(np.percentile(rel_l2s, 95))
        rel_l2_thresh = thresh.get("logit_rel_l2_p95", 0.05)
        metrics["logit_rel_l2_p95"] = MetricResult(
            value=logit_rel_l2_p95, threshold=rel_l2_thresh,
            operator="<=", passed=logit_rel_l2_p95 <= rel_l2_thresh,
        )

        # --- Per-step argmax and margin analysis ---
        trt_argmax = trt_clean.argmax(axis=1)
        ref_argmax = ref_clean.argmax(axis=1)

        ref_sorted = np.sort(ref_clean, axis=1)
        hf_margin = ref_sorted[:, -1] - ref_sorted[:, -2]

        stable_margin = thresh.get("stable_margin", _DEFAULT_STABLE_MARGIN)
        top_k = int(thresh.get("top_k", _DEFAULT_TOP_K))

        stable_mask = hf_margin >= stable_margin
        unstable_mask = ~stable_mask
        n_stable = int(stable_mask.sum())
        n_unstable = int(unstable_mask.sum())

        # --- Metric 3: stable_top1_match_rate ---
        if n_stable > 0:
            stable_matches = int((trt_argmax[stable_mask] == ref_argmax[stable_mask]).sum())
            stable_top1_match_rate = stable_matches / n_stable
        else:
            stable_top1_match_rate = 1.0
        stable_thresh = thresh.get("stable_top1_match_rate", 0.9)
        metrics["stable_top1_match_rate"] = MetricResult(
            value=stable_top1_match_rate, threshold=stable_thresh,
            operator=">=", passed=stable_top1_match_rate >= stable_thresh,
            note=f"{n_stable} stable steps",
        )

        # --- Metric 4: unstable_topk_hit_rate ---
        if n_unstable > 0:
            ref_topk = np.argsort(ref_clean, axis=1)[:, -top_k:]
            hits = 0
            unstable_indices = np.where(unstable_mask)[0]
            for idx in unstable_indices:
                if trt_argmax[idx] in ref_topk[idx]:
                    hits += 1
            unstable_topk_hit_rate = hits / n_unstable
        else:
            unstable_topk_hit_rate = 1.0
        topk_thresh = thresh.get("unstable_topk_hit_rate", 0.8)
        metrics["unstable_topk_hit_rate"] = MetricResult(
            value=unstable_topk_hit_rate, threshold=topk_thresh,
            operator=">=", passed=unstable_topk_hit_rate >= topk_thresh,
            note=f"{n_unstable} unstable steps",
        )

        # --- Metric 5: token_agreement_rate ---
        token_agreement_rate = float((trt_argmax == ref_argmax).mean())
        ta_thresh = thresh.get("token_agreement_rate", 0.8)
        metrics["token_agreement_rate"] = MetricResult(
            value=token_agreement_rate, threshold=ta_thresh,
            operator=">=", passed=token_agreement_rate >= ta_thresh,
        )

        # --- Metric 6: normalized_text_edit_distance ---
        trt_text = (trt.text or "").strip()
        ref_text = (ref.text or "").strip()

        prompt = (trt.data or {}).get("prompt", "")
        # Prompt echo handling is TRT-side only. HF reference text is decoded
        # from generated tokens and should not include prompt prefill; stripping
        # prompt from reference can incorrectly remove legitimate generated text
        # if the model naturally repeats the prompt phrase later.
        trt_text_for_ned = _normalize_for_ned(
            _truncate_after_first_turn(
                _strip_leading_role_prefix(_strip_prompt_echo(trt_text, prompt))
            )
        )
        ref_text_for_ned = _normalize_for_ned(
            _truncate_after_first_turn(
                _strip_leading_role_prefix(ref_text)
            )
        )
        trt_text_for_ned = _strip_prompt_echo_normalized(trt_text_for_ned, prompt)
        # Seq2seq models output text that may start with the prompt (e.g.
        # BART reconstructing its input).  Strip the prompt prefix from ref
        # only when it appears at the very start of the normalized text.
        # This avoids accidentally removing prompt substrings that appear
        # later in naturally generated text from causal models.
        norm_prompt = _normalize_for_ned(prompt)
        if norm_prompt and ref_text_for_ned.startswith(norm_prompt):
            ref_text_for_ned = ref_text_for_ned[len(norm_prompt):].lstrip()

        if trt_text_for_ned or ref_text_for_ned:
            ned = normalized_edit_distance(trt_text_for_ned, ref_text_for_ned)
            # Some TRT CLI paths stop decoding early on EOS while the debug/HF
            # text path keeps fixed-length continuation tokens. If token/logit
            # metrics already agree, compare on the common prefix to avoid
            # false NED hard-fails caused purely by suffix length mismatch.
            ta_thresh = thresh.get("token_agreement_rate", 0.8)
            if token_agreement_rate >= ta_thresh:
                if len(trt_text_for_ned) <= len(ref_text_for_ned):
                    short, long = trt_text_for_ned, ref_text_for_ned
                else:
                    short, long = ref_text_for_ned, trt_text_for_ned
                if len(short) >= _MIN_PREFIX_FALLBACK_CHARS and long.startswith(short):
                    prefix_ned = normalized_edit_distance(short, long[:len(short)])
                    if prefix_ned < ned:
                        notes.append(
                            "NED prefix fallback applied (matching continuation prefix; "
                            "suffix length mismatch likely due EOS stopping behavior)"
                        )
                        ned = prefix_ned
        else:
            ned = 0.0
        ned_thresh = thresh.get("normalized_text_edit_distance", 0.2)
        metrics["normalized_text_edit_distance"] = MetricResult(
            value=ned, threshold=ned_thresh,
            operator="<=", passed=ned <= ned_thresh,
        )

        # --- Composite gating ---
        logit_quality_ok = (
            metrics["logit_cosine_p5"].passed
            or metrics["logit_rel_l2_p95"].passed
        )

        token_level_ok = (
            metrics["token_agreement_rate"].passed
            or (
                metrics["stable_top1_match_rate"].passed
                and metrics["unstable_topk_hit_rate"].passed
            )
        )

        text_ok = metrics["normalized_text_edit_distance"].passed

        passed = logit_quality_ok and token_level_ok and text_ok

        message = (
            f"{'PASS' if passed else 'FAIL'}: "
            f"cosine_p5={logit_cosine_p5:.4f}, "
            f"agreement={token_agreement_rate:.4f}, "
            f"ned={ned:.4f}"
        )

        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule=_COMPOSITE_RULE,
            message=message,
        )


    def _compare_text_only(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
        metrics: dict[str, MetricResult],
    ) -> CompareResult:
        """Text-only comparison when logits are unavailable (seq2seq models)."""
        thresh = threshold.metrics

        prompt = (trt.data or {}).get("prompt", "")
        trt_text = _normalize_for_ned(
            _strip_leading_role_prefix(_strip_prompt_echo((trt.text or "").strip(), prompt))
        )
        ref_text = _normalize_for_ned(
            _strip_leading_role_prefix(_strip_prompt_echo((ref.text or "").strip(), prompt))
        )

        if trt_text and ref_text:
            ned = normalized_edit_distance(trt_text, ref_text)
        elif not trt_text and not ref_text:
            ned = 0.0
        else:
            ned = 1.0

        ned_thresh = thresh.get("normalized_text_edit_distance", 0.2)
        metrics["normalized_text_edit_distance"] = MetricResult(
            value=ned, threshold=ned_thresh,
            operator="<=", passed=ned <= ned_thresh,
        )

        passed = ned <= ned_thresh
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="text-only (logits unavailable for seq2seq): ned <= threshold",
            message=f"{'PASS' if passed else 'FAIL'}: text-only ned={ned:.4f}",
        )


plugin = TextComparator()
