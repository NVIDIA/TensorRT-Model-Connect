"""Contract test plugin protocol and shared verification helpers.

Each contract test plugin handles one or more reference families and defines:
1. Which reference families it covers.
2. How to configure the HF reference invocation (chat template, processor, etc.).
3. How to verify the user-facing contract (exact text, ranking, mask overlap, etc.).

Plugins are auto-discovered from this directory by __init__.py, following the
same pattern as builder family plugins in tensorrt_model_connect/tensorrt_model_connect/families/.

Adding a new contract = adding one .py file with a module-level ``plugin``
attribute.  Zero edits to the orchestrator or registry.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Protocol, runtime_checkable

from ..contracts import (
    CompareResult,
    E2ECase,
    MetricResult,
    PluginRuntimeContext,
    StageOutput,
    StageStatus,
    ThresholdProfile,
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ContractTestPlugin(Protocol):
    """Plugin that defines the user-contract test for a group of reference families.

    Similar to builder FamilyPlugin: one file, one class, handles a group
    of related reference families end-to-end.
    """

    @property
    def reference_families(self) -> List[str]:
        """Reference family values this plugin handles (ReferenceFamily enum values)."""
        ...

    @property
    def user_contract(self) -> str:
        """The UserContract enum value this plugin verifies."""
        ...

    def configure_reference(self, case: E2ECase) -> Dict[str, Any]:
        """Return configuration for the reference backend invocation.

        The returned dict is passed to the reference backend's run_stage()
        via case.metadata["contract_config"].  Keys are reference-specific:

            {"use_chat_template": True, "enable_thinking": False}
            {"auto_class": "AutoModelForSeq2SeqLM", "task_prefix": "translate:"}
            {"use_processor": True, "closed_qa_question": "What color is the car?"}

        Returns empty dict if no special configuration is needed.
        """
        ...

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
        *,
        runtime_context: PluginRuntimeContext | None = None,
    ) -> CompareResult:
        """Verify the user-facing contract.

        Called in the acceptance CI lane.  Should check user-visible behavior
        (exact text match, correct ranking, valid transcript, etc.) rather
        than numeric tensor parity.

        Migrated plugins may accept ``runtime_context`` for resolved runtime
        paths such as engine directories, binaries, and Python interpreters.
        The orchestrator omits this keyword for legacy plugins that have not
        declared it yet.

        Returns a CompareResult with contract-level metrics.
        """
        ...


# ---------------------------------------------------------------------------
# Shared text helpers (used by multiple plugins)
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Lightweight text normalization: collapse whitespace, lowercase, strip."""
    if not text:
        return ""
    return " ".join(text.split()).strip().lower()


def strip_prompt_echo(text: str, prompt: str) -> str:
    """Remove echoed prompt from generated text.

    Some C++ runs echo the prompt before the generated continuation.
    Handles warning preambles up to 2048 chars before the prompt.
    """
    if not text or not prompt:
        return text
    idx = text.find(prompt)
    if 0 <= idx <= 2048:
        return text[idx + len(prompt):].lstrip()
    # Fallback: try normalized match
    norm_text = normalize_text(text)
    norm_prompt = normalize_text(prompt)
    if norm_prompt and norm_text.startswith(norm_prompt):
        # Map back to original text position
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
    """Remove leading chat role prefixes and truncate after first turn."""
    if not text:
        return ""
    out = text.lstrip()
    # Strip leading role prefix
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
    # Truncate after first turn
    lowered = out.lower()
    cut = len(out)
    for marker in _CHAT_TURN_MARKERS:
        idx = lowered.find(marker)
        if idx > 0:
            cut = min(cut, idx)
    if cut < len(out):
        out = out[:cut]
    # Strip trailing markdown stubs
    out = re.sub(r"(?:\s*#{2,}\s*)+$", "", out).strip()
    return out


def extract_answer(output: StageOutput, prompt: str = "") -> str:
    """Extract clean answer text from a StageOutput.

    Strips prompt echo, chat markup, and normalizes.
    """
    raw = output.text or ""
    if prompt:
        raw = strip_prompt_echo(raw, prompt)
    raw = strip_chat_markup(raw)
    return raw.strip()


def levenshtein_ned(a: str, b: str) -> float:
    """Normalized edit distance (0.0 = identical, 1.0 = completely different)."""
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    # DP computation
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


# ---------------------------------------------------------------------------
# Shared result builders
# ---------------------------------------------------------------------------


def make_pass(stage_name: str, metrics: Dict[str, MetricResult],
              rule: str = "") -> CompareResult:
    """Build a passing CompareResult."""
    return CompareResult(
        stage_name=stage_name,
        status=StageStatus.PASSED.value,
        metrics=metrics,
        composite_rule=rule,
        message="Contract verified",
    )


def make_fail(stage_name: str, metrics: Dict[str, MetricResult],
              rule: str = "", message: str = "") -> CompareResult:
    """Build a failing CompareResult."""
    return CompareResult(
        stage_name=stage_name,
        status=StageStatus.FAILED.value,
        metrics=metrics,
        composite_rule=rule,
        message=message or "Contract verification failed",
    )


def make_skip(stage_name: str, metrics: Dict[str, MetricResult],
              rule: str = "", message: str = "") -> CompareResult:
    """Build a skipped CompareResult for an unvalidated required contract."""
    return CompareResult(
        stage_name=stage_name,
        status=StageStatus.SKIPPED.value,
        metrics=metrics,
        composite_rule=rule,
        message=message or "Contract validation skipped",
    )


def make_error(stage_name: str, error: str) -> CompareResult:
    """Build an error CompareResult."""
    return CompareResult(
        stage_name=stage_name,
        status=StageStatus.ERROR.value,
        message=f"Contract verification error: {error}",
    )
