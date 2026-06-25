"""roberta-owned encoder/embedding contract plugins."""
from __future__ import annotations

import numpy as np

from tests.e2e_harness.contracts import MetricResult
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

_MIN_CONTRACT_COSINE = 0.80

class RobertaEncoderFeaturesPlugin:
    reference_families = ['encoder_base_features']
    user_contract = "representation_parity"

    def configure_reference(self, case):
        return contract_config(case)

    def verify(self, trt_output, ref_output, case, threshold):
        del case
        trt_emb = trt_output.data.get("embedding") or trt_output.data.get("cls_embedding")
        ref_emb = ref_output.data.get("embedding") or ref_output.data.get("cls_embedding")

        if trt_emb is None or ref_emb is None:
            return make_error("full_inference", "Missing embedding in output data")

        trt_arr = np.asarray(trt_emb, dtype=np.float32).flatten()
        ref_arr = np.asarray(ref_emb, dtype=np.float32).flatten()

        norm_t = np.linalg.norm(trt_arr)
        norm_r = np.linalg.norm(ref_arr)
        if norm_t < 1e-12 or norm_r < 1e-12:
            cosine = 0.0
        else:
            cosine = float(np.dot(trt_arr, ref_arr) / (norm_t * norm_r))

        configured_cosine_threshold = threshold.metrics.get(
            "contract_cosine_threshold",
            threshold.metrics.get("cls_embedding_cosine", 0.80),
        )
        cosine_threshold = max(configured_cosine_threshold, _MIN_CONTRACT_COSINE)
        note = ""
        if cosine_threshold != configured_cosine_threshold:
            note = (
                f"configured threshold {configured_cosine_threshold} raised to "
                f"{_MIN_CONTRACT_COSINE} floor"
            )

        metrics = {
            "cosine_similarity": MetricResult(
                value=cosine,
                threshold=cosine_threshold,
                operator=">=",
                passed=cosine >= cosine_threshold,
                note=note,
            ),
        }

        if cosine >= cosine_threshold:
            return make_pass("full_inference", metrics, "cosine >= threshold")
        return make_fail(
            "full_inference",
            metrics,
            "cosine >= threshold",
            f"Representation diverged: cosine={cosine:.4f}",
        )

plugin = RobertaEncoderFeaturesPlugin()
