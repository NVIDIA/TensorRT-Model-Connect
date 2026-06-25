"""patchtst-owned E2E contract plugins."""
from __future__ import annotations

import numpy as np
from functools import reduce
from operator import mul
from typing import Any

from tests.e2e_harness.contracts import (
    MetricResult,
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

def _load_output_field(data: dict[str, Any]) -> np.ndarray | None:
    if "output_field" not in data:
        return None
    return np.asarray(data["output_field"], dtype=np.float64)

def _declared_shape(data: dict[str, Any]) -> tuple[int, ...] | None:
    raw = data.get("output_shape")
    if not isinstance(raw, list) or not raw:
        return None
    try:
        shape = tuple(int(dim) for dim in raw)
    except (TypeError, ValueError):
        return None
    if any(dim <= 0 for dim in shape):
        return None
    return shape

def _numel(shape: tuple[int, ...] | None) -> int | None:
    if not shape:
        return None
    return reduce(mul, shape, 1)

def _reshape_trt_like_reference(
    trt_data: dict[str, Any], ref_data: dict[str, Any]
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    trt_arr = _load_output_field(trt_data)
    ref_arr = _load_output_field(ref_data)
    if trt_arr is None or ref_arr is None:
        return trt_arr, ref_arr, "Missing output_field"

    ref_shape = _declared_shape(ref_data)
    trt_shape = _declared_shape(trt_data)
    if ref_shape is not None:
        expected_numel = _numel(ref_shape)
        if expected_numel is not None and trt_arr.size != expected_numel:
            return trt_arr, ref_arr, (
                f"TRT output element count {trt_arr.size} does not match "
                f"reference shape {list(ref_shape)} ({expected_numel} elements)"
            )
        if trt_arr.shape != ref_shape and expected_numel == trt_arr.size:
            trt_arr = trt_arr.reshape(ref_shape)
        if ref_arr.shape != ref_shape and expected_numel == ref_arr.size:
            ref_arr = ref_arr.reshape(ref_shape)

    if trt_shape is not None and ref_shape is not None and trt_shape != ref_shape:
        return trt_arr, ref_arr, (
            f"Declared output shapes differ: TRT {list(trt_shape)} vs ref {list(ref_shape)}"
        )

    if trt_arr.shape != ref_arr.shape:
        return trt_arr, ref_arr, (
            f"Output tensor shapes differ after normalization: TRT {list(trt_arr.shape)} "
            f"vs ref {list(ref_arr.shape)}"
        )

    return trt_arr, ref_arr, None

def _relative_l2(trt_arr: np.ndarray, ref_arr: np.ndarray) -> float:
    ref_norm = np.linalg.norm(ref_arr)
    if ref_norm < 1e-12:
        return float(np.linalg.norm(trt_arr - ref_arr))
    return float(np.linalg.norm(trt_arr - ref_arr) / ref_norm)

def _max_pointwise_error(trt_arr: np.ndarray, ref_arr: np.ndarray) -> float:
    return float(np.max(np.abs(trt_arr - ref_arr)))

def _finite_metric(name: str, ok: bool, note: str = "") -> MetricResult:
    del name
    return MetricResult(
        value=1.0 if ok else 0.0,
        threshold=1.0,
        operator="==",
        passed=ok,
        note=note,
    )

def _check_all_finite(arr: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(arr)))

class PatchtstTimeSeriesPointForecastPlugin:
    reference_families = ["time_series_point_forecast"]
    user_contract = "time_series_point_forecast"

    def configure_reference(self, case):
        del case
        return {}

    def verify(self, trt_output, ref_output, case, threshold):
        trt_arr, ref_arr, shape_error = _reshape_trt_like_reference(
            trt_output.data, ref_output.data)
        if shape_error is not None:
            return make_fail("full_inference", {}, message=shape_error)

        ref_name = str(ref_output.data.get("reference_output_name", ""))
        ref_shape = _declared_shape(ref_output.data)
        rank_ok = ref_shape is not None and len(ref_shape) in (2, 3)
        output_name_ok = ref_name in {"prediction_outputs", "mean_predictions"}
        finite_ok = _check_all_finite(trt_arr) and _check_all_finite(ref_arr)

        rel_l2 = _relative_l2(trt_arr, ref_arr)
        max_err = _max_pointwise_error(trt_arr, ref_arr)
        rel_l2_thresh = threshold.metrics.get("relative_l2", 0.01)
        max_err_thresh = threshold.metrics.get("max_pointwise_error", 0.1)
        metrics = {
            "output_name_supported": _finite_metric(
                "output_name_supported",
                output_name_ok,
                note=f"reference_output_name={ref_name}",
            ),
            "forecast_rank_supported": _finite_metric(
                "forecast_rank_supported",
                rank_ok,
                note=f"reference_shape={list(ref_shape) if ref_shape else None}",
            ),
            "finite_outputs": _finite_metric("finite_outputs", finite_ok),
            "relative_l2": MetricResult(
                value=rel_l2,
                threshold=rel_l2_thresh,
                operator="<=",
                passed=rel_l2 <= rel_l2_thresh,
            ),
            "max_pointwise_error": MetricResult(
                value=max_err,
                threshold=max_err_thresh,
                operator="<=",
                passed=max_err <= max_err_thresh,
            ),
        }

        passed = all(metric.passed for metric in metrics.values())
        if passed:
            return make_pass(
                "full_inference",
                metrics,
                "supported point-forecast output shape + finite outputs + numeric parity",
            )
        return make_fail(
            "full_inference",
            metrics,
            "supported point-forecast output shape + finite outputs + numeric parity",
            f"Point forecast contract failed for {case.name}",
        )

class PatchtstTimeSeriesRegressionPlugin:
    reference_families = ["time_series_regression"]
    user_contract = "time_series_regression"

    def configure_reference(self, case):
        del case
        return {}

    def verify(self, trt_output, ref_output, case, threshold):
        trt_arr, ref_arr, shape_error = _reshape_trt_like_reference(
            trt_output.data, ref_output.data)
        if shape_error is not None:
            return make_fail("full_inference", {}, message=shape_error)

        ref_name = str(ref_output.data.get("reference_output_name", ""))
        ref_shape = _declared_shape(ref_output.data)
        rank_ok = ref_shape is not None and len(ref_shape) in (1, 2, 3)
        output_name_ok = "regression" in ref_name or ref_name in {
            "prediction_outputs",
            "output",
        }
        finite_ok = _check_all_finite(trt_arr) and _check_all_finite(ref_arr)
        rel_l2 = _relative_l2(trt_arr, ref_arr)
        max_err = _max_pointwise_error(trt_arr, ref_arr)
        sign_match = bool(np.all(np.sign(trt_arr) == np.sign(ref_arr)))
        rel_l2_thresh = threshold.metrics.get("relative_l2", 0.01)
        max_err_thresh = threshold.metrics.get("max_pointwise_error", 0.1)
        metrics = {
            "output_name_supported": _finite_metric(
                "output_name_supported",
                output_name_ok,
                note=f"reference_output_name={ref_name}",
            ),
            "regression_rank_supported": _finite_metric(
                "regression_rank_supported",
                rank_ok,
                note=f"reference_shape={list(ref_shape) if ref_shape else None}",
            ),
            "finite_outputs": _finite_metric("finite_outputs", finite_ok),
            "sign_match": MetricResult(
                value=1.0 if sign_match else 0.0,
                threshold=1.0,
                operator="==",
                passed=sign_match,
            ),
            "relative_l2": MetricResult(
                value=rel_l2,
                threshold=rel_l2_thresh,
                operator="<=",
                passed=rel_l2 <= rel_l2_thresh,
            ),
            "max_pointwise_error": MetricResult(
                value=max_err,
                threshold=max_err_thresh,
                operator="<=",
                passed=max_err <= max_err_thresh,
            ),
        }

        passed = all(metric.passed for metric in metrics.values())
        if passed:
            return make_pass(
                "full_inference",
                metrics,
                "supported regression output semantics + finite outputs + numeric parity",
            )
        return make_fail(
            "full_inference",
            metrics,
            "supported regression output semantics + finite outputs + numeric parity",
            f"Regression contract failed for {case.name}",
        )

plugin = [PatchtstTimeSeriesPointForecastPlugin(), PatchtstTimeSeriesRegressionPlugin()]
