# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM model-owned prompted segmentation contract."""

from __future__ import annotations

from pathlib import Path

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

def _resolve_mask_list(data):
    masks = data.get("masks") or []
    if masks:
        return masks

    masks_path = data.get("masks_path")
    if not masks_path:
        return []

    path = Path(masks_path)
    if not path.is_file():
        return []

    loaded = np.load(path, allow_pickle=False)
    if loaded.ndim == 2:
        return [loaded]
    return [loaded[i] for i in range(loaded.shape[0])]

def _compute_binary_iou(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)

def _verify_sam_prompted_masks(trt_output, ref_output, threshold):
    trt_masks = _resolve_mask_list(trt_output.data)
    ref_masks = _resolve_mask_list(ref_output.data)

    if not trt_masks or not ref_masks:
        return make_error("full_inference", "Missing prompted segmentation masks")

    metrics = {
        "trt_num_masks": MetricResult(
            value=float(len(trt_masks)), threshold=None, operator=">=", passed=True),
        "ref_num_masks": MetricResult(
            value=float(len(ref_masks)), threshold=None, operator=">=", passed=True),
    }

    num_masks_threshold = threshold.metrics.get("num_masks_consistency")
    if num_masks_threshold is not None:
        same_count = len(trt_masks) == len(ref_masks)
        metrics["num_masks_consistency"] = MetricResult(
            value=1.0 if same_count else 0.0,
            threshold=1.0,
            operator="==",
            passed=same_count,
        )

    iou_values = []
    for i in range(min(len(trt_masks), len(ref_masks))):
        trt_mask = np.asarray(trt_masks[i], dtype=bool)
        ref_mask = np.asarray(ref_masks[i], dtype=bool)
        if trt_mask.shape != ref_mask.shape:
            try:
                from PIL import Image
                trt_img = Image.fromarray(trt_mask.astype(np.uint8) * 255)
                trt_img = trt_img.resize(
                    (ref_mask.shape[1], ref_mask.shape[0]), Image.NEAREST)
                trt_mask = np.asarray(trt_img, dtype=np.uint8).astype(bool)
            except ImportError:
                return make_error(
                    "full_inference",
                    f"Shape mismatch {trt_mask.shape} vs {ref_mask.shape} and PIL unavailable",
                )
        iou = _compute_binary_iou(trt_mask, ref_mask)
        iou_values.append(iou)
        metrics[f"mask_{i}_iou"] = MetricResult(
            value=iou, threshold=None, operator=">=", passed=True,
            note="per-mask informational",
        )

    if not iou_values:
        return make_error("full_inference", "No prompted segmentation masks were comparable")

    mean_iou = sum(iou_values) / len(iou_values)
    iou_threshold = threshold.metrics.get("iou_per_prompt", 0.5)
    metrics["iou_per_prompt"] = MetricResult(
        value=mean_iou,
        threshold=iou_threshold,
        operator=">=",
        passed=mean_iou >= iou_threshold,
    )

    rule = "mean prompted-mask IoU >= threshold"
    gated = [m for m in metrics.values() if m.threshold is not None]
    passed = all(m.passed for m in gated)
    if passed:
        return make_pass("full_inference", metrics, rule)
    return make_fail(
        "full_inference",
        metrics,
        rule,
        f"SAM prompted segmentation quality: mean_iou={mean_iou:.3f}",
    )

class SamSegmentationPlugin:
    reference_families = ["prompted_segmentation_sam"]
    user_contract = "prompted_mask"

    def configure_reference(self, case):
        return {"sam_mode": True}

    def verify(self, trt_output, ref_output, case, threshold):
        return _verify_sam_prompted_masks(trt_output, ref_output, threshold)

plugin = SamSegmentationPlugin()
