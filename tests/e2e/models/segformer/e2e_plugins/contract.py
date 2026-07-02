# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""segformer-owned E2E contract plugins."""
from __future__ import annotations

import numpy as np

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

def _compute_iou(pred, gt):
    pred = np.asarray(pred, dtype=np.int32)
    gt = np.asarray(gt, dtype=np.int32)
    classes = np.union1d(np.unique(pred), np.unique(gt))
    if len(classes) == 0:
        return 1.0
    ious = []
    for c in classes:
        p = pred == c
        g = gt == c
        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union > 0:
            ious.append(float(intersection / union))
    return float(np.mean(ious)) if ious else 0.0

def _pixel_accuracy(pred, gt):
    pred = np.asarray(pred, dtype=np.int32).flatten()
    gt = np.asarray(gt, dtype=np.int32).flatten()
    if len(pred) != len(gt):
        return 0.0
    return float((pred == gt).mean())

class SegformerSegmentationPlugin:
    reference_families = ["semantic_segmentation"]
    user_contract = "segmentation_mask"

    def configure_reference(self, case):
        del case
        return {}

    def verify(self, trt_output, ref_output, case, threshold):
        del case
        trt_mask = trt_output.data.get("class_map")
        if trt_mask is None:
            trt_mask = trt_output.data.get("mask")
        ref_mask = ref_output.data.get("class_map")
        if ref_mask is None:
            ref_mask = ref_output.data.get("mask")

        if trt_mask is None or ref_mask is None:
            return make_error("full_inference", "Missing mask/class_map in output data")

        trt_arr = np.asarray(trt_mask, dtype=np.int32)
        ref_arr = np.asarray(ref_mask, dtype=np.int32)
        if trt_arr.shape != ref_arr.shape:
            try:
                from PIL import Image
            except ImportError:
                return make_error(
                    "full_inference",
                    f"Shape mismatch {trt_arr.shape} vs {ref_arr.shape} and PIL unavailable",
                )
            ref_img = Image.fromarray(ref_arr.astype(np.uint8))
            ref_img = ref_img.resize((trt_arr.shape[1], trt_arr.shape[0]), Image.NEAREST)
            ref_arr = np.array(ref_img, dtype=np.int32)

        miou = _compute_iou(trt_arr, ref_arr)
        pixel_acc = _pixel_accuracy(trt_arr, ref_arr)
        miou_threshold = threshold.metrics.get("contract_miou_threshold", 0.5)
        pixel_threshold = threshold.metrics.get("contract_pixel_accuracy", 0.85)
        metrics = {
            "mIoU": MetricResult(
                value=miou,
                threshold=miou_threshold,
                operator=">=",
                passed=miou >= miou_threshold,
            ),
            "pixel_accuracy": MetricResult(
                value=pixel_acc,
                threshold=pixel_threshold,
                operator=">=",
                passed=pixel_acc >= pixel_threshold,
            ),
        }

        passed = miou >= miou_threshold and pixel_acc >= pixel_threshold
        rule = "mIoU >= threshold AND pixel_accuracy >= threshold"
        if passed:
            return make_pass("full_inference", metrics, rule)
        return make_fail(
            "full_inference",
            metrics,
            rule,
            f"Segmentation quality: mIoU={miou:.3f} pixel_acc={pixel_acc:.3f}",
        )

plugin = SegformerSegmentationPlugin()
