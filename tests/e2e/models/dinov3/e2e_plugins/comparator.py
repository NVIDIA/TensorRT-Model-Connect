# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict semantic comparator for complete DINOv3 feature tensors."""

from __future__ import annotations

import numpy as np

from tests.e2e_harness.contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)

_DEFAULT_THRESHOLDS = {
    "full_cosine": 0.999,
    "cls_cosine": 0.999,
    "pooler_cosine": 0.999,
    "register_cosine": 0.999,
    "mean_patch_cosine": 0.999,
    "p01_patch_cosine": 0.995,
    "relative_frobenius": 0.01,
    "shape_match": 1.0,
    "register_count_match": 1.0,
    "pooler_token_invariant": 1.0,
    "finite_tensors": 1.0,
}


def _threshold(profile: ThresholdProfile, name: str) -> float:
    return float(profile.metrics.get(name, _DEFAULT_THRESHOLDS[name]))


def _tensor(data: dict, name: str) -> np.ndarray:
    payload = data.get(name)
    if not isinstance(payload, dict):
        raise ValueError(f"missing tensor object {name!r}")
    shape = payload.get("shape")
    values = payload.get("data")
    if (
        not isinstance(shape, list)
        or not shape
        or not all(isinstance(dim, int) and dim > 0 for dim in shape)
    ):
        raise ValueError(f"{name} has invalid shape {shape!r}")
    array = np.asarray(values, dtype=np.float64)
    expected = int(np.prod(shape, dtype=np.int64))
    if array.ndim != 1 or array.size != expected:
        raise ValueError(f"{name} data size {array.size} does not match shape {shape} ({expected})")
    return array.reshape(tuple(shape))


def _cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left, right = lhs.reshape(-1), rhs.reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def _minimum_metric(value: float, threshold: float) -> MetricResult:
    return MetricResult(value=value, threshold=threshold, operator=">=", passed=value >= threshold)


def _maximum_metric(value: float, threshold: float) -> MetricResult:
    return MetricResult(value=value, threshold=threshold, operator="<=", passed=value <= threshold)


def _exact_metric(value: bool, threshold: float = 1.0) -> MetricResult:
    return MetricResult(value=float(value), threshold=threshold, operator="==", passed=value)


class ImageFeatureExtractionComparator:
    @property
    def task_strategy(self) -> str:
        return "image_feature_extraction"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        metrics: dict[str, MetricResult] = {}
        try:
            trt_hidden = _tensor(trt.data, "last_hidden_state")
            ref_hidden = _tensor(ref.data, "last_hidden_state")
            trt_pooler = _tensor(trt.data, "pooler_output")
            ref_pooler = _tensor(ref.data, "pooler_output")
        except (TypeError, ValueError) as error:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics=metrics,
                message=f"Invalid DINOv3 feature payload: {error}",
            )

        shape_match = (
            trt_hidden.shape == ref_hidden.shape
            and trt_pooler.shape == ref_pooler.shape
            and trt_hidden.ndim == 3
            and trt_pooler.ndim == 2
            and trt_hidden.shape[0] == trt_pooler.shape[0]
            and trt_hidden.shape[-1] == trt_pooler.shape[-1]
        )
        metrics["shape_match"] = _exact_metric(shape_match, _threshold(threshold, "shape_match"))
        if not shape_match:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.FAILED.value,
                metrics=metrics,
                composite_rule="all tensor shapes must match exactly",
                message=(
                    "DINOv3 shape mismatch: "
                    f"TRT hidden={trt_hidden.shape}, pooler={trt_pooler.shape}; "
                    f"HF hidden={ref_hidden.shape}, pooler={ref_pooler.shape}"
                ),
            )

        trt_registers = int(trt.data.get("num_register_tokens", -1))
        ref_registers = int(ref.data.get("num_register_tokens", -1))
        register_count_match = (
            trt_registers == ref_registers
            and trt_registers >= 0
            and trt_hidden.shape[1] > 1 + trt_registers
        )
        metrics["register_count_match"] = _exact_metric(
            register_count_match, _threshold(threshold, "register_count_match")
        )
        if not register_count_match:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.FAILED.value,
                metrics=metrics,
                composite_rule="register count must match and leave at least one patch token",
                message=(
                    f"DINOv3 register mismatch: TRT={trt_registers}, "
                    f"HF={ref_registers}, tokens={trt_hidden.shape[1]}"
                ),
            )

        finite = all(
            np.isfinite(array).all() for array in (trt_hidden, ref_hidden, trt_pooler, ref_pooler)
        )
        metrics["finite_tensors"] = _exact_metric(finite, _threshold(threshold, "finite_tensors"))
        metrics["pooler_token_invariant"] = _exact_metric(
            np.array_equal(trt_pooler, trt_hidden[:, 0, :])
            and np.array_equal(ref_pooler, ref_hidden[:, 0, :]),
            _threshold(threshold, "pooler_token_invariant"),
        )

        full_cosine = _cosine(trt_hidden, ref_hidden)
        cls_cosine = _cosine(trt_hidden[:, 0, :], ref_hidden[:, 0, :])
        pooler_cosine = _cosine(trt_pooler, ref_pooler)
        relative_frobenius = float(
            np.linalg.norm(trt_hidden - ref_hidden) / max(float(np.linalg.norm(ref_hidden)), 1e-12)
        )
        for name, value in (
            ("full_cosine", full_cosine),
            ("cls_cosine", cls_cosine),
            ("pooler_cosine", pooler_cosine),
        ):
            metrics[name] = _minimum_metric(value, _threshold(threshold, name))
        metrics["relative_frobenius"] = _maximum_metric(
            relative_frobenius, _threshold(threshold, "relative_frobenius")
        )

        patch_start = 1 + trt_registers
        trt_patches = trt_hidden[:, patch_start:, :].reshape(-1, trt_hidden.shape[-1])
        ref_patches = ref_hidden[:, patch_start:, :].reshape(-1, ref_hidden.shape[-1])
        patch_denominators = np.linalg.norm(trt_patches, axis=1) * np.linalg.norm(
            ref_patches, axis=1
        )
        patch_cosines = np.divide(
            np.sum(trt_patches * ref_patches, axis=1),
            patch_denominators,
            out=np.zeros_like(patch_denominators),
            where=patch_denominators != 0.0,
        )
        zero_norm = patch_denominators == 0.0
        patch_cosines[zero_norm] = np.all(trt_patches[zero_norm] == ref_patches[zero_norm], axis=1)
        mean_patch_cosine = float(np.mean(patch_cosines))
        p01_patch_cosine = float(np.percentile(patch_cosines, 1.0))
        metrics["mean_patch_cosine"] = _minimum_metric(
            mean_patch_cosine, _threshold(threshold, "mean_patch_cosine")
        )
        metrics["p01_patch_cosine"] = _minimum_metric(
            p01_patch_cosine, _threshold(threshold, "p01_patch_cosine")
        )

        if trt_registers:
            register_cosine = _cosine(
                trt_hidden[:, 1:patch_start, :], ref_hidden[:, 1:patch_start, :]
            )
            metrics["register_cosine"] = _minimum_metric(
                register_cosine, _threshold(threshold, "register_cosine")
            )

        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule=(
                "exact shapes/counts/pooler invariant and finite tensors; "
                "full, CLS, register, mean-patch cosine >= 0.999; "
                "p01 patch cosine >= 0.995; relative Frobenius <= "
                f"{_threshold(threshold, 'relative_frobenius'):g}"
            ),
            message=(
                f"DINOv3 semantic parity: full_cos={full_cosine:.8f}, "
                f"cls_cos={cls_cosine:.8f}, mean_patch_cos={mean_patch_cosine:.8f}, "
                f"p01_patch_cos={p01_patch_cosine:.8f}, rel_frob={relative_frobenius:.8f}"
            ),
        )


comparator = ImageFeatureExtractionComparator()
