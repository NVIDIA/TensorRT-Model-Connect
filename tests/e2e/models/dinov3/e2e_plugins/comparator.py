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
_KNN_K_VALUES = (10, 20, 100, 200)
_KNN_QUERY_COSINE_MIN = 0.999
_KNN_QUERY_RELATIVE_L2_MAX = 0.01


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


def _informational_metric(value: float, note: str) -> MetricResult:
    return MetricResult(
        value=value,
        threshold=None,
        operator="informational",
        passed=True,
        note=note,
    )


def _prediction_vector(data: dict, k: int, query_count: int) -> np.ndarray:
    predictions = data.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError("missing weighted k-NN predictions")
    values = np.asarray(predictions.get(str(k)), dtype=np.int64)
    if values.ndim != 1 or len(values) != query_count:
        raise ValueError(f"invalid {k}-NN predictions")
    return values


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
        if trt.data.get("knn_task_accuracy") or ref.data.get("knn_task_accuracy"):
            return self._compare_knn(trt, ref, stage)
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


    def _compare_knn(
        self,
        trt: StageOutput,
        ref: StageOutput,
        stage: StageSpec,
    ) -> CompareResult:
        metrics: dict[str, MetricResult] = {}
        try:
            if not trt.data.get("knn_task_accuracy") or not ref.data.get(
                "knn_task_accuracy"
            ):
                raise ValueError("candidate/reference k-NN task markers differ")
            query_count = int(trt.data.get("query_count", 0))
            reference_query_count = int(ref.data.get("query_count", 0))
            bank_size = int(trt.data.get("bank_size", 0))
            reference_bank_size = int(ref.data.get("bank_size", 0))
            if query_count <= 0 or query_count != reference_query_count:
                raise ValueError("candidate/reference query counts differ")
            if bank_size <= 0 or bank_size != reference_bank_size:
                raise ValueError("candidate/reference bank sizes differ")
            if trt.data.get("class_names") != ref.data.get("class_names"):
                raise ValueError("candidate/reference class maps differ")
            labels = np.asarray(trt.data.get("labels"), dtype=np.int64)
            reference_labels = np.asarray(ref.data.get("labels"), dtype=np.int64)
            if (
                labels.ndim != 1
                or len(labels) != query_count
                or not np.array_equal(labels, reference_labels)
            ):
                raise ValueError("candidate/reference ground-truth labels differ")
            trt_pooler = _tensor(trt.data, "query_pooler_output")
            ref_pooler = _tensor(ref.data, "query_pooler_output")
            if (
                trt_pooler.shape != ref_pooler.shape
                or trt_pooler.shape[0] != query_count
            ):
                raise ValueError("candidate/reference query pooler shapes differ")
            if not np.isfinite(trt_pooler).all() or not np.isfinite(ref_pooler).all():
                raise ValueError("candidate/reference query poolers are non-finite")
        except (TypeError, ValueError) as error:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics=metrics,
                message=f"Invalid DINOv3 k-NN task payload: {error}",
            )

        denominators = np.linalg.norm(trt_pooler, axis=1) * np.linalg.norm(
            ref_pooler, axis=1
        )
        cosines = np.divide(
            np.sum(trt_pooler * ref_pooler, axis=1),
            denominators,
            out=np.zeros_like(denominators),
            where=denominators != 0.0,
        )
        zero_norm = denominators == 0.0
        cosines[zero_norm] = np.all(
            trt_pooler[zero_norm] == ref_pooler[zero_norm], axis=1
        )
        relative_l2 = np.linalg.norm(trt_pooler - ref_pooler, axis=1) / np.maximum(
            np.linalg.norm(ref_pooler, axis=1), 1e-12
        )
        minimum_cosine = float(np.min(cosines))
        maximum_relative_l2 = float(np.max(relative_l2))
        metrics.update(
            {
                "query_count": _informational_metric(
                    float(query_count),
                    "Number of ground-truth test images in this shard",
                ),
                "bank_size": _informational_metric(
                    float(bank_size),
                    "Number of train images in the independent feature bank",
                ),
                "query_pooler_cosine_min": _minimum_metric(
                    minimum_cosine, _KNN_QUERY_COSINE_MIN
                ),
                "query_pooler_relative_l2_max": _maximum_metric(
                    maximum_relative_l2, _KNN_QUERY_RELATIVE_L2_MAX
                ),
            }
        )
        try:
            for k in _KNN_K_VALUES:
                candidate = _prediction_vector(trt.data, k, query_count)
                reference = _prediction_vector(ref.data, k, query_count)
                metrics[f"candidate_correct_{k}nn"] = _informational_metric(
                    float(np.sum(candidate == labels)),
                    "Ground-truth correct predictions",
                )
                metrics[f"reference_correct_{k}nn"] = _informational_metric(
                    float(np.sum(reference == labels)),
                    "Ground-truth correct predictions",
                )
                metrics[f"prediction_agreement_{k}nn"] = _informational_metric(
                    float(np.sum(candidate == reference)),
                    "Candidate/reference identical predictions",
                )
        except ValueError as error:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics=metrics,
                message=f"Invalid DINOv3 k-NN predictions: {error}",
            )

        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule=(
                "all query poolers must preserve cosine >= 0.999 and relative L2 <= 0.01; "
                "ground-truth k-NN task gates are evaluated over the complete test split"
            ),
            message=(
                f"DINOv3 Beans shard: queries={query_count}, bank={bank_size}, "
                f"min_pooler_cos={minimum_cosine:.8f}, max_pooler_rel_l2={maximum_relative_l2:.8f}"
            ),
        )

    def aggregate(self, cases: list[dict], gates: dict) -> dict:
        task_gate = "max_candidate_20nn_top1_accuracy_drop_from_reference"
        if task_gate not in gates:
            return {"evaluated": False, "passed": True}
        required_gates = (
            "exact_task_query_count",
            "min_reference_20nn_top1_accuracy",
            task_gate,
            "min_candidate_reference_20nn_top1_agreement",
            "min_task_query_pooler_cosine",
            "max_task_query_pooler_relative_l2",
        )
        missing_gates = [name for name in required_gates if name not in gates]
        if missing_gates:
            return {
                "evaluated": True,
                "passed": False,
                "gate_failures": [
                    "missing DINOv3 task gates: " + ", ".join(missing_gates)
                ],
            }
        required_metrics = [
            "query_count",
            "query_pooler_cosine_min",
            "query_pooler_relative_l2_max",
        ]
        for k in _KNN_K_VALUES:
            required_metrics.extend(
                (
                    f"candidate_correct_{k}nn",
                    f"reference_correct_{k}nn",
                    f"prediction_agreement_{k}nn",
                )
            )
        missing_cases = [
            str(case.get("sample_id", ""))
            for case in cases
            if any(name not in case.get("metrics", {}) for name in required_metrics)
        ]
        if missing_cases:
            return {
                "evaluated": True,
                "passed": False,
                "gate_failures": [
                    "DINOv3 task sufficient statistics are missing for: "
                    + ", ".join(missing_cases)
                ],
            }
        query_count = int(
            sum(float(case["metrics"]["query_count"]["value"]) for case in cases)
        )
        expected_query_count = int(gates["exact_task_query_count"])
        task_accuracy: dict[str, float | int] = {"task_query_count": query_count}
        for k in _KNN_K_VALUES:
            for side in ("candidate", "reference"):
                correct = sum(
                    float(case["metrics"][f"{side}_correct_{k}nn"]["value"])
                    for case in cases
                )
                task_accuracy[f"{side}_{k}nn_top1_accuracy"] = (
                    correct / query_count if query_count else 0.0
                )
            agreement = sum(
                float(case["metrics"][f"prediction_agreement_{k}nn"]["value"])
                for case in cases
            )
            task_accuracy[f"candidate_reference_{k}nn_top1_agreement"] = (
                agreement / query_count if query_count else 0.0
            )
        task_accuracy["candidate_20nn_top1_accuracy_drop_from_reference"] = max(
            0.0,
            task_accuracy["reference_20nn_top1_accuracy"]
            - task_accuracy["candidate_20nn_top1_accuracy"],
        )
        task_accuracy["task_query_pooler_cosine"] = min(
            float(case["metrics"]["query_pooler_cosine_min"]["value"]) for case in cases
        )
        task_accuracy["task_query_pooler_relative_l2"] = max(
            float(case["metrics"]["query_pooler_relative_l2_max"]["value"])
            for case in cases
        )

        reference_floor = float(gates["min_reference_20nn_top1_accuracy"])
        drop_allowance = float(gates[task_gate])
        agreement_floor = float(gates["min_candidate_reference_20nn_top1_agreement"])
        cosine_floor = float(gates["min_task_query_pooler_cosine"])
        relative_l2_limit = float(gates["max_task_query_pooler_relative_l2"])
        gate_results = {
            "complete_test_split": query_count == expected_query_count,
            "reference_20nn_top1_accuracy": (
                task_accuracy["reference_20nn_top1_accuracy"] >= reference_floor
            ),
            "candidate_20nn_top1_accuracy": (
                task_accuracy["candidate_20nn_top1_accuracy_drop_from_reference"]
                <= drop_allowance
            ),
            "candidate_reference_20nn_top1_agreement": (
                task_accuracy["candidate_reference_20nn_top1_agreement"]
                >= agreement_floor
            ),
            "query_pooler_cosine": task_accuracy["task_query_pooler_cosine"]
            >= cosine_floor,
            "query_pooler_relative_l2": (
                task_accuracy["task_query_pooler_relative_l2"] <= relative_l2_limit
            ),
        }
        failures = [name for name, passed in gate_results.items() if not passed]
        return {
            "evaluated": True,
            "passed": not failures,
            "task_accuracy": task_accuracy,
            "gates": {
                "exact_task_query_count": expected_query_count,
                "min_reference_20nn_top1_accuracy": reference_floor,
                task_gate: drop_allowance,
                "min_candidate_reference_20nn_top1_agreement": agreement_floor,
                "min_task_query_pooler_cosine": cosine_floor,
                "max_task_query_pooler_relative_l2": relative_l2_limit,
            },
            "gate_results": gate_results,
            "gate_failures": failures,
        }


comparator = ImageFeatureExtractionComparator()
