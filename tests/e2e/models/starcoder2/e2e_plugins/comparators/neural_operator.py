"""Neural operator comparator — compare TRT vs reference field outputs.

Metrics: relative L2 on output fields, max pointwise error, optional
PDE residual constraints.
"""

from __future__ import annotations

import logging
import math
from functools import reduce
from operator import mul
from typing import Any, Dict

from ..contracts import CompareResult, MetricResult, StageOutput, StageSpec, StageStatus, ThresholdProfile

logger = logging.getLogger(__name__)


def _try_import_numpy():
    """Lazily import numpy; return None if unavailable."""
    try:
        import numpy as np
        return np
    except ImportError:
        return None


def _relative_l2(trt_field: Any, ref_field: Any) -> float:
    """Compute relative L2 error: ||trt - ref||_2 / ||ref||_2."""
    np = _try_import_numpy()
    if np is not None:
        trt_arr = np.asarray(trt_field, dtype=np.float64)
        ref_arr = np.asarray(ref_field, dtype=np.float64)
        ref_norm = np.linalg.norm(ref_arr)
        if ref_norm < 1e-12:
            return float(np.linalg.norm(trt_arr - ref_arr))
        return float(np.linalg.norm(trt_arr - ref_arr) / ref_norm)

    # Pure Python fallback for flat lists
    if isinstance(trt_field, list) and isinstance(ref_field, list):
        diff_sq = sum((a - b) ** 2 for a, b in zip(trt_field, ref_field))
        ref_sq = sum(b ** 2 for b in ref_field)
        if ref_sq < 1e-24:
            return math.sqrt(diff_sq)
        return math.sqrt(diff_sq / ref_sq)

    return float("inf")


def _max_pointwise_error(trt_field: Any, ref_field: Any) -> float:
    """Compute max pointwise absolute error."""
    np = _try_import_numpy()
    if np is not None:
        trt_arr = np.asarray(trt_field, dtype=np.float64)
        ref_arr = np.asarray(ref_field, dtype=np.float64)
        return float(np.max(np.abs(trt_arr - ref_arr)))

    if isinstance(trt_field, list) and isinstance(ref_field, list):
        return max(abs(a - b) for a, b in zip(trt_field, ref_field))

    return float("inf")


def _declared_shape(data: Dict[str, Any]) -> tuple[int, ...] | None:
    raw = data.get("output_shape")
    if not isinstance(raw, list) or not raw:
        return None
    try:
        shape = tuple(int(dim) for dim in raw)
    except (TypeError, ValueError):
        return None
    if any(dim < 0 for dim in shape):
        return None
    return shape


def _numel_for_shape(shape: tuple[int, ...] | None) -> int | None:
    if not shape:
        return None
    return reduce(mul, shape, 1)


def _reshape_like_reference(trt_field: Any, trt_data: Dict[str, Any], ref_data: Dict[str, Any]) -> tuple[Any, Any, str | None]:
    np = _try_import_numpy()
    if np is None:
        return trt_field, _load_field(ref_data), None

    ref_field = _load_field(ref_data)
    if trt_field is None or ref_field is None:
        return trt_field, ref_field, None

    trt_arr = np.asarray(trt_field, dtype=np.float64)
    ref_arr = np.asarray(ref_field, dtype=np.float64)
    ref_shape = _declared_shape(ref_data)
    trt_shape = _declared_shape(trt_data)

    if ref_shape is not None:
        expected_numel = _numel_for_shape(ref_shape)
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
            f"Output tensor shapes differ after normalization: TRT "
            f"{list(trt_arr.shape)} vs ref {list(ref_arr.shape)}"
        )

    return trt_arr, ref_arr, None


def _load_field(data: Dict[str, Any], key: str = "output_field_path") -> Any:
    """Load field data from StageOutput data dict.

    Supports numpy files via path or inline arrays.
    """
    np = _try_import_numpy()

    # Try loading from file path
    field_path = data.get(key)
    if field_path and np is not None:
        try:
            return np.load(field_path)
        except Exception:
            pass

    # Try inline arrays
    if "output_field" in data:
        return data["output_field"]
    if "field" in data:
        return data["field"]

    return None


class NeuralOperatorComparator:
    """Compare TRT vs reference neural operator field outputs."""

    @property
    def task_strategy(self) -> str:
        return "neural_operator"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        trt_field = _load_field(trt.data)
        ref_field = _load_field(ref.data)

        if trt_field is None or ref_field is None:
            missing = []
            if trt_field is None:
                missing.append("TRT")
            if ref_field is None:
                missing.append("ref")
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"Missing field data from {', '.join(missing)}",
            )

        trt_field, ref_field, shape_error = _reshape_like_reference(
            trt_field, trt.data, ref.data)
        if shape_error is not None:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.FAILED.value,
                metrics={},
                message=shape_error,
            )

        rel_l2 = _relative_l2(trt_field, ref_field)
        max_error = _max_pointwise_error(trt_field, ref_field)

        th = threshold.metrics

        rel_l2_thresh = th.get("relative_l2", 0.01)
        max_err_thresh = th.get("max_pointwise_error", 0.1)

        metrics: dict[str, MetricResult] = {
            "relative_l2": MetricResult(
                value=rel_l2, threshold=rel_l2_thresh, operator="<=",
                passed=rel_l2 <= rel_l2_thresh,
            ),
            "max_pointwise_error": MetricResult(
                value=max_error, threshold=max_err_thresh, operator="<=",
                passed=max_error <= max_err_thresh,
            ),
        }

        # Optional PDE residual constraint (if provided in threshold)
        pde_residual_thresh = th.get("pde_residual")
        if pde_residual_thresh is not None:
            pde_residual = trt.data.get("pde_residual")
            if pde_residual is not None:
                metrics["pde_residual"] = MetricResult(
                    value=float(pde_residual), threshold=pde_residual_thresh,
                    operator="<=", passed=float(pde_residual) <= pde_residual_thresh,
                )

        passed = all(m.passed for m in metrics.values())

        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=(
                f"Neural operator comparison: relative_l2={rel_l2:.6e}, "
                f"max_pointwise_error={max_error:.6e}"
            ),
        )


plugin = NeuralOperatorComparator()
