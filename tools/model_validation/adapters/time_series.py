# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native time-series workload identity and fidelity reduction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..contracts import digest_value


@dataclass(frozen=True)
class TimeSeriesSample:
    sample_id: str
    inputs: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedTimeSeriesWorkload:
    suite_id: str
    adapter_kind: str
    adapter_version: str
    ordered_sample_ids: tuple[str, ...]
    samples: tuple[TimeSeriesSample, ...]
    workload_digest: str


class TimeSeriesTaskAdapter:
    kind = "time_series_csv"
    version = "1"

    def prepare(self, work_dir: Path, *, suite_id: str) -> PreparedTimeSeriesWorkload:
        path = work_dir / "prompts.jsonl"
        rows: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(row)
        if not rows:
            raise ValueError("Time-series workload must contain at least one sample")
        samples: list[TimeSeriesSample] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            sample_id = str(row.get("sample_id", f"time_series_{index:06d}"))
            if not sample_id:
                raise ValueError(f"Time-series sample {index} has an empty sample ID")
            if sample_id in seen:
                raise ValueError(f"Time-series workload has duplicate sample ID {sample_id!r}")
            seen.add(sample_id)
            inputs = row.get("inputs")
            if not isinstance(inputs, Mapping) or not inputs:
                raise ValueError(f"Time-series sample {sample_id!r} has no numeric inputs")
            samples.append(TimeSeriesSample(sample_id, dict(inputs)))
        digest = digest_value(
            {
                "suite_id": suite_id,
                "adapter_kind": self.kind,
                "adapter_version": self.version,
                "samples": [
                    {"sample_id": sample.sample_id, "inputs": sample.inputs} for sample in samples
                ],
            }
        )
        return PreparedTimeSeriesWorkload(
            suite_id=suite_id,
            adapter_kind=self.kind,
            adapter_version=self.version,
            ordered_sample_ids=tuple(sample.sample_id for sample in samples),
            samples=tuple(samples),
            workload_digest=digest,
        )

    def fidelity_metrics(
        self,
        hf_data: Mapping[str, Any],
        trtfb_data: Mapping[str, Any],
        *,
        gates: Mapping[str, Any],
    ) -> dict[str, Any]:
        hf_rows = _response_rows(hf_data, "HF")
        trtfb_rows = _response_rows(trtfb_data, "TRTMC")
        if len(hf_rows) != len(trtfb_rows):
            raise ValueError(
                "Time-series HF/TRTMC prediction count mismatch: "
                f"{len(hf_rows)} != {len(trtfb_rows)}"
            )
        max_relative_l2 = float(gates.get("max_relative_l2", 0.01))
        max_absolute_error = float(gates.get("max_absolute_error", 0.1))
        min_sample_agreement_rate = float(gates.get("min_sample_agreement_rate", 1.0))
        cases: list[dict[str, Any]] = []
        for index, (hf_row, trtfb_row) in enumerate(zip(hf_rows, trtfb_rows, strict=True)):
            hf_id = str(hf_row.get("sample_id", index))
            trtfb_id = str(trtfb_row.get("sample_id", index))
            if hf_id != trtfb_id:
                raise ValueError(
                    f"Time-series sample id mismatch at {index}: {hf_id!r} != {trtfb_id!r}"
                )
            hf_values = hf_row.get("output_values")
            trtfb_values = trtfb_row.get("output_values")
            if not isinstance(hf_values, list) or not isinstance(trtfb_values, list):
                raise ValueError(f"Time-series prediction {hf_id!r} is missing output_values")
            if len(hf_values) != len(trtfb_values) or not hf_values:
                cases.append(
                    {
                        "sample_id": hf_id,
                        "passed": False,
                        "error": (
                            "output element count mismatch: "
                            f"HF={len(hf_values)} TRTMC={len(trtfb_values)}"
                        ),
                    }
                )
                continue
            hf_vector = [float(value) for value in hf_values]
            trtfb_vector = [float(value) for value in trtfb_values]
            if not all(math.isfinite(value) for value in hf_vector + trtfb_vector):
                cases.append(
                    {
                        "sample_id": hf_id,
                        "passed": False,
                        "error": "non-finite output value",
                    }
                )
                continue
            squared_error = sum(
                (trtfb - hf) ** 2 for hf, trtfb in zip(hf_vector, trtfb_vector, strict=True)
            )
            reference_squared_norm = sum(value * value for value in hf_vector)
            relative_l2 = (
                math.sqrt(squared_error / reference_squared_norm)
                if reference_squared_norm >= 1e-24
                else math.sqrt(squared_error)
            )
            absolute_error = max(
                abs(trtfb - hf) for hf, trtfb in zip(hf_vector, trtfb_vector, strict=True)
            )
            cases.append(
                {
                    "sample_id": hf_id,
                    "output_numel": len(hf_vector),
                    "hf_output_shape": hf_row.get("output_shape", []),
                    "trtfb_output_shape": trtfb_row.get("output_shape", []),
                    "relative_l2": relative_l2,
                    "max_absolute_error": absolute_error,
                    "passed": (
                        relative_l2 <= max_relative_l2 and absolute_error <= max_absolute_error
                    ),
                }
            )
        valid_cases = [case for case in cases if "relative_l2" in case]
        passed_count = sum(bool(case.get("passed")) for case in cases)
        agreement_rate = passed_count / len(cases) if cases else 0.0
        status = (
            "passed"
            if cases
            and len(valid_cases) == len(cases)
            and agreement_rate >= min_sample_agreement_rate
            else "failed"
        )
        return {
            "status": status,
            "sample_count": len(cases),
            "valid_count": len(valid_cases),
            "passed_count": passed_count,
            "sample_agreement_rate": agreement_rate,
            "mean_relative_l2": (
                sum(float(case["relative_l2"]) for case in valid_cases) / len(valid_cases)
                if valid_cases
                else float("inf")
            ),
            "max_relative_l2": max(
                (float(case["relative_l2"]) for case in valid_cases),
                default=float("inf"),
            ),
            "max_absolute_error": max(
                (float(case["max_absolute_error"]) for case in valid_cases),
                default=float("inf"),
            ),
            "gates": {
                "max_relative_l2": max_relative_l2,
                "max_absolute_error": max_absolute_error,
                "min_sample_agreement_rate": min_sample_agreement_rate,
            },
            "cases": cases,
        }

    def measurement_units(self) -> tuple[str, ...]:
        return ("sample",)

    def prediction_output_digests(
        self,
        data: Mapping[str, Any],
        *,
        label: str,
    ) -> dict[str, str]:
        """Identify outputs accepted by the correctness/fidelity stage."""

        digests: dict[str, str] = {}
        for index, row in enumerate(_response_rows(data, label)):
            sample_id = str(row.get("sample_id", index))
            if sample_id in digests:
                raise ValueError(f"{label} predictions contain duplicate ID {sample_id!r}")
            values = row.get("output_values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"{label} prediction {sample_id!r} has no output_values")
            numeric_values = [float(value) for value in values]
            if not all(math.isfinite(value) for value in numeric_values):
                raise ValueError(f"{label} prediction {sample_id!r} has non-finite output")
            shape = row.get("output_shape", [])
            if not isinstance(shape, list):
                raise ValueError(f"{label} prediction {sample_id!r} has invalid output_shape")
            digests[sample_id] = digest_value(
                {"output_values": numeric_values, "output_shape": shape}
            )
        return digests


def _response_rows(data: Mapping[str, Any], label: str) -> list[Mapping[str, Any]]:
    rows = data.get("responses", [])
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{label} predictions must contain an object response list")
    return rows
