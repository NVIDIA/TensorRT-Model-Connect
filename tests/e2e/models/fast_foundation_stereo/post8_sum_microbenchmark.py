# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-event control-candidate-control tile sweep for the post8 sum plugin."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tests.e2e.models.fast_foundation_stereo.post8_sum_oracle import (
    _Runner,
    _SHAPE,
    _TILE_POSITIONS,
    _build_engine,
    _pin_plugin_library,
    _sha256,
    _validate_tile_positions,
)
from tests.e2e.models.fast_foundation_stereo.trt_runner import (
    load_native_plugin_libraries,
)


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    return {
        "count": int(array.size),
        "mean_ms": mean,
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "stddev_ms": float(array.std()),
        "coefficient_of_variation": float(array.std() / mean),
    }


class _TimedRunner(_Runner):
    def run(
        self,
        linear: torch.Tensor,
        skip: torch.Tensor,
        *,
        warmup: int,
        iterations: int,
    ) -> tuple[list[float], torch.Tensor]:
        outputs = self._bind(linear, skip)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(warmup):
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("post8 warmup enqueue failed")
        self.stream.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        with torch.cuda.stream(self.stream):
            for start, end in zip(starts, ends):
                start.record(self.stream)
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("post8 timed enqueue failed")
                end.record(self.stream)
        ends[-1].synchronize()
        timings = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
        return timings, next(iter(outputs.values()))


def _input_sha256(linear: torch.Tensor, skip: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(linear.cpu().numpy().tobytes())
    digest.update(skip.cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-library", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--tiles", type=int, nargs="+", default=list(_TILE_POSITIONS))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.warmup < 20 or args.iterations < 100:
        raise ValueError("post8 microbenchmark requires warmup >= 20 and iterations >= 100")
    tiles = tuple(dict.fromkeys(_validate_tile_positions(value) for value in args.tiles))

    plugin_library = _pin_plugin_library(args.plugin_library)
    loaded = load_native_plugin_libraries([plugin_library])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    control_plan = args.out_dir / "control.plan"
    candidate_plans = {
        tile_positions: args.out_dir / f"candidate-tile{tile_positions}.plan"
        for tile_positions in tiles
    }
    if args.build:
        control_plan.write_bytes(_build_engine(32, ("reference",)))
        for tile_positions, plan in candidate_plans.items():
            plan.write_bytes(_build_engine(tile_positions, ("candidate",)))
    missing = [path for path in (control_plan, *candidate_plans.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing standalone post8 plans: {missing}")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260817)
    linear = torch.randn(_SHAPE, generator=generator, dtype=torch.float16, device="cuda")
    skip = torch.randn(_SHAPE, generator=generator, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()

    control = _TimedRunner(control_plan.resolve(), ("reference",))
    results = {}
    for tile_positions, plan in candidate_plans.items():
        candidate = _TimedRunner(plan.resolve(), ("candidate",))
        reference_once = control.run_once(linear, skip)["reference"]
        candidate_once = candidate.run_once(linear, skip)["candidate"]
        if not torch.equal(reference_once.view(torch.int16), candidate_once.view(torch.int16)):
            mismatch_count = int(
                (reference_once.view(torch.int16) != candidate_once.view(torch.int16)).sum()
            )
            raise RuntimeError(
                f"tile {tile_positions} failed pre-benchmark bitwise gate: "
                f"{mismatch_count} mismatches"
            )

        control_first, _ = control.run(
            linear,
            skip,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        candidate_values, _ = candidate.run(
            linear,
            skip,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        control_second, _ = control.run(
            linear,
            skip,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        pooled_control = _summary(control_first + control_second)
        candidate_summary = _summary(candidate_values)
        gain_ms = float(pooled_control["mean_ms"]) - float(candidate_summary["mean_ms"])
        results[str(tile_positions)] = {
            "protocol": "control-candidate-control",
            "control_first": _summary(control_first),
            "candidate": candidate_summary,
            "control_second": _summary(control_second),
            "pooled_control": pooled_control,
            "control_drift_ms": float(_summary(control_second)["mean_ms"])
            - float(_summary(control_first)["mean_ms"]),
            "drift_corrected_gain_ms": gain_ms,
            "drift_corrected_gain_fraction": gain_ms / float(pooled_control["mean_ms"]),
            "candidate_plan": str(plan.resolve()),
            "candidate_plan_sha256": _sha256(plan.resolve()),
        }

    winner = min(results, key=lambda tile: float(results[tile]["candidate"]["mean_ms"]))
    receipt = {
        "protocol": "per-tile-control-candidate-control",
        "shape": list(_SHAPE),
        "warmup": args.warmup,
        "iterations_per_segment": args.iterations,
        "fixed_inputs_sha256": _input_sha256(linear, skip),
        "plugin_libraries": loaded,
        "plugin_library_sha256": _sha256(plugin_library),
        "control_plan": str(control_plan.resolve()),
        "control_plan_sha256": _sha256(control_plan.resolve()),
        "tiles": results,
        "winner_tile_positions": int(winner),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
