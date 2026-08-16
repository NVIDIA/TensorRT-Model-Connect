# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize Nsight Compute SpeedOfLight CSV exports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_samples(paths: list[Path]) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for path in paths:
        launches: dict[str, dict[str, object]] = {}
        with path.open(newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                launch = launches.setdefault(
                    row["ID"],
                    {
                        "kernel": row["Kernel Name"],
                        "source": path.name,
                    },
                )
                metric_name = row["Metric Name"]
                metric_value = row["Metric Value"]
                if metric_name.startswith("Duration ("):
                    launch["duration_us"] = float(metric_value)
                elif metric_name.startswith("Memory Throughput ("):
                    launch["memory_pct"] = float(metric_value)
                elif metric_name.startswith("Compute (SM) Throughput ("):
                    launch["compute_pct"] = float(metric_value)
        samples.extend(launches.values())
    return [
        sample
        for sample in samples
        if {"duration_us", "memory_pct", "compute_pct"} <= sample.keys()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    samples = _load_samples(args.csv)
    total_duration = sum(float(sample["duration_us"]) for sample in samples)

    def weighted(field: str) -> float:
        return (
            sum(float(sample["duration_us"]) * float(sample[field]) for sample in samples)
            / total_duration
        )

    summary = {
        "sampled_launches": len(samples),
        "sampled_duration_us": total_duration,
        "duration_weighted_compute_pct": weighted("compute_pct"),
        "duration_weighted_memory_pct": weighted("memory_pct"),
        "compute_limited_launches": sum(
            float(sample["compute_pct"]) >= float(sample["memory_pct"]) for sample in samples
        ),
        "memory_limited_launches": sum(
            float(sample["memory_pct"]) > float(sample["compute_pct"]) for sample in samples
        ),
        "launches_at_or_above_80_pct": sum(
            max(float(sample["compute_pct"]), float(sample["memory_pct"])) >= 80.0
            for sample in samples
        ),
        "max_compute_pct": max(float(sample["compute_pct"]) for sample in samples),
        "max_memory_pct": max(float(sample["memory_pct"]) for sample in samples),
        "top_duration_launches": sorted(
            samples,
            key=lambda sample: float(sample["duration_us"]),
            reverse=True,
        )[: args.top],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
