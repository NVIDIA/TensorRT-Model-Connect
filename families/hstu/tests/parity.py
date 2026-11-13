# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The HSTU parity judge: native logits against the reference's golden.

Numpy only (a target device's build environment need not have torch): the
golden was written on a host with the PyTorch reference
(``families.hstu.tests.golden``), and the tolerance was declared from the
reference's own BF16 spread before any TensorRT output was compared
(``tests/data/README.md``).

``python -m families.hstu.tests.parity (--probe <probe.json> | --observations
<observations.jsonl>) --golden <golden.json> --thresholds <thresholds.json>
--checkpoint <checkpoint_dir> --out <parity.json>`` judges every repeat (a task
probe's results, or each observation that ``trtmc-bench``'s ``decide``
operation recorded) and exits 0 when every one passes, 1 otherwise; the record
states both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import load_config
from ..weights import weights_record

SCHEMA = "trtmc.hstu.parity.v1"
#: The tolerance's quantities, as the case's threshold sidecar names them.
LIMITS = ("max_abs", "min_task_cosine")


def probe_logits(result: dict) -> np.ndarray:
    """One result's logits as [candidates, tasks], by task index."""
    scores = sorted(result["scores"], key=lambda score: int(score["question_id"].split("_")[1]))
    if [score["question_id"] for score in scores] != [f"task_{i}" for i in range(len(scores))]:
        raise ValueError("result tasks are not task_0..task_{n-1}")
    return np.asarray([score["logits"] for score in scores], dtype=np.float64).T


def compare(native: np.ndarray, golden: np.ndarray) -> dict:
    if native.shape != golden.shape:
        raise ValueError(f"native logits {native.shape} vs golden {golden.shape}")
    error = np.abs(native - golden)
    cosine = (native * golden).sum(0) / (
        np.linalg.norm(native, axis=0) * np.linalg.norm(golden, axis=0)
    )
    return {
        "max_abs": float(error.max()),
        "mean_abs": float(error.mean()),
        "min_task_cosine": float(cosine.min()),
    }


def judge(stats: dict, limits: dict) -> bool:
    return (
        stats["max_abs"] <= limits["max_abs"]
        and stats["min_task_cosine"] >= limits["min_task_cosine"]
    )


def checkpoint_of(model_dir: str | Path) -> dict:
    """A synthetic checkpoint's name and seed, as its config.json states."""
    config = load_config(model_dir)
    return {"name": config.raw["name"], "seed": weights_record(config)["seed"]}


def parity_record(native: dict, golden: dict, limits: dict, *, checkpoint: dict) -> dict:
    """Every repeat judged against the golden's FP32 logits under ``limits``."""
    if checkpoint != golden["checkpoint"]:
        raise ValueError(
            f"the bundle's checkpoint {checkpoint} is not the golden's {golden['checkpoint']}"
        )
    if set(limits) != set(LIMITS):
        raise ValueError(f"the tolerance names {sorted(limits)}, not {sorted(LIMITS)}")
    reference = np.asarray(golden["logits_fp32"], dtype=np.float64)
    repeats = []
    for result in native["results"]:
        stats = compare(probe_logits(result), reference)
        repeats.append(
            {
                **stats,
                "passed": judge(stats, limits),
                "public_task_call_wall_ms": result.get("public_task_call_wall_ms"),
            }
        )
    return {
        "schema": SCHEMA,
        "checkpoint": checkpoint,
        "tolerance": limits,
        "repeats": repeats,
        "passed": bool(repeats) and all(repeat["passed"] for repeat in repeats),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--probe", type=Path)
    source.add_argument("--observations", type=Path)
    for name in ("golden", "thresholds", "checkpoint", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.probe is not None:
        native = json.loads(args.probe.read_text(encoding="utf-8"))
    else:
        native = {
            "results": [
                {**row, "public_task_call_wall_ms": row.get("runtime_e2e_wall_ms")}
                for row in map(
                    json.loads, args.observations.read_text(encoding="utf-8").splitlines()
                )
                if row
            ]
        }
    overrides = json.loads(args.thresholds.read_text(encoding="utf-8"))["threshold_overrides"]
    record = parity_record(
        native,
        json.loads(args.golden.read_text(encoding="utf-8")),
        {name: overrides[name] for name in LIMITS},
        checkpoint=checkpoint_of(args.checkpoint),
    )
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("checkpoint", "tolerance", "passed")}))
    for repeat in record["repeats"]:
        print(json.dumps(repeat))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
