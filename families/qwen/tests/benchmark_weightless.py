# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure current Qwen builds, optionally against a separate baseline checkout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _source_identity(root: Path) -> dict:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return {
        "root": str(root),
        "head": result.stdout.strip() if result.returncode == 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--precisions", nargs="+", default=["bf16"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument(
        "--baseline-root", type=Path,
        help="Source checkout to use for baseline builds.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.model_dir = args.model_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.model_dir.is_dir():
        parser.error("model_dir must be an existing local checkpoint directory")
    current_root = Path(__file__).resolve().parents[3]
    roots = {"current": current_root}
    if args.baseline_root is not None:
        baseline = args.baseline_root.resolve()
        if baseline == current_root:
            parser.error("--baseline-root must name a separate source checkout")
        if not (baseline / "families/qwen/model.py").is_file() or not (
            baseline / "core/builder/tensorrt_model_connect/__main__.py"
        ).is_file():
            parser.error("--baseline-root does not contain the Qwen family and build CLI")
        roots = {"baseline": baseline, "candidate": current_root}
    sources = {arm: _source_identity(root) for arm, root in roots.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for precision in args.precisions:
        for repeat in range(args.repeats):
            arms = list(roots)
            if repeat % 2:
                arms.reverse()
            for arm in arms:
                label = f"{precision}-{len(records)}-{arm}"
                log_path = args.output_dir / f"{label}.log"
                bundle = args.output_dir / f"{label}.bundle"
                if bundle.exists() or log_path.exists():
                    raise FileExistsError(f"Refusing to overwrite benchmark artifacts: {label}")
                command = [sys.executable, "-m", "tensorrt_model_connect", "build",
                           str(args.model_dir), "--precision", precision, "--output", str(bundle)]
                if args.max_sequence_length is not None:
                    command.extend(["--max-sequence-length", str(args.max_sequence_length)])
                root = roots[arm]
                environment = dict(
                    os.environ,
                    PYTHONPATH=os.pathsep.join((str(root / "core/builder"), str(root))),
                    PYTHONDONTWRITEBYTECODE="1",
                )
                start = time.perf_counter()
                with log_path.open("w") as log:
                    process = subprocess.run(
                        command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    )
                record = {
                    "label": label, "precision": precision, "arm": arm,
                    "command": command, "returncode": process.returncode,
                    "wall_seconds": time.perf_counter() - start,
                    "source": sources[arm],
                }
                records.append(record)
                (args.output_dir / "benchmark.json").write_text(json.dumps(records, indent=2) + "\n")
                print(json.dumps(record), flush=True)
                if process.returncode:
                    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
