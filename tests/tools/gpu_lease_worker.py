# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess worker for focused cross-process GPU lease tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tools.ci.context import CiContext
from tools.ci.gpu_lease import GpuLease
from tools.ci.process import CiError


REPO_ROOT = Path(__file__).resolve().parents[2]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--resource-class", choices=("shared", "exclusive_gpu"), required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--release-file", type=Path)
    parser.add_argument("--revision", default="test-revision")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    lease = GpuLease(
        CiContext(REPO_ROOT, os.environ.copy()),
        args.model,
        args.resource_class,
        args.artifacts_dir,
    )
    try:
        lease.acquire()
        evidence = lease.evidence(args.revision)
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (args.artifacts_dir / "gpu-id.txt").write_text(
            str(evidence["gpu_id"]) + "\n", encoding="utf-8"
        )
        (args.artifacts_dir / "gpu-lease.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
        if args.release_file is not None:
            deadline = time.monotonic() + 600
            while not args.release_file.exists():
                if time.monotonic() >= deadline:
                    raise CiError(f"timed out waiting for release file: {args.release_file}")
                time.sleep(0.01)
        return 0
    except CiError as error:
        print(error, file=sys.stderr)
        return 2
    finally:
        lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
