# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU recorded-control example checks using the actual family loader and C SDK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def bundle(path: Path, mode: str) -> None:
    header = json.dumps(
        {
            "format": 1,
            "family": "action_fixture",
            "task": mode,
            "backend": "fake",
            "sections": {"engine.plan": {"offset": 0, "length": 4}},
        }
    ).encode()
    path.write_bytes(b"BUNDLE\x01\x00" + struct.pack("<Q", len(header)) + header + b"PLAN")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="trtmc-recorded-example-") as temporary:
        root = Path(temporary)
        image = root / "frame.ppm"
        image.write_bytes(b"P6\n640 480\n255\n" + bytes((255, 0, 0)) * (640 * 480))
        state = root / "state.f32"
        state.write_bytes(struct.pack("=14f", *range(14)))

        def run(mode: str) -> subprocess.CompletedProcess[str]:
            path = root / f"{mode}.bundle"
            bundle(path, mode)
            return subprocess.run(
                [
                    str(options.binary),
                    str(path),
                    "--image",
                    str(image),
                    "--state",
                    str(state),
                    "--runtime-root",
                    str(options.runtime_root),
                    "--control-hz",
                    "1000000",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )

        for mode in ("example_recorded", "robot_control"):
            completed = run(mode)
            assert completed.returncode == 0, completed.stderr
            rows = completed.stdout.splitlines()
            assert len(rows) == 100
            for index, row in enumerate(rows):
                step, values = row.split(",", 1)
                assert int(step) == index
                assert [float(value) for value in values.split()] == [
                    100 * index + joint - 700 for joint in range(14)
                ]
            assert "inference_ms=101; within_training_bounds=false" in completed.stderr
            assert "Emitted 100 recorded-replay actions" in completed.stderr
        for mode in (
            "example_recorded_bad_shape",
            "example_recorded_unsupported",
            "example_recorded_fail",
        ):
            completed = run(mode)
            assert completed.returncode != 0 and completed.stdout == "", mode
            if mode == "example_recorded_fail":
                assert "recorded fixture provider failed" in completed.stderr
        state.write_bytes(struct.pack("=13f", *range(13)))
        completed = run("example_recorded")
        assert completed.returncode != 0 and completed.stdout == ""
        assert "exactly 14" in completed.stderr
        state.write_bytes(struct.pack("=14f", *range(14)))
        image.write_bytes(b"P6\n1 1\n255\n" + bytes((255, 0, 0)))
        completed = run("example_recorded")
        assert completed.returncode != 0 and completed.stdout == ""
        assert "640x480" in completed.stderr
    print("ALL PASSED")


if __name__ == "__main__":
    main()
