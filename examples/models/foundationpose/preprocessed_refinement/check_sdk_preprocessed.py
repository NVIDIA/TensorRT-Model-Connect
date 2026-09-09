# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU callback/output checks for the actual preprocessed-refinement example."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def bundle(path: Path, mode: str) -> None:
    header = json.dumps(
        {
            "format": 1,
            "family": "perception_fixture",
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
    parser.add_argument("--sdk-library", type=Path, required=True)
    options = parser.parse_args()
    sdk, runtime = options.sdk_library.resolve(), options.runtime_root.resolve()
    with tempfile.TemporaryDirectory(prefix="trtmc-preprocessed-example-") as temporary:
        root = Path(temporary)
        inputs = root / "inputs"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("prepare_synthetic_inputs.py")),
                str(inputs),
            ],
            check=True,
            timeout=20,
        )
        # Distinguish each hypothesis and both refinement iterations using
        # valid deterministic crop values, without changing the example itself.
        for name, scale in (("rendered_features.f32", 0.1), ("observed_features.f32", 0.05)):
            path = inputs / name
            data = bytearray(path.read_bytes())
            for index in range(3):
                struct.pack_into("=f", data, index * 160 * 160 * 6 * 4, scale * (index + 1))
            path.write_bytes(data)
        original = list(struct.unpack("=48f", (inputs / "candidate_poses.f32").read_bytes()))
        expected = original.copy()
        for index in range(3):
            expected[index * 16 + 3] += 2 * 0.18 * 0.1 * (index + 1)
            expected[index * 16 + 7] += 2 * 0.05 * (index + 1)

        layout = root / "installed-lib"
        layout.mkdir()
        for name, source in {
            "libtrtmc_c.so.1": sdk,
            "libtrtmc_core.so": sdk.parent / "libtrtmc_core.so",
            "libtrtmc_runtime.so": sdk.parent / "libtrtmc_runtime.so",
            "libtrtmc_backend_fake.so": runtime / "libtrtmc_backend_fake.so",
            "libtrtmc_model_perception_fixture.so": runtime
            / "libtrtmc_model_perception_fixture.so",
        }.items():
            assert source.is_file(), source
            (layout / name).symlink_to(source)
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = str(layout) + (
            ":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else ""
        )

        def run(
            mode: str, *, omit_root: bool = False
        ) -> tuple[subprocess.CompletedProcess[str], Path]:
            path = root / f"{mode}.bundle"
            output = root / f"{mode}.f32"
            bundle(path, mode)
            command = [str(options.binary), str(path)]
            if not omit_root:
                command.append(str(runtime))
            command += [str(inputs), str(output)]
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
                env=environment if omit_root else None,
            ), output

        for mode, omit_root in (
            ("pose_only", False),
            ("pose_only", True),
            ("pose_hypothesis_refinement", False),
        ):
            completed, output = run(mode, omit_root=omit_root)
            assert completed.returncode == 0, completed.stderr
            actual = struct.unpack("=48f", output.read_bytes())
            assert all(
                math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
                for a, b in zip(actual, expected, strict=True)
            )
            assert completed.stdout == "best_index=2 score=0.6 rigid=true\n"
        for mode in (
            "example_pose_unsupported",
            "example_pose_fail",
            "example_pose_missing_score",
            "example_pose_bad_result",
        ):
            completed, output = run(mode)
            assert completed.returncode != 0 and not output.exists(), mode
            assert completed.stdout == "", mode
            if mode == "example_pose_fail":
                assert "preprocessed fixture provider failed" in completed.stderr
        prior = (root / "pose_only.f32").read_bytes()
        (inputs / "candidate_poses.f32").write_bytes(b"short")
        completed, output = run("pose_only")
        assert completed.returncode != 0 and output.read_bytes() == prior
        assert "unexpected size" in completed.stderr
    print("ALL PASSED")


if __name__ == "__main__":
    main()
