# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the actual example against the existing loaded-DSO speech fixture."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import json
import os
import subprocess
import tempfile


def bundle(path: Path, mode: str) -> None:
    header = json.dumps(
        {
            "format": 1,
            "family": "speech_fixture",
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
    binary = str(options.binary.resolve())
    runtime = str(options.runtime_root.resolve())
    sdk = options.sdk_library.resolve()
    prompts = b"  \nfirst prompt\n\nsecond prompt\n"
    with tempfile.TemporaryDirectory(prefix="trtmc-audio-example-") as temporary:
        root = Path(temporary)
        # CMake keeps the SDK in the build root and fixtures in api-runtime.
        # Reproduce an installed co-located layout instead of letting a scratch
        # directory accidentally make the default-root test pass.
        layout = root / "installed-lib"
        layout.mkdir()
        dependencies = {
            "libtrtmc_c.so.1": sdk,
            "libtrtmc_core.so": sdk.parent / "libtrtmc_core.so",
            "libtrtmc_runtime.so": sdk.parent / "libtrtmc_runtime.so",
            "libtrtmc_backend_fake.so": Path(runtime) / "libtrtmc_backend_fake.so",
            "libtrtmc_model_speech_fixture.so": Path(runtime) / "libtrtmc_model_speech_fixture.so",
        }
        for name, source in dependencies.items():
            assert source.is_file(), source
            (layout / name).symlink_to(source)
        installed_environment = dict(os.environ)
        installed_environment["LD_LIBRARY_PATH"] = str(layout) + (
            ":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""
        )

        def command(mode: str, *, defaults: bool = False) -> list[str]:
            path = root / f"{mode}.bundle"
            bundle(path, mode)
            arguments = [binary, str(path)]
            if not defaults:
                arguments += [
                    "--runtime-root",
                    runtime,
                    "--chunk-frames",
                    "2",
                    "--max-new-tokens",
                    "7",
                ]
            return arguments

        for defaults in (False, True):
            mode = "example_streaming_defaults" if defaults else "example_streaming"
            completed = subprocess.run(
                command(mode, defaults=defaults),
                input=prompts,
                capture_output=True,
                check=False,
                timeout=10,
                env=installed_environment if defaults else None,
            )
            assert completed.returncode == 0, completed.stderr.decode()
            assert completed.stdout == struct.pack("=7f", 0.25, -0.5, 0.75, 0, -1, 0.5, 0)
            status = completed.stderr.decode()
            assert "utterance 1 done; samples=3; sample_rate=22050; reported_samples=3" in status
            assert "utterance 2 done; samples=2; sample_rate=22050; reported_samples=2" in status
            assert "EOF; utterances=2" in status

        for mode in (
            "example_streaming_unsupported",
            "example_streaming_fail",
            "example_streaming_stereo",
            "example_streaming_bad_rate",
            "example_streaming_empty",
        ):
            completed = subprocess.run(
                command(mode),
                input=b"first prompt\n",
                capture_output=True,
                check=False,
                timeout=10,
            )
            assert completed.returncode != 0, mode
            assert completed.stdout == b"", mode
            assert b"utterance 1 done" not in completed.stderr, mode
            assert b"does not implement streaming audio generation" not in completed.stderr, mode
            if mode == "example_streaming_fail":
                assert b"streaming example provider failed" in completed.stderr

        # The callback's write failure must cross C and terminate the example;
        # no completed utterance may be reported and no old Task may be retried.
        if Path("/dev/full").exists():
            with Path("/dev/full").open("wb") as output:
                completed = subprocess.run(
                    command("example_streaming"),
                    input=b"first prompt\n",
                    stdout=output,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=10,
                )
            assert completed.returncode != 0
            assert b"FP32 PCM" in completed.stderr
            assert b"utterance 1 done" not in completed.stderr
        else:
            print("Output-failure check unrun: /dev/full is unavailable.")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
