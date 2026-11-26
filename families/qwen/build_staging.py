# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep one Qwen build's private staging directory alive until its parent exits."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


@contextmanager
def worker_staging(output_dir: Path):
    # This process owns the temporary directory. It can clean up even when the
    # builder receives SIGKILL and cannot run Python finally blocks. The pipe's
    # write end belongs only to the builder, never to its worker subprocesses.
    with subprocess.Popen(
        [sys.executable, "-I", "-S", str(Path(__file__).resolve()), str(output_dir.resolve())],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=os.name == "posix",
    ) as owner:
        encoded = owner.stdout.readline()
        if not encoded:
            raise RuntimeError("Could not create the Qwen worker staging directory")
        yield Path(json.loads(encoded))
        owner.stdin.close()
        if owner.wait() != 0:
            raise RuntimeError("Could not remove the Qwen worker staging directory")


def _hold_directory(output_dir: str) -> None:
    stage = tempfile.TemporaryDirectory(prefix="trtmc-qwen-build-", dir=output_dir)
    try:
        print(json.dumps(stage.name), flush=True)
        sys.stdin.buffer.read(1)
    finally:
        # Windows forbids removal while a terminating worker still has an open
        # handle. Retry only this owner's unique directory for a bounded time.
        deadline = time.monotonic() + 60
        while True:
            try:
                stage.cleanup()
                break
            except PermissionError:
                if os.name != "nt" or time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)


if __name__ == "__main__":
    _hold_directory(sys.argv[1])
