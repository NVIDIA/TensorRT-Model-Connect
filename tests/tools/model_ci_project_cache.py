# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse an immutable source projection across model-proof runner tests."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _value(command: list[str], option: str) -> str:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit(f"cached project command is missing {option}") from error


def main() -> int:
    if len(sys.argv) < 5 or sys.argv[1] != "--cache-root" or sys.argv[3] != "--":
        raise SystemExit(
            "usage: model_ci_project_cache.py --cache-root PATH -- MODEL_CI project ..."
        )
    cache_root = Path(sys.argv[2])
    command = sys.argv[4:]
    model = _value(command, "--model")
    revision = _value(command, "--revision")
    output = Path(_value(command, "--output-dir"))
    key = hashlib.sha256(f"{Path.cwd().resolve()}\0{revision}\0{model}".encode()).hexdigest()
    cached = cache_root / key
    lock_path = cache_root / f"{key}.lock"
    cache_root.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not cached.is_dir():
            temporary = cache_root / f"{key}.tmp.{os.getpid()}"
            shutil.rmtree(temporary, ignore_errors=True)
            projected = list(command)
            projected[projected.index("--output-dir") + 1] = str(temporary)
            result = subprocess.run(
                [sys.executable, *projected],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode:
                sys.stdout.write(result.stdout)
                sys.stderr.write(result.stderr)
                shutil.rmtree(temporary, ignore_errors=True)
                return result.returncode
            manifest = temporary / ".trtmc-model-projection.json"
            if not manifest.is_file():
                shutil.rmtree(temporary, ignore_errors=True)
                raise SystemExit("cached project command did not produce a projection manifest")
            temporary.replace(cached)

    shutil.rmtree(output, ignore_errors=True)
    shutil.copytree(cached, output, symlinks=True, copy_function=os.link)
    manifest = json.loads((cached / ".trtmc-model-projection.json").read_text(encoding="utf-8"))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
