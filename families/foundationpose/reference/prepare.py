# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the FoundationPose model directory the bundles build from and the native reference runs: the NGC ONNX
refiner and scorer the catalog manifests declare (``external_files``), under
``<environment>/trtmc-reference/foundationpose-ngc``."""

from pathlib import Path
import json
import shutil
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
DESTINATION = Path(sys.prefix) / "trtmc-reference" / "foundationpose-ngc"


def main() -> int:
    manifest = json.loads((ROOT / "families/foundationpose/tests/manifests/foundationpose-ngc-1.0.1.json").read_text())
    files = manifest["external_files"]
    if all((DESTINATION / entry["path"]).is_file() for entry in files):
        return 0
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="foundationpose.", dir=DESTINATION.parent))
    try:
        for entry in files:
            with urllib.request.urlopen(entry["url"], timeout=600) as response, (work / entry["path"]).open("wb") as out:
                shutil.copyfileobj(response, out)
        shutil.rmtree(DESTINATION, ignore_errors=True)
        work.rename(DESTINATION)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
