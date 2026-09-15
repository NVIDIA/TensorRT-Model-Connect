# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse the common GPT-2 reference environment, or prepare an isolated CUDA reference."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def _compatible(python: str) -> bool:
    code = (
        "import torch; from transformers import GPT2LMHeadModel, AutoTokenizer; "
        "assert torch.cuda.is_available(); assert callable(torch.compile); "
        "x = torch.ones(1, device='cuda'); assert x.item() == 1"
    )
    return subprocess.run([python, "-c", code], check=False).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    request = json.loads(arguments.request.read_text())
    common = request["common_python"]
    reference = common
    if not _compatible(common):
        root = Path(request["environment_directory"])
        receipt = root / "reference.json"
        if receipt.is_file():
            previous = json.loads(receipt.read_text())["python"]
            if _compatible(previous):
                reference = previous
        if reference == common:
            if request["allow_create"] is not True:
                raise RuntimeError(
                    "GPT-2 needs a CUDA reference; enable environment creation during preparation"
                )
            root.mkdir(parents=True, exist_ok=True)
            target = Path(tempfile.mkdtemp(prefix="reference-", dir=root))
            subprocess.run([common, "-m", "venv", str(target)], check=True)
            reference = str(target / "bin" / "python")
            subprocess.run(
                [
                    reference,
                    "-m",
                    "pip",
                    "install",
                    "torch==2.14.0",
                    "transformers==5.2.0",
                ],
                check=True,
            )
            if not _compatible(reference):
                raise RuntimeError("prepared GPT-2 reference failed its CUDA import check")
            receipt.write_text(json.dumps({"python": reference}) + "\n")
    arguments.output.write_text(
        json.dumps({"python": common, "reference_python": reference}) + "\n"
    )


if __name__ == "__main__":
    main()
