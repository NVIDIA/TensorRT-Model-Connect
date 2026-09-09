# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse or create the isolated Chronos-Bolt build and reference environment."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT.parent.parent / "requirements.txt"
VERIFICATION = ROOT / "verify_environment.py"


def _compatible(python: str) -> bool:
    return (
        subprocess.run(
            [python, str(VERIFICATION)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    request = json.loads(arguments.request.read_text(encoding="utf-8"))
    common = str(request["common_python"])
    selected = common
    if not _compatible(common):
        root = Path(request["environment_directory"])
        receipt = root / "environment.json"
        if receipt.is_file():
            previous = str(json.loads(receipt.read_text(encoding="utf-8"))["python"])
            if _compatible(previous):
                selected = previous
        if selected == common:
            if request.get("allow_create") is not True:
                raise RuntimeError(
                    "Chronos-Bolt requires its pinned family environment; "
                    "enable environment creation during preparation"
                )
            root.mkdir(parents=True, exist_ok=True)
            target = Path(tempfile.mkdtemp(prefix="chronos-", dir=root))
            subprocess.run(
                [common, "-m", "venv", "--system-site-packages", str(target)], check=True
            )
            selected = str(target / "bin" / "python")
            subprocess.run(
                [
                    selected,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--no-build-isolation",
                    "--requirement",
                    str(REQUIREMENTS),
                ],
                check=True,
            )
            if not _compatible(selected):
                raise RuntimeError("prepared Chronos-Bolt environment failed verification")
            receipt.write_text(json.dumps({"python": selected}) + "\n", encoding="utf-8")
    arguments.output.write_text(
        json.dumps({"python": selected, "reference_python": selected}) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
