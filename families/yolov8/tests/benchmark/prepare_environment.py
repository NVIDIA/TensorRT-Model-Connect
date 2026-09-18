# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the Ultralytics family environment independent of system OpenCV."""

from importlib import metadata
import subprocess
import sys


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "--yes", "opencv-python"],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--force-reinstall",
            "--no-deps",
            "opencv-python-headless==4.10.0.84",
        ],
        check=True,
    )
    if metadata.version("opencv-python-headless") != "4.10.0.84":
        raise RuntimeError("the headless OpenCV environment was not prepared")


if __name__ == "__main__":
    main()
