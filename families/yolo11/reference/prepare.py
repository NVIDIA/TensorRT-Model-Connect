# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the native reference environment: Ultralytics pulls the GUI OpenCV build, which shadows the
pinned headless one the family requires; keep only the headless build."""

from importlib import metadata
import subprocess
import sys

HEADLESS = "opencv-python-headless==4.10.0.84"


def main() -> None:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "opencv-python"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--force-reinstall",
                    "--no-deps", HEADLESS], check=True)
    if metadata.version("opencv-python-headless") != HEADLESS.split("==")[1]:
        raise RuntimeError("the headless OpenCV environment was not prepared")


if __name__ == "__main__":
    main()
