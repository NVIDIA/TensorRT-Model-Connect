# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata as metadata

import numpy
import scipy
import torch
from PIL import Image


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"MoGe reference profile check failed: {message}")


for package, expected in (
    ("huggingface-hub", "0.29.1"),
    ("numpy", "1.26.4"),
    ("Pillow", "12.2.0"),
    ("scipy", "1.12.0"),
):
    installed = metadata.version(package)
    _require(installed == expected, f"{package}=={installed}, expected {expected}")

_require(numpy.__version__ == "1.26.4", f"numpy runtime {numpy.__version__}")
_require(scipy.__version__ == "1.12.0", f"scipy runtime {scipy.__version__}")
torch_minor = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
_require(torch_minor >= (2, 4), f"torch {torch.__version__} is older than 2.4")
_require(callable(Image.open), "Pillow cannot open images")
print(
    f"torch={torch.__version__} numpy={numpy.__version__} scipy={scipy.__version__}"
)
