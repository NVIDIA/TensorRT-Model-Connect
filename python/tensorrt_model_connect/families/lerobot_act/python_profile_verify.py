# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata as metadata

import imageio_ffmpeg
import numpy
import pyarrow
import pyarrow.parquet
import safetensors
import safetensors.torch
import torch
import torchvision
from PIL import Image


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"LeRobot ACT reference profile check failed: {message}")


_require(
    metadata.version("imageio-ffmpeg") == "0.6.0",
    f"imageio-ffmpeg=={metadata.version('imageio-ffmpeg')}",
)
_require(imageio_ffmpeg.get_ffmpeg_exe(), "bundled FFmpeg executable is unavailable")
_require(
    tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 4),
    f"torch {torch.__version__} is older than 2.4",
)
_require(callable(torchvision.models.resnet18), "torchvision ResNet-18 is unavailable")
_require(callable(pyarrow.parquet.read_table), "pyarrow parquet support is unavailable")
_require(callable(safetensors.torch.load_model), "safetensors torch loading is unavailable")
_require(callable(Image.open), "Pillow image loading is unavailable")
print(
    f"torch={torch.__version__} torchvision={torchvision.__version__} "
    f"numpy={numpy.__version__} pyarrow={pyarrow.__version__} "
    f"imageio-ffmpeg={metadata.version('imageio-ffmpeg')}"
)
