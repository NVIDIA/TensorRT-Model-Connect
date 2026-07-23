# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata as metadata

import cv2
import imageio
import numpy
from flash_attn import flash_attn_varlen_func
from transformers.cache_utils import SlidingWindowCache

assert metadata.version("flash-attn") == "2.8.3"
assert metadata.version("huggingface-hub") == "0.29.1"
assert metadata.version("imageio") == "2.34.0"
assert metadata.version("numpy") == "1.26.4"
assert metadata.version("opencv-python") == "4.7.0.72"
assert metadata.version("tokenizers") == "0.21.4"
assert metadata.version("transformers") == "4.49.0"
assert callable(cv2.imread)
assert callable(imageio.imread)
assert numpy.__version__ == "1.26.4"
assert callable(flash_attn_varlen_func)
assert SlidingWindowCache is not None
print(
    f"flash-attn={metadata.version('flash-attn')} "
    f"transformers={metadata.version('transformers')}"
)
