# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata as metadata

import cv2
import imageio
import numpy
import scipy
import sklearn
from transformers.cache_utils import DynamicCache

try:
    metadata.version("flash-attn")
except metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("the vulnerable flash-attn distribution must not be installed")
assert metadata.version("huggingface-hub") == "1.26.0"
assert metadata.version("imageio") == "2.34.0"
assert metadata.version("numpy") == "1.26.4"
assert metadata.version("opencv-python-headless") == "4.8.1.78"
assert metadata.version("scikit-learn") == "1.5.0"
assert metadata.version("scipy") == "1.12.0"
assert metadata.version("tokenizers") == "0.22.2"
assert metadata.version("transformers") == "5.5.0"
assert callable(cv2.imread)
assert callable(imageio.imread)
assert numpy.__version__ == "1.26.4"
assert scipy.__version__ == "1.12.0"
assert sklearn.__version__ == "1.5.0"
assert hasattr(DynamicCache, "get_max_cache_shape")
print(
    f"transformers={metadata.version('transformers')}"
)
