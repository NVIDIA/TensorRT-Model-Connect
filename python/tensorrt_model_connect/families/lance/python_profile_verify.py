# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata as metadata

import cv2
import imageio
import numpy
import scipy
import sklearn
from transformers.cache_utils import SlidingWindowCache

assert metadata.version("huggingface-hub") == "0.29.1"
assert metadata.version("imageio") == "2.34.0"
assert metadata.version("numpy") == "1.26.4"
assert metadata.version("opencv-python-headless") == "4.7.0.72"
assert metadata.version("scikit-learn") == "1.4.2"
assert metadata.version("scipy") == "1.12.0"
assert metadata.version("tokenizers") == "0.21.4"
assert metadata.version("transformers") == "4.49.0"
assert callable(cv2.imread)
assert callable(imageio.imread)
assert numpy.__version__ == "1.26.4"
assert scipy.__version__ == "1.12.0"
assert sklearn.__version__ == "1.4.2"
assert SlidingWindowCache is not None
print(
    f"attention=torch-sdpa transformers={metadata.version('transformers')}"
)
