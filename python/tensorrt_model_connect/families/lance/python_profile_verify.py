# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata as metadata
import platform

import cv2
import imageio
import numpy
import scipy
import sklearn
import torch.nn.functional as functional
from transformers.cache_utils import SlidingWindowCache

if platform.machine() == "aarch64":
    assert callable(functional.scaled_dot_product_attention)
    attention = "torch-sdpa"
else:
    from flash_attn import flash_attn_varlen_func

    assert metadata.version("flash-attn") == "2.8.3"
    assert callable(flash_attn_varlen_func)
    attention = f"flash-attn={metadata.version('flash-attn')}"
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
print(f"attention={attention} transformers={metadata.version('transformers')}")
