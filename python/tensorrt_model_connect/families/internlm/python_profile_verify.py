# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import einops
import tiktoken
import transformers
from transformers.cache_utils import DynamicCache

assert hasattr(DynamicCache, "from_legacy_cache")
assert hasattr(DynamicCache, "get_max_cache_shape")
assert version("tiktoken") == "0.12.0"
assert callable(tiktoken.get_encoding)
print(f"einops={einops.__version__} transformers={transformers.__version__}")
