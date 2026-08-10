# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import einops
import transformers
from transformers.cache_utils import DynamicCache


if not hasattr(DynamicCache, "from_legacy_cache"):

    @classmethod
    def _from_legacy_cache(cls, past_key_values=None):
        return cls(past_key_values) if past_key_values else cls()

    DynamicCache.from_legacy_cache = _from_legacy_cache

assert version("einops") == "0.8.1"
assert version("huggingface-hub") == "1.5.0"
assert version("tokenizers") == "0.22.2"
assert transformers.__version__ == "5.5.4"
assert callable(DynamicCache.get_max_cache_shape)
assert isinstance(DynamicCache.from_legacy_cache(), DynamicCache)
print(f"einops={einops.__version__} transformers={transformers.__version__}")
