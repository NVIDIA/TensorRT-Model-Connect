# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import einops
import transformers
from transformers.cache_utils import DynamicCache

assert hasattr(DynamicCache, "from_legacy_cache")
assert hasattr(DynamicCache, "get_max_cache_shape")
print(f"einops={einops.__version__} transformers={transformers.__version__}")
