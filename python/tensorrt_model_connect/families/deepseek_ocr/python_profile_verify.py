# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention

assert transformers.__version__ == "5.5.0"
assert hasattr(DynamicCache, "get_max_cache_shape")
assert LlamaAttention is not None
print(f"transformers={transformers.__version__}")
