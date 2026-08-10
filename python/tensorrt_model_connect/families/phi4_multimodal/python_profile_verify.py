# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import backoff
import transformers
from transformers.cache_utils import DynamicCache, StaticCache

assert transformers.__version__ == "5.5.0"
assert hasattr(DynamicCache, "get_max_cache_shape")
assert StaticCache is not None
print(f"backoff={backoff.__version__} transformers={transformers.__version__}")
