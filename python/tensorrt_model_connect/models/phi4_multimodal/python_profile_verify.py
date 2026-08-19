# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import backoff
import transformers
from transformers.cache_utils import SlidingWindowCache

assert transformers.__version__ == "4.48.2"
assert SlidingWindowCache is not None
print(f"backoff={backoff.__version__} transformers={transformers.__version__}")
