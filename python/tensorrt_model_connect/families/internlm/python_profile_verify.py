# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import einops
import sentencepiece
import transformers
from transformers.cache_utils import DynamicCache

assert sentencepiece.__version__ == "0.2.0", sentencepiece.__version__
assert hasattr(DynamicCache, "from_legacy_cache")
assert hasattr(DynamicCache, "get_max_cache_shape")
print(
    f"einops={einops.__version__} "
    f"sentencepiece={sentencepiece.__version__} "
    f"transformers={transformers.__version__}"
)
