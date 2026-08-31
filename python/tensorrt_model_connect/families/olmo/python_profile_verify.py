# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the OLMo tokenizer-compatible build and reference environment."""

from importlib.metadata import version

import transformers
from transformers import AutoTokenizer


assert transformers.__version__ == "5.2.0"
assert version("tokenizers") == "0.22.2"
assert version("huggingface-hub") == "1.22.0"
assert AutoTokenizer is not None
print(
    f"transformers={transformers.__version__} "
    f"tokenizers={version('tokenizers')} "
    f"huggingface-hub={version('huggingface-hub')}"
)
