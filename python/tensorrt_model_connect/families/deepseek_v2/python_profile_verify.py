# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the DeepSeek-V2 tokenizer-compatible Python environment."""

from importlib.metadata import version

import transformers
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM


assert transformers.__version__ == "4.57.6"
assert version("tokenizers") == "0.22.2"
assert version("huggingface-hub") == "0.36.2"
assert DeepseekV2Config.model_type == "deepseek_v2"
assert DeepseekV2ForCausalLM.config_class is DeepseekV2Config
print(
    f"transformers={transformers.__version__} "
    f"tokenizers={version('tokenizers')} "
    f"huggingface-hub={version('huggingface-hub')} "
    "deepseek_v2=available"
)
