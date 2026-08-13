# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the DeepSeek-V2 tokenizer-compatible Python environment."""

from importlib.metadata import version

import transformers
from transformers import (
    DeepseekV2Config,
    DeepseekV2ForCausalLM,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
)


assert transformers.__version__ == "5.2.0"
assert version("tokenizers") == "0.22.2"
assert version("huggingface-hub") == "1.22.0"
assert DeepseekV2Config.model_type == "deepseek_v2"
assert DeepseekV2ForCausalLM.config_class is DeepseekV2Config
assert DeepseekV3ForCausalLM.config_class is DeepseekV3Config
deepseek_v3_config = DeepseekV3Config(experts_implementation="batched_mm")
assert deepseek_v3_config._experts_implementation == "batched_mm"
print(
    f"transformers={transformers.__version__} "
    f"tokenizers={version('tokenizers')} "
    f"huggingface-hub={version('huggingface-hub')} "
    "deepseek_v2=available experts_implementation=batched_mm"
)
