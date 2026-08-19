# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import transformers
from transformers.models.llama.modeling_llama import LlamaFlashAttention2

assert transformers.__version__ == "4.46.3"
assert LlamaFlashAttention2 is not None
print(f"transformers={transformers.__version__}")
