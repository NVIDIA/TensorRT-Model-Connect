# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the official PyTorch ELF reference environment."""

import huggingface_hub
import tokenizers
import transformers


assert transformers.__version__ == "4.44.2"
assert tokenizers.__version__ == "0.19.1"
assert huggingface_hub.__version__ == "0.24.7"
