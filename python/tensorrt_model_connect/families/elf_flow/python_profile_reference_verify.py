# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the official PyTorch ELF reference environment."""

import huggingface_hub
import tokenizers
import transformers


assert transformers.__version__ == "5.5.0"
assert tokenizers.__version__ == "0.22.2"
assert huggingface_hub.__version__ == "1.26.0"
