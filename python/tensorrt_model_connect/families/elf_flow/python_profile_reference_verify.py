# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the official PyTorch ELF reference environment."""

from importlib.metadata import version

import transformers


assert version("huggingface-hub") == "1.5.0"
assert version("tokenizers") == "0.22.2"
assert transformers.__version__ == "5.5.4"
