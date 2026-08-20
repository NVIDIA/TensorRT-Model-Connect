# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the official PyTorch ELF reference environment."""

from importlib.metadata import version

import colorama  # noqa: F401
import einops
import huggingface_hub
import sacrebleu  # noqa: F401
import tokenizers
import transformers


assert transformers.__version__ == "4.44.2"
assert tokenizers.__version__ == "0.19.1"
assert huggingface_hub.__version__ == "0.24.7"
assert version("colorama") == "0.4.6"
assert einops.__version__ == "0.8.1"
assert version("sacrebleu") == "2.5.1"
assert version("portalocker") == "3.2.0"
assert version("tabulate") == "0.10.0"
assert version("lxml") == "6.1.1"
