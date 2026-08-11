# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the official SANA-WM reference execution profile."""

from importlib.metadata import version

import diffusers
import torch
import torchvision
import transformers
import yaml


assert version("PyYAML") == "6.0.3"
assert transformers.__version__ == "5.2.0", transformers.__version__
assert version("huggingface-hub") == "1.22.0"
assert version("tokenizers") == "0.22.2"
assert callable(torch.cuda.is_available)
assert hasattr(torchvision.transforms, "Compose")
assert hasattr(diffusers, "DiffusionPipeline")
assert hasattr(transformers, "AutoTokenizer")
assert callable(yaml.safe_load)
