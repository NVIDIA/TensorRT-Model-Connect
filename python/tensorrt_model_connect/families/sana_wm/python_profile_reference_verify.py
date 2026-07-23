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
assert callable(torch.cuda.is_available)
assert hasattr(torchvision.transforms, "Compose")
assert hasattr(diffusers, "DiffusionPipeline")
assert hasattr(transformers, "AutoTokenizer")
assert callable(yaml.safe_load)
