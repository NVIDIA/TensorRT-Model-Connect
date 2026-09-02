# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the official SANA-WM reference execution profile."""

from importlib.metadata import version

import diffusers
import einops
import torch
import torchvision
import transformers
import yaml


assert version("PyYAML") == "6.0.3"
required_versions = {
    "transformers": "5.2.0",
    "huggingface-hub": "1.26.0",
    "tokenizers": "0.22.2",
    "einops": "0.8.2",
}
for distribution, expected in required_versions.items():
    actual = version(distribution)
    if actual != expected:
        raise RuntimeError(
            f"Sana-WM profile requires {distribution}=={expected}, found {actual}"
        )
assert callable(torch.cuda.is_available)
assert hasattr(torchvision.transforms, "Compose")
assert hasattr(diffusers, "DiffusionPipeline")
assert einops is not None
assert hasattr(transformers, "AutoTokenizer")
assert callable(yaml.safe_load)
