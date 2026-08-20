# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the Nemotron Labs Diffusion LoRA reference profile."""

from importlib.metadata import version

import accelerate
import peft
import safetensors
import torch
import transformers


assert version("accelerate") == "1.14.0"
assert version("huggingface-hub") == "1.22.0"
assert version("peft") == "0.20.0"
assert version("safetensors") == "0.8.0"
assert version("tokenizers") == "0.22.2"
assert transformers.__version__ == "5.2.0"
assert accelerate is not None
assert peft is not None
assert safetensors is not None
assert callable(torch.cuda.is_available)
