# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify dependencies imported by the Qwen-Image HF reference."""

from importlib.metadata import version

import accelerate
import diffusers
import ftfy
from diffusers.utils import is_torch_available, is_transformers_available


assert version("accelerate") == "1.14.0"
assert version("diffusers") == "0.39.0"
assert version("ftfy") == "6.3.1"
assert version("wcwidth") == "0.8.2"
assert accelerate is not None
if not is_torch_available() or not is_transformers_available():
    raise RuntimeError("QwenImagePipeline requires PyTorch and Transformers backends")
pipeline = getattr(diffusers, "QwenImagePipeline", None)
if pipeline is None or "dummy" in pipeline.__module__.lower():
    raise RuntimeError(
        f"QwenImagePipeline resolved to an unavailable placeholder: {pipeline!r}"
    )
assert ftfy.fix_text("Qwen-Image") == "Qwen-Image"
