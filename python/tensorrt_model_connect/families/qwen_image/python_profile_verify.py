# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify dependencies imported by the Qwen-Image HF reference."""

from importlib.metadata import version

import accelerate
import diffusers
import ftfy


assert version("accelerate") == "1.14.0"
assert version("diffusers") == "0.39.0"
assert version("ftfy") == "6.3.1"
assert version("wcwidth") == "0.8.2"
assert accelerate is not None
assert getattr(diffusers, "QwenImagePipeline", None) is not None
assert ftfy.fix_text("Qwen-Image") == "Qwen-Image"
