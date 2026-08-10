# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the Nemotron-H checkpoint's declared reference environment."""

from importlib.metadata import version

import causal_conv1d  # noqa: F401
import transformers
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn  # noqa: F401


assert transformers.__version__ == "5.5.0"
assert version("causal-conv1d") == "1.6.2.post1"
assert version("mamba-ssm") == "2.3.2.post1"
print(
    f"transformers={transformers.__version__} "
    f"causal-conv1d={version('causal-conv1d')} "
    f"mamba-ssm={version('mamba-ssm')}")
