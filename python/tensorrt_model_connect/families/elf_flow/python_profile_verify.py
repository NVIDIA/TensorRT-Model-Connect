# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the family-owned ELF checkpoint-loading profile."""

import flax
import jax
import orbax.checkpoint
import yaml


assert flax.__version__ == "0.10.2"
assert jax.__version__ == "0.4.38"
assert hasattr(orbax.checkpoint, "PyTreeCheckpointer")
assert hasattr(yaml, "safe_load")
