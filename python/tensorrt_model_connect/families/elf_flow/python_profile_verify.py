# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the family-owned ELF checkpoint-loading profile."""

from importlib.metadata import version

import flax
import jax
import numpy
import optax
import orbax.checkpoint
import yaml


assert flax.__version__ == "0.10.2"
assert jax.__version__ == "0.4.38"
assert version("absl-py") == "2.5.0"
assert version("msgpack") == "1.2.1"
assert numpy.__version__ == "1.26.4"
assert optax.__version__ == "0.2.5"
assert hasattr(orbax.checkpoint, "PyTreeCheckpointer")
assert hasattr(yaml, "safe_load")
