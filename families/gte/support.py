# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for gte.

The ``model_type`` this checkpoint declares is the bare string ``"new"``.
That is not a placeholder to be tidied up: it is the identifier the GTE-v1.5
configs ship with, the one ``auto_map`` keys off, and the one a cached
checkpoint will present offline. The architecture name is ``NewModel`` for
the same reason.
"""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("new",),
    architectures=("NewModel",),
    tasks=("text_to_embedding",),
    default_task="text_to_embedding",
)
