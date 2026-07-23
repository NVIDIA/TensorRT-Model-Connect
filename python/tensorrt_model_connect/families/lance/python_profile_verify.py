# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata as metadata

from flash_attn import flash_attn_varlen_func

assert metadata.version("flash-attn") == "2.8.3"
assert callable(flash_attn_varlen_func)
print(f"flash-attn={metadata.version('flash-attn')}")
