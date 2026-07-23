# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata as metadata

import sphn

assert metadata.version("sphn") == "0.1.12"
assert callable(sphn.read)
print(f"sphn={metadata.version('sphn')}")
