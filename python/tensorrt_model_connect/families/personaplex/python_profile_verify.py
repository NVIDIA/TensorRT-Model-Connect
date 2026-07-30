# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import sphn

assert version("sphn") == "0.1.4"
assert sphn is not None
print(f"sphn={version('sphn')}")
