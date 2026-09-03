# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import timm

assert version("timm") == "1.0.28"
assert timm.__version__ == "1.0.28"
assert callable(timm.create_model)
print(f"timm={timm.__version__} create_model=ok")
