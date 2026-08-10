# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import backoff
import jinja2
import transformers
from transformers.cache_utils import SlidingWindowCache

assert transformers.__version__ == "4.48.2"
assert version("jinja2") == "3.1.6"
assert jinja2 is not None
assert SlidingWindowCache is not None
print(
    f"backoff={backoff.__version__} "
    f"jinja2={version('jinja2')} "
    f"transformers={transformers.__version__}"
)
