# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import backoff
import peft
import transformers
from transformers.cache_utils import SlidingWindowCache


assert version("huggingface-hub") == "1.5.0"
assert version("peft") == "0.13.2"
assert version("tokenizers") == "0.22.2"
assert transformers.__version__ == "5.5.4"
assert peft.__version__ == "0.13.2"
assert SlidingWindowCache is not None
print(f"backoff={backoff.__version__} transformers={transformers.__version__}")
