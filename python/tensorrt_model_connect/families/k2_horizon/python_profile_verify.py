# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import safetensors
import transformers

expected = {
    "transformers": "5.15.0",
    "safetensors metadata": "0.8.0",
    "safetensors module": "0.8.0",
}
observed = {
    "transformers": transformers.__version__,
    "safetensors metadata": version("safetensors"),
    "safetensors module": safetensors.__version__,
}
mismatches = [
    f"{name}: expected {expected[name]}, observed {value}"
    for name, value in observed.items()
    if value != expected[name]
]
if mismatches:
    raise RuntimeError("K2-Horizon reference profile mismatch: " + "; ".join(mismatches))
print(f"transformers={transformers.__version__} safetensors={safetensors.__version__}")
