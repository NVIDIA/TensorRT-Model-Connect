# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the dependency ABI used by the SAM3 compiled tracker bridge."""

import tensorrt
import torch
import transformers
import tvm_ffi

assert torch.__version__ == "2.12.0+cu130"
assert torch.version.cuda == "13.0"
assert torch._C._GLIBCXX_USE_CXX11_ABI is True
assert transformers.__version__ == "5.2.0"
assert tvm_ffi.__version__ == "0.1.12"
assert tensorrt.__version__ == "11.2.0.113"

# This profile is materialized while the CI image is built without device
# access.  The tracker exporter performs the live-device and compute-capability
# checks when a user actually builds a SAM3 bundle.

print(
    f"torch={torch.__version__} transformers={transformers.__version__} "
    f"tvm_ffi={tvm_ffi.__version__} tensorrt={tensorrt.__version__}"
)
