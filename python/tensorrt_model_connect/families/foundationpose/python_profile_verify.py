# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import onnxruntime

if onnxruntime.__version__ != "1.29.0":
    raise RuntimeError(
        f"FoundationPose reference requires onnxruntime 1.29.0, got {onnxruntime.__version__}"
    )
