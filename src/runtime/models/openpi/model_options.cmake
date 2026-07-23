# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(TRTMC_MODEL_PROOF_MODEL STREQUAL "openpi")
  # OpenPI's qualified runtime contains only Model Connect C++, CUDA, and TensorRT.
  set(TRTMC_ENABLE_LIBTORCH_MULTINOMIAL OFF CACHE BOOL
    "Disable libtorch in the OpenPI native runtime" FORCE)
  set(TRTMC_ENABLE_TVM_FFI OFF CACHE BOOL
    "Disable TVM-FFI in the OpenPI native runtime" FORCE)
endif()
