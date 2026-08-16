/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

namespace trtmc {

cudaError_t launch_fast_foundation_stereo_gwc(const void* reference, const void* target,
                                              void* reference_norm, void* target_norm, void* output,
                                              cudaStream_t stream);

} // namespace trtmc
