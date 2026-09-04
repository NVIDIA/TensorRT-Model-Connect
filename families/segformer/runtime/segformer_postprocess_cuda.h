/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace trtmc {

// Bilinearly resize NCHW logits to the requested source geometry and select
// the highest-scoring class without materializing the resized logits tensor.
// All pointers refer to device memory and execution is asynchronous on stream.
cudaError_t launch_segformer_bilinear_argmax(const float* logits, int32_t num_classes,
                                             int32_t logits_h, int32_t logits_w, int32_t target_h,
                                             int32_t target_w, int32_t* class_map,
                                             cudaStream_t stream);

} // namespace trtmc
