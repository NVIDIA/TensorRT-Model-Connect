/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace trtmc {

void qwen3_omni_gpu_argmax(const float* logits, int32_t vocab_size, int32_t* token_id,
                           cudaStream_t stream);

} // namespace trtmc
