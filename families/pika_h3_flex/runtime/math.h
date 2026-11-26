/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime_api.h>

namespace trtmc::pika_h3_flex {
void initialize_math();
cudaStream_t execution_stream();
} // namespace trtmc::pika_h3_flex

extern "C" const char* trtmc_pika_h3_flex_initialize_math() noexcept;
