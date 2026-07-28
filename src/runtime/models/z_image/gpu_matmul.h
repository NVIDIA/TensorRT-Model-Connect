/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>

namespace trtmc {

bool z_image_should_use_gpu_matmul(int32_t rows, int32_t inner, int32_t columns);

class ZImageGpuMatmul {
  public:
    ZImageGpuMatmul();
    ~ZImageGpuMatmul();

    ZImageGpuMatmul(const ZImageGpuMatmul&) = delete;
    ZImageGpuMatmul& operator=(const ZImageGpuMatmul&) = delete;

    bool run(const float* lhs, const float* rhs, const float* bias, float* output, int32_t rows,
             int32_t inner, int32_t columns);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace trtmc
