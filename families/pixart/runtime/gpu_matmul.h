/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>

namespace trtmc {

class PixArtGpuMatmul {
  public:
    PixArtGpuMatmul();
    ~PixArtGpuMatmul();

    PixArtGpuMatmul(const PixArtGpuMatmul&) = delete;
    PixArtGpuMatmul& operator=(const PixArtGpuMatmul&) = delete;

    bool run(const float* lhs, const float* rhs, const float* bias, float* output, int32_t rows,
             int32_t inner, int32_t columns);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace trtmc
