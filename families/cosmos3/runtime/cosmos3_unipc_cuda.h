/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

namespace trtmc::cosmos3 {

// CUDA implementation of the order-2 BH2 flow UniPC scheduler used by
// Cosmos3-Nano. Inputs and output are host FP32 arrays. Tensor arithmetic is
// performed on the caller-supplied CUDA stream; step() synchronizes that stream
// before returning so output is ready for immediate host use.
//
// An instance is stateful and must receive exactly one call per timestep. It is
// not safe to call an instance concurrently. The stream remains owned by the
// caller and must outlive this object.
class FlowUniPCCuda final {
  public:
    explicit FlowUniPCCuda(cudaStream_t stream, int32_t num_inference_steps = 35,
                           float shift = 10.0F, int32_t num_train_timesteps = 1000);
    ~FlowUniPCCuda();

    FlowUniPCCuda(const FlowUniPCCuda&) = delete;
    FlowUniPCCuda& operator=(const FlowUniPCCuda&) = delete;

    const std::vector<int64_t>& timesteps() const noexcept { return timesteps_; }

    // model_output, sample, and output point to host FP32 arrays of count
    // elements. output may alias either input.
    void step(const float* model_output, const float* sample, float* output, std::size_t count);

  private:
    struct Impl;

    std::vector<int64_t> timesteps_;
    std::unique_ptr<Impl> impl_;
};

} // namespace trtmc::cosmos3
