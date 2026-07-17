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

namespace trtmc::wan2_2_ti2v {

// CUDA implementation of the order-2 BH2 flow UniPC scheduler used by
// Wan2.2-TI2V-5B. Inputs and output are host FP32 arrays. Tensor arithmetic is
// performed on the caller-supplied CUDA stream; step() synchronizes that stream
// before returning so output is ready for immediate host use.
//
// An instance is stateful and must receive exactly one call per timestep. It is
// not safe to call an instance concurrently. The stream remains owned by the
// caller and must outlive this object.
class FlowUniPCCuda final {
  public:
    explicit FlowUniPCCuda(cudaStream_t stream, int32_t num_inference_steps = 50,
                           float shift = 5.0F, int32_t num_train_timesteps = 1000);
    FlowUniPCCuda(int32_t num_inference_steps, float shift, int32_t num_train_timesteps,
                  cudaStream_t stream);
    ~FlowUniPCCuda();

    FlowUniPCCuda(const FlowUniPCCuda&) = delete;
    FlowUniPCCuda& operator=(const FlowUniPCCuda&) = delete;
    FlowUniPCCuda(FlowUniPCCuda&&) noexcept;
    FlowUniPCCuda& operator=(FlowUniPCCuda&&) noexcept;

    const std::vector<int64_t>& timesteps() const noexcept { return timesteps_; }
    const std::vector<float>& sigmas() const noexcept { return sigmas_; }
    int32_t step_index() const noexcept;

    // Drop solver history while retaining allocated CUDA storage.
    void reset();

    // model_output, sample, and output point to host FP32 arrays of count
    // elements. output may alias either input.
    void step(const float* model_output, const float* sample, float* output, std::size_t count,
              float* corrected_sample = nullptr);

  private:
    struct Impl;

    void make_schedule(float shift, int32_t num_inference_steps, int32_t num_train_timesteps);

    std::vector<int64_t> timesteps_;
    std::vector<float> sigmas_;
    std::unique_ptr<Impl> impl_;
};

} // namespace trtmc::wan2_2_ti2v
