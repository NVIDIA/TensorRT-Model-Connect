/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// ITrtModule: virtual interface for TRT engine execution.
// Concrete implementations live in backend DSOs (libtrtmc_backend_*.so).
// No TRT headers — only CUDA runtime types and our own tensor types.

#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/tensor.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

enum class ProfileShapeSelector {
    kMin,
    kOpt,
    kMax,
};

class ITrtModule {
  public:
    virtual ~ITrtModule() = default;

    // Forward passes
    virtual TensorMap forward(const TensorMap& inputs) = 0;
    virtual DeviceTensorMap forward_device(const DeviceTensorMap& inputs) = 0;
    virtual void forward_device_async(const DeviceTensorMap& inputs) = 0;
    virtual void forward_async(const TensorMap& inputs) = 0;
    virtual void sync() = 0;

    // Introspection
    virtual cudaStream_t stream() const = 0;
    virtual void enable_cuda_graph() = 0;
    virtual bool cuda_graph_active() const = 0;
    virtual bool cuda_graph_captured() const = 0;
    virtual int32_t profile_idx() const = 0;
    virtual std::vector<TensorInfo> input_info() const = 0;
    virtual std::vector<TensorInfo> output_info() const = 0;
    virtual bool has_input(const std::string& name) const = 0;
    virtual bool has_output(const std::string& name) const = 0;
    virtual DType tensor_dtype(const std::string& name) const = 0;
    virtual std::vector<int64_t> tensor_shape(const std::string& name) const = 0;
    virtual std::vector<int64_t> input_profile_shape(const std::string& name, int32_t profile_idx,
                                                     ProfileShapeSelector selector) const = 0;
    virtual int32_t optimization_profile_count() const = 0;

    // Direct buffer access (KV cache binding)
    virtual void* device_ptr(const std::string& name) const = 0;
    virtual void bind_external(const std::string& name, void* ptr) = 0;

    virtual void bind_external(const std::string& name, void* ptr,
                               const std::vector<int64_t>& shape) = 0;

    virtual int32_t input_rank(const std::string& name) const = 0;
    virtual bool input_is_dynamic(const std::string& name) const = 0;

    // Reset module-owned per-generation state without replacing the loaded
    // execution context. Sequence state belongs to the pipeline/state object;
    // stable TensorRT contexts, profiles, bindings, and CUDA graphs are reused.
    virtual void reset_execution_context() = 0;

    // Human-readable label used for automatic runtime timing logs.
    virtual void set_timing_label(std::string label) = 0;

    virtual bool ok() const = 0;
    virtual void keep_alive(std::shared_ptr<void> resource) = 0;
};

} // namespace trtmc
