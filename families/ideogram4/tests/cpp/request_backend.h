/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/runtime/trt_backend.h"

#include <condition_variable>
#include <mutex>
#include <stdexcept>

namespace trtmc::ideogram4::test {

struct EngineState {
    std::mutex mutex;
    std::condition_variable changed;
    bool block{false}, entered{false}, release{false};
    unsigned calls{0};
};

class RequestModule final : public ITrtModule {
  public:
    explicit RequestModule(std::shared_ptr<EngineState> state) : state_(std::move(state)) {}
    TensorMap forward(const TensorMap&) override {
        std::unique_lock<std::mutex> lock(state_->mutex);
        ++state_->calls;
        state_->entered = true;
        state_->changed.notify_all();
        if (state_->block)
            state_->changed.wait(lock, [&] { return state_->release; });
        // No weights or GPU execution are needed to test request admission.
        throw std::runtime_error("injected engine stop");
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override { return {}; }
    void forward_device_async(const DeviceTensorMap&) override {}
    void forward_async(const TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return {}; }
    std::vector<TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return true; }
    bool has_output(const std::string&) const override { return true; }
    DType tensor_dtype(const std::string&) const override { return DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {1}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             ProfileShapeSelector) const override {
        return {1};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 1; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::shared_ptr<EngineState> state_;
};

class RequestBackend final : public IBackend {
  public:
    std::shared_ptr<EngineState> state = std::make_shared<EngineState>();
    std::unique_ptr<ITrtModule> create_module(const void*, std::size_t,
                                              const ModuleCreateOptions&) override {
        return std::make_unique<RequestModule>(state);
    }
    std::unique_ptr<ITrtModule>
    create_module_prebound(const void*, std::size_t, const ModuleCreateOptions&,
                           const std::vector<ModuleExternalBinding>&) override {
        return nullptr;
    }
    BackendDualProfileModules create_dual_profile_modules(const void*, std::size_t,
                                                          const ModuleCreateOptions&) override {
        return {};
    }
    const char* name() const override { return "trt"; }
};
} // namespace trtmc::ideogram4::test
