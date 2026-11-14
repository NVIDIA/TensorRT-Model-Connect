/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/runtime/trt_module.h"
using namespace trtmc;

class FakeModule final : public ITrtModule {
  public:
    explicit FakeModule(int role) : role_(role) { ++live_instances; }
    ~FakeModule() override { --live_instances; }
    inline static int live_instances{0};
    int calls{0};
    bool reset_seen{false};
    bool malformed{false};
    bool missing{false};
    TensorMap forward(const TensorMap& in) override {
        ++calls;
        if (missing)
            return {};
        if (role_ == 0)
            return {
                {"encoder_output", {values_, {1, 2}, malformed ? DType::kInt32 : DType::kFloat32}}};
        if (role_ == 1) {
            if (*static_cast<int*>(in.at("token_id").data) == 2)
                reset_seen = *static_cast<float*>(in.at("state_h_0").data) == 0;
            return {{"pred_output", {values_, {1, malformed ? 1 : 2}, DType::kFloat32}},
                    {"next_h_0", {values_, {1, 2}, DType::kFloat32}},
                    {"next_c_0", {values_, {1, 2}, DType::kFloat32}}};
        }
        return {{"token_logits", {malformed ? nullptr : tokens_, {1, 3}, DType::kFloat32}},
                {"duration_logits", {durations_, {1, 2}, DType::kFloat32}}};
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
    bool has_input(const std::string&) const override { return false; }
    bool has_output(const std::string&) const override { return true; }
    DType tensor_dtype(const std::string&) const override { return DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 0; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    int role_;
    float values_[2]{1, 1};
    float tokens_[3]{2, 0, -1};
    float durations_[2]{0, 1};
};
