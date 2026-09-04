/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/chronos_bolt/runtime/pipeline.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

class RecordingModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto& input = inputs.at("context");
        const auto* data = static_cast<const float*>(input.data);
        context.assign(data, data + input.numel());
        context_shape = input.shape;
        return {{"quantile_preds", {output.data(), {1, 2, 3}, trtmc::DType::kFloat32}}};
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    std::int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override { return name == "context"; }
    bool has_output(const std::string& name) const override { return name == "quantile_preds"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<std::int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<std::int64_t> input_profile_shape(const std::string&, std::int32_t,
                                                  trtmc::ProfileShapeSelector) const override {
        return {};
    }
    std::int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<std::int64_t>&) override {}
    std::int32_t input_rank(const std::string&) const override { return 2; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    std::vector<float> context;
    std::vector<std::int64_t> context_shape;
    std::vector<float> output = std::vector<float>(6, 0.0F);
};

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

trtmc::ForecastRequest request(const std::vector<float>& values, const std::vector<float>& mask) {
    return {{values.data(), values.size()}, {mask.data(), mask.size()}, 0};
}

void test_short_series_is_left_padded() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::chronos_bolt::Pipeline pipeline(std::move(module), {4, 2, 3, 1});
    const std::vector<float> values{11.0F, 12.0F};
    const std::vector<float> mask;

    pipeline.forecast(request(values, mask));

    require(recording->context_shape == std::vector<std::int64_t>({1, 4}),
            "Chronos-Bolt must send its configured context length");
    require(recording->context.size() == 4 && std::isnan(recording->context[0]) &&
                std::isnan(recording->context[1]) && recording->context[2] == 11.0F &&
                recording->context[3] == 12.0F,
            "Chronos-Bolt must left-pad a short series as unobserved");
}

void test_frequency_is_rejected() {
    auto module = std::make_unique<RecordingModule>();
    trtmc::chronos_bolt::Pipeline pipeline(std::move(module), {4, 2, 3, 1});
    const std::vector<float> values{1.0F};
    const std::vector<float> mask{1.0F};
    bool rejected = false;
    try {
        pipeline.forecast({{values.data(), values.size()}, {mask.data(), mask.size()}, 1});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Chronos-Bolt must reject an unsupported frequency category");
}

void test_overlong_series_is_left_truncated() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::chronos_bolt::Pipeline pipeline(std::move(module), {4, 2, 3, 1});
    const std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const std::vector<float> mask(values.size(), 1.0F);

    pipeline.forecast(request(values, mask));

    require(recording->context == std::vector<float>({3.0F, 4.0F, 5.0F, 6.0F}),
            "Chronos-Bolt must retain the newest context window");
}

} // namespace

int main() {
    test_short_series_is_left_padded();
    test_overlong_series_is_left_truncated();
    test_frequency_is_rejected();
    std::cerr << "ALL PASSED\n";
    return 0;
}
