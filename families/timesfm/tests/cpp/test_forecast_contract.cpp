/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timesfm/runtime/pipeline.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class RecordingModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto& value_tensor = inputs.at("past_values");
        const auto* value_data = static_cast<const float*>(value_tensor.data);
        values.assign(value_data, value_data + value_tensor.numel());
        shape = value_tensor.shape;

        const auto& padding_tensor = inputs.at("past_values_padding");
        const auto* padding_data = static_cast<const std::int32_t*>(padding_tensor.data);
        padding.assign(padding_data, padding_data + padding_tensor.numel());

        const auto& frequency_tensor = inputs.at("freq");
        frequency = *static_cast<const std::int32_t*>(frequency_tensor.data);
        return {{"mean_predictions", {output.data(), {1, 2}, trtmc::DType::kFloat32}}};
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
    bool has_input(const std::string& name) const override {
        return name == "past_values" || name == "past_values_padding" || name == "freq";
    }
    bool has_output(const std::string& name) const override { return name == "mean_predictions"; }
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

    std::vector<float> values;
    std::vector<std::int32_t> padding;
    std::vector<std::int64_t> shape;
    std::int32_t frequency{-1};
    std::vector<float> output = std::vector<float>(2, 0.0F);
};

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

trtmc::ForecastRequest request(const std::vector<float>& values, const std::vector<float>& mask,
                               std::int32_t frequency) {
    return {{values.data(), values.size()}, {mask.data(), mask.size()}, frequency};
}

void test_short_series_is_left_padded_and_frequency_is_explicit() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::timesfm::Pipeline pipeline(std::move(module), {4, 2, 3, 1});
    const std::vector<float> values{11.0F, 12.0F};
    const std::vector<float> mask;

    pipeline.forecast(request(values, mask, 1));

    require(recording->shape == std::vector<std::int64_t>({1, 4}),
            "TimesFM must send its configured context length");
    require(recording->values == std::vector<float>({0.0F, 0.0F, 11.0F, 12.0F}),
            "TimesFM must left-pad a short series");
    require(recording->padding == std::vector<std::int32_t>({1, 1, 0, 0}),
            "TimesFM must mark left padding separately from observed values");
    require(recording->frequency == 1, "TimesFM must use the explicit request frequency");
}

void test_out_of_range_frequency_is_rejected() {
    auto module = std::make_unique<RecordingModule>();
    trtmc::timesfm::Pipeline pipeline(std::move(module), {4, 2, 3, 1});
    const std::vector<float> values{1.0F};
    const std::vector<float> mask{1.0F};
    bool rejected = false;
    try {
        pipeline.forecast(request(values, mask, 3));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "TimesFM must reject an out-of-range frequency category");
}

void test_overlong_series_is_left_truncated() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::timesfm::Pipeline pipeline(std::move(module), {4, 2, 3, 1});
    const std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const std::vector<float> mask(values.size(), 1.0F);

    pipeline.forecast(request(values, mask, 2));

    require(recording->values == std::vector<float>({3.0F, 4.0F, 5.0F, 6.0F}),
            "TimesFM must retain the newest context window");
    require(recording->padding == std::vector<std::int32_t>({0, 0, 0, 0}),
            "TimesFM must truncate the mask with the values");
    require(recording->frequency == 2, "TimesFM must forward each request frequency");
}

} // namespace

int main() {
    test_short_series_is_left_padded_and_frequency_is_explicit();
    test_overlong_series_is_left_truncated();
    test_out_of_range_frequency_is_rejected();
    std::cerr << "ALL PASSED\n";
    return 0;
}
