/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/patchtsmixer/runtime/pipeline.h"

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
        record(inputs.at("past_values"), values, shape);
        std::vector<std::int64_t> ignored_shape;
        record(inputs.at("observed_mask"), mask, ignored_shape);
        return {{"prediction_outputs", {output.data(), {1, 2, 2}, trtmc::DType::kFloat32}}};
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
        return name == "past_values" || name == "observed_mask";
    }
    bool has_output(const std::string& name) const override { return name == "prediction_outputs"; }
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
    std::int32_t input_rank(const std::string&) const override { return 3; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    std::vector<float> values;
    std::vector<float> mask;
    std::vector<std::int64_t> shape;
    std::vector<float> output = std::vector<float>(4, 0.0F);

  private:
    static void record(const trtmc::Tensor& tensor, std::vector<float>& destination,
                       std::vector<std::int64_t>& destination_shape) {
        const auto* data = static_cast<const float*>(tensor.data);
        destination.assign(data, data + tensor.numel());
        destination_shape = tensor.shape;
    }
};

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

trtmc::ForecastRequest request(const std::vector<float>& values, const std::vector<float>& mask) {
    return {{values.data(), values.size()}, {mask.data(), mask.size()}, 0};
}

trtmc::patchtsmixer::RuntimeConfig config() {
    return {4, 2, 2};
}

void test_short_multichannel_series_is_left_padded() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::patchtsmixer::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{11.0F, 21.0F, 12.0F, 22.0F};
    const std::vector<float> mask;

    pipeline.forecast(request(values, mask));

    require(recording->shape == std::vector<std::int64_t>({1, 4, 2}),
            "PatchTSMixer must preserve its configured channel count");
    require(recording->values ==
                std::vector<float>({0.0F, 0.0F, 0.0F, 0.0F, 11.0F, 21.0F, 12.0F, 22.0F}),
            "PatchTSMixer must left-pad complete timesteps");
    require(recording->mask == std::vector<float>({0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F}),
            "PatchTSMixer must mark padded timesteps unobserved");
}

void test_frequency_is_rejected() {
    auto module = std::make_unique<RecordingModule>();
    trtmc::patchtsmixer::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{1.0F, 2.0F};
    const std::vector<float> mask(values.size(), 1.0F);
    bool rejected = false;
    try {
        pipeline.forecast({{values.data(), values.size()}, {mask.data(), mask.size()}, 1});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "PatchTSMixer must reject an unsupported frequency category");
}

void test_overlong_multichannel_series_is_left_truncated() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::patchtsmixer::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{1.0F, 11.0F, 2.0F, 12.0F, 3.0F, 13.0F,
                                    4.0F, 14.0F, 5.0F, 15.0F, 6.0F, 16.0F};
    const std::vector<float> mask(values.size(), 1.0F);

    pipeline.forecast(request(values, mask));

    require(recording->values ==
                std::vector<float>({3.0F, 13.0F, 4.0F, 14.0F, 5.0F, 15.0F, 6.0F, 16.0F}),
            "PatchTSMixer must retain the newest complete timesteps");
}

void test_partial_multichannel_timestep_is_rejected() {
    auto module = std::make_unique<RecordingModule>();
    trtmc::patchtsmixer::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{1.0F, 2.0F, 3.0F};
    const std::vector<float> mask(values.size(), 1.0F);
    bool rejected = false;
    try {
        pipeline.forecast(request(values, mask));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "PatchTSMixer must reject a partial channel timestep");
}

} // namespace

int main() {
    test_short_multichannel_series_is_left_padded();
    test_overlong_multichannel_series_is_left_truncated();
    test_partial_multichannel_timestep_is_rejected();
    test_frequency_is_rejected();
    std::cerr << "ALL PASSED\n";
    return 0;
}
