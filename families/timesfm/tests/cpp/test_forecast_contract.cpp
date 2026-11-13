/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timesfm/runtime/pipeline.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class RecordingModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& value_tensor = inputs.at("past_values");
        const auto* value_data = static_cast<const float*>(value_tensor.data);
        values.assign(value_data, value_data + value_tensor.numel());
        shape = value_tensor.shape;

        const auto& padding_tensor = inputs.at("past_values_padding");
        const auto* padding_data = static_cast<const std::int32_t*>(padding_tensor.data);
        padding.assign(padding_data, padding_data + padding_tensor.numel());

        const auto& frequency_tensor = inputs.at("freq");
        frequency = *static_cast<const std::int32_t*>(frequency_tensor.data);
        return {{"mean_predictions", {output.data(), output_shape, output_dtype}}};
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
    std::vector<std::int64_t> output_shape{1, 2};
    trtmc::DType output_dtype{trtmc::DType::kFloat32};
    int calls{0};
};

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

trtmc::internal::SeriesToPointForecastRequest request(const std::vector<float>& values,
                                                      const std::vector<std::uint8_t>& mask) {
    return {{{{values.data(), values.size()}, 0, 0}, {mask.data(), mask.size()}}};
}

trtmc::internal::PointForecastResult forecast(trtmc::timesfm::Pipeline& pipeline,
                                              const std::vector<float>& values,
                                              const std::vector<std::uint8_t>& mask,
                                              std::int64_t frequency) {
    const trtmc::internal::ConfigEntry config[] = {{"frequency", frequency}};
    return pipeline.run(request(values, mask), config);
}

void test_short_series_is_left_padded_and_frequency_is_explicit() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::timesfm::Pipeline pipeline(std::move(module), {4, 2, 3, 1});
    const std::vector<float> values{11.0F, 12.0F};
    const std::vector<std::uint8_t> mask;

    forecast(pipeline, values, mask, 1);

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
    const std::vector<std::uint8_t> mask{1};
    bool rejected = false;
    try {
        forecast(pipeline, values, mask, 3);
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
    const std::vector<std::uint8_t> mask(values.size(), 1);

    forecast(pipeline, values, mask, 2);

    require(recording->values == std::vector<float>({3.0F, 4.0F, 5.0F, 6.0F}),
            "TimesFM must retain the newest context window");
    require(recording->padding == std::vector<std::int32_t>({0, 0, 0, 0}),
            "TimesFM must truncate the mask with the values");
    require(recording->frequency == 2, "TimesFM must forward each request frequency");
}

template <class Exception, class Function>
void rejects(Function operation) {
    bool rejected = false;
    try {
        operation();
    } catch (const Exception&) {
        rejected = true;
    }
    require(rejected, "invalid request must be rejected");
}

void test_sdk_binding_defaults_and_result_ownership() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    recording->output = {1.25F, 2.5F};
    trtmc::timesfm::Pipeline pipeline(std::move(module), {4, 2, 3, 1});
    require(std::string(pipeline.task()) == "series_to_point_forecast", "semantic primary task");
    const auto bindings = pipeline.task_bindings();
    require(bindings.size() == 1 && bindings[0].key.id == pipeline.task(), "exact Task binding");
    require(bindings[0].implementation ==
                static_cast<trtmc::internal::ISeriesToPointForecast*>(&pipeline),
            "binding must retain the adjusted interface pointer");
    require(bindings[0].fields.size() == 1 && bindings[0].fields[0].name == "frequency",
            "frequency is declared by the family");
    const std::vector<float> values{7, 8};
    auto first = pipeline.run(request(values, {}), {});
    require(recording->frequency == 0, "omitted frequency keeps the original zero default");
    require(first.values.rows == 2 && first.values.columns == 1 &&
                first.values.values == std::vector<float>({1.25F, 2.5F}) &&
                first.axes.horizon_steps == std::vector<std::int64_t>({1, 2}),
            "result keeps the complete point forecast and positive horizon steps");
    require(first.axes.channel_names.empty() && first.axes.channel_units.empty(),
            "unnamed checkpoint channels stay unspecified");
    recording->output = {3, 4};
    forecast(pipeline, values, {}, 2);
    require(recording->frequency == 2, "frequency is resolved for each request");
    pipeline.run(request(values, {}), {});
    require(recording->frequency == 0, "a previous override must not become the next default");
    require(first.values.values == std::vector<float>({1.25F, 2.5F}),
            "later calls cannot overwrite a retained result");
}

void test_history_shape_mask_and_invalid_config() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::timesfm::Pipeline pipeline(std::move(module), {4, 2, 3, 1});
    const std::vector<float> values{1, 2, 3, 4, 5, 6};
    const std::vector<std::uint8_t> mask{1, 0, 0, 1, 0, 1};
    auto input = request(values, mask);
    input.history.past_values.rows = values.size();
    input.history.past_values.columns = 1;
    pipeline.run(input, {});
    require(recording->padding == std::vector<std::int32_t>({1, 0, 1, 0}),
            "an explicit observed mask is truncated with its values");
    const auto calls = recording->calls;
    for (const auto shape :
         {std::pair<std::uint64_t, std::uint64_t>{0, 1}, {6, 0}, {3, 2}, {5, 1}}) {
        input.history.past_values.rows = shape.first;
        input.history.past_values.columns = shape.second;
        rejects<std::invalid_argument>([&] { pipeline.run(input, {}); });
    }
    rejects<std::invalid_argument>([&] { pipeline.run(request({}, {}), {}); });
    rejects<std::invalid_argument>([&] { pipeline.run(request(values, {1}), {}); });
    rejects<std::invalid_argument>([&] { pipeline.run(request(values, {1, 1, 1, 1, 1, 2}), {}); });
    auto missing = request(values, {});
    missing.history.past_values.values = {nullptr, values.size()};
    rejects<std::invalid_argument>([&] { pipeline.run(missing, {}); });
    for (const std::int64_t frequency : {-1LL, 3LL, std::numeric_limits<long long>::max()})
        rejects<trtmc::internal::ConfigError>([&] { forecast(pipeline, values, {}, frequency); });
    for (const auto name : {"frequency", "unknown"}) {
        const trtmc::internal::ConfigEntry wrong[] = {{name, true}};
        rejects<trtmc::internal::ConfigError>([&] { pipeline.run(request(values, {}), wrong); });
    }
    const trtmc::internal::ConfigEntry duplicate[] = {{"frequency", std::int64_t{0}},
                                                      {"frequency", std::int64_t{1}}};
    rejects<trtmc::internal::ConfigError>([&] { pipeline.run(request(values, {}), duplicate); });
    require(recording->calls == calls, "invalid requests must fail before engine execution");
    recording->output_shape = {2, 1};
    rejects<std::runtime_error>([&] { pipeline.run(request(values, {}), {}); });
    recording->output_shape = {1, 2};
    recording->output_dtype = trtmc::DType::kInt32;
    rejects<std::runtime_error>([&] { pipeline.run(request(values, {}), {}); });
}

} // namespace

int main() {
    test_short_series_is_left_padded_and_frequency_is_explicit();
    test_overlong_series_is_left_truncated();
    test_out_of_range_frequency_is_rejected();
    test_sdk_binding_defaults_and_result_ownership();
    test_history_shape_mask_and_invalid_config();
    std::cerr << "ALL PASSED\n";
    return 0;
}
