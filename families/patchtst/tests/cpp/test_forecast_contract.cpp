/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/patchtst/runtime/pipeline.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace numeric = trtmc::internal;

class RecordingModule final : public trtmc::ITrtModule {
  public:
    explicit RecordingModule(std::string output_name = "prediction_outputs",
                             std::vector<std::int64_t> output_shape = {1, 2, 2})
        : output_name_(std::move(output_name)), output_shape_(std::move(output_shape)) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        record(inputs.at("past_values"), values, shape);
        std::vector<std::int64_t> ignored_shape;
        record(inputs.at("past_observed_mask"), mask, ignored_shape);
        return {{output_name_, {output.data(), output_shape_, output_dtype}}};
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
        return name == "past_values" || name == "past_observed_mask";
    }
    bool has_output(const std::string& name) const override { return name == output_name_; }
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

    int calls{0};
    trtmc::DType output_dtype{trtmc::DType::kFloat32};
    std::vector<float> values;
    std::vector<float> mask;
    std::vector<std::int64_t> shape;
    std::vector<float> output = std::vector<float>(4, 0.0F);

  private:
    std::string output_name_;
    std::vector<std::int64_t> output_shape_;

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

numeric::SeriesToPointForecastRequest request(const std::vector<float>& values,
                                              const std::vector<std::uint8_t>& mask) {
    return {{{{values.data(), values.size()}, 0, 0}, {mask.data(), mask.size()}}};
}

trtmc::patchtst::RuntimeConfig config() {
    return {4, 2, 2, 1, "prediction_outputs"};
}

void test_short_multichannel_series_is_left_padded() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::patchtst::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{11.0F, 21.0F, 12.0F, 22.0F};
    const std::vector<std::uint8_t> mask;

    pipeline.run(request(values, mask), {});

    require(recording->shape == std::vector<std::int64_t>({1, 4, 2}),
            "PatchTST must preserve its configured channel count");
    require(recording->values ==
                std::vector<float>({0.0F, 0.0F, 0.0F, 0.0F, 11.0F, 21.0F, 12.0F, 22.0F}),
            "PatchTST must left-pad complete timesteps");
    require(recording->mask == std::vector<float>({0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F}),
            "PatchTST must mark padded timesteps unobserved");
}

void test_frequency_is_rejected() {
    auto module = std::make_unique<RecordingModule>();
    trtmc::patchtst::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{1.0F, 2.0F};
    const std::vector<std::uint8_t> mask(values.size(), 1);
    bool rejected = false;
    try {
        const numeric::ConfigEntry frequency[] = {{"frequency", std::int64_t{1}}};
        pipeline.run(request(values, mask), frequency);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "PatchTST must reject an unsupported frequency category");
}

void test_overlong_multichannel_series_is_left_truncated() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::patchtst::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{1.0F, 11.0F, 2.0F, 12.0F, 3.0F, 13.0F,
                                    4.0F, 14.0F, 5.0F, 15.0F, 6.0F, 16.0F};
    const std::vector<std::uint8_t> mask(values.size(), 1);

    pipeline.run(request(values, mask), {});

    require(recording->values ==
                std::vector<float>({3.0F, 13.0F, 4.0F, 14.0F, 5.0F, 15.0F, 6.0F, 16.0F}),
            "PatchTST must retain the newest complete timesteps");
}

void test_partial_multichannel_timestep_is_rejected() {
    auto module = std::make_unique<RecordingModule>();
    trtmc::patchtst::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{1.0F, 2.0F, 3.0F};
    const std::vector<std::uint8_t> mask(values.size(), 1);
    bool rejected = false;
    try {
        pipeline.run(request(values, mask), {});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "PatchTST must reject a partial channel timestep");
}

template <class Exception, class Function>
void rejects(Function function) {
    bool rejected = false;
    try {
        function();
    } catch (const Exception&) {
        rejected = true;
    }
    require(rejected, "invalid request must be rejected");
}

void test_single_target_regression_uses_the_logical_reference_shape() {
    auto module =
        std::make_unique<RecordingModule>("regression_outputs", std::vector<std::int64_t>{1, 1, 2});
    module->output = {0.25F, 0.5F};
    trtmc::patchtst::Pipeline pipeline(
        std::move(module),
        {4, 1, 1, 1, "regression_outputs", "series_to_regression_distribution", 1, "normal"});
    const std::vector<float> values{1, 2, 3, 4};
    const auto input = request(values, {});
    const auto result =
        pipeline.run(numeric::SeriesToRegressionDistributionRequest{input.history}, {});
    require(result.target_count == 1 && result.parameters.size() == 2 &&
                result.parameters[0].name == "location" &&
                result.parameters[0].values == std::vector<float>{0.25F} &&
                result.parameters[1].name == "scale" &&
                result.parameters[1].values == std::vector<float>{0.5F},
            "single-target parameters are scalars over targets, not a horizon axis");
    const auto bindings = pipeline.task_bindings();
    require(bindings.size() == 1 && bindings[0].key.id == "series_to_regression_distribution" &&
                bindings[0].implementation ==
                    static_cast<numeric::ISeriesToRegressionDistribution*>(&pipeline),
            "the loaded distribution head must expose exactly its adjusted Task pointer");
    rejects<numeric::UnsupportedTask>([&] { pipeline.run(input, {}); });
}

void test_complete_multitarget_parameters_and_result_lifetime() {
    auto module =
        std::make_unique<RecordingModule>("regression_outputs", std::vector<std::int64_t>{1, 2, 3});
    auto* recording = module.get();
    recording->output = {3, -1, 0.5F, 7, 9, 1.5F};
    trtmc::patchtst::Pipeline pipeline(
        std::move(module),
        {4, 1, 1, 1, "regression_outputs", "series_to_regression_distribution", 2, "student_t"});
    const std::vector<float> values{1, 2};
    const auto input = request(values, {});
    auto result = pipeline.run(numeric::SeriesToRegressionDistributionRequest{input.history}, {});
    require(result.distribution == numeric::DistributionKind::StudentT &&
                result.target_count == 2 &&
                result.parameters[0].values == std::vector<float>({3, 7}) &&
                result.parameters[1].values == std::vector<float>({-1, 9}) &&
                result.parameters[2].values == std::vector<float>({0.5F, 1.5F}),
            "parameter-major views must retain every target from the target-major engine output");
    recording->output.assign(6, 1);
    pipeline.run(numeric::SeriesToRegressionDistributionRequest{input.history}, {});
    require(result.parameters[1].values == std::vector<float>({-1, 9}),
            "results must own their parameter storage");
    require(result.target_names.empty() && result.target_units.empty(),
            "unknown target identity stays unspecified");
}

void test_regression_values_and_negative_binomial_bindings() {
    const std::vector<float> values{1, 2};
    const auto input = request(values, {});
    {
        auto module = std::make_unique<RecordingModule>("regression_outputs",
                                                        std::vector<std::int64_t>{1, 3});
        module->output = {-2, 0, 4};
        trtmc::patchtst::Pipeline pipeline(
            std::move(module),
            {4, 1, 1, 1, "regression_outputs", "series_to_regression_values", 3, {}});
        const auto result =
            pipeline.run(numeric::SeriesToRegressionValuesRequest{input.history}, {});
        require(result.values == std::vector<float>({-2, 0, 4}),
                "point regression must retain all targets");
        const auto bindings = pipeline.task_bindings();
        require(bindings.size() == 1 && bindings[0].key.id == "series_to_regression_values" &&
                    bindings[0].implementation ==
                        static_cast<numeric::ISeriesToRegressionValues*>(&pipeline),
                "values binding");
        require(result.target_names.empty() && result.target_units.empty(), "unnamed targets");
    }
    {
        auto module = std::make_unique<RecordingModule>("regression_outputs",
                                                        std::vector<std::int64_t>{1, 1, 2});
        module->output = {2, -1};
        trtmc::patchtst::Pipeline pipeline(std::move(module), {4, 1, 1, 1, "regression_outputs",
                                                               "series_to_regression_distribution",
                                                               1, "negative_binomial"});
        const auto result =
            pipeline.run(numeric::SeriesToRegressionDistributionRequest{input.history}, {});
        require(result.distribution == numeric::DistributionKind::NegativeBinomial &&
                    result.parameters[0].name == "total_count" &&
                    result.parameters[1].name == "logits" &&
                    result.parameters[1].values == std::vector<float>{-1},
                "negative-binomial parameter identities");
    }
}

void test_shapes_masks_configuration_and_owned_forecast() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    recording->output = {1, 2, 3, 4};
    trtmc::patchtst::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{1, 11, 2, 12, 3, 13, 4, 14, 5, 15};
    const std::vector<std::uint8_t> observed{1, 0, 1, 1, 0, 1, 1, 0, 1, 1};
    auto input = request(values, observed);
    input.history.past_values.rows = 5;
    input.history.past_values.columns = 2;
    const numeric::ConfigEntry zero[] = {{"frequency", std::int64_t{0}}};
    auto first = pipeline.run(input, zero);
    require(recording->mask == std::vector<float>({1, 1, 0, 1, 1, 0, 1, 1}),
            "truncate masks with full timesteps");
    require(first.values.rows == 2 && first.values.columns == 2 &&
                first.values.values == std::vector<float>({1, 2, 3, 4}) &&
                first.axes.horizon_steps == std::vector<std::int64_t>({1, 2}),
            "complete forecast axes");
    require(first.axes.channel_names.empty() && first.axes.channel_units.empty(),
            "unspecified channel identity");
    const auto bindings = pipeline.task_bindings();
    require(bindings.size() == 1 && bindings[0].key.id == "series_to_point_forecast" &&
                bindings[0].implementation ==
                    static_cast<numeric::ISeriesToPointForecast*>(&pipeline),
            "forecast binding");
    const auto calls = recording->calls;
    for (const auto& shape :
         {std::pair<std::uint64_t, std::uint64_t>{0, 2}, {5, 0}, {10, 1}, {4, 2}}) {
        input.history.past_values.rows = shape.first;
        input.history.past_values.columns = shape.second;
        rejects<std::invalid_argument>([&] { pipeline.run(input, {}); });
    }
    rejects<std::invalid_argument>([&] { pipeline.run(request({}, {}), {}); });
    rejects<std::invalid_argument>([&] { pipeline.run(request(values, {1}), {}); });
    rejects<std::invalid_argument>(
        [&] { pipeline.run(request(values, {2, 1, 1, 1, 1, 1, 1, 1, 1, 1}), {}); });
    auto missing = request(values, {});
    missing.history.past_values.values = {nullptr, values.size()};
    rejects<std::invalid_argument>([&] { pipeline.run(missing, {}); });
    const numeric::ConfigEntry wrong[] = {{"frequency", true}};
    const numeric::ConfigEntry unknown[] = {{"unknown", std::int64_t{0}}};
    const numeric::ConfigEntry duplicate[] = {{"frequency", std::int64_t{0}},
                                              {"frequency", std::int64_t{0}}};
    for (auto fields : {trtmc::Span<const numeric::ConfigEntry>{wrong}, {unknown}, {duplicate}})
        rejects<numeric::ConfigError>([&] { pipeline.run(request(values, {}), fields); });
    require(recording->calls == calls, "invalid inputs must fail before engine evaluation");
    recording->output = {5, 6, 7, 8};
    pipeline.run(request(values, {}), {});
    require(first.values.values == std::vector<float>({1, 2, 3, 4}),
            "later calls cannot overwrite owned results");
    recording->output_dtype = trtmc::DType::kFloat16;
    rejects<std::runtime_error>([&] { pipeline.run(request(values, {}), {}); });
}

} // namespace

int main() {
    test_short_multichannel_series_is_left_padded();
    test_overlong_multichannel_series_is_left_truncated();
    test_partial_multichannel_timestep_is_rejected();
    test_frequency_is_rejected();
    test_single_target_regression_uses_the_logical_reference_shape();
    test_complete_multitarget_parameters_and_result_lifetime();
    test_regression_values_and_negative_binomial_bindings();
    test_shapes_masks_configuration_and_owned_forecast();
    std::cerr << "ALL PASSED\n";
    return 0;
}
