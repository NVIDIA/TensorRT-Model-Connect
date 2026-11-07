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
        return {{"quantile_preds", {output.data(), output_shape, trtmc::DType::kFloat32}}};
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
    std::vector<std::int64_t> output_shape{1, 3, 2};
};

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

using Request = trtmc::internal::SeriesToQuantileForecastRequest;

Request request(const std::vector<float>& values, const std::vector<std::uint8_t>& mask) {
    return {{{{values.data(), values.size()}, 0, 0}, {mask.data(), mask.size()}}};
}

trtmc::chronos_bolt::RuntimeConfig config() {
    return {4, 2, {0.1, 0.5, 0.9}, 1};
}

template <typename Function>
bool throws_invalid_argument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_short_series_is_left_padded() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    recording->output = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    trtmc::chronos_bolt::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{11.0F, 12.0F};
    const std::vector<std::uint8_t> mask;

    const auto result = pipeline.run(request(values, mask), {});

    require(recording->context_shape == std::vector<std::int64_t>({1, 4}),
            "Chronos-Bolt must send its configured context length");
    require(recording->context.size() == 4 && std::isnan(recording->context[0]) &&
                std::isnan(recording->context[1]) && recording->context[2] == 11.0F &&
                recording->context[3] == 12.0F,
            "Chronos-Bolt must left-pad a short series as unobserved");
    require(result.values == recording->output && result.horizon == 2 && result.channels == 1 &&
                result.quantile_levels == std::vector<double>({0.1, 0.5, 0.9}) &&
                result.axes.horizon_steps == std::vector<std::int64_t>({1, 2}),
            "Chronos-Bolt must report [quantile, horizon, channel] values with their levels");
}

void test_frequency_is_rejected() {
    auto module = std::make_unique<RecordingModule>();
    trtmc::chronos_bolt::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{1.0F};
    const std::vector<std::uint8_t> mask{1};
    const trtmc::internal::ConfigEntry fields[] = {{"frequency", std::int64_t{1}}};
    require(throws_invalid_argument([&] { pipeline.run(request(values, mask), fields); }),
            "Chronos-Bolt must reject an unsupported frequency category");
}

void test_overlong_series_is_left_truncated() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::chronos_bolt::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const std::vector<std::uint8_t> mask(values.size(), 1);

    pipeline.run(request(values, mask), {});

    require(recording->context == std::vector<float>({3.0F, 4.0F, 5.0F, 6.0F}),
            "Chronos-Bolt must retain the newest context window");
}

void test_observed_mask_survives_padding_and_truncation() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::chronos_bolt::Pipeline pipeline(std::move(module), config());
    const std::vector<float> short_values{1.0F, 2.0F};
    const std::vector<std::uint8_t> short_mask{0, 1};
    pipeline.run(request(short_values, short_mask), {});
    require(recording->context.size() == 4 && std::isnan(recording->context[0]) &&
                std::isnan(recording->context[1]) && std::isnan(recording->context[2]) &&
                recording->context[3] == 2.0F,
            "a caller-provided missing observation must reach the engine as NaN");
    const std::vector<float> long_values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const std::vector<std::uint8_t> long_mask{1, 1, 0, 1, 0, 1};
    pipeline.run(request(long_values, long_mask), {});
    require(std::isnan(recording->context[0]) && recording->context[1] == 4.0F &&
                std::isnan(recording->context[2]) && recording->context[3] == 6.0F,
            "truncation must crop the mask at exactly the same timestep as the values");
    require(throws_invalid_argument([&] { pipeline.run(request(long_values, short_mask), {}); }),
            "observed mask must contain one entry per input value");
}

void test_history_shape_must_be_a_single_channel() {
    auto module = std::make_unique<RecordingModule>();
    trtmc::chronos_bolt::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<std::uint8_t> mask;
    auto input = request(values, mask);
    input.history.past_values.rows = 4;
    input.history.past_values.columns = 1;
    pipeline.run(input, {});
    for (const auto& shape : {std::pair{0U, 4U}, std::pair{4U, 0U}, std::pair{2U, 2U},
                              std::pair{1U, 4U}, std::pair{3U, 1U}}) {
        input.history.past_values.rows = shape.first;
        input.history.past_values.columns = shape.second;
        require(throws_invalid_argument([&] { pipeline.run(input, {}); }),
                "history shape must be complete and describe exactly one channel");
    }
    const std::vector<float> none;
    require(throws_invalid_argument([&] { pipeline.run(request(none, mask), {}); }),
            "Chronos-Bolt must reject an empty history");
}

void test_task_binding_declares_family_config() {
    auto module = std::make_unique<RecordingModule>();
    trtmc::chronos_bolt::Pipeline pipeline(std::move(module), config());
    const auto bindings = pipeline.task_bindings();
    require(bindings.size() == 1 && bindings[0].key.id == "series_to_quantile_forecast" &&
                bindings[0].key.major == 1 && bindings[0].key.minor == 0,
            "Chronos-Bolt must bind only its quantile forecast Task");
    require(bindings[0].implementation ==
                static_cast<trtmc::internal::ISeriesToQuantileForecast*>(&pipeline),
            "Task binding must point to the adjusted interface subobject");
    require(std::string(pipeline.task()) == bindings[0].key.id,
            "bundle primary Task must match the family binding");
    const auto fields = bindings[0].fields;
    require(fields.size() == 1 && fields[0].name == "frequency" &&
                fields[0].kind == trtmc::internal::ConfigKind::I64 &&
                trtmc::internal::config_get<std::int64_t>({}, fields, "frequency") == 0,
            "frequency must be family-declared with its unchanged zero default");
    const std::vector<float> values{1.0F, 2.0F};
    const std::vector<std::uint8_t> mask;
    const trtmc::internal::ConfigEntry explicit_zero[] = {{"frequency", std::int64_t{0}}};
    auto* task =
        static_cast<trtmc::internal::ISeriesToQuantileForecast*>(bindings[0].implementation);
    require(task->run(request(values, mask), explicit_zero).values.size() == 6,
            "bound Task must accept the declared zero frequency");
}

void test_engine_output_must_match_runtime_config() {
    auto module = std::make_unique<RecordingModule>();
    module->output = std::vector<float>(4, 0.0F);
    module->output_shape = {1, 2, 2};
    trtmc::chronos_bolt::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{1.0F, 2.0F};
    const std::vector<std::uint8_t> mask;
    bool rejected = false;
    try {
        pipeline.run(request(values, mask), {});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "an engine output that disagrees with runtime.json must be rejected");
}

void test_result_owns_every_value_after_next_call_and_pipeline_destruction() {
    trtmc::internal::QuantileForecastResult first;
    {
        auto module = std::make_unique<RecordingModule>();
        auto* recording = module.get();
        recording->output = {1.25F, -2.5F, 3.75F, 4.5F, 5.5F, -6.5F};
        trtmc::chronos_bolt::Pipeline pipeline(std::move(module), config());
        const std::vector<float> values{1.0F, 2.0F};
        const std::vector<std::uint8_t> mask;
        first = pipeline.run(request(values, mask), {});
        recording->output.assign(6, 99.0F);
        const auto next = pipeline.run(request(values, mask), {});
        require(next.values == std::vector<float>(6, 99.0F),
                "each call must return its own engine output");
    }
    require(first.values == std::vector<float>({1.25F, -2.5F, 3.75F, 4.5F, 5.5F, -6.5F}) &&
                first.quantile_levels == std::vector<double>({0.1, 0.5, 0.9}) &&
                first.horizon == 2 && first.channels == 1 &&
                first.axes.horizon_steps == std::vector<std::int64_t>({1, 2}),
            "result must own all values, levels and axes beyond the engine lifetime");
}

} // namespace

int main() {
    test_short_series_is_left_padded();
    test_overlong_series_is_left_truncated();
    test_frequency_is_rejected();
    test_observed_mask_survives_padding_and_truncation();
    test_history_shape_must_be_a_single_channel();
    test_task_binding_declares_family_config();
    test_engine_output_must_match_runtime_config();
    test_result_owns_every_value_after_next_call_and_pipeline_destruction();
    std::cerr << "ALL PASSED\n";
    return 0;
}
