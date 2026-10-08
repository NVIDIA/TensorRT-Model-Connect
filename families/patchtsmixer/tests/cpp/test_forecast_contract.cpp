/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/patchtsmixer/runtime/pipeline.h"
#include "families/patchtsmixer/runtime/plugin_helpers.h"

#include <cstdint>
#include <cstdlib>
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

trtmc::internal::SeriesToPointForecastRequest request(const std::vector<float>& values,
                                                      const std::vector<std::uint8_t>& mask) {
    return {{{{values.data(), values.size()}, 0, 0}, {mask.data(), mask.size()}}};
}

trtmc::patchtsmixer::RuntimeConfig config() {
    return {4, 2, 2, 1};
}

void test_tensor_parallel_rank_contract_is_strict() {
    unsetenv("OMPI_COMM_WORLD_SIZE");
    unsetenv("OMPI_COMM_WORLD_RANK");
    unsetenv("OMPI_COMM_WORLD_LOCAL_RANK");
    require(trtmc::patchtsmixer::require_rank(1) == 0,
            "single-rank PatchTSMixer must not require an MPI launcher");

    auto rejected = [] {
        try {
            (void)trtmc::patchtsmixer::require_rank(2);
            return false;
        } catch (const std::runtime_error&) {
            return true;
        }
    };

    require(rejected(), "multi-rank PatchTSMixer must require the OpenMPI world size");
    setenv("OMPI_COMM_WORLD_SIZE", "2x", 1);
    setenv("OMPI_COMM_WORLD_RANK", "0", 1);
    setenv("OMPI_COMM_WORLD_LOCAL_RANK", "0", 1);
    require(rejected(), "PatchTSMixer must reject a malformed OpenMPI world size");
    setenv("OMPI_COMM_WORLD_SIZE", "3", 1);
    require(rejected(), "PatchTSMixer OpenMPI world size must match tensor_parallel_size");
    setenv("OMPI_COMM_WORLD_SIZE", "2", 1);
    setenv("OMPI_COMM_WORLD_RANK", "2", 1);
    require(rejected(), "PatchTSMixer must reject an out-of-range global rank");
    setenv("OMPI_COMM_WORLD_RANK", "0", 1);
    setenv("OMPI_COMM_WORLD_LOCAL_RANK", "-1", 1);
    require(rejected(), "PatchTSMixer must reject a negative local rank");
}

void test_runtime_config_requires_tensor_parallel_size() {
    const auto parsed = trtmc::patchtsmixer::parse_runtime_config(
        R"({"context_length":4,"num_input_channels":2,"prediction_length":2,"tensor_parallel_size":4})");
    require(parsed.tensor_parallel_size == 4, "PatchTSMixer must parse tensor_parallel_size");

    bool rejected = false;
    try {
        (void)trtmc::patchtsmixer::parse_runtime_config(
            R"({"context_length":4,"num_input_channels":2,"prediction_length":2})");
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "PatchTSMixer must reject runtime.json without tensor_parallel_size");
}

void test_short_multichannel_series_is_left_padded() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::patchtsmixer::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{11.0F, 21.0F, 12.0F, 22.0F};
    const std::vector<std::uint8_t> mask;

    const auto result = pipeline.run(request(values, mask), {});
    require(result.values.rows == 2 && result.values.columns == 2 &&
                result.axes.horizon_steps == std::vector<std::int64_t>({1, 2}),
            "forecast must retain its horizon and channel axes");

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
    const std::vector<std::uint8_t> mask(values.size(), 1);
    bool rejected = false;
    try {
        const trtmc::internal::ConfigEntry fields[] = {{"frequency", std::int64_t{1}}};
        pipeline.run(request(values, mask), fields);
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
    const std::vector<std::uint8_t> mask(values.size(), 1);

    pipeline.run(request(values, mask), {});

    require(recording->values ==
                std::vector<float>({3.0F, 13.0F, 4.0F, 14.0F, 5.0F, 15.0F, 6.0F, 16.0F}),
            "PatchTSMixer must retain the newest complete timesteps");
}

void test_partial_multichannel_timestep_is_rejected() {
    auto module = std::make_unique<RecordingModule>();
    trtmc::patchtsmixer::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{1.0F, 2.0F, 3.0F};
    const std::vector<std::uint8_t> mask(values.size(), 1);
    bool rejected = false;
    try {
        pipeline.run(request(values, mask), {});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "PatchTSMixer must reject a partial channel timestep");
}

void test_task_binding_declares_family_config() {
    auto module = std::make_unique<RecordingModule>();
    trtmc::patchtsmixer::Pipeline pipeline(std::move(module), config());
    const auto bindings = pipeline.task_bindings();
    require(bindings.size() == 1 && bindings[0].key.id == "series_to_point_forecast" &&
                bindings[0].key.major == 1 && bindings[0].key.minor == 0,
            "PatchTSMixer must bind only its supported point forecast Task");
    require(bindings[0].implementation ==
                static_cast<trtmc::internal::ISeriesToPointForecast*>(&pipeline),
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
    auto* task = static_cast<trtmc::internal::ISeriesToPointForecast*>(bindings[0].implementation);
    require(task->run(request(values, mask), explicit_zero).values.values.size() == 4,
            "bound Task must accept the declared zero frequency");
}

void test_explicit_history_shape_matches_bundle_channels() {
    auto module = std::make_unique<RecordingModule>();
    trtmc::patchtsmixer::Pipeline pipeline(std::move(module), config());
    const std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<std::uint8_t> mask;
    auto input = request(values, mask);
    input.history.past_values.rows = 2;
    input.history.past_values.columns = 2;
    pipeline.run(input, {});
    for (const auto& shape :
         {std::pair{0U, 2U}, std::pair{2U, 0U}, std::pair{1U, 4U}, std::pair{3U, 2U}}) {
        input.history.past_values.rows = shape.first;
        input.history.past_values.columns = shape.second;
        bool rejected = false;
        try {
            pipeline.run(input, {});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "history shape must be complete and match the bundle channels");
    }
}

void test_observed_mask_survives_padding_and_truncation() {
    auto module = std::make_unique<RecordingModule>();
    auto* recording = module.get();
    trtmc::patchtsmixer::Pipeline pipeline(std::move(module), config());
    const std::vector<float> short_values{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<std::uint8_t> short_mask{1, 0, 0, 1};
    pipeline.run(request(short_values, short_mask), {});
    require(recording->mask == std::vector<float>({0, 0, 0, 0, 1, 0, 0, 1}),
            "padding must preserve caller-provided missing observations");
    const std::vector<float> long_values(12, 1.0F);
    const std::vector<std::uint8_t> long_mask{1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1};
    pipeline.run(request(long_values, long_mask), {});
    require(recording->mask == std::vector<float>({1, 0, 0, 1, 1, 1, 0, 1}),
            "truncation must crop the mask at exactly the same timestep as the values");
    bool rejected = false;
    try {
        pipeline.run(request(long_values, short_mask), {});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "observed mask must contain one entry per input value");
}

void test_result_owns_every_value_after_next_call_and_pipeline_destruction() {
    trtmc::internal::PointForecastResult first;
    {
        auto module = std::make_unique<RecordingModule>();
        auto* recording = module.get();
        recording->output = {1.25F, -2.5F, 3.75F, 4.5F};
        trtmc::patchtsmixer::Pipeline pipeline(std::move(module), config());
        const std::vector<float> values{1.0F, 2.0F};
        const std::vector<std::uint8_t> mask;
        first = pipeline.run(request(values, mask), {});
        recording->output.assign(4, 99.0F);
        const auto next = pipeline.run(request(values, mask), {});
        require(next.values.values == std::vector<float>(4, 99.0F),
                "each call must return its own engine output");
    }
    require(first.values.values == std::vector<float>({1.25F, -2.5F, 3.75F, 4.5F}) &&
                first.values.rows == 2 && first.values.columns == 2 &&
                first.axes.horizon_steps == std::vector<std::int64_t>({1, 2}),
            "result must own all values and axes beyond the engine lifetime");
}

} // namespace

int main() {
    test_runtime_config_requires_tensor_parallel_size();
    test_tensor_parallel_rank_contract_is_strict();
    test_short_multichannel_series_is_left_padded();
    test_overlong_multichannel_series_is_left_truncated();
    test_partial_multichannel_timestep_is_rejected();
    test_frequency_is_rejected();
    test_task_binding_declares_family_config();
    test_explicit_history_shape_matches_bundle_channels();
    test_observed_mask_survives_padding_and_truncation();
    test_result_owns_every_value_after_next_call_and_pipeline_destruction();
    std::cerr << "ALL PASSED\n";
    return 0;
}
