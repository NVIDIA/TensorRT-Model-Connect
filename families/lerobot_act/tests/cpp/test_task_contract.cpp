/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lerobot_act/runtime/pipeline.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace internal = trtmc::internal;

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

class FakeActModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++forward_calls;
        const auto& image = inputs.at("observation_image");
        const auto& state = inputs.at("observation_state");
        image_shape = image.shape;
        state_shape = state.shape;
        const auto* image_values = static_cast<const float*>(image.data);
        const auto* state_values = static_cast<const float*>(state.data);
        image_copy.assign(image_values, image_values + image.numel());
        state_copy.assign(state_values, state_values + state.numel());
        return {{"actions", {actions.data(), {1, 3, 2}, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return name == "observation_image" || name == "observation_state";
    }
    bool has_output(const std::string& name) const override { return name == "actions"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return name == "actions" ? std::vector<int64_t>{1, 3, 2} : std::vector<int64_t>{};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 0; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override { ++reset_calls; }
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    std::vector<float> actions{0.25F, -0.25F, 0.5F, -0.5F, 1.25F, 0.0F};
    std::vector<int64_t> image_shape;
    std::vector<int64_t> state_shape;
    std::vector<float> image_copy;
    std::vector<float> state_copy;
    int forward_calls{0};
    int reset_calls{0};
};

using Pipeline = trtmc::lerobot_act::Pipeline;

std::unique_ptr<Pipeline> make_pipeline(FakeActModule** module_ptr, float bound) {
    auto module = std::make_unique<FakeActModule>();
    if (module_ptr)
        *module_ptr = module.get();
    return std::make_unique<Pipeline>(std::move(module), 2, 2, 3, 2, 2, 3,
                                      std::vector<float>{-bound, -bound},
                                      std::vector<float>{bound, bound});
}

internal::ImageStateObservation observation(const std::vector<float>& image,
                                            const std::vector<float>& state) {
    return {{image.data(), image.size() * sizeof(float), 2, 2, 3, internal::ImageFormat::Float32},
            {state.data(), state.size()}};
}

internal::ImageStateActionChunkResult run_chunk(Pipeline& pipeline,
                                                const internal::ImageStateObservation& value,
                                                internal::ConfigView config = {}) {
    return pipeline.run(internal::ImageStateToActionChunkRequest{value}, config);
}

bool schema_is_unnamed_act(const internal::ActionSchema& schema) {
    return schema.domain == "lerobot.act" && schema.component_names.empty() &&
           schema.units.empty() && schema.coordinate_frame.empty() &&
           schema.normalization == "unnormalized";
}

std::vector<float> row(const FakeActModule& module, std::size_t index) {
    return {module.actions[index * 2], module.actions[index * 2 + 1]};
}

void test_bindings_advertise_both_tasks() {
    auto pipeline = make_pipeline(nullptr, 1.0F);
    check(std::string_view(pipeline->task()) == "image_state_to_action_chunk",
          "ACT primary Task is the stateless chunk");
    const auto bindings = pipeline->task_bindings();
    check(bindings.size() == 2, "ACT binds the chunk and queue Tasks");
    if (bindings.size() != 2)
        return;
    check(bindings[0].key.id == internal::IImageStateToActionChunk::kTask &&
              bindings[0].key.major == 1 && bindings[0].key.minor == 0,
          "ACT chunk binding key");
    check(bindings[0].implementation ==
              static_cast<void*>(static_cast<internal::IImageStateToActionChunk*>(pipeline.get())),
          "ACT chunk binding points at its interface");
    check(bindings[1].key.id == internal::IImageStateActionQueue::kTask &&
              bindings[1].key.major == 1 && bindings[1].key.minor == 0,
          "ACT queue binding key");
    check(bindings[1].implementation ==
              static_cast<void*>(static_cast<internal::IImageStateActionQueue*>(pipeline.get())),
          "ACT queue binding points at its interface");
    check(bindings[0].fields.empty() && bindings[1].fields.empty(),
          "ACT declares no runtime Config fields");
}

void test_chunk_contract_and_bounds() {
    FakeActModule* module = nullptr;
    auto pipeline = make_pipeline(&module, 1.0F);
    const std::vector<float> image(12, 0.5F);
    const std::vector<float> state{0.1F, -0.2F};
    const auto result = run_chunk(*pipeline, observation(image, state));

    check(result.actions.values.rows == 3 && result.actions.values.columns == 2,
          "ACT action chunk shape");
    check(result.actions.values.values == module->actions, "ACT action chunk values");
    check(schema_is_unnamed_act(result.actions.schema), "ACT action chunk schema");
    check(result.actions.timestamps_seconds.empty() && result.actions.frame_spans.empty(),
          "ACT invents no action clock or frame association");
    check(!result.within_training_bounds, "ACT reports out-of-training-range action");
    check(std::isfinite(result.inference_ms) && result.inference_ms >= 0.0,
          "ACT reports its engine time");
    check(module->image_shape == std::vector<int64_t>({1, 2, 2, 3}),
          "ACT engine receives RGB HWC image");
    check(module->state_shape == std::vector<int64_t>({1, 2}), "ACT engine receives state vector");
    check(module->image_copy == image && module->state_copy == state,
          "ACT engine receives observation values unchanged");

    auto in_range = make_pipeline(nullptr, 2.0F);
    check(run_chunk(*in_range, observation(image, state)).within_training_bounds,
          "ACT reports an in-range chunk");
}

void test_uint8_images_are_scaled() {
    FakeActModule* module = nullptr;
    auto pipeline = make_pipeline(&module, 1.0F);
    const std::vector<std::uint8_t> image{0, 1, 51, 127, 128, 200, 254, 255, 3, 17, 99, 240};
    const std::vector<float> state{0.1F, -0.2F};
    const internal::ImageStateObservation value{
        {image.data(), image.size(), 2, 2, 3, internal::ImageFormat::UInt8},
        {state.data(), state.size()}};
    const auto result = run_chunk(*pipeline, value);

    std::vector<float> expected;
    for (const auto byte : image)
        expected.push_back(static_cast<float>(byte) / 255.0F);
    check(module->image_copy == expected, "ACT divides UInt8 pixels by 255");
    check(result.actions.values.values == module->actions, "ACT UInt8 chunk values");
}

void test_action_queue_refill_and_reset() {
    FakeActModule* module = nullptr;
    auto pipeline = make_pipeline(&module, 2.0F);
    const std::vector<float> image(12, 0.5F);
    std::vector<float> state{0.1F, -0.2F};
    const auto value = observation(image, state);
    auto session = pipeline->create_action_session({});

    const auto first = session->act(value, {});
    check(first.started_new_chunk && first.inference_ms >= 0.0, "ACT queue starts a chunk");
    check(first.action.values.rows == 1 && first.action.values.columns == 2,
          "ACT queued step is one action row");
    check(first.action.values.values == row(*module, 0), "ACT first queued action");
    check(schema_is_unnamed_act(first.action.schema), "ACT queued step schema");
    check(first.within_training_bounds, "ACT reports an in-range queued action");

    state[0] = std::numeric_limits<float>::quiet_NaN();
    bool rejected_queued_observation = false;
    try {
        (void)session->act(value, {});
    } catch (const std::invalid_argument&) {
        rejected_queued_observation = true;
    }
    check(rejected_queued_observation, "ACT validates state while serving a queued action");
    state[0] = 0.1F;

    const auto independent = run_chunk(*pipeline, value);
    check(independent.actions.values.values == module->actions && module->forward_calls == 2,
          "ACT runs an independent chunk while a queue is alive");
    const auto second = session->act(value, {});
    check(!second.started_new_chunk && second.inference_ms == 0.0,
          "ACT queued step reports no new inference");
    check(second.action.values.values == row(*module, 1),
          "ACT independent chunk leaves the queue in place");
    const auto third = session->act(value, {});
    check(third.action.values.values == row(*module, 2), "ACT third queued action");
    module->actions[0] = 1.5F;
    const auto refilled = session->act(value, {});
    check(refilled.started_new_chunk && refilled.action.values.values == row(*module, 0),
          "ACT spent queue refills from a new chunk");
    check(module->forward_calls == 3, "ACT action queue avoids redundant inference");

    session->reset();
    const auto restarted = session->act(value, {});
    check(restarted.started_new_chunk && restarted.action.values.values == row(*module, 0),
          "ACT reset forces a new chunk from its first action");
    check(module->forward_calls == 4 && module->reset_calls == 1,
          "ACT reset preserves engine and clears execution state");

    auto other = pipeline->create_action_session({});
    check(other->act(value, {}).started_new_chunk, "ACT sessions own separate queues");

    auto narrow = make_pipeline(&module, 2.0F);
    module->actions[2] = 9.0F;
    auto narrow_session = narrow->create_action_session({});
    check(narrow_session->act(value, {}).within_training_bounds,
          "ACT bounds each queued action separately (first row)");
    check(!narrow_session->act(value, {}).within_training_bounds,
          "ACT bounds each queued action separately (second row)");
}

template <class Call>
bool throws_invalid(Call&& call) {
    try {
        call();
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

void test_invalid_observations() {
    auto pipeline = make_pipeline(nullptr, 1.0F);
    auto session = pipeline->create_action_session({});
    std::vector<float> image(12, 0.5F);
    std::vector<float> state{0.1F, -0.2F};
    auto expect_invalid = [&](internal::ImageStateObservation value, const char* name) {
        check(throws_invalid([&] { (void)run_chunk(*pipeline, value); }), name);
        check(throws_invalid([&] { (void)session->act(value, {}); }), name);
    };

    auto value = observation(image, state);
    value.image.data = nullptr;
    expect_invalid(value, "ACT rejects missing image");
    value = observation(image, state);
    value.state = {};
    expect_invalid(value, "ACT rejects missing state");
    value = observation(image, state);
    value.image.width = 3;
    expect_invalid(value, "ACT rejects wrong image shape");
    value = observation(image, state);
    value.image.channels = 4;
    expect_invalid(value, "ACT rejects RGBA images");
    value = observation(image, state);
    value.image.byte_size -= sizeof(float);
    expect_invalid(value, "ACT rejects image storage that disagrees with its shape");
    value = observation(image, state);
    value.image.format = static_cast<internal::ImageFormat>(7);
    expect_invalid(value, "ACT rejects an unknown pixel format");
    value = observation(image, state);
    value.state = {state.data(), 1};
    expect_invalid(value, "ACT rejects wrong state shape");
    state[0] = std::numeric_limits<float>::infinity();
    expect_invalid(observation(image, state), "ACT rejects non-finite state");
    state[0] = 0.1F;

    image[3] = 1.1F;
    check(throws_invalid([&] { (void)run_chunk(*pipeline, observation(image, state)); }),
          "ACT rejects image outside [0,1]");
    image[3] = std::numeric_limits<float>::quiet_NaN();
    check(throws_invalid([&] { (void)run_chunk(*pipeline, observation(image, state)); }),
          "ACT rejects non-finite image pixels");
    check(throws_invalid([&] { (void)session->act(observation(image, state), {}); }),
          "ACT scans pixels when a queue refills");
}

void test_runtime_config_is_rejected() {
    auto pipeline = make_pipeline(nullptr, 1.0F);
    const std::vector<float> image(12, 0.5F);
    const std::vector<float> state{0.1F, -0.2F};
    const internal::ConfigEntry entry{"seed", internal::ConfigValue{std::int64_t{1}}};
    const internal::ConfigView config{&entry, 1};
    auto rejects = [](auto&& call) {
        try {
            call();
            return false;
        } catch (const internal::ConfigError&) {
            return true;
        }
    };
    check(rejects([&] { (void)run_chunk(*pipeline, observation(image, state), config); }),
          "ACT chunk rejects runtime Config");
    check(rejects([&] { (void)pipeline->create_action_session(config); }),
          "ACT queue creation rejects runtime Config");
    auto session = pipeline->create_action_session({});
    check(rejects([&] { (void)session->act(observation(image, state), config); }),
          "ACT queued step rejects runtime Config");
}

void test_invalid_construction() {
    bool threw = false;
    try {
        Pipeline pipeline(std::make_unique<FakeActModule>(), 2, 2, 3, 2, 2, 3, {-1.0F}, {1.0F});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "ACT rejects malformed training bounds");
}

} // namespace

int main() {
    test_bindings_advertise_both_tasks();
    test_chunk_contract_and_bounds();
    test_uint8_images_are_scaled();
    test_action_queue_refill_and_reset();
    test_invalid_observations();
    test_runtime_config_is_rejected();
    test_invalid_construction();
    if (g_failures != 0) {
        std::cerr << g_failures << " LeRobot ACT Task contract test(s) failed\n";
        return 1;
    }
    return 0;
}
