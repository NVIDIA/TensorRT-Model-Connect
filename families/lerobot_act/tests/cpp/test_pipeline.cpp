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
#include <utility>
#include <vector>

namespace {

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

trtmc::RobotObservation observation(std::vector<float>& image, std::vector<float>& state) {
    return {{image.data(), image.size()}, 2, 2, 3, {state.data(), state.size()}};
}

void test_chunk_contract_and_bounds() {
    auto module = std::make_unique<FakeActModule>();
    auto* module_ptr = module.get();
    trtmc::lerobot_act::Pipeline pipeline(std::move(module), 2, 2, 3, 2, 2, 3, {-1.0F, -1.0F},
                                          {1.0F, 1.0F});
    std::vector<float> image(12, 0.5F);
    std::vector<float> state{0.1F, -0.2F};
    auto result = pipeline.predict_action_chunk(observation(image, state));

    check(result.num_actions == 3 && result.action_dim == 2, "ACT action chunk shape");
    check(result.actions == module_ptr->actions, "ACT action chunk values");
    check(!result.within_training_bounds, "ACT reports out-of-training-range action");
    check(module_ptr->image_shape == std::vector<int64_t>({1, 2, 2, 3}),
          "ACT engine receives RGB HWC image");
    check(module_ptr->state_shape == std::vector<int64_t>({1, 2}),
          "ACT engine receives state vector");
    check(module_ptr->image_copy == image && module_ptr->state_copy == state,
          "ACT engine receives observation values unchanged");
}

void test_action_queue_and_reset() {
    auto module = std::make_unique<FakeActModule>();
    auto* module_ptr = module.get();
    trtmc::lerobot_act::Pipeline pipeline(std::move(module), 2, 2, 3, 2, 2, 3, {-2.0F, -2.0F},
                                          {2.0F, 2.0F});
    std::vector<float> image(12, 0.5F);
    std::vector<float> state{0.1F, -0.2F};
    const auto obs = observation(image, state);

    const auto first = pipeline.act(obs);
    state[0] = std::numeric_limits<float>::quiet_NaN();
    bool rejected_queued_observation = false;
    try {
        (void)pipeline.act(obs);
    } catch (const std::invalid_argument&) {
        rejected_queued_observation = true;
    }
    check(rejected_queued_observation, "ACT validates state while serving a queued action");
    state[0] = 0.1F;
    const auto second = pipeline.act(obs);
    check(first.started_new_chunk && !second.started_new_chunk, "ACT queue refill marker");
    check(first.values == std::vector<float>({0.25F, -0.25F}), "ACT first queued action");
    check(second.values == std::vector<float>({0.5F, -0.5F}), "ACT second queued action");
    check(module_ptr->forward_calls == 1, "ACT action queue avoids redundant inference");

    pipeline.reset();
    const auto restarted = pipeline.act(obs);
    check(restarted.started_new_chunk, "ACT reset forces a new chunk");
    check(restarted.values == first.values, "ACT reset restarts at the first action");
    check(module_ptr->forward_calls == 2 && module_ptr->reset_calls == 1,
          "ACT reset preserves engine and clears execution state");
}

void test_invalid_observations() {
    auto module = std::make_unique<FakeActModule>();
    trtmc::lerobot_act::Pipeline pipeline(std::move(module), 2, 2, 3, 2, 2, 3, {-1.0F, -1.0F},
                                          {1.0F, 1.0F});
    std::vector<float> image(12, 0.5F);
    std::vector<float> state{0.1F, -0.2F};

    auto expect_invalid = [&](trtmc::RobotObservation value) {
        try {
            (void)pipeline.predict_action_chunk(value);
            return false;
        } catch (const std::invalid_argument&) {
            return true;
        }
    };
    auto value = observation(image, state);
    value.image_pixels = {};
    check(expect_invalid(value), "ACT rejects missing image");
    value = observation(image, state);
    value.image_width = 3;
    check(expect_invalid(value), "ACT rejects wrong image shape");
    value = observation(image, state);
    value.state = {state.data(), 1};
    check(expect_invalid(value), "ACT rejects wrong state shape");
    image[3] = 1.1F;
    check(expect_invalid(observation(image, state)), "ACT rejects image outside [0,1]");
    image[3] = 0.5F;
    state[0] = std::numeric_limits<float>::quiet_NaN();
    check(expect_invalid(observation(image, state)), "ACT rejects non-finite state");
}

void test_invalid_construction() {
    bool threw = false;
    try {
        trtmc::lerobot_act::Pipeline pipeline(std::make_unique<FakeActModule>(), 2, 2, 3, 2, 2, 3,
                                              {-1.0F}, {1.0F});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "ACT rejects malformed training bounds");
}

} // namespace

int main() {
    test_chunk_contract_and_bounds();
    test_action_queue_and_reset();
    test_invalid_observations();
    test_invalid_construction();
    if (g_failures != 0) {
        std::cerr << g_failures << " LeRobot ACT pipeline test(s) failed\n";
        return 1;
    }
    return 0;
}
