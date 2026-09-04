/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lerobot_act/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace trtmc::lerobot_act {

namespace {

const Tensor* find_actions(const TensorMap& outputs) {
    if (const auto it = outputs.find("actions"); it != outputs.end())
        return &it->second;
    return nullptr;
}

void validate_policy_module(const ITrtModule* policy) {
    if (!policy || !policy->ok())
        throw std::runtime_error("LeRobotActPipeline: invalid policy module");
}

void validate_contract_dimensions(int32_t image_height, int32_t image_width, int32_t image_channels,
                                  int32_t state_dim, int32_t action_dim, int32_t chunk_size) {
    if (image_height <= 0 || image_width <= 0 || image_channels != 3 || state_dim <= 0 ||
        action_dim <= 0 || chunk_size <= 0)
        throw std::runtime_error("LeRobotActPipeline: invalid observation/action contract");
}

void validate_action_bounds(const std::vector<float>& action_min,
                            const std::vector<float>& action_max, int32_t action_dim) {
    if (action_min.size() != static_cast<std::size_t>(action_dim) ||
        action_max.size() != static_cast<std::size_t>(action_dim))
        throw std::runtime_error("LeRobotActPipeline: invalid action training bounds");
    for (int32_t index = 0; index < action_dim; ++index) {
        if (!std::isfinite(action_min[index]) || !std::isfinite(action_max[index]) ||
            action_min[index] > action_max[index])
            throw std::runtime_error("LeRobotActPipeline: malformed action training bounds");
    }
}

void validate_observation_shape(const RobotObservation& observation, int32_t image_height,
                                int32_t image_width, int32_t image_channels, int32_t state_dim) {
    if (observation.image_pixels.empty())
        throw std::invalid_argument("LeRobot ACT observation image is required");
    if (observation.state.empty())
        throw std::invalid_argument("LeRobot ACT observation state is required");
    if (observation.image_height != image_height || observation.image_width != image_width ||
        observation.image_channels != image_channels)
        throw std::invalid_argument(
            "LeRobot ACT observation image shape does not match the bundle");
    const auto expected_pixels =
        static_cast<std::size_t>(image_height) * image_width * image_channels;
    if (observation.image_pixels.size() != expected_pixels)
        throw std::invalid_argument(
            "LeRobot ACT observation image values do not match the declared shape");
    if (observation.state.size() != static_cast<std::size_t>(state_dim))
        throw std::invalid_argument(
            "LeRobot ACT observation state dimension does not match the bundle");
}

void validate_image_values(const float* pixels, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        const float value = pixels[index];
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
            throw std::invalid_argument(
                "LeRobot ACT image pixels must be finite RGB HWC values in [0, 1]");
    }
}

void validate_state_values(const float* state, int32_t count) {
    for (int32_t index = 0; index < count; ++index) {
        if (!std::isfinite(state[index]))
            throw std::invalid_argument("LeRobot ACT state values must be finite");
    }
}

} // namespace

Pipeline::Pipeline(std::unique_ptr<ITrtModule> policy, int32_t image_height, int32_t image_width,
                   int32_t image_channels, int32_t state_dim, int32_t action_dim,
                   int32_t chunk_size, std::vector<float> action_min, std::vector<float> action_max)
    : policy_(std::move(policy)), image_height_(image_height), image_width_(image_width),
      image_channels_(image_channels), state_dim_(state_dim), action_dim_(action_dim),
      chunk_size_(chunk_size), action_min_(std::move(action_min)),
      action_max_(std::move(action_max)) {
    validate_policy_module(policy_.get());
    validate_contract_dimensions(image_height_, image_width_, image_channels_, state_dim_,
                                 action_dim_, chunk_size_);
    validate_action_bounds(action_min_, action_max_, action_dim_);
    next_action_ = static_cast<std::size_t>(chunk_size_);
}

void Pipeline::validate_observation(const RobotObservation& observation) const {
    validate_observation_shape(observation, image_height_, image_width_, image_channels_,
                               state_dim_);
    const auto image_values =
        static_cast<std::size_t>(image_height_) * image_width_ * image_channels_;
    validate_image_values(observation.image_pixels.data(), image_values);
    validate_state_values(observation.state.data(), state_dim_);
}

bool Pipeline::action_within_bounds(const float* action) const {
    if (!action)
        return false;
    for (int32_t index = 0; index < action_dim_; ++index) {
        const float value = action[index];
        if (!std::isfinite(value) || value < action_min_[index] || value > action_max_[index])
            return false;
    }
    return true;
}

RobotActionChunk Pipeline::predict_action_chunk(const RobotObservation& observation) {
    validate_observation(observation);

    Tensor image;
    image.data = const_cast<float*>(observation.image_pixels.data());
    image.shape = {1, image_height_, image_width_, image_channels_};
    image.dtype = DType::kFloat32;
    Tensor state;
    state.data = const_cast<float*>(observation.state.data());
    state.shape = {1, state_dim_};
    state.dtype = DType::kFloat32;
    TensorMap inputs{{"observation_image", image}, {"observation_state", state}};

    const auto begin = std::chrono::steady_clock::now();
    auto outputs = policy_->forward(inputs);
    const auto end = std::chrono::steady_clock::now();
    const auto* actions = find_actions(outputs);
    if (!actions || !actions->data)
        throw std::runtime_error("LeRobot ACT engine returned no action tensor");
    if (actions->dtype != DType::kFloat32)
        throw std::runtime_error("LeRobot ACT engine action tensor must be float32");
    const auto expected_values = static_cast<std::size_t>(chunk_size_) * action_dim_;
    if (actions->numel() != expected_values)
        throw std::runtime_error("LeRobot ACT engine returned an unexpected action shape");

    RobotActionChunk result;
    result.actions.resize(expected_values);
    std::memcpy(result.actions.data(), actions->data, expected_values * sizeof(float));
    result.num_actions = chunk_size_;
    result.action_dim = action_dim_;
    result.within_training_bounds = true;
    for (int32_t step = 0; step < chunk_size_; ++step) {
        result.within_training_bounds =
            result.within_training_bounds &&
            action_within_bounds(result.actions.data() +
                                 static_cast<std::size_t>(step) * action_dim_);
    }
    result.inference_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    return result;
}

RobotAction Pipeline::act(const RobotObservation& observation) {
    // A queued action does not consume new image pixels, but every public call
    // still enforces pointer/shape and finite-state contracts. The full pixel
    // scan remains on chunk refill to preserve the 50 Hz action-step budget.
    validate_observation_shape(observation, image_height_, image_width_, image_channels_,
                               state_dim_);
    validate_state_values(observation.state.data(), state_dim_);
    bool started_new_chunk = false;
    double inference_ms = 0.0;
    if (next_action_ >= static_cast<std::size_t>(chunk_size_)) {
        auto chunk = predict_action_chunk(observation);
        queued_actions_ = std::move(chunk.actions);
        next_action_ = 0;
        started_new_chunk = true;
        inference_ms = chunk.inference_ms;
    }

    const auto offset = next_action_ * static_cast<std::size_t>(action_dim_);
    RobotAction result;
    result.values.assign(queued_actions_.begin() + static_cast<std::ptrdiff_t>(offset),
                         queued_actions_.begin() +
                             static_cast<std::ptrdiff_t>(offset + action_dim_));
    result.action_dim = action_dim_;
    result.within_training_bounds = action_within_bounds(result.values.data());
    result.started_new_chunk = started_new_chunk;
    result.inference_ms = inference_ms;
    ++next_action_;
    return result;
}

void Pipeline::reset() {
    queued_actions_.clear();
    next_action_ = static_cast<std::size_t>(chunk_size_);
    policy_->reset_execution_context();
}

} // namespace trtmc::lerobot_act
