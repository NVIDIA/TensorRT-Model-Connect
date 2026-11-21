/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lerobot_act/runtime/pipeline.h"

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

void validate_image_values(const float* pixels, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        const float value = pixels[index];
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
            throw std::invalid_argument(
                "LeRobot ACT image pixels must be finite RGB HWC values in [0, 1]");
    }
}

void reject_config(internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("LeRobot ACT accepts no runtime configuration");
}

// The checkpoint orders the action components but names none of them and
// states no units or frame, so the schema leaves those empty.
internal::ActionSchema action_schema() {
    return {"lerobot.act", {}, {}, "", "unnormalized"};
}

} // namespace

// The family-owned action queue: one chunk inference, then one queued step per
// call until the chunk is spent or the session is reset.
class Pipeline::ActionQueue final : public internal::IImageStateActionSession {
  public:
    explicit ActionQueue(Pipeline& pipeline)
        : pipeline_(pipeline), next_action_(static_cast<std::size_t>(pipeline.chunk_size_)) {}

    internal::ActionStepResult act(const internal::ImageStateObservation& observation,
                                   internal::ConfigView config) override {
        reject_config(config);
        // A queued action does not consume new image pixels, but every call
        // still enforces shape and finite-state contracts. The full pixel scan
        // remains on chunk refill to preserve the 50 Hz action-step budget.
        pipeline_.validate_shape(observation);
        pipeline_.validate_state(observation.state);
        internal::ActionStepResult result;
        if (next_action_ >= static_cast<std::size_t>(pipeline_.chunk_size_)) {
            auto chunk = pipeline_.predict(observation);
            queued_actions_ = std::move(chunk.actions.values.values);
            next_action_ = 0;
            result.started_new_chunk = true;
            result.inference_ms = chunk.inference_ms;
        }

        const auto action_dim = static_cast<std::size_t>(pipeline_.action_dim_);
        const float* action = queued_actions_.data() + next_action_ * action_dim;
        result.action.values.values.assign(action, action + action_dim);
        result.action.values.rows = 1;
        result.action.values.columns = action_dim;
        result.action.schema = action_schema();
        result.within_training_bounds = pipeline_.action_within_bounds(action);
        ++next_action_;
        return result;
    }

    void reset() override {
        queued_actions_.clear();
        next_action_ = static_cast<std::size_t>(pipeline_.chunk_size_);
        pipeline_.policy_->reset_execution_context();
    }

  private:
    Pipeline& pipeline_;
    std::vector<float> queued_actions_;
    std::size_t next_action_{0};
};

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
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {internal::bind<internal::IImageStateToActionChunk>(*this),
            internal::bind<internal::IImageStateActionQueue>(*this)};
}

void Pipeline::validate_shape(const internal::ImageStateObservation& observation) const {
    const auto& image = observation.image;
    if (image.data == nullptr)
        throw std::invalid_argument("LeRobot ACT observation image is required");
    if (observation.state.empty())
        throw std::invalid_argument("LeRobot ACT observation state is required");
    if (image.height != static_cast<uint32_t>(image_height_) ||
        image.width != static_cast<uint32_t>(image_width_) ||
        image.channels != static_cast<uint32_t>(image_channels_))
        throw std::invalid_argument(
            "LeRobot ACT observation image shape does not match the bundle");
    if (image.format != internal::ImageFormat::Float32 &&
        image.format != internal::ImageFormat::UInt8)
        throw std::invalid_argument("LeRobot ACT accepts Float32 or UInt8 RGB images");
    const auto element_size =
        image.format == internal::ImageFormat::Float32 ? sizeof(float) : sizeof(std::uint8_t);
    const auto expected_pixels =
        static_cast<std::size_t>(image_height_) * image_width_ * image_channels_;
    if (image.byte_size != expected_pixels * element_size)
        throw std::invalid_argument(
            "LeRobot ACT observation image values do not match the declared shape");
    if (observation.state.size() != static_cast<std::size_t>(state_dim_))
        throw std::invalid_argument(
            "LeRobot ACT observation state dimension does not match the bundle");
}

void Pipeline::validate_state(Span<const float> state) const {
    for (const float value : state) {
        if (!std::isfinite(value))
            throw std::invalid_argument("LeRobot ACT state values must be finite");
    }
}

const float* Pipeline::image_pixels(const internal::ImageView& image) {
    const auto count = static_cast<std::size_t>(image_height_) * image_width_ * image_channels_;
    if (image.format == internal::ImageFormat::Float32) {
        const auto* pixels = static_cast<const float*>(image.data);
        validate_image_values(pixels, count);
        return pixels;
    }
    const auto* bytes = static_cast<const std::uint8_t*>(image.data);
    converted_pixels_.resize(count);
    for (std::size_t index = 0; index < count; ++index)
        converted_pixels_[index] = static_cast<float>(bytes[index]) / 255.0F;
    return converted_pixels_.data();
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

internal::ImageStateActionChunkResult
Pipeline::predict(const internal::ImageStateObservation& observation) {
    validate_shape(observation);
    const float* pixels = image_pixels(observation.image);
    validate_state(observation.state);

    Tensor image;
    image.data = const_cast<float*>(pixels);
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

    internal::ImageStateActionChunkResult result;
    auto& values = result.actions.values;
    values.values.resize(expected_values);
    std::memcpy(values.values.data(), actions->data, expected_values * sizeof(float));
    values.rows = static_cast<std::uint64_t>(chunk_size_);
    values.columns = static_cast<std::uint64_t>(action_dim_);
    result.actions.schema = action_schema();
    result.within_training_bounds = true;
    for (int32_t step = 0; step < chunk_size_; ++step) {
        result.within_training_bounds =
            result.within_training_bounds &&
            action_within_bounds(values.values.data() +
                                 static_cast<std::size_t>(step) * action_dim_);
    }
    result.inference_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    return result;
}

internal::ImageStateActionChunkResult
Pipeline::run(const internal::ImageStateToActionChunkRequest& request,
              internal::ConfigView config) {
    reject_config(config);
    return predict(request.observation);
}

std::unique_ptr<internal::IImageStateActionSession>
Pipeline::create_action_session(internal::ConfigView config) {
    reject_config(config);
    return std::make_unique<ActionQueue>(*this);
}

} // namespace trtmc::lerobot_act
