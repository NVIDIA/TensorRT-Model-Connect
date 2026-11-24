/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lerobot_pi05/runtime/pipeline.h"

#include "families/lerobot_pi05/runtime/openpi.h"
#include "families/lerobot_pi05/runtime/request.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace trtmc::lerobot_pi05 {
namespace {

// LeRobot's additive attention mask for blocked keys.
constexpr float kBlocked = -2.3819763e38F;

Tensor tensor_of(void* data, std::vector<std::int64_t> shape, DType dtype) {
    Tensor tensor;
    tensor.data = data;
    tensor.shape = std::move(shape);
    tensor.dtype = dtype;
    return tensor;
}

const Tensor& require_output(const TensorMap& outputs, const char* name, std::size_t numel) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || found->second.data == nullptr)
        throw std::runtime_error(std::string("lerobot_pi05 engine did not return ") + name);
    if (found->second.numel() != numel)
        throw std::runtime_error(
            std::string("lerobot_pi05 engine returned an unexpected size for ") + name);
    return found->second;
}

void require_shape(const ITrtModule& engine, const char* name, std::vector<std::int64_t> expected,
                   const char* what) {
    if (!engine.has_input(name) && !engine.has_output(name))
        throw std::runtime_error(std::string("lerobot_pi05 ") + what + " engine has no tensor " +
                                 name);
    if (engine.tensor_shape(name) != expected)
        throw std::runtime_error(std::string("lerobot_pi05 ") + what + " engine tensor " + name +
                                 " does not match runtime.json");
}

} // namespace

Pipeline::Pipeline(std::unique_ptr<ITrtModule> prefix, std::unique_ptr<ITrtModule> denoise,
                   PaligemmaBpeTokenizer tokenizer, RuntimeConfig config, std::string primary_task)
    : prefix_(std::move(prefix)), denoise_(std::move(denoise)), tokenizer_(std::move(tokenizer)),
      config_(std::move(config)), primary_task_(std::move(primary_task)),
      rng_(std::random_device{}()) {
    if (!prefix_ || !prefix_->ok() || !denoise_ || !denoise_->ok())
        throw std::runtime_error("lerobot_pi05: invalid engine");
    if (config_.cameras.empty() || config_.camera_height <= 0 || config_.camera_width <= 0 ||
        config_.tokens_per_image <= 0 || config_.prompt_tokens <= 0 || config_.chunk_size <= 0 ||
        config_.state_dim <= 0 || config_.action_dim <= 0 ||
        config_.action_dim > config_.max_action_dim || config_.num_inference_steps <= 0 ||
        config_.prefix_tokens !=
            config_.camera_slots() * config_.tokens_per_image + config_.prompt_tokens) {
        throw std::runtime_error("lerobot_pi05 runtime.json does not match its runtime contract");
    }
    if (!config_.openpi_policy.empty()) {
        const bool libero = config_.openpi_policy == "pi05_libero";
        const bool droid = config_.openpi_policy == "pi05_droid";
        const std::vector<std::string> expected =
            libero
                ? std::vector<std::string>{"observation/image", "observation/wrist_image"}
                : (droid
                       ? std::vector<std::string>{"observation/exterior_image_1_left",
                                                  "observation/wrist_image_left"}
                       : std::vector<std::string>{"cam_high", "cam_left_wrist", "cam_right_wrist"});
        if ((!libero && !droid && config_.openpi_policy != "pi05_aloha") ||
            config_.cameras != expected || config_.camera_height != 224 ||
            config_.camera_width != 224 || config_.empty_cameras != 0 ||
            config_.state_dim != (libero || droid ? 8 : 14) ||
            config_.action_dim != (libero ? 7 : (droid ? 8 : 14)) ||
            config_.chunk_size != (libero ? 10 : (droid ? 15 : 50)) ||
            config_.max_action_dim != 32 || config_.prompt_tokens != 200 ||
            config_.num_inference_steps != 10 || config_.tokens_per_image != 256)
            throw std::runtime_error("Invalid openpi runtime contract");
    }
    const auto check_stats = [](const Normalization& norm, std::int32_t dim, const char* what) {
        if (!norm.identity() && (norm.offset.size() != static_cast<std::size_t>(dim) ||
                                 norm.scale.size() != norm.offset.size()))
            throw std::runtime_error(std::string("lerobot_pi05 ") + what +
                                     " statistics do not match the feature size");
    };
    check_stats(config_.state_normalization, config_.state_dim, "state");
    check_stats(config_.action_normalization, config_.action_dim, "action");
    const auto range_size = static_cast<std::size_t>(config_.action_dim);
    if (config_.action_min.size() != config_.action_max.size() ||
        (!config_.action_min.empty() && config_.action_min.size() != range_size))
        throw std::runtime_error("lerobot_pi05 action range does not match the action size");
    validate_engines();
}

// Both engines must agree with runtime.json and with each other on the cache they hand over.
void Pipeline::validate_engines() {
    const std::int64_t prefix = config_.prefix_tokens;
    require_shape(*prefix_, "pixel_values",
                  {config_.camera_slots(), 3, config_.camera_height, config_.camera_width},
                  "prefix");
    require_shape(*prefix_, "token_ids", {config_.prompt_tokens}, "prefix");
    require_shape(*prefix_, "position_ids", {prefix}, "prefix");
    require_shape(*prefix_, "attention_bias", {prefix, prefix}, "prefix");
    require_shape(*denoise_, "x_t", {config_.chunk_size, config_.max_action_dim}, "denoise");
    require_shape(*denoise_, "prefix_bias", {prefix}, "denoise");
    for (const char* name : {"time", "dt", "position_offset"})
        require_shape(*denoise_, name, {1}, "denoise");
    for (const char* name : {"k_cache", "v_cache"}) {
        const auto shape = denoise_->tensor_shape(name);
        if (!prefix_->has_output(name) || prefix_->tensor_shape(name) != shape ||
            prefix_->tensor_dtype(name) != denoise_->tensor_dtype(name) || shape.size() != 3 ||
            shape[1] != prefix)
            throw std::runtime_error(
                std::string("lerobot_pi05 prefix and denoise engines disagree on ") + name);
        cache_numel_ = static_cast<std::size_t>(shape[0] * shape[1] * shape[2]);
    }
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {internal::bind<internal::IImagesStateTextToActionChunk>(
        *this, Span<const internal::ConfigField>{kFields, std::size(kFields)})};
}

std::vector<float> Pipeline::pack_pixels(const internal::ImagesStateTextObservation& observation,
                                         std::vector<bool>& seen) const {
    const auto height = static_cast<std::size_t>(config_.camera_height);
    const auto width = static_cast<std::size_t>(config_.camera_width);
    const std::size_t plane = height * width;
    std::vector<float> pixels(static_cast<std::size_t>(config_.camera_slots()) * 3 * plane, 0.0F);

    seen.assign(config_.cameras.size(), false);
    for (const auto& named : observation.images) {
        if (config_.openpi_policy == "pi05_aloha" && named.name == "cam_low")
            continue;
        const auto found =
            std::find(config_.cameras.begin(), config_.cameras.end(), std::string(named.name));
        if (found == config_.cameras.end())
            throw std::invalid_argument("lerobot_pi05 does not accept a camera named '" +
                                        std::string(named.name) + "'");
        const auto slot = static_cast<std::size_t>(found - config_.cameras.begin());
        const auto& image = named.image;
        if (seen[slot])
            throw std::invalid_argument("duplicate pi05 camera: " + std::string(named.name));
        if (!config_.openpi_policy.empty()) {
            openpi_image(image, pixels.data() + slot * 3 * plane);
            seen[slot] = true;
            continue;
        }
        if (image.height != height || image.width != width || image.channels != 3)
            throw std::invalid_argument("lerobot_pi05 camera '" + std::string(named.name) +
                                        "' must be " + std::to_string(height) + "x" +
                                        std::to_string(width) + "x3");
        seen[slot] = true;
        float* destination = pixels.data() + slot * 3 * plane;
        const bool floating = image.format == internal::ImageFormat::Float32;
        for (std::size_t index = 0; index < plane; ++index) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const float value =
                    floating ? static_cast<const float*>(image.data)[index * 3 + channel]
                             : static_cast<const std::uint8_t*>(image.data)[index * 3 + channel] /
                                   255.0F;
                if (!std::isfinite(value))
                    throw std::invalid_argument("lerobot_pi05 camera '" + std::string(named.name) +
                                                "' contains a non-finite pixel");
                destination[channel * plane + index] = value;
            }
        }
    }
    for (std::size_t slot = 0; slot < seen.size(); ++slot)
        if (!seen[slot] && (config_.openpi_policy != "pi05_aloha" || slot == 0))
            throw std::invalid_argument("lerobot_pi05 requires camera '" + config_.cameras[slot] +
                                        "'");
    return pixels;
}

std::vector<float> Pipeline::normalized_state(Span<const float> state) const {
    if (state.size() != static_cast<std::size_t>(config_.state_dim))
        throw std::invalid_argument("lerobot_pi05 state must have " +
                                    std::to_string(config_.state_dim) + " values");
    auto values = openpi_state(config_.openpi_policy, state);
    const auto& norm = config_.state_normalization;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index]))
            throw std::invalid_argument("lerobot_pi05 state must be finite");
        values[index] = normalize_value(norm, index, values[index]);
    }
    return values;
}

Pipeline::PrefixInputs
Pipeline::prefix_inputs(const internal::ImagesStateTextObservation& observation) const {
    if (observation.text.empty())
        throw std::invalid_argument("lerobot_pi05 requires a non-empty instruction");
    PrefixInputs inputs;
    std::vector<bool> seen;
    inputs.pixels = pack_pixels(observation, seen);
    const auto state = normalized_state(observation.state);
    auto prompt = config_.openpi_policy == "pi05_libero"
                      ? tokenizer_.tokenize_task_only(observation.text, config_.prompt_tokens)
                      : tokenizer_.tokenize_pi05(observation.text, state,
                                                 static_cast<std::size_t>(config_.prompt_tokens));
    if (prompt.truncated && config_.openpi_policy.empty())
        throw std::invalid_argument("lerobot_pi05 instruction and state exceed " +
                                    std::to_string(config_.prompt_tokens) + " prompt tokens");
    inputs.token_ids = std::move(prompt.token_ids);

    // Masks and positions follow LeRobot: a masked camera slot or a padded prompt token neither
    // attends nor advances the position counter. Real cameras fill the first slots.
    const auto prefix = static_cast<std::size_t>(config_.prefix_tokens);
    const auto tokens_per_image = static_cast<std::size_t>(config_.tokens_per_image);
    std::vector<std::uint8_t> valid(prefix, 0);
    for (std::size_t i = 0; i < seen.size(); ++i)
        if (seen[i])
            std::fill_n(valid.begin() + i * tokens_per_image, tokens_per_image, 1);
    const auto prompt_start = static_cast<std::size_t>(config_.camera_slots()) * tokens_per_image;
    for (std::size_t index = 0; index < prompt.token_mask.size(); ++index)
        valid[prompt_start + index] = prompt.token_mask[index] != 0 ? 1 : 0;

    inputs.positions.resize(prefix);
    for (std::size_t index = 0; index < prefix; ++index) {
        inputs.real_tokens += valid[index];
        inputs.positions[index] = inputs.real_tokens - 1;
    }
    inputs.attention_bias.assign(prefix * prefix, kBlocked);
    inputs.prefix_bias.assign(prefix, kBlocked);
    for (std::size_t row = 0; row < prefix; ++row) {
        if (valid[row] == 0)
            continue;
        inputs.prefix_bias[row] = 0.0F;
        for (std::size_t column = 0; column < prefix; ++column)
            if (valid[column] != 0)
                inputs.attention_bias[row * prefix + column] = 0.0F;
    }
    return inputs;
}

std::vector<float> Pipeline::denoise(PrefixInputs& inputs, const Tensor& k_cache,
                                     const Tensor& v_cache, std::vector<float> sample) {
    // LeRobot's schedule: t = 1 + step * dt with dt = -1 / steps, evaluated in double.
    const double dt = -1.0 / config_.num_inference_steps;
    float step_size = static_cast<float>(dt);
    for (std::int32_t step = 0; step < config_.num_inference_steps; ++step) {
        float time = static_cast<float>(1.0 + step * dt);
        auto outputs = denoise_->forward(
            {{"x_t", tensor_of(sample.data(), {config_.chunk_size, config_.max_action_dim},
                               DType::kFloat32)},
             {"time", tensor_of(&time, {1}, DType::kFloat32)},
             {"dt", tensor_of(&step_size, {1}, DType::kFloat32)},
             {"position_offset", tensor_of(&inputs.real_tokens, {1}, DType::kInt32)},
             {"prefix_bias",
              tensor_of(inputs.prefix_bias.data(), {config_.prefix_tokens}, DType::kFloat32)},
             {"k_cache", k_cache},
             {"v_cache", v_cache}});
        const Tensor& next = require_output(outputs, "x_next", sample.size());
        std::memcpy(sample.data(), next.data, sample.size() * sizeof(float));
    }
    return sample;
}

internal::ImagesStateTextActionChunkResult
Pipeline::pack_actions(const std::vector<float>& sample) const {
    // Keep the leading action_dim columns of the padded action and undo the normalization.
    internal::ImagesStateTextActionChunkResult result;
    const auto rows = static_cast<std::size_t>(config_.chunk_size);
    const auto columns = static_cast<std::size_t>(config_.action_dim);
    const auto padded = static_cast<std::size_t>(config_.max_action_dim);
    result.actions.values.rows = rows;
    result.actions.values.columns = columns;
    result.actions.values.values.reserve(rows * columns);
    // Within range only when the checkpoint knows its dataset range and every action is inside.
    bool within = !config_.action_min.empty();
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            const float value = unnormalize_value(config_.action_normalization, column,
                                                  sample[row * padded + column]);
            if (!std::isfinite(value))
                throw std::runtime_error("lerobot_pi05 produced a non-finite action");
            if (within &&
                (value < config_.action_min[column] || value > config_.action_max[column]))
                within = false;
            result.actions.values.values.push_back(value);
        }
    }
    result.actions.schema = {"lerobot.pi05", {}, {}, "", "unnormalized"};
    result.within_training_bounds = within;
    return result;
}

internal::ImagesStateTextActionChunkResult
Pipeline::run(const internal::ImagesStateTextToActionChunkRequest& request,
              internal::ConfigView config) {
    const auto started = std::chrono::steady_clock::now();
    const auto options = parse_options(config, static_cast<std::size_t>(config_.chunk_size) *
                                                   config_.max_action_dim);
    auto inputs = prefix_inputs(request.observation);

    auto prefix_outputs = prefix_->forward(
        {{"pixel_values",
          tensor_of(inputs.pixels.data(),
                    {config_.camera_slots(), 3, config_.camera_height, config_.camera_width},
                    DType::kFloat32)},
         {"token_ids", tensor_of(inputs.token_ids.data(), {config_.prompt_tokens}, DType::kInt32)},
         {"position_ids",
          tensor_of(inputs.positions.data(), {config_.prefix_tokens}, DType::kInt32)},
         {"attention_bias",
          tensor_of(inputs.attention_bias.data(), {config_.prefix_tokens, config_.prefix_tokens},
                    DType::kFloat32)}});
    const Tensor& k_cache = require_output(prefix_outputs, "k_cache", cache_numel_);
    const Tensor& v_cache = require_output(prefix_outputs, "v_cache", cache_numel_);

    auto result = pack_actions(denoise(
        inputs, k_cache, v_cache,
        initial_noise(
            options, static_cast<std::size_t>(config_.chunk_size) * config_.max_action_dim, rng_)));
    if (config_.openpi_policy == "pi05_aloha")
        openpi_actions(config_.openpi_policy,
                       openpi_state(config_.openpi_policy, request.observation.state),
                       result.actions.values.values);
    result.inference_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    return result;
}

} // namespace trtmc::lerobot_pi05
