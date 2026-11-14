/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lerobot_groot/runtime/pipeline.h"

#include "families/lerobot_groot/runtime/image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace trtmc::lerobot_groot {
namespace {

// Additive mask for keys a query must not see; exp() of it underflows to exactly zero.
constexpr float kMasked = -1e30F;

const internal::ConfigField kFields[] = {
    {"seed", internal::ConfigKind::I64, std::nullopt,
     "Seed for the initial flow noise. Omit for a fresh random draw on every call."},
    {"noise", internal::ConfigKind::F64List, std::nullopt,
     "Explicit initial noise of action_horizon x max_action_dim values (40 x 132), row-major. It "
     "overrides seed; use it to reproduce a reference run."},
};

Tensor tensor_of(void* data, std::vector<std::int64_t> shape, DType dtype) {
    Tensor tensor;
    tensor.data = data;
    tensor.shape = std::move(shape);
    tensor.dtype = dtype;
    return tensor;
}

std::int32_t special_token(const ITokenizer& tokenizer, std::string_view token) {
    const std::int32_t id = tokenizer.id_for_token(token);
    if (id < 0)
        throw std::runtime_error("lerobot_groot tokenizer has no " + std::string(token) + " token");
    return id;
}

void require(bool condition, const std::string& what) {
    if (!condition)
        throw std::runtime_error("lerobot_groot " + what);
}

} // namespace

std::int32_t RuntimeConfig::grid() const {
    return camera_size / (kPatch * kMerge);
}

Pipeline::Pipeline(std::unique_ptr<ITrtModule> condition, std::unique_ptr<ITrtModule> denoise,
                   std::unique_ptr<ITokenizer> tokenizer, RuntimeConfig config,
                   std::string primary_task)
    : condition_(std::move(condition)), denoise_(std::move(denoise)),
      tokenizer_(std::move(tokenizer)), config_(std::move(config)),
      primary_task_(std::move(primary_task)), rng_(std::random_device{}()) {
    require(condition_ && condition_->ok() && denoise_ && denoise_->ok() && tokenizer_,
            "pipeline has an invalid engine or tokenizer");
    const auto& c = config_;
    require(!c.cameras.empty() && c.camera_size > 0 && c.camera_size % (kPatch * kMerge) == 0 &&
                c.crop > 0 && c.crop <= c.camera_size && c.chunk_size > 0 &&
                c.chunk_size <= c.action_horizon && c.state_dim > 0 &&
                c.state_dim <= c.max_state_dim && c.action_dim > 0 &&
                c.action_dim <= c.max_action_dim && c.num_inference_steps > 0 &&
                c.max_instruction_tokens > 0 &&
                c.state_min.size() == static_cast<std::size_t>(c.state_dim) &&
                c.state_max.size() == c.state_min.size() &&
                c.action_min.size() == static_cast<std::size_t>(c.action_dim) &&
                c.action_max.size() == c.action_min.size(),
            "runtime.json does not match its runtime contract");
    template_.im_start = special_token(*tokenizer_, "<|im_start|>");
    template_.im_end = special_token(*tokenizer_, "<|im_end|>");
    template_.vision_start = special_token(*tokenizer_, "<|vision_start|>");
    template_.vision_end = special_token(*tokenizer_, "<|vision_end|>");
    template_.image_pad = special_token(*tokenizer_, "<|image_pad|>");
    template_.user = tokenizer_->encode("user\n");
    template_.newline = tokenizer_->encode("\n");
    validate_engines();
}

// Both engines must agree with runtime.json and with each other on what they hand over. The
// condition engine writes the cross-attention keys/values and the state token into device
// buffers sized for the longest prompt, which the denoising steps read in place.
void Pipeline::validate_engines() {
    const auto patches = static_cast<std::int64_t>(config_.cameras.size()) * config_.grid() *
                         config_.grid() * kMerge * kMerge;
    require(condition_->tensor_shape("pixel_values") ==
                std::vector<std::int64_t>{patches, kPatchValues},
            "condition engine pixel_values does not match runtime.json");
    require(condition_->tensor_shape("state") ==
                std::vector<std::int64_t>{1, config_.max_state_dim},
            "condition engine state does not match runtime.json");
    for (const char* name :
         {"token_ids", "image_index", "image_mask", "position_ids", "causal_bias"})
        require(condition_->input_is_dynamic(name), std::string("condition engine input ") + name +
                                                        " must have a dynamic sequence length");
    max_tokens_ = condition_->input_profile_shape("token_ids", 0, ProfileShapeSelector::kMax).at(0);
    // <|im_start|> user\n, per camera <|vision_start|> pads <|vision_end|>, <|im_end|> \n.
    const std::int64_t fixed =
        2 + static_cast<std::int64_t>(template_.user.size()) +
        static_cast<std::int64_t>(template_.newline.size()) +
        static_cast<std::int64_t>(config_.cameras.size()) * (config_.grid() * config_.grid() + 2);
    require(max_tokens_ >= fixed + config_.max_instruction_tokens,
            "condition engine is shorter than the prompts runtime.json allows");

    const auto cross = condition_->tensor_shape("cross_k");
    require(cross.size() == 3 && cross[1] == max_tokens_ &&
                condition_->tensor_shape("cross_v") == cross &&
                condition_->tensor_dtype("cross_v") == condition_->tensor_dtype("cross_k"),
            "condition engine cross-attention outputs are inconsistent");
    cross_blocks_ = cross[0];
    cross_width_ = cross[2];
    require(condition_->tensor_shape("state_features") ==
                    std::vector<std::int64_t>{1, cross_width_} &&
                condition_->tensor_dtype("state_features") == DType::kFloat32,
            "condition engine state_features is inconsistent");

    require(denoise_->tensor_shape("x_t") ==
                    std::vector<std::int64_t>{config_.action_horizon, config_.max_action_dim} &&
                denoise_->tensor_shape("x_next") == denoise_->tensor_shape("x_t"),
            "denoise engine x_t/x_next does not match runtime.json");
    for (const char* name : {"cross_k", "cross_v"}) {
        const auto max = denoise_->input_profile_shape(name, 0, ProfileShapeSelector::kMax);
        require(denoise_->input_is_dynamic(name) && max.size() == 3 && max[0] == cross_blocks_ &&
                    max[1] >= max_tokens_ && max[2] == cross_width_ &&
                    denoise_->tensor_dtype(name) == condition_->tensor_dtype(name),
                std::string("condition and denoise engines disagree on ") + name);
    }
    require(denoise_->tensor_shape("state_features") == std::vector<std::int64_t>{1, cross_width_},
            "condition and denoise engines disagree on state_features");

    const cudaStream_t stream = condition_->stream();
    cross_k_ = DeviceTensor({cross_blocks_, max_tokens_, cross_width_},
                            condition_->tensor_dtype("cross_k"), stream);
    cross_v_ = DeviceTensor({cross_blocks_, max_tokens_, cross_width_},
                            condition_->tensor_dtype("cross_v"), stream);
    state_features_ = DeviceTensor({1, cross_width_}, DType::kFloat32, stream);
    require(cross_k_.ok() && cross_v_.ok() && state_features_.ok(),
            "could not allocate device buffers");
    condition_->bind_external("cross_k", cross_k_.data());
    condition_->bind_external("cross_v", cross_v_.data());
    condition_->bind_external("state_features", state_features_.data());
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {internal::bind<internal::IImagesStateTextToActionChunk>(
        *this, Span<const internal::ConfigField>{kFields, std::size(kFields)})};
}

Pipeline::Options Pipeline::parse_options(internal::ConfigView config) const {
    Options options;
    for (const auto& entry : config) {
        if (entry.name == "seed") {
            const auto seed = std::get<std::int64_t>(entry.value);
            if (seed < 0)
                throw std::invalid_argument("lerobot_groot seed must not be negative");
            options.has_seed = true;
            options.seed = static_cast<std::uint64_t>(seed);
        } else if (entry.name == "noise") {
            const auto values = std::get<Span<const double>>(entry.value);
            const auto expected =
                static_cast<std::size_t>(config_.action_horizon) * config_.max_action_dim;
            if (values.size() != expected)
                throw std::invalid_argument(
                    "lerobot_groot noise must have action_horizon x max_action_dim values");
            options.noise.reserve(expected);
            for (const double value : values) {
                if (!std::isfinite(value))
                    throw std::invalid_argument("lerobot_groot noise must be finite");
                options.noise.push_back(static_cast<float>(value));
            }
        }
    }
    return options;
}

std::vector<float> Pipeline::initial_noise(const Options& options) {
    if (!options.noise.empty())
        return options.noise;
    std::vector<float> noise(static_cast<std::size_t>(config_.action_horizon) *
                             config_.max_action_dim);
    std::mt19937_64 seeded(options.seed);
    auto& engine = options.has_seed ? seeded : rng_;
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (float& value : noise)
        value = normal(engine);
    return noise;
}

std::vector<float>
Pipeline::pack_pixels(const internal::ImagesStateTextObservation& observation) const {
    const auto size = static_cast<std::uint32_t>(config_.camera_size);
    const std::size_t values = static_cast<std::size_t>(size) * size * 3;
    std::vector<const internal::ImageView*> views(config_.cameras.size(), nullptr);
    for (const auto& named : observation.images) {
        const auto found =
            std::find(config_.cameras.begin(), config_.cameras.end(), std::string(named.name));
        if (found == config_.cameras.end())
            throw std::invalid_argument("lerobot_groot does not accept a camera named '" +
                                        std::string(named.name) + "'");
        const auto& image = named.image;
        const std::size_t element =
            image.format == internal::ImageFormat::Float32 ? sizeof(float) : 1;
        if (image.height != size || image.width != size || image.channels != 3 ||
            image.data == nullptr || image.byte_size != values * element)
            throw std::invalid_argument("lerobot_groot camera '" + std::string(named.name) +
                                        "' must be " + std::to_string(size) + "x" +
                                        std::to_string(size) + "x3");
        views[static_cast<std::size_t>(found - config_.cameras.begin())] = &image;
    }

    std::vector<float> pixels;
    std::vector<std::uint8_t> bytes(values);
    for (std::size_t slot = 0; slot < views.size(); ++slot) {
        const internal::ImageView* image = views[slot];
        if (image == nullptr)
            throw std::invalid_argument("lerobot_groot requires camera '" + config_.cameras[slot] +
                                        "'");
        const std::uint8_t* rgb = static_cast<const std::uint8_t*>(image->data);
        if (image->format == internal::ImageFormat::Float32) {
            // LeRobot's float image path: clamp to [0, 1], scale by 255 and truncate to uint8.
            const auto* source = static_cast<const float*>(image->data);
            for (std::size_t index = 0; index < values; ++index) {
                if (!std::isfinite(source[index]))
                    throw std::invalid_argument("lerobot_groot camera '" + config_.cameras[slot] +
                                                "' contains a non-finite pixel");
                bytes[index] =
                    static_cast<std::uint8_t>(std::clamp(source[index], 0.0F, 1.0F) * 255.0F);
            }
            rgb = bytes.data();
        }
        append_patches(rgb, config_.camera_size, config_.crop, pixels);
    }
    return pixels;
}

std::vector<float> Pipeline::normalized_state(Span<const float> state) const {
    if (state.size() != static_cast<std::size_t>(config_.state_dim))
        throw std::invalid_argument("lerobot_groot state must have " +
                                    std::to_string(config_.state_dim) + " values");
    // Min/max to [-1, 1] (0 where the range is empty), clipped, zero-padded to max_state_dim.
    std::vector<float> values(static_cast<std::size_t>(config_.max_state_dim), 0.0F);
    for (std::size_t index = 0; index < state.size(); ++index) {
        if (!std::isfinite(state[index]))
            throw std::invalid_argument("lerobot_groot state must be finite");
        const float denom = config_.state_max[index] - config_.state_min[index];
        float value = 0.0F;
        if (denom != 0.0F) {
            value = 2.0F * (state[index] - config_.state_min[index]);
            value = value / denom;
            value = value - 1.0F;
        }
        values[index] = config_.clip_state ? std::clamp(value, -1.0F, 1.0F) : value;
    }
    return values;
}

Prompt Pipeline::prompt(std::string_view text) const {
    const auto max_bytes =
        static_cast<std::size_t>(config_.max_instruction_tokens) * kInstructionBytesPerToken;
    if (text.size() > max_bytes)
        throw std::invalid_argument("lerobot_groot instruction exceeds " +
                                    std::to_string(max_bytes) + " bytes");
    const auto instruction =
        tokenizer_->encode(formalize_instruction(text, config_.formalize_language));
    if (instruction.size() > static_cast<std::size_t>(config_.max_instruction_tokens))
        throw std::invalid_argument("lerobot_groot instruction exceeds " +
                                    std::to_string(config_.max_instruction_tokens) + " tokens");
    return build_prompt(instruction, static_cast<std::int32_t>(config_.cameras.size()),
                        config_.grid(), template_);
}

internal::ImagesStateTextActionChunkResult
Pipeline::pack_actions(const std::vector<float>& sample) const {
    // The leading chunk_size steps and action_dim columns, clipped and min/max-unnormalized; the
    // LIBERO gripper becomes -sign(2g - 1). Within bounds when no component needed the clip; a
    // LIBERO gripper is a binary command whose prediction routinely overshoots, so it is exempt.
    internal::ImagesStateTextActionChunkResult result;
    const auto rows = static_cast<std::size_t>(config_.chunk_size);
    const auto columns = static_cast<std::size_t>(config_.action_dim);
    const auto padded = static_cast<std::size_t>(config_.max_action_dim);
    result.actions.values.rows = rows;
    result.actions.values.columns = columns;
    result.actions.values.values.reserve(rows * columns);
    bool within = true;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            float value = sample[row * padded + column];
            if (!std::isfinite(value))
                throw std::runtime_error("lerobot_groot produced a non-finite action");
            const bool gripper = config_.libero_gripper && column + 1 == columns;
            within = within && (gripper || (value >= -1.0F && value <= 1.0F));
            if (config_.clip_action)
                value = std::clamp(value, -1.0F, 1.0F);
            const float low = config_.action_min[column];
            const float denom = config_.action_max[column] - low;
            if (denom != 0.0F) {
                value = (value + 1.0F) * 0.5F;
                value = value * denom;
                value = value + low;
            } else {
                value = low;
            }
            if (gripper) {
                const float centred = 2.0F * value - 1.0F;
                value = centred > 0.0F ? -1.0F : (centred < 0.0F ? 1.0F : 0.0F);
            }
            result.actions.values.values.push_back(value);
        }
    }
    result.actions.schema = {"lerobot.groot", {}, {}, "", "unnormalized"};
    result.within_training_bounds = within;
    return result;
}

internal::ImagesStateTextActionChunkResult
Pipeline::run(const internal::ImagesStateTextToActionChunkRequest& request,
              internal::ConfigView config) {
    const auto started = std::chrono::steady_clock::now();
    const auto options = parse_options(config);
    const auto& observation = request.observation;
    auto pixels = pack_pixels(observation);
    auto state = normalized_state(observation.state);
    auto tokens = prompt(observation.text);
    const auto length = static_cast<std::int64_t>(tokens.token_ids.size());
    if (length > max_tokens_)
        throw std::invalid_argument("lerobot_groot prompt exceeds the engine's sequence length");

    std::vector<float> causal(static_cast<std::size_t>(length * length), 0.0F);
    for (std::int64_t row = 0; row < length; ++row)
        std::fill(causal.begin() + row * length + row + 1, causal.begin() + (row + 1) * length,
                  kMasked);
    condition_->forward(
        {{"pixel_values",
          tensor_of(pixels.data(),
                    {static_cast<std::int64_t>(pixels.size()) / kPatchValues, kPatchValues},
                    DType::kFloat32)},
         {"token_ids", tensor_of(tokens.token_ids.data(), {length}, DType::kInt32)},
         {"image_index", tensor_of(tokens.image_index.data(), {length}, DType::kInt32)},
         {"image_mask", tensor_of(tokens.image_mask.data(), {length}, DType::kInt32)},
         {"position_ids", tensor_of(tokens.positions.data(), {3, length}, DType::kInt32)},
         {"causal_bias", tensor_of(causal.data(), {1, 1, length, length}, DType::kFloat32)},
         {"state", tensor_of(state.data(), {1, config_.max_state_dim}, DType::kFloat32)}});

    // Text-reading cross-attention blocks see the non-image tokens; the others the image tokens.
    std::vector<float> text_bias(static_cast<std::size_t>(length)),
        image_bias(static_cast<std::size_t>(length));
    for (std::size_t index = 0; index < text_bias.size(); ++index) {
        const bool image = tokens.image_mask[index] != 0;
        text_bias[index] = image ? kMasked : 0.0F;
        image_bias[index] = image ? 0.0F : kMasked;
    }
    denoise_->bind_external("cross_k", cross_k_.data(), {cross_blocks_, length, cross_width_});
    denoise_->bind_external("cross_v", cross_v_.data(), {cross_blocks_, length, cross_width_});
    denoise_->bind_external("state_features", state_features_.data(), {1, cross_width_});

    auto sample = initial_noise(options);
    const auto steps = config_.num_inference_steps;
    float step_size = static_cast<float>(1.0 / steps);
    for (std::int32_t step = 0; step < steps; ++step) {
        // GR00T's schedule: t = step / steps, discretized to int(t * 1000) buckets.
        auto timestep = static_cast<float>(
            static_cast<std::int64_t>(static_cast<double>(step) / steps * 1000.0));
        auto outputs = denoise_->forward(
            {{"x_t", tensor_of(sample.data(), {config_.action_horizon, config_.max_action_dim},
                               DType::kFloat32)},
             {"timestep", tensor_of(&timestep, {1}, DType::kFloat32)},
             {"dt", tensor_of(&step_size, {1}, DType::kFloat32)},
             {"text_bias", tensor_of(text_bias.data(), {1, 1, 1, length}, DType::kFloat32)},
             {"image_bias", tensor_of(image_bias.data(), {1, 1, 1, length}, DType::kFloat32)}});
        const auto found = outputs.find("x_next");
        if (found == outputs.end() || found->second.data == nullptr ||
            found->second.numel() != sample.size())
            throw std::runtime_error("lerobot_groot denoise engine did not return x_next");
        std::memcpy(sample.data(), found->second.data, sample.size() * sizeof(float));
    }

    auto result = pack_actions(sample);
    result.inference_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    return result;
}

} // namespace trtmc::lerobot_groot
