/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_audio_3/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace trtmc::stable_audio_3 {
namespace {

namespace in = trtmc::internal;

// StableAudioModel.generate defaults.
const in::ConfigField kConfigFields[] = {
    {"duration", in::ConfigKind::F64, in::ConfigValue{120.0},
     "Seconds of audio to render; the render length is capped at 120 s"},
    {"steps", in::ConfigKind::I64, in::ConfigValue{std::int64_t{8}}, "Sampler steps"},
    {"seed", in::ConfigKind::I64, in::ConfigValue{std::int64_t{-1}},
     "Noise seed; -1 draws a random one"},
};
constexpr std::int64_t kMaxSteps = 1000;

const Tensor& require_output(const TensorMap& outputs, const char* name) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || found->second.data == nullptr)
        throw std::runtime_error(std::string("stable_audio_3 engine did not return ") + name);
    return found->second;
}

std::vector<float> copy_floats(const Tensor& tensor, std::size_t expected, const char* name) {
    if (tensor.dtype != DType::kFloat32 || tensor.numel() != expected)
        throw std::runtime_error(std::string("stable_audio_3 engine output has the wrong shape: ") +
                                 name);
    const auto* data = static_cast<const float*>(tensor.data);
    return std::vector<float>(data, data + expected);
}

Tensor host_tensor(void* data, std::vector<int64_t> shape, DType dtype) {
    Tensor tensor;
    tensor.data = data;
    tensor.shape = std::move(shape);
    tensor.dtype = dtype;
    return tensor;
}

} // namespace

Pipeline::Pipeline(std::unique_ptr<ITrtModule> conditioner, std::unique_ptr<ITrtModule> dit,
                   std::unique_ptr<ITrtModule> decoder, std::unique_ptr<ITokenizer> tokenizer,
                   RuntimeConfig config)
    : conditioner_(std::move(conditioner)), dit_(std::move(dit)), decoder_(std::move(decoder)),
      tokenizer_(std::move(tokenizer)), config_(std::move(config)) {
    if (!conditioner_ || !conditioner_->ok() || !dit_ || !dit_->ok() || !decoder_ ||
        !decoder_->ok())
        throw std::runtime_error("stable_audio_3: an engine failed to load");
    if (!tokenizer_)
        throw std::runtime_error("stable_audio_3: missing tokenizer");
}

const char* Pipeline::task() const noexcept {
    return in::ITextToAudio::kTask.data();
}

std::vector<in::TaskInstance> Pipeline::task_bindings() {
    return {in::bind<in::ITextToAudio>(*this, kConfigFields)};
}

Pipeline::Conditioning Pipeline::condition(const std::string& prompt, double seconds) {
    const auto length = static_cast<std::size_t>(config_.prompt_tokens);
    auto ids = tokenizer_->encode(prompt);
    if (ids.size() > length)
        ids.resize(length); // truncation=True, max_length=prompt_tokens
    std::vector<std::int32_t> input_ids(length, config_.pad_token_id);
    std::vector<std::int32_t> mask(length, 0);
    std::copy(ids.begin(), ids.end(), input_ids.begin());
    std::fill_n(mask.begin(), ids.size(), 1);
    float seconds_total = static_cast<float>(seconds);
    const std::vector<int64_t> token_shape{1, config_.prompt_tokens};
    TensorMap inputs{
        {"input_ids", host_tensor(input_ids.data(), token_shape, DType::kInt32)},
        {"attention_mask", host_tensor(mask.data(), token_shape, DType::kInt32)},
        {"seconds_total", host_tensor(&seconds_total, {1, 1}, DType::kFloat32)},
    };
    const auto outputs = conditioner_->forward(inputs);
    const auto cond = static_cast<std::size_t>(config_.cond_dim);
    Conditioning result;
    result.context =
        copy_floats(require_output(outputs, "cross_attn_cond"),
                    static_cast<std::size_t>(config_.context_tokens) * cond, "cross_attn_cond");
    result.global = copy_floats(require_output(outputs, "global_cond"), cond, "global_cond");
    return result;
}

std::vector<float> Pipeline::velocity(std::vector<float>& x, float t,
                                      const Conditioning& conditioning, const RenderPlan& plan) {
    auto context = conditioning.context;
    auto global = conditioning.global;
    std::int32_t valid = plan.valid_latents;
    float time = t;
    TensorMap inputs{
        {"x", host_tensor(x.data(), {1, config_.latent_dim, plan.latents}, DType::kFloat32)},
        {"t", host_tensor(&time, {1}, DType::kFloat32)},
        {"cross_attn_cond",
         host_tensor(context.data(), {1, config_.context_tokens, config_.cond_dim},
                     DType::kFloat32)},
        {"global_cond", host_tensor(global.data(), {1, config_.cond_dim}, DType::kFloat32)},
        {"valid_latents", host_tensor(&valid, {1}, DType::kInt32)},
    };
    const auto outputs = dit_->forward(inputs);
    return copy_floats(require_output(outputs, "v"), x.size(), "v");
}

std::vector<float> Pipeline::sample(const Conditioning& conditioning, const RenderPlan& plan,
                                    std::int32_t steps, std::uint64_t seed) {
    const auto count =
        static_cast<std::size_t>(config_.latent_dim) * static_cast<std::size_t>(plan.latents);
    std::vector<float> x(count);
    gaussian_noise(seed, 0, x.data(), count);
    auto times = schedule(steps, plan.effective_latents, config_.shift);
    const bool half = config_.time_precision == "fp16";
    if (half)
        for (auto& value : times)
            value = round_to_half(value);
    // Pingpong sampler of rf_denoiser models:
    // denoised = x - t * v; x = (1 - t') * denoised + t' * noise
    std::vector<float> noise(count);
    for (std::int32_t step = 0; step < steps; ++step) {
        const float t_curr = times[static_cast<std::size_t>(step)];
        const float t_next = times[static_cast<std::size_t>(step) + 1];
        const auto v = velocity(x, t_curr, conditioning, plan);
        const bool renoise = t_next > 0.0F;
        if (renoise)
            gaussian_noise(seed, static_cast<std::uint64_t>(step) + 1, noise.data(), count);
        for (std::size_t index = 0; index < count; ++index) {
            const float denoised = x[index] - t_curr * v[index];
            x[index] = (1.0F - t_next) * denoised + (renoise ? t_next * noise[index] : 0.0F);
        }
    }
    return x;
}

std::vector<float> Pipeline::decode(std::vector<float>& latents, const RenderPlan& plan) {
    TensorMap inputs{{"latents", host_tensor(latents.data(), {1, config_.latent_dim, plan.latents},
                                             DType::kFloat32)}};
    const auto outputs = decoder_->forward(inputs);
    const auto ratio = static_cast<std::size_t>(config_.limits.downsampling_ratio);
    const auto length = static_cast<std::size_t>(plan.latents) * ratio;
    const auto channels = static_cast<std::size_t>(config_.audio_channels);
    auto planar = copy_floats(require_output(outputs, "audio"), channels * length, "audio");
    const auto valid = static_cast<std::size_t>(plan.valid_latents) * ratio;
    const auto frames = static_cast<std::size_t>(plan.output_samples);
    std::vector<float> interleaved(frames * channels);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const float value = frame < valid ? planar[channel * length + frame] : 0.0F;
            interleaved[frame * channels + channel] = std::clamp(value, -1.0F, 1.0F);
        }
    }
    return interleaved;
}

in::AudioResult Pipeline::run(const in::TextToAudioRequest& request, in::ConfigView config) {
    const auto start = std::chrono::steady_clock::now();
    const auto duration = in::config_get<double>(config, kConfigFields, "duration").value();
    const auto steps = in::config_get<std::int64_t>(config, kConfigFields, "steps").value();
    auto seed = in::config_get<std::int64_t>(config, kConfigFields, "seed").value();
    if (!(duration > 0.0) || !(duration <= kMaxDurationSeconds))
        throw in::ConfigError("stable_audio_3: duration must be in (0, 1e9] seconds");
    if (steps < 1 || steps > kMaxSteps)
        throw in::ConfigError("stable_audio_3: steps must be in [1, 1000]");
    if (seed < -1)
        throw in::ConfigError("stable_audio_3: seed must be non-negative, or -1 for a random seed");
    if (seed == -1) {
        std::random_device device;
        seed = static_cast<std::int64_t>(device() % 100000U);
        std::cerr << "[trtmc] stable_audio_3: seed=" << seed << std::endl;
    }

    const auto plan = plan_render(duration, config_.limits);
    const auto conditioning = condition(std::string(request.prompt), duration);
    auto latents = sample(conditioning, plan, static_cast<std::int32_t>(steps),
                          static_cast<std::uint64_t>(seed));
    in::AudioResult result;
    result.samples = decode(latents, plan);
    result.sample_rate = static_cast<std::uint32_t>(config_.sample_rate);
    result.channels = static_cast<std::uint32_t>(config_.audio_channels);
    result.inference_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}

} // namespace trtmc::stable_audio_3
