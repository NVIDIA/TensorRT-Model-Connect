/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_audio/runtime/pipeline.h"

#include "families/stable_audio/runtime/noise.h"
#include "trtmc/internal/config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace trtmc::stable_audio {
namespace {

const Tensor& require_output(const TensorMap& outputs, const char* name, std::size_t count) {
    const auto it = outputs.find(name);
    if (it == outputs.end())
        throw std::runtime_error(std::string("stable_audio engine did not produce ") + name);
    if (it->second.dtype != DType::kFloat32 || it->second.numel() != count)
        throw std::runtime_error(std::string("stable_audio engine output has the wrong shape: ") +
                                 name);
    return it->second;
}

bool debug_tokens() {
    const char* value = std::getenv("TRTMC_STABLE_AUDIO_DEBUG_TOKENS");
    return value != nullptr && std::string_view(value) == "1";
}

void print_ids(const char* label, const std::vector<std::int32_t>& ids,
               const std::vector<float>& mask) {
    std::ostringstream line;
    line << "[trtmc] stable_audio: " << label << "=[";
    bool first = true;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (mask[i] == 0.0F)
            break;
        line << (first ? "" : ",") << ids[i];
        first = false;
    }
    line << "]";
    std::cerr << line.str() << std::endl;
}

double milliseconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

} // namespace

StableAudioPipeline::StableAudioPipeline(StableAudioEngines engines,
                                         std::unique_ptr<ITokenizer> tokenizer,
                                         StableAudioConfig config)
    : engines_(std::move(engines)), tokenizer_(std::move(tokenizer)), config_(std::move(config)),
      scheduler_(config_.scheduler) {
    if (!engines_.conditioner || !engines_.transformer || !engines_.decoder || !tokenizer_)
        throw std::invalid_argument("stable_audio pipeline needs its three engines and tokenizer");
}

const char* StableAudioPipeline::task() const noexcept {
    return trtmc::internal::ITextToAudio::kTask.data();
}

std::vector<trtmc::internal::TaskInstance> StableAudioPipeline::task_bindings() {
    return {trtmc::internal::bind<trtmc::internal::ITextToAudio>(*this, config_fields())};
}

StableAudioPipeline::Conditioning StableAudioPipeline::condition(const std::string& text,
                                                                 double start, double end) {
    std::vector<std::int32_t> ids;
    std::vector<float> mask;
    frame_tokens(tokenizer_->encode(text), config_, ids, mask);
    if (debug_tokens())
        print_ids("input_ids", ids, mask);
    // torch.tensor of the Python floats: float32.
    float seconds_start = static_cast<float>(start);
    float seconds_end = static_cast<float>(end);
    const auto length = static_cast<std::int64_t>(config_.text_length);
    TensorMap inputs;
    inputs["input_ids"] = Tensor{ids.data(), {1, length}, DType::kInt32};
    inputs["attention_mask"] = Tensor{mask.data(), {1, length}, DType::kFloat32};
    inputs["seconds_start"] = Tensor{&seconds_start, {1}, DType::kFloat32};
    inputs["seconds_end"] = Tensor{&seconds_end, {1}, DType::kFloat32};
    const auto outputs = engines_.conditioner->forward(inputs);
    const auto encoder_count = static_cast<std::size_t>(config_.encoder_tokens) *
                               static_cast<std::size_t>(config_.conditioning_dim);
    const auto global_count = static_cast<std::size_t>(config_.global_dim);
    const auto& encoder = require_output(outputs, "encoder_hidden_states", encoder_count);
    const auto& global = require_output(outputs, "global_hidden_states", global_count);
    Conditioning result;
    const auto* encoder_values = static_cast<const float*>(encoder.data);
    const auto* global_values = static_cast<const float*>(global.data);
    result.encoder_states.assign(encoder_values, encoder_values + encoder_count);
    result.global_states.assign(global_values, global_values + global_count);
    return result;
}

std::vector<float> StableAudioPipeline::denoise(const GenerationPlan& plan,
                                                const Conditioning& positive,
                                                const std::optional<Conditioning>& negative) {
    const bool guided = plan.guidance_scale > 1.0;
    const std::int64_t batch = guided ? 2 : 1;
    const std::size_t encoder_count = positive.encoder_states.size();
    const std::size_t global_count = positive.global_states.size();
    const auto latent_count = static_cast<std::size_t>(config_.latent_channels) *
                              static_cast<std::size_t>(config_.latent_length);

    // Branch order is [unconditional, conditional]. Without a negative prompt the
    // unconditional text-and-duration tokens are zeros while its global (duration)
    // states repeat the conditional ones, exactly as StableAudioPipeline stacks them.
    std::vector<float> encoder(static_cast<std::size_t>(batch) * encoder_count, 0.0F);
    std::vector<float> global(static_cast<std::size_t>(batch) * global_count);
    const auto conditional = static_cast<std::size_t>(batch - 1);
    std::copy(positive.encoder_states.begin(), positive.encoder_states.end(),
              encoder.begin() + static_cast<std::ptrdiff_t>(conditional * encoder_count));
    std::copy(positive.global_states.begin(), positive.global_states.end(),
              global.begin() + static_cast<std::ptrdiff_t>(conditional * global_count));
    if (guided) {
        const auto& unconditional_global =
            negative ? negative->global_states : positive.global_states;
        std::copy(unconditional_global.begin(), unconditional_global.end(), global.begin());
        if (negative)
            std::copy(negative->encoder_states.begin(), negative->encoder_states.end(),
                      encoder.begin());
    }

    scheduler_.set_timesteps(plan.num_inference_steps);
    std::vector<float> latents(latent_count);
    fill_gaussian(plan.seed, 0, latents.data(), latent_count);
    const float initial_scale = scheduler_.init_noise_sigma();
    for (auto& value : latents)
        value *= initial_scale;

    std::vector<float> model_input(static_cast<std::size_t>(batch) * latent_count);
    std::vector<float> guided_output(latent_count);
    std::vector<float> noise(latent_count);
    const auto guidance = static_cast<float>(plan.guidance_scale);
    const auto channels = static_cast<std::int64_t>(config_.latent_channels);
    const auto frames = static_cast<std::int64_t>(config_.latent_length);
    const auto tokens = static_cast<std::int64_t>(config_.encoder_tokens);
    const auto width = static_cast<std::int64_t>(config_.conditioning_dim);
    const auto global_width = static_cast<std::int64_t>(config_.global_dim);
    for (int index = 0; index < plan.num_inference_steps; ++index) {
        const float scale = scheduler_.input_scale(index);
        for (std::int64_t branch = 0; branch < batch; ++branch) {
            auto* row = model_input.data() + static_cast<std::size_t>(branch) * latent_count;
            for (std::size_t i = 0; i < latent_count; ++i)
                row[i] = latents[i] * scale;
        }
        float timestep = scheduler_.timesteps()[static_cast<std::size_t>(index)];
        TensorMap inputs;
        inputs["hidden_states"] =
            Tensor{model_input.data(), {batch, channels, frames}, DType::kFloat32};
        inputs["timestep"] = Tensor{&timestep, {1}, DType::kFloat32};
        inputs["encoder_hidden_states"] =
            Tensor{encoder.data(), {batch, tokens, width}, DType::kFloat32};
        inputs["global_hidden_states"] =
            Tensor{global.data(), {batch, 1, global_width}, DType::kFloat32};
        const auto outputs = engines_.transformer->forward(inputs);
        const auto* values = static_cast<const float*>(
            require_output(outputs, "sample", static_cast<std::size_t>(batch) * latent_count).data);
        if (guided) {
            const float* unconditional = values;
            const float* conditional_values = values + latent_count;
            for (std::size_t i = 0; i < latent_count; ++i)
                guided_output[i] =
                    unconditional[i] + guidance * (conditional_values[i] - unconditional[i]);
        } else {
            std::copy(values, values + latent_count, guided_output.begin());
        }
        fill_gaussian(plan.seed, static_cast<std::uint64_t>(index) + 1U, noise.data(),
                      latent_count);
        scheduler_.step(index, guided_output.data(), latents.data(), noise.data(), latent_count);
    }
    return latents;
}

std::vector<float> StableAudioPipeline::decode(const std::vector<float>& latents) {
    TensorMap inputs;
    inputs["latents"] = Tensor{const_cast<float*>(latents.data()),
                               {1, config_.latent_channels, config_.latent_length},
                               DType::kFloat32};
    const auto count = static_cast<std::size_t>(config_.audio_channels) *
                       static_cast<std::size_t>(config_.audio_samples);
    const auto outputs = engines_.decoder->forward(inputs);
    const auto* values = static_cast<const float*>(require_output(outputs, "audio", count).data);
    return {values, values + count};
}

trtmc::internal::AudioResult
StableAudioPipeline::run(const trtmc::internal::TextToAudioRequest& request,
                         trtmc::internal::ConfigView config) {
    const auto started = std::chrono::steady_clock::now();
    const auto request_plan = plan_request(config_, request, config);
    const bool guided = request_plan.guidance_scale > 1.0;
    const auto positive =
        condition(request_plan.prompt, request_plan.audio_start_in_s, request_plan.audio_end_in_s);
    std::optional<Conditioning> negative;
    // The pipeline encodes a negative prompt only when guidance is on.
    if (guided && request_plan.negative_prompt)
        negative = condition(*request_plan.negative_prompt, request_plan.audio_start_in_s,
                             request_plan.audio_end_in_s);
    const auto conditioned = std::chrono::steady_clock::now();

    const auto latents = denoise(request_plan, positive, negative);
    const auto planar = decode(latents);

    const auto first = request_plan.first_sample;
    const auto last = request_plan.last_sample;
    const auto channels = static_cast<std::size_t>(config_.audio_channels);
    const auto frames = static_cast<std::size_t>(last - first);
    trtmc::internal::AudioResult result;
    result.sample_rate = static_cast<std::uint32_t>(config_.sample_rate);
    result.channels = static_cast<std::uint32_t>(config_.audio_channels);
    result.samples.resize(frames * channels);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (std::size_t channel = 0; channel < channels; ++channel)
            result.samples[frame * channels + channel] =
                planar[channel * static_cast<std::size_t>(config_.audio_samples) +
                       static_cast<std::size_t>(first) + frame];
    }
    const auto finished = std::chrono::steady_clock::now();
    result.setup_ms = milliseconds(conditioned - started);
    result.inference_ms = milliseconds(finished - conditioned);
    std::cerr << "[trtmc] stable_audio: " << request_plan.num_inference_steps << " steps, guidance "
              << request_plan.guidance_scale << (negative ? " (negative prompt)" : "") << ", seed "
              << request_plan.seed << ", " << frames << " frames x " << channels << " channels @ "
              << config_.sample_rate << " Hz in " << milliseconds(finished - started) << " ms"
              << std::endl;
    return result;
}

} // namespace trtmc::stable_audio
