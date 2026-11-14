/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Built with -ffp-contract=off so host arithmetic rounds as the reference's
// separate float32 tensor operations do.

#include "families/stable_video_diffusion/runtime/pipeline.h"

#include "families/stable_video_diffusion/runtime/preprocess.h"
#include "families/stable_video_diffusion/runtime/sampling.h"
#include "families/stable_video_diffusion/runtime/torch_random.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::stable_video_diffusion {

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::int64_t kCfgBatch = 2;
constexpr std::int64_t kMaxSteps = 1000;
// Integers the reference passes through float32 time ids must stay exact.
constexpr std::int64_t kExactFloatIntegers = 1 << 24;

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::size_t product(const std::vector<std::int64_t>& shape) {
    std::size_t count = 1;
    for (const auto extent : shape)
        count *= static_cast<std::size_t>(extent);
    return count;
}

void require_tensor(const ITrtModule& module, const std::string& name,
                    const std::vector<std::int64_t>& shape, const char* stage) {
    if ((!module.has_input(name) && !module.has_output(name)) ||
        module.tensor_dtype(name) != DType::kFloat32 || module.tensor_shape(name) != shape) {
        throw std::runtime_error(std::string("Stable Video Diffusion ") + stage +
                                 " engine does not match runtime.json at tensor " + name);
    }
}

Tensor host_tensor(const std::vector<float>& values, std::vector<std::int64_t> shape) {
    if (values.size() != product(shape))
        throw std::logic_error("Stable Video Diffusion host tensor has the wrong size");
    return Tensor{const_cast<float*>(values.data()), std::move(shape), DType::kFloat32};
}

const float* output_values(const TensorMap& outputs, const char* name, std::size_t count,
                           const char* stage) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
        found->second.numel() != count || found->second.data == nullptr) {
        throw std::runtime_error(std::string("Stable Video Diffusion ") + stage +
                                 " engine returned no valid " + name);
    }
    return static_cast<const float*>(found->second.data);
}

} // namespace

Pipeline::Pipeline(ModuleLoader loader, RuntimeConfig config)
    : loader_(std::move(loader)), config_(std::move(config)) {
    if (!loader_)
        throw std::invalid_argument("Stable Video Diffusion needs an engine loader");
    using internal::ConfigKind;
    fields_ = {
        {"seed", ConfigKind::I64, std::int64_t{config_.default_seed},
         "Noise seed, as torch.Generator(\"cpu\").manual_seed(seed); 0 to 4294967295"},
        {"num_steps", ConfigKind::I64, std::int64_t{config_.default_num_steps},
         "Euler denoising steps over the Karras schedule, 1 to 1000"},
        {"min_guidance_scale", ConfigKind::F64, double{config_.default_min_guidance_scale},
         "Classifier-free guidance scale of the first frame"},
        {"max_guidance_scale", ConfigKind::F64, double{config_.default_max_guidance_scale},
         "Guidance scale of the last frame; at most 1 disables guidance"},
        {"fps", ConfigKind::I64, std::int64_t{config_.default_fps},
         "Frame rate of the result; the model is conditioned on fps - 1"},
        {"motion_bucket_id", ConfigKind::I64, std::int64_t{config_.default_motion_bucket_id},
         "Motion conditioning; higher values request more motion"},
        {"noise_aug_strength", ConfigKind::F64, double{config_.default_noise_aug_strength},
         "Noise added to the conditioning image before its VAE encode"},
    };
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    const Span<const internal::ConfigField> fields{fields_.data(), fields_.size()};
    return {internal::bind<internal::IInitialImageTextToVideo>(*this, fields)};
}

GenerationSettings Pipeline::resolve(internal::ConfigView config) const {
    const Span<const internal::ConfigField> fields{fields_.data(), fields_.size()};
    internal::validate_config(fields, config);
    const auto integer = [&](std::string_view name, std::int64_t minimum, std::int64_t maximum) {
        const auto value = *internal::config_get<std::int64_t>(config, fields, name);
        if (value < minimum || value > maximum)
            throw internal::ConfigError("Stable Video Diffusion config " + std::string(name) +
                                        " is out of range");
        return value;
    };
    const auto real = [&](std::string_view name, double minimum) {
        const auto value = *internal::config_get<double>(config, fields, name);
        if (!std::isfinite(value) || value < minimum)
            throw internal::ConfigError("Stable Video Diffusion config " + std::string(name) +
                                        " must be finite and at least " + std::to_string(minimum));
        return value;
    };
    GenerationSettings settings;
    settings.seed =
        static_cast<std::uint32_t>(integer("seed", 0, std::numeric_limits<std::uint32_t>::max()));
    settings.num_steps = static_cast<std::int32_t>(integer("num_steps", 1, kMaxSteps));
    const double minimum = real("min_guidance_scale", 0.0);
    const double maximum = real("max_guidance_scale", 0.0);
    settings.guided = maximum > 1.0;
    // The reference prepares unguided conditioning when the last frame's scale is
    // at most one, then guides inside the loop when any frame's scale exceeds it.
    if (!settings.guided && minimum > 1.0)
        throw internal::ConfigError(
            "Stable Video Diffusion min_guidance_scale cannot exceed 1 when max_guidance_scale "
            "is at most 1");
    settings.min_guidance_scale = static_cast<float>(minimum);
    settings.max_guidance_scale = static_cast<float>(maximum);
    settings.fps = static_cast<std::int32_t>(integer("fps", 1, kExactFloatIntegers));
    settings.motion_bucket_id =
        static_cast<std::int32_t>(integer("motion_bucket_id", 0, kExactFloatIntegers));
    settings.noise_aug_strength = static_cast<float>(real("noise_aug_strength", 0.0));
    return settings;
}

std::unique_ptr<ITrtModule> Pipeline::load(const char* section) const {
    auto module = loader_(section);
    if (!module || !module->ok())
        throw std::runtime_error(std::string("Stable Video Diffusion could not load ") + section);
    module->set_timing_label(section);
    return module;
}

std::vector<float> Pipeline::encode_image(const std::vector<float>& pixels) {
    const std::int64_t size = config_.clip_image_size;
    const auto input = clip_pixels(pixels, config_.height, config_.width, config_.clip_image_size,
                                   config_.clip_image_mean, config_.clip_image_std);
    const std::vector<std::int64_t> input_shape{1, 3, size, size};
    const std::vector<std::int64_t> output_shape{1, config_.cross_attention_dim};
    auto encoder = load(kImageEncoderPlan);
    require_tensor(*encoder, "pixel_values", input_shape, "image encoder");
    require_tensor(*encoder, "image_embeds", output_shape, "image encoder");
    const auto outputs = encoder->forward({{"pixel_values", host_tensor(input, input_shape)}});
    const float* values =
        output_values(outputs, "image_embeds", product(output_shape), "image encoder");
    return {values, values + product(output_shape)};
}

std::vector<float> Pipeline::encode_latent(const std::vector<float>& noisy_image) {
    const std::vector<std::int64_t> input_shape{1, 3, config_.height, config_.width};
    const std::vector<std::int64_t> output_shape{1, config_.latent_channels, config_.latent_height,
                                                 config_.latent_width};
    auto encoder = load(kVaeEncoderPlan);
    require_tensor(*encoder, "image", input_shape, "VAE encoder");
    require_tensor(*encoder, "image_latent", output_shape, "VAE encoder");
    const auto outputs = encoder->forward({{"image", host_tensor(noisy_image, input_shape)}});
    const float* values =
        output_values(outputs, "image_latent", product(output_shape), "VAE encoder");
    return {values, values + product(output_shape)};
}

void Pipeline::denoise(std::vector<float>& latents, const std::vector<float>& image_embedding,
                       const std::vector<float>& image_latent, const GenerationSettings& settings) {
    const std::int64_t frames = config_.num_frames;
    const std::int64_t channels = config_.latent_channels;
    const std::size_t frame_size = product({channels, config_.latent_height, config_.latent_width});
    const std::size_t latent_count = static_cast<std::size_t>(frames) * frame_size;
    const std::vector<std::int64_t> sample_shape{kCfgBatch, frames, 2 * channels,
                                                 config_.latent_height, config_.latent_width};
    const std::vector<std::int64_t> context_shape{kCfgBatch, 1, config_.cross_attention_dim};
    const std::vector<std::int64_t> time_ids_shape{1, config_.added_time_ids};
    const std::vector<std::int64_t> noise_shape{kCfgBatch, frames, channels, config_.latent_height,
                                                config_.latent_width};

    // Batch 0 is unconditional: a zero image embedding and a zero image latent.
    std::vector<float> sample(product(sample_shape), 0.0F);
    std::vector<float> context(product(context_shape), 0.0F);
    std::copy(image_embedding.begin(), image_embedding.end(),
              context.begin() + static_cast<std::ptrdiff_t>(image_embedding.size()));
    const auto slot = [&](std::int64_t batch, std::int64_t frame, std::int64_t half) {
        return static_cast<std::size_t>((batch * frames + frame) * 2 + half) * frame_size;
    };
    for (std::int64_t frame = 0; frame < frames; ++frame)
        std::copy(image_latent.begin(), image_latent.end(),
                  sample.begin() + static_cast<std::ptrdiff_t>(slot(1, frame, 1)));
    // [fps - 1, motion_bucket_id, noise_aug_strength]: the model was trained on fps - 1.
    const std::vector<float> time_ids{static_cast<float>(settings.fps - 1),
                                      static_cast<float>(settings.motion_bucket_id),
                                      settings.noise_aug_strength};
    const auto schedule = karras_schedule(settings.num_steps, config_.sigma_min, config_.sigma_max);
    const auto scales =
        guidance_ramp(settings.min_guidance_scale, settings.max_guidance_scale, config_.num_frames);

    auto unet = load(kUnetPlan);
    require_tensor(*unet, "sample", sample_shape, "UNet");
    require_tensor(*unet, "timestep", {1}, "UNet");
    require_tensor(*unet, "encoder_hidden_states", context_shape, "UNet");
    require_tensor(*unet, "added_time_ids", time_ids_shape, "UNet");
    require_tensor(*unet, "noise_pred", noise_shape, "UNet");
    std::vector<float> timestep(1);
    std::vector<float> guided(latent_count);
    for (std::int32_t step = 0; step < settings.num_steps; ++step) {
        const float sigma = schedule.sigmas[static_cast<std::size_t>(step)];
        const float scale = input_scale(sigma);
        for (std::int64_t batch = 0; batch < kCfgBatch; ++batch) {
            for (std::int64_t frame = 0; frame < frames; ++frame) {
                const float* source = latents.data() + static_cast<std::size_t>(frame) * frame_size;
                float* target = sample.data() + slot(batch, frame, 0);
                for (std::size_t index = 0; index < frame_size; ++index)
                    target[index] = source[index] * scale;
            }
        }
        timestep[0] = schedule.timesteps[static_cast<std::size_t>(step)];
        const auto outputs =
            unet->forward({{"sample", host_tensor(sample, sample_shape)},
                           {"timestep", host_tensor(timestep, {1})},
                           {"encoder_hidden_states", host_tensor(context, context_shape)},
                           {"added_time_ids", host_tensor(time_ids, time_ids_shape)}});
        const float* prediction = output_values(outputs, "noise_pred", 2 * latent_count, "UNet");
        const float* conditional = prediction + latent_count;
        if (settings.guided)
            apply_guidance(prediction, conditional, scales, frame_size, guided.data());
        else
            std::copy(conditional, conditional + latent_count, guided.begin());
        euler_v_step(guided.data(), latents.data(), latent_count, sigma,
                     schedule.sigmas[static_cast<std::size_t>(step) + 1]);
    }
}

ImageResult Pipeline::decode(const std::vector<float>& latents) {
    const std::vector<std::int64_t> input_shape{config_.num_frames, config_.latent_channels,
                                                config_.latent_height, config_.latent_width};
    const std::vector<std::int64_t> output_shape{config_.num_frames, 3, config_.height,
                                                 config_.width};
    // 1 / scaling_factor is a Python float the float32 tensor multiplies by.
    const auto factor = static_cast<float>(1.0 / config_.scaling_factor);
    std::vector<float> scaled(latents.size());
    for (std::size_t index = 0; index < latents.size(); ++index)
        scaled[index] = factor * latents[index];
    auto decoder = load(kVaeDecoderPlan);
    require_tensor(*decoder, "latents", input_shape, "VAE decoder");
    require_tensor(*decoder, "frames", output_shape, "VAE decoder");
    const auto outputs = decoder->forward({{"latents", host_tensor(scaled, input_shape)}});
    const float* values = output_values(outputs, "frames", product(output_shape), "VAE decoder");

    ImageResult result;
    result.num_frames = config_.num_frames;
    result.height = config_.height;
    result.width = config_.width;
    result.channels = 3;
    const std::size_t plane = product({config_.height, config_.width});
    result.pixels.resize(static_cast<std::size_t>(config_.num_frames) * plane * 3);
    // The engine emits FCHW in [-1, 1]; the Task returns THWC in [0, 1].
    for (std::size_t frame = 0; frame < static_cast<std::size_t>(config_.num_frames); ++frame) {
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const float value = values[(frame * 3 + channel) * plane + pixel];
                result.pixels[(frame * plane + pixel) * 3 + channel] =
                    std::clamp(value * 0.5F + 0.5F, 0.0F, 1.0F);
            }
        }
    }
    return result;
}

internal::VideoResult Pipeline::run(const internal::InitialImageTextToVideoRequest& request,
                                    internal::ConfigView config) {
    if (!request.prompt.empty())
        throw std::invalid_argument("Stable Video Diffusion is conditioned on the initial image "
                                    "only; pass an empty prompt");
    const auto settings = resolve(config);
    const auto pixels = initial_image_pixels(request.initial_image, config_.height, config_.width);
    std::lock_guard<std::mutex> lock(generation_mutex_);
    const auto begin = Clock::now();

    TorchCpuGenerator generator(settings.seed);
    const auto embedding = encode_image(pixels);
    // The generator's first draw augments the conditioning image, its second
    // draw is the initial latent noise.
    auto image = vae_image(pixels, config_.height, config_.width);
    std::vector<float> noise(image.size());
    generator.normal(noise.data(), noise.size());
    for (std::size_t index = 0; index < image.size(); ++index)
        image[index] = image[index] + settings.noise_aug_strength * noise[index];
    const auto image_latent = encode_latent(image);

    std::vector<float> latents(product({config_.num_frames, config_.latent_channels,
                                        config_.latent_height, config_.latent_width}));
    generator.normal(latents.data(), latents.size());
    const float initial_sigma =
        init_noise_sigma(karras_schedule(settings.num_steps, config_.sigma_min, config_.sigma_max),
                         config_.init_noise_sigma_includes_unit);
    for (float& value : latents)
        value = value * initial_sigma;
    denoise(latents, embedding, image_latent, settings);

    internal::VideoResult result;
    result.frames = decode(latents);
    result.timestamps_seconds.reserve(static_cast<std::size_t>(config_.num_frames));
    for (std::int32_t frame = 0; frame < config_.num_frames; ++frame)
        result.timestamps_seconds.push_back(static_cast<double>(frame) / settings.fps);
    result.conditioned_prefix_frames = 0;
    result.inference_ms = milliseconds(begin, Clock::now());
    return result;
}

} // namespace trtmc::stable_video_diffusion
