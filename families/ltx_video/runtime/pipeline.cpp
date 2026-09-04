/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ltx_video/runtime/pipeline.h"

#include "families/ltx_video/runtime/ltx_video_scheduler_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace trtmc {
namespace {

using half_bits_t = uint16_t;

half_bits_t fp32_to_fp16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    const uint32_t mant = bits & 0x7FFFFFU;
    if (exp <= 0)
        return static_cast<half_bits_t>(sign);
    if (exp >= 31)
        return static_cast<half_bits_t>(sign | 0x7C00U);
    return static_cast<half_bits_t>(sign | (static_cast<uint32_t>(exp) << 10U) | (mant >> 13U));
}

float fp16_to_fp32(half_bits_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000U) << 16U;
    const uint32_t exp = (h >> 10U) & 0x1FU;
    const uint32_t mant = h & 0x3FFU;
    uint32_t bits = sign;
    if (exp == 31U) {
        bits |= 0x7F800000U | (mant << 13U);
    } else if (exp != 0U) {
        bits |= (exp - 15U + 127U) << 23U;
        bits |= mant << 13U;
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

half_bits_t fp32_to_bf16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return static_cast<half_bits_t>(bits >> 16U);
}

float bf16_to_fp32(half_bits_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16U;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

DType require_input_dtype(const ITrtModule& module, const std::string& name) {
    for (const auto& info : module.input_info()) {
        if (info.name == name)
            return info.dtype;
    }
    throw std::runtime_error("LTX engine is missing required input: " + name);
}

std::vector<half_bits_t> convert_float_to_16(const std::vector<float>& src, DType dtype) {
    std::vector<half_bits_t> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        dst[i] = (dtype == DType::kBFloat16) ? fp32_to_bf16(src[i]) : fp32_to_fp16(src[i]);
    }
    return dst;
}

std::vector<float> tensor_to_float_vector(const Tensor& tensor, std::size_t count) {
    if (tensor.data == nullptr || tensor.numel() != count)
        throw std::runtime_error("LTX engine output size does not match its contract");
    std::vector<float> out(count);
    if (tensor.dtype == DType::kFloat32) {
        const auto* src = static_cast<const float*>(tensor.data);
        std::copy_n(src, count, out.data());
    } else if (tensor.dtype == DType::kFloat16) {
        const auto* src = static_cast<const half_bits_t*>(tensor.data);
        for (std::size_t i = 0; i < count; ++i)
            out[i] = fp16_to_fp32(src[i]);
    } else if (tensor.dtype == DType::kBFloat16) {
        const auto* src = static_cast<const half_bits_t*>(tensor.data);
        for (std::size_t i = 0; i < count; ++i)
            out[i] = bf16_to_fp32(src[i]);
    } else
        throw std::runtime_error("LTX engine output has an unsupported dtype");
    return out;
}

Tensor make_float_tensor(const std::vector<float>& values, const std::vector<int64_t>& shape) {
    return Tensor{const_cast<float*>(values.data()), shape, DType::kFloat32};
}

Tensor make_model_tensor(const std::vector<float>& values, std::vector<half_bits_t>& scratch16,
                         DType dtype, const std::vector<int64_t>& shape) {
    if (dtype == DType::kFloat32)
        return make_float_tensor(values, shape);
    scratch16 = convert_float_to_16(values, dtype);
    return Tensor{scratch16.data(), shape, dtype};
}

bool should_log_progress(int32_t step, int32_t num_steps) {
    return (step + 1) % 5 == 0 || step + 1 == num_steps;
}

int32_t latent_frames(const LTXVideoDiffusionConfig& config) {
    return (config.video_num_frames - 1) / std::max(config.scale_factor_temporal, 1) + 1;
}

int32_t latent_height(const LTXVideoDiffusionConfig& config) {
    return config.video_height / std::max(config.scale_factor_spatial, 1);
}

int32_t latent_width(const LTXVideoDiffusionConfig& config) {
    return config.video_width / std::max(config.scale_factor_spatial, 1);
}

int32_t ltx_sequence_length(const LTXVideoDiffusionConfig& config) {
    return latent_frames(config) * latent_height(config) * latent_width(config);
}

std::vector<int32_t> tokenize_t5(const ITokenizer& tokenizer, const std::string& text,
                                 int32_t max_len) {
    constexpr int32_t kT5EosTokenId = 1;
    std::vector<int32_t> ids = tokenizer.encode(text);
    if (!ids.empty() && ids.front() == kT5EosTokenId)
        ids.erase(ids.begin());
    if (ids.empty() || ids.back() != kT5EosTokenId)
        ids.push_back(kT5EosTokenId);
    if (static_cast<int32_t>(ids.size()) > max_len) {
        ids.resize(static_cast<std::size_t>(max_len));
        ids.back() = kT5EosTokenId;
    }
    return ids;
}

std::vector<float> make_initial_packed_latents(std::size_t count, int32_t seed) {
    std::vector<float> latents(count);
    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::normal_distribution<float> dist(0.0F, 1.0F);
    for (auto& v : latents)
        v = dist(gen);
    return latents;
}

void unpack_ltx_latents_for_vae(const std::vector<float>& packed,
                                const LTXVideoDiffusionConfig& config, std::vector<float>& out) {
    const int32_t channels = config.z_dim;
    const int32_t frames = latent_frames(config);
    const int32_t height = latent_height(config);
    const int32_t width = latent_width(config);
    const auto channel_stride = static_cast<std::size_t>(frames) *
                                static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    out.assign(static_cast<std::size_t>(channels) * channel_stride, 0.0F);

    for (int32_t f = 0; f < frames; ++f) {
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                const auto token = (static_cast<std::size_t>(f) * static_cast<std::size_t>(height) +
                                    static_cast<std::size_t>(y)) *
                                       static_cast<std::size_t>(width) +
                                   static_cast<std::size_t>(x);
                const float* src = packed.data() + token * static_cast<std::size_t>(channels);
                for (int32_t c = 0; c < channels; ++c) {
                    const auto dst =
                        static_cast<std::size_t>(c) * channel_stride +
                        (static_cast<std::size_t>(f) * static_cast<std::size_t>(height) +
                         static_cast<std::size_t>(y)) *
                            static_cast<std::size_t>(width) +
                        static_cast<std::size_t>(x);
                    out[dst] = src[static_cast<std::size_t>(c)];
                }
            }
        }
    }
}

void denormalize_ltx_latents_for_vae(std::vector<float>& latents,
                                     const LTXVideoDiffusionConfig& config) {
    const int32_t channels = config.z_dim;
    if (channels <= 0 || config.latents_mean.size() < static_cast<std::size_t>(channels) ||
        config.latents_std.size() < static_cast<std::size_t>(channels)) {
        return;
    }

    const int32_t frames = latent_frames(config);
    const int32_t height = latent_height(config);
    const int32_t width = latent_width(config);
    const auto channel_stride = static_cast<std::size_t>(frames) *
                                static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    const float scaling = (config.vae_scaling_factor > 0.0F) ? config.vae_scaling_factor : 1.0F;

    for (int32_t c = 0; c < channels; ++c) {
        const float mean = config.latents_mean[static_cast<std::size_t>(c)];
        const float std_val = config.latents_std[static_cast<std::size_t>(c)];
        const auto offset = static_cast<std::size_t>(c) * channel_stride;
        for (std::size_t i = 0; i < channel_stride && offset + i < latents.size(); ++i)
            latents[offset + i] = latents[offset + i] * std_val / scaling + mean;
    }
}

void vae_output_to_video(const std::vector<float>& raw, const LTXVideoDiffusionConfig& config,
                         LTXVideoResult& result) {
    const int32_t frames = config.video_num_frames;
    const int32_t height = config.video_height;
    const int32_t width = config.video_width;
    const auto frame_stride = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    const auto channel_stride = static_cast<std::size_t>(frames) * frame_stride;

    result.frames.assign(static_cast<std::size_t>(frames) * frame_stride * 3U, 0.0F);
    result.num_frames = frames;
    result.height = height;
    result.width = width;

    for (int32_t f = 0; f < frames; ++f) {
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                const auto spatial = static_cast<std::size_t>(f) * frame_stride +
                                     static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                     static_cast<std::size_t>(x);
                const auto dst = spatial * 3U;
                for (int32_t c = 0; c < 3; ++c) {
                    const auto src = static_cast<std::size_t>(c) * channel_stride + spatial;
                    const float v = (raw[std::min(src, raw.size() - 1)] + 1.0F) * 0.5F;
                    result.frames[dst + static_cast<std::size_t>(c)] = std::clamp(v, 0.0F, 1.0F);
                }
            }
        }
    }
}

void apply_cfg_and_rescale(const std::vector<float>& cond, const std::vector<float>& uncond,
                           float guidance_scale, float guidance_rescale, std::vector<float>& out) {
    out.resize(cond.size());
    for (std::size_t i = 0; i < cond.size(); ++i)
        out[i] = uncond[i] + guidance_scale * (cond[i] - uncond[i]);

    if (guidance_rescale <= 0.0F || out.empty())
        return;

    const auto mean_std = [](const std::vector<float>& values) {
        const double mean =
            std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
        double var = 0.0;
        for (float v : values) {
            const double d = static_cast<double>(v) - mean;
            var += d * d;
        }
        return std::sqrt(var / static_cast<double>(values.size()) + 1e-12);
    };

    const double std_text = mean_std(cond);
    const double std_cfg = mean_std(out);
    const double scale = std_text / std::max(std_cfg, 1e-12);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const float rescaled = static_cast<float>(static_cast<double>(out[i]) * scale);
        out[i] = guidance_rescale * rescaled + (1.0F - guidance_rescale) * out[i];
    }
}

ImageResult video_to_image_result(LTXVideoResult&& video) {
    ImageResult image;
    image.pixels = std::move(video.frames);
    image.height = video.height;
    image.width = video.width;
    image.channels = 3;
    image.num_frames = video.num_frames;
    return image;
}

} // namespace

LTXVideoOptions parse_ltx_video_options(const std::string& config_json) {
    const auto document = nlohmann::json::parse(config_json);
    if (!document.is_object())
        throw std::runtime_error("LTX runtime.json must be a JSON object");
    const auto& frame_rate = document.at("frame_rate");
    if (!frame_rate.is_number_integer() && !frame_rate.is_number_unsigned())
        throw std::runtime_error("LTX runtime.json frame_rate must be an integer");
    const auto& guidance_rescale = document.at("guidance_rescale");
    if (!guidance_rescale.is_number())
        throw std::runtime_error("LTX runtime.json guidance_rescale must be numeric");
    LTXVideoOptions options;
    options.negative_prompt = document.at("negative_prompt").get<std::string>();
    options.frame_rate = frame_rate.get<int32_t>();
    options.guidance_rescale = guidance_rescale.get<float>();
    return options;
}

LTXVideoPipeline::LTXVideoPipeline(std::unique_ptr<ITrtModule> text_encoder,
                                   std::unique_ptr<ITrtModule> denoiser,
                                   std::unique_ptr<ITrtModule> vae, LTXVideoDiffusionConfig config,
                                   LTXVideoOptions options, std::shared_ptr<ITokenizer> tokenizer,
                                   std::string model_id_str)
    : text_encoder_(std::move(text_encoder)), denoiser_(std::move(denoiser)), vae_(std::move(vae)),
      config_(std::move(config)), options_(std::move(options)), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id_str)) {}

LTXVideoPipeline::~LTXVideoPipeline() = default;

bool LTXVideoPipeline::run_t5_encoder(const std::vector<int32_t>& input_ids,
                                      std::vector<float>& text_embeddings, int32_t& real_tokens) {
    if (!text_encoder_ || !text_encoder_->ok())
        return false;

    const int32_t seq_len = config_.text_seq_len;
    const int32_t text_dim = config_.text_encoder_dim;
    real_tokens = std::min<int32_t>(seq_len, static_cast<int32_t>(input_ids.size()));

    std::vector<int32_t> padded(static_cast<std::size_t>(seq_len), 0);
    std::copy_n(input_ids.begin(), static_cast<std::size_t>(real_tokens), padded.begin());

    std::vector<float> attention_mask(static_cast<std::size_t>(seq_len), -1e9F);
    for (int32_t i = 0; i < real_tokens; ++i) {
        if (padded[static_cast<std::size_t>(i)] != 0)
            attention_mask[static_cast<std::size_t>(i)] = 0.0F;
    }

    TensorMap inputs;
    inputs["input_ids"] = Tensor{padded.data(), {1, static_cast<int64_t>(seq_len)}, DType::kInt32};
    inputs["attention_mask"] =
        Tensor{attention_mask.data(), {1, static_cast<int64_t>(seq_len)}, DType::kFloat32};

    TensorMap outputs = text_encoder_->forward(inputs);
    auto it = outputs.find("text_embeddings");
    if (it == outputs.end())
        throw std::runtime_error("LTX text encoder output text_embeddings is missing");

    const auto count = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(text_dim);
    text_embeddings = tensor_to_float_vector(it->second, count);
    return true;
}

bool LTXVideoPipeline::encode_prompt(const std::string& prompt,
                                     std::vector<float>& prompt_embeddings, int32_t& prompt_tokens,
                                     std::vector<float>& negative_embeddings,
                                     int32_t& negative_tokens) {
    if (!tokenizer_) {
        std::cerr << "[ltx-video] No tokenizer available\n";
        return false;
    }

    const auto prompt_ids = tokenize_t5(*tokenizer_, prompt, config_.text_seq_len);
    const auto negative_ids =
        tokenize_t5(*tokenizer_, options_.negative_prompt, config_.text_seq_len);

    if (!run_t5_encoder(prompt_ids, prompt_embeddings, prompt_tokens))
        return false;
    if (!run_t5_encoder(negative_ids, negative_embeddings, negative_tokens))
        return false;

    std::cerr << "[ltx-video] Tokenized prompt=" << prompt_tokens << " negative=" << negative_tokens
              << "\n";
    return true;
}

bool LTXVideoPipeline::run_denoiser(const std::vector<float>& packed_latents,
                                    const std::vector<float>& text_embeddings, int32_t real_tokens,
                                    float timestep, std::vector<float>& output) {
    if (!denoiser_ || !denoiser_->ok())
        return false;

    const int32_t seq = ltx_sequence_length(config_);
    const int32_t channels = config_.z_dim;
    const int32_t text_seq = config_.text_seq_len;
    const int32_t text_dim = config_.text_encoder_dim;

    const DType latent_dtype = require_input_dtype(*denoiser_, "hidden_states");
    const DType text_dtype = require_input_dtype(*denoiser_, "encoder_hidden_states");
    const DType timestep_dtype = require_input_dtype(*denoiser_, "timestep");
    if (text_dtype != latent_dtype || timestep_dtype != DType::kFloat32 ||
        require_input_dtype(*denoiser_, "encoder_attention_mask") != DType::kFloat32)
        throw std::runtime_error("LTX denoiser input dtypes do not match its builder contract");

    std::vector<half_bits_t> latent16;
    std::vector<half_bits_t> text16;
    std::vector<half_bits_t> timestep16;
    std::vector<float> timestep_vec{timestep};

    std::vector<float> encoder_mask(static_cast<std::size_t>(text_seq), 0.0F);
    for (int32_t i = 0; i < std::min(real_tokens, text_seq); ++i)
        encoder_mask[static_cast<std::size_t>(i)] = 1.0F;

    TensorMap inputs;
    inputs["hidden_states"] =
        make_model_tensor(packed_latents, latent16, latent_dtype,
                          {1, static_cast<int64_t>(seq), static_cast<int64_t>(channels)});
    inputs["encoder_hidden_states"] =
        make_model_tensor(text_embeddings, text16, text_dtype,
                          {1, static_cast<int64_t>(text_seq), static_cast<int64_t>(text_dim)});
    inputs["timestep"] = make_model_tensor(timestep_vec, timestep16, timestep_dtype, {1});
    inputs["encoder_attention_mask"] =
        Tensor{encoder_mask.data(), {1, static_cast<int64_t>(text_seq)}, DType::kFloat32};

    TensorMap outputs = denoiser_->forward(inputs);
    auto it = outputs.find("sample");
    if (it == outputs.end())
        throw std::runtime_error("LTX denoiser output sample is missing");

    const auto count = static_cast<std::size_t>(seq) * static_cast<std::size_t>(channels);
    output = tensor_to_float_vector(it->second, count);
    return true;
}

bool LTXVideoPipeline::decode_vae(const std::vector<float>& packed_latents,
                                  LTXVideoResult& result) {
    if (!vae_ || !vae_->ok())
        return false;

    std::vector<float> vae_latents;
    unpack_ltx_latents_for_vae(packed_latents, config_, vae_latents);
    denormalize_ltx_latents_for_vae(vae_latents, config_);

    const int32_t frames = latent_frames(config_);
    const int32_t height = latent_height(config_);
    const int32_t width = latent_width(config_);
    const DType latent_dtype = require_input_dtype(*vae_, "latents");
    std::vector<half_bits_t> latent16;

    TensorMap inputs;
    inputs["latents"] =
        make_model_tensor(vae_latents, latent16, latent_dtype,
                          {1, static_cast<int64_t>(config_.z_dim), static_cast<int64_t>(frames),
                           static_cast<int64_t>(height), static_cast<int64_t>(width)});

    TensorMap outputs = vae_->forward(inputs);
    auto it = outputs.find("sample");
    if (it == outputs.end())
        throw std::runtime_error("LTX VAE output sample is missing");

    const auto raw_count = static_cast<std::size_t>(3) *
                           static_cast<std::size_t>(config_.video_num_frames) *
                           static_cast<std::size_t>(config_.video_height) *
                           static_cast<std::size_t>(config_.video_width);
    const std::vector<float> raw = tensor_to_float_vector(it->second, raw_count);
    vae_output_to_video(raw, config_, result);
    return true;
}

bool LTXVideoPipeline::compute_velocity_for_step(
    const std::vector<float>& latents, const std::vector<float>& prompt_embeddings,
    int32_t prompt_tokens, const std::vector<float>& negative_embeddings, int32_t negative_tokens,
    float timestep, float guidance_scale, std::vector<float>& cond, std::vector<float>& uncond,
    std::vector<float>& velocity) {
    if (!run_denoiser(latents, prompt_embeddings, prompt_tokens, timestep, cond))
        return false;

    if (guidance_scale <= 1.0F) {
        velocity = cond;
        return true;
    }

    if (!run_denoiser(latents, negative_embeddings, negative_tokens, timestep, uncond))
        return false;
    apply_cfg_and_rescale(cond, uncond, guidance_scale, options_.guidance_rescale, velocity);
    return true;
}

bool LTXVideoPipeline::denoise_loop(std::vector<float>& latents,
                                    const std::vector<float>& prompt_embeddings,
                                    int32_t prompt_tokens,
                                    const std::vector<float>& negative_embeddings,
                                    int32_t negative_tokens, int32_t num_steps,
                                    float guidance_scale) {
    diffusion::ltx_video_scheduler::FlowMatchEulerState scheduler;
    scheduler.num_train_timesteps = 1000;
    scheduler.shift = config_.flow_shift;
    scheduler.use_dynamic_shifting = config_.use_dynamic_shifting;
    scheduler.base_shift = config_.base_shift;
    scheduler.max_shift = config_.max_shift;
    scheduler.base_image_seq_len = config_.base_image_seq_len;
    scheduler.max_image_seq_len = config_.max_image_seq_len;
    scheduler.shift_terminal = config_.shift_terminal;
    scheduler.image_seq_len = ltx_sequence_length(config_);
    scheduler.set_timesteps(num_steps);

    if (scheduler.last_used_dynamic_shifting) {
        std::cerr << "[ltx-video] Dynamic shift mu=" << scheduler.last_dynamic_mu
                  << " terminal=" << config_.shift_terminal << "\n";
    }

    std::vector<float> cond;
    std::vector<float> uncond;
    std::vector<float> velocity;
    std::vector<float> next(latents.size());

    for (int32_t step = 0; step < num_steps; ++step) {
        const float timestep = scheduler.timesteps[static_cast<std::size_t>(step)];
        if (!compute_velocity_for_step(latents, prompt_embeddings, prompt_tokens,
                                       negative_embeddings, negative_tokens, timestep,
                                       guidance_scale, cond, uncond, velocity))
            return false;

        scheduler.step(velocity.data(), latents.data(), next.data(), latents.size(), step);
        latents.swap(next);

        if (should_log_progress(step, num_steps)) {
            std::cerr << "[ltx-video] Step " << (step + 1) << "/" << num_steps << "\n";
        }
    }
    return true;
}

ImageResult LTXVideoPipeline::generate_image(const std::string& prompt,
                                             const ImageGenerationConfig& cfg) {
    const int32_t num_steps = (cfg.num_steps > 0) ? cfg.num_steps : config_.num_inference_steps;
    const float guidance_scale =
        (cfg.guidance_scale >= 0.0F) ? cfg.guidance_scale : config_.guidance_scale;
    const int32_t seed = (cfg.seed >= 0) ? cfg.seed : 0;

    const int32_t seq = ltx_sequence_length(config_);
    const auto latent_count =
        static_cast<std::size_t>(seq) * static_cast<std::size_t>(config_.z_dim);

    std::cerr << "[ltx-video] Generating " << config_.video_num_frames << " frames at "
              << config_.video_width << "x" << config_.video_height << " (steps=" << num_steps
              << ", guidance=" << guidance_scale << ")\n";

    std::vector<float> prompt_embeddings;
    std::vector<float> negative_embeddings;
    int32_t prompt_tokens = 0;
    int32_t negative_tokens = 0;
    if (!encode_prompt(prompt, prompt_embeddings, prompt_tokens, negative_embeddings,
                       negative_tokens)) {
        return {};
    }

    std::vector<float> latents;
    if (!cfg.initial_latents.empty()) {
        if (cfg.initial_latents.size() != latent_count) {
            std::cerr << "[ltx-video] Initial latents size mismatch: got "
                      << cfg.initial_latents.size() << ", expected " << latent_count << "\n";
            return {};
        }
        latents = cfg.initial_latents;
    } else {
        latents = make_initial_packed_latents(latent_count, seed);
    }
    if (!denoise_loop(latents, prompt_embeddings, prompt_tokens, negative_embeddings,
                      negative_tokens, num_steps, guidance_scale)) {
        std::cerr << "[ltx-video] Denoising failed\n";
        return {};
    }

    LTXVideoResult video;
    if (!decode_vae(latents, video)) {
        std::cerr << "[ltx-video] VAE decode failed\n";
        return {};
    }

    std::cerr << "[ltx-video] Video generation complete\n";
    return video_to_image_result(std::move(video));
}

} // namespace trtmc
