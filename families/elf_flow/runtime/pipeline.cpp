/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/elf_flow/runtime/pipeline.h"

#include "families/elf_flow/runtime/sampling_kernels.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

int32_t require_static_dim(const ITrtModule& module, const std::string& input_name, int32_t axis) {
    for (const auto& info : module.input_info()) {
        if (info.name != input_name)
            continue;
        if (axis < 0 || axis >= static_cast<int32_t>(info.shape.size()))
            throw std::runtime_error("ElfFlowPipeline: engine input rank is invalid");
        const int64_t value = info.shape[static_cast<std::size_t>(axis)];
        if (value > 0)
            return static_cast<int32_t>(value);
        throw std::runtime_error("ElfFlowPipeline: engine input dimension must be static");
    }
    throw std::runtime_error("ElfFlowPipeline: required engine input is missing");
}

float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

std::uint32_t normalize_seed(int32_t seed) {
    return seed >= 0 ? static_cast<std::uint32_t>(seed) : 42U;
}

int32_t argmax_row(const float* row, int32_t width) {
    int32_t best = 0;
    float best_value = row[0];
    for (int32_t i = 1; i < width; ++i) {
        if (row[i] > best_value) {
            best_value = row[i];
            best = i;
        }
    }
    return best;
}

const Tensor* find_named_output(const TensorMap& outputs, const char* name) {
    auto it = outputs.find(name);
    if (it == outputs.end() || !it->second.data || it->second.numel() == 0)
        return nullptr;
    return &it->second;
}

const Tensor* find_text_embedding_output(const TensorMap& outputs) {
    return find_named_output(outputs, "text_embeddings");
}

std::vector<float> zero_vector_like(const std::vector<float>& values) {
    return std::vector<float>(values.size(), 0.0F);
}

int32_t elf_default_num_steps(bool has_condition) {
    if (has_condition)
        return 64;
    return 32;
}

float elf_default_self_cond_cfg_scale(bool has_condition) {
    if (has_condition)
        return 1.0F;
    return 3.0F;
}

float elf_default_cfg_scale(bool has_condition) {
    if (has_condition)
        return 2.0F;
    return 1.0F;
}

float elf_default_sde_gamma(bool has_condition) {
    if (has_condition)
        return 0.0F;
    return 1.5F;
}

int32_t resolve_positive_int(int32_t value, int32_t default_value) {
    if (value > 0)
        return value;
    return default_value;
}

float resolve_positive_float(float value, float default_value) {
    if (value > 0.0F)
        return value;
    return default_value;
}

float resolve_nonnegative_float(float value, float default_value) {
    if (value >= 0.0F)
        return value;
    return default_value;
}

void check_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess)
        return;
    throw std::runtime_error(std::string("ElfFlowPipeline: ") + operation +
                             " failed: " + cudaGetErrorString(error));
}

} // namespace

struct ElfFlowPipeline::DeviceSamplingWorkspace {
    DeviceSamplingWorkspace(int32_t max_length, int32_t input_dim, int32_t text_dim,
                            cudaStream_t stream_value)
        : stream(stream_value), z({max_length, text_dim}, DType::kFloat32, stream),
          self_condition({max_length, text_dim}, DType::kFloat32, stream),
          z_eval({max_length, text_dim}, DType::kFloat32, stream),
          condition({max_length, text_dim}, DType::kFloat32, stream),
          condition_mask({max_length}, DType::kFloat32, stream),
          model_latent({max_length, input_dim}, DType::kFloat32, stream),
          timestep({1}, DType::kFloat32, stream), decoder_mode({1}, DType::kFloat32, stream),
          self_cond_cfg_scale({1}, DType::kFloat32, stream),
          conditional_denoised({max_length, text_dim}, DType::kFloat32, stream),
          token_ids({max_length}, DType::kInt32, stream) {}

    bool ok() const {
        const std::array<const DeviceTensor*, 11> required = {
            &z,
            &self_condition,
            &z_eval,
            &condition,
            &condition_mask,
            &model_latent,
            &timestep,
            &decoder_mode,
            &self_cond_cfg_scale,
            &conditional_denoised,
            &token_ids,
        };
        return std::all_of(required.begin(), required.end(),
                           [](const DeviceTensor* tensor) { return tensor->ok(); });
    }

    void ensure_sde_noises(std::size_t numel) {
        if (numel == sde_noise_numel)
            return;
        sde_noises = numel == 0
                         ? DeviceTensor{}
                         : DeviceTensor({static_cast<int64_t>(numel)}, DType::kFloat32, stream);
        sde_noise_numel = numel;
        if (numel > 0 && !sde_noises.ok())
            throw std::runtime_error("ElfFlowPipeline: failed to allocate device SDE noise");
    }

    cudaStream_t stream{nullptr};
    DeviceTensor z;
    DeviceTensor self_condition;
    DeviceTensor z_eval;
    DeviceTensor condition;
    DeviceTensor condition_mask;
    DeviceTensor model_latent;
    DeviceTensor timestep;
    DeviceTensor decoder_mode;
    DeviceTensor self_cond_cfg_scale;
    DeviceTensor conditional_denoised;
    DeviceTensor token_ids;
    DeviceTensor sde_noises;
    std::size_t sde_noise_numel{0};
    float host_timestep{0.0F};
    float host_decoder_mode{0.0F};
    float host_self_cond_cfg_scale{1.0F};
};

ElfFlowPipeline::ElfFlowPipeline(std::unique_ptr<ITrtModule> model, int32_t max_length,
                                 int32_t max_input_length, int32_t input_dim, int32_t text_dim,
                                 int32_t vocab_size, float denoiser_noise_scale,
                                 float denoiser_p_mean, float denoiser_p_std, float t_eps,
                                 std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str,
                                 std::unique_ptr<ITrtModule> text_encoder, float latent_mean,
                                 float latent_std, int32_t encoder_pad_token_id)
    : model_(std::move(model)), text_encoder_(std::move(text_encoder)), max_length_(max_length),
      max_input_length_(max_input_length), input_dim_(input_dim), text_dim_(text_dim),
      vocab_size_(vocab_size), denoiser_noise_scale_(denoiser_noise_scale),
      denoiser_p_mean_(denoiser_p_mean), denoiser_p_std_(denoiser_p_std), t_eps_(t_eps),
      latent_mean_(latent_mean), latent_std_(latent_std),
      encoder_pad_token_id_(encoder_pad_token_id), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id_str)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("ElfFlowPipeline: invalid TRT module");
    configure_model_dimensions();
    configure_text_encoder();
}

ElfFlowPipeline::~ElfFlowPipeline() = default;

void ElfFlowPipeline::configure_model_dimensions() {
    max_length_ = require_static_dim(*model_, "latent", 0);
    input_dim_ = require_static_dim(*model_, "latent", 1);
    if (max_length_ <= 0 || input_dim_ <= 0)
        throw std::runtime_error("ElfFlowPipeline: invalid latent input shape");
    if (max_input_length_ < 0)
        max_input_length_ = 0;
    if (max_input_length_ >= max_length_)
        max_input_length_ = 0;
    if (text_dim_ <= 0)
        text_dim_ = input_dim_;
    if (input_dim_ != text_dim_ && input_dim_ != 2 * text_dim_)
        throw std::runtime_error("ElfFlowPipeline: input_dim must equal text_dim or 2 * text_dim");
    if (latent_std_ == 0.0F)
        throw std::runtime_error("ElfFlowPipeline: latent_std must be non-zero");
}

void ElfFlowPipeline::configure_text_encoder() {
    encoder_seq_length_ = max_length_;
    if (!text_encoder_)
        return;
    if (!text_encoder_->ok())
        throw std::runtime_error("ElfFlowPipeline: invalid text encoder TRT module");
    encoder_seq_length_ = require_static_dim(*text_encoder_, "input_ids", 1);
    if (encoder_seq_length_ != max_length_) {
        throw std::runtime_error(
            "ElfFlowPipeline: text encoder sequence length must match elf_max_length");
    }
}

const Tensor* ElfFlowPipeline::select_output(const TensorMap& outputs, bool decoder_mode) {
    return find_named_output(outputs, decoder_mode ? "decoder_logits" : "denoised");
}

std::vector<float> ElfFlowPipeline::copy_tensor_data(const Tensor& tensor) {
    std::vector<float> data(tensor.numel());
    if (!data.empty() && tensor.data)
        std::memcpy(data.data(), tensor.data, data.size() * sizeof(float));
    return data;
}

void ElfFlowPipeline::add_self_cond_cfg_input(TensorMap& inputs, Tensor& tensor) const {
    if (model_->has_input("self_cond_cfg_scale"))
        inputs["self_cond_cfg_scale"] = tensor;
}

int32_t ElfFlowPipeline::resolve_forward_row_dim(const Tensor& tensor, bool decoder_mode) const {
    int32_t row_dim = tensor.shape.empty() ? static_cast<int32_t>(tensor.numel())
                                           : static_cast<int32_t>(tensor.shape.back());
    if (decoder_mode && vocab_size_ > 0)
        row_dim = vocab_size_;
    if (!decoder_mode && text_dim_ > 0)
        row_dim = text_dim_;
    return row_dim;
}

ElfFlowPipeline::ForwardOutput ElfFlowPipeline::forward_model(const std::vector<float>& latent,
                                                              float timestep,
                                                              float self_cond_cfg_scale,
                                                              bool decoder_mode) {
    const int32_t expected = max_length_ * input_dim_;
    if (static_cast<int32_t>(latent.size()) != expected)
        throw std::runtime_error("ElfFlowPipeline: latent size does not match engine input shape");

    float decoder_mode_value = decoder_mode ? 1.0F : 0.0F;
    Tensor latent_t{const_cast<float*>(latent.data()), {max_length_, input_dim_}, DType::kFloat32};
    Tensor timestep_t{&timestep, {1}, DType::kFloat32};
    Tensor decoder_mode_t{&decoder_mode_value, {1}, DType::kFloat32};
    Tensor self_cond_cfg_t{&self_cond_cfg_scale, {1}, DType::kFloat32};

    TensorMap inputs;
    inputs["latent"] = latent_t;
    inputs["timestep"] = timestep_t;
    inputs["decoder_mode"] = decoder_mode_t;
    add_self_cond_cfg_input(inputs, self_cond_cfg_t);

    auto outputs = model_->forward(inputs);
    const Tensor* selected = select_output(outputs, decoder_mode);
    if (!selected)
        throw std::runtime_error("ElfFlowPipeline: no ELF output found in engine");

    ForwardOutput out;
    out.data = copy_tensor_data(*selected);
    out.row_dim = resolve_forward_row_dim(*selected, decoder_mode);
    return out;
}

std::vector<float> ElfFlowPipeline::build_model_latent(const std::vector<float>& z,
                                                       const std::vector<float>& self_cond) const {
    const std::size_t text_numel =
        static_cast<std::size_t>(max_length_) * static_cast<std::size_t>(text_dim_);
    if (z.size() != text_numel || self_cond.size() != text_numel)
        throw std::runtime_error("ElfFlowPipeline: invalid text latent size");

    if (input_dim_ == text_dim_)
        return z;
    if (input_dim_ != 2 * text_dim_)
        throw std::runtime_error("ElfFlowPipeline: unsupported self-conditioning input shape");

    std::vector<float> model_latent(static_cast<std::size_t>(max_length_) *
                                    static_cast<std::size_t>(input_dim_));
    for (int32_t pos = 0; pos < max_length_; ++pos) {
        const std::size_t src = static_cast<std::size_t>(pos) * static_cast<std::size_t>(text_dim_);
        const std::size_t dst =
            static_cast<std::size_t>(pos) * static_cast<std::size_t>(input_dim_);
        std::copy(z.begin() + static_cast<std::ptrdiff_t>(src),
                  z.begin() + static_cast<std::ptrdiff_t>(src + text_dim_),
                  model_latent.begin() + static_cast<std::ptrdiff_t>(dst));
        std::copy(self_cond.begin() + static_cast<std::ptrdiff_t>(src),
                  self_cond.begin() + static_cast<std::ptrdiff_t>(src + text_dim_),
                  model_latent.begin() + static_cast<std::ptrdiff_t>(dst + text_dim_));
    }
    return model_latent;
}

std::vector<float> ElfFlowPipeline::make_sampling_steps(const TextGenerationConfig& cfg,
                                                        int32_t num_steps, int32_t seed) const {
    if (!cfg.sampling_steps.empty()) {
        if (cfg.sampling_steps.size() < 2U)
            throw std::runtime_error("ElfFlowPipeline: sampling_steps must contain at least "
                                     "start and end timesteps");
        for (std::size_t i = 1; i < cfg.sampling_steps.size(); ++i) {
            if (cfg.sampling_steps[i] < cfg.sampling_steps[i - 1])
                throw std::runtime_error("ElfFlowPipeline: sampling_steps must be sorted");
        }
        return cfg.sampling_steps;
    }
    if (num_steps <= 0)
        num_steps = 32;
    std::vector<float> steps;
    steps.reserve(static_cast<std::size_t>(num_steps) + 1U);
    steps.push_back(0.0F);
    if (num_steps > 1) {
        std::mt19937 rng(normalize_seed(seed) ^ 0x9E3779B9U);
        std::normal_distribution<float> normal(denoiser_p_mean_, denoiser_p_std_);
        std::vector<float> middle;
        middle.reserve(static_cast<std::size_t>(num_steps - 1));
        for (int32_t i = 0; i < num_steps - 1; ++i)
            middle.push_back(sigmoid(normal(rng)));
        std::sort(middle.begin(), middle.end());
        steps.insert(steps.end(), middle.begin(), middle.end());
    }
    steps.push_back(1.0F);
    return steps;
}

std::vector<float> ElfFlowPipeline::make_initial_latent(const TextGenerationConfig& cfg,
                                                        int32_t seed) const {
    const std::size_t n =
        static_cast<std::size_t>(max_length_) * static_cast<std::size_t>(text_dim_);
    if (!cfg.initial_latents.empty()) {
        if (cfg.initial_latents.size() != n)
            throw std::runtime_error("ElfFlowPipeline: initial_latents must have elf_max_length * "
                                     "elf_text_encoder_dim values");
        return cfg.initial_latents;
    }
    std::vector<float> latent(n);
    std::mt19937 rng(normalize_seed(seed));
    std::normal_distribution<float> normal(0.0F, denoiser_noise_scale_);
    for (float& value : latent)
        value = normal(rng);
    return latent;
}

std::vector<float> ElfFlowPipeline::make_condition_latents(const TextGenerationConfig& cfg) const {
    const std::size_t n =
        static_cast<std::size_t>(max_length_) * static_cast<std::size_t>(text_dim_);
    if (cfg.condition_latents.empty())
        return std::vector<float>(n, 0.0F);
    if (cfg.condition_latents.size() != n)
        throw std::runtime_error("ElfFlowPipeline: condition_latents must have elf_max_length * "
                                 "elf_text_encoder_dim values");
    return cfg.condition_latents;
}

std::vector<float> ElfFlowPipeline::make_condition_mask(const TextGenerationConfig& cfg) const {
    if (cfg.condition_mask.empty())
        return std::vector<float>(static_cast<std::size_t>(max_length_), 0.0F);
    if (cfg.condition_mask.size() != static_cast<std::size_t>(max_length_))
        throw std::runtime_error("ElfFlowPipeline: condition_mask must have elf_max_length values");
    return cfg.condition_mask;
}

ElfFlowPipeline::PromptTokenSpan
ElfFlowPipeline::trim_prompt_tokens(const std::vector<int32_t>& encoded,
                                    int32_t eos_token_id) const {
    PromptTokenSpan span;
    span.end = encoded.size();
    auto is_generated_special = [&](int32_t id) {
        return (eos_token_id >= 0 && id == eos_token_id) || id == encoder_pad_token_id_;
    };
    while (span.begin < span.end && is_generated_special(encoded[span.begin]))
        ++span.begin;
    while (span.end > span.begin && is_generated_special(encoded[span.end - 1]))
        --span.end;

    const int32_t prompt_limit =
        max_input_length_ > 0 ? std::min(max_input_length_, max_length_) : max_length_;
    span.length = std::min(prompt_limit, static_cast<int32_t>(span.end - span.begin));
    if (span.length <= 0)
        throw std::runtime_error("ElfFlowPipeline: prompt encoded to no ELF condition tokens");
    return span;
}

ElfFlowPipeline::PromptEncoderInputs
ElfFlowPipeline::make_prompt_encoder_inputs(const std::vector<int32_t>& encoded,
                                            const PromptTokenSpan& span) const {
    PromptEncoderInputs inputs;
    inputs.input_ids.assign(static_cast<std::size_t>(encoder_seq_length_), encoder_pad_token_id_);
    inputs.attention_mask.assign(static_cast<std::size_t>(encoder_seq_length_), -1e9F);
    for (int32_t i = 0; i < span.length; ++i) {
        inputs.input_ids[static_cast<std::size_t>(i)] =
            encoded[span.begin + static_cast<std::size_t>(i)];
        inputs.attention_mask[static_cast<std::size_t>(i)] = 0.0F;
    }
    return inputs;
}

std::vector<float> ElfFlowPipeline::normalize_prompt_latents(const Tensor& embeddings) const {
    const std::size_t expected =
        static_cast<std::size_t>(max_length_) * static_cast<std::size_t>(text_dim_);
    if (embeddings.numel() < expected)
        throw std::runtime_error("ElfFlowPipeline: invalid text encoder output shape");

    std::vector<float> latents(expected, 0.0F);
    const auto* raw = static_cast<const float*>(embeddings.data);
    for (std::size_t i = 0; i < expected; ++i)
        latents[i] = (raw[i] - latent_mean_) / latent_std_;
    return latents;
}

std::vector<float>
ElfFlowPipeline::run_prompt_encoder(const PromptEncoderInputs& prompt_inputs) const {
    TensorMap inputs;
    inputs["input_ids"] = Tensor{const_cast<int32_t*>(prompt_inputs.input_ids.data()),
                                 {1, static_cast<int64_t>(encoder_seq_length_)},
                                 DType::kInt32};
    inputs["attention_mask"] = Tensor{const_cast<float*>(prompt_inputs.attention_mask.data()),
                                      {1, static_cast<int64_t>(encoder_seq_length_)},
                                      DType::kFloat32};

    auto outputs = text_encoder_->forward(inputs);
    const Tensor* embeddings = find_text_embedding_output(outputs);
    if (!embeddings || embeddings->dtype != DType::kFloat32)
        throw std::runtime_error("ElfFlowPipeline: text encoder must output fp32 text_embeddings");
    return normalize_prompt_latents(*embeddings);
}

ElfFlowPipeline::PromptCondition ElfFlowPipeline::make_prompt_condition(const std::string& prompt,
                                                                        int32_t eos_token_id) {
    if (!text_encoder_) {
        throw std::runtime_error(
            "ElfFlowPipeline: prompt-conditioned generation requires text_encoder.plan in "
            "the bundle; pass condition_latents/condition_mask for the raw API path");
    }
    if (!tokenizer_)
        throw std::runtime_error(
            "ElfFlowPipeline: prompt-conditioned generation requires tokenizer.json");

    const auto encoded = tokenizer_->encode(prompt);
    const auto span = trim_prompt_tokens(encoded, eos_token_id);
    const auto prompt_inputs = make_prompt_encoder_inputs(encoded, span);

    PromptCondition out;
    out.latents = run_prompt_encoder(prompt_inputs);
    out.mask.assign(static_cast<std::size_t>(max_length_), 0.0F);
    for (int32_t i = 0; i < span.length; ++i)
        out.mask[static_cast<std::size_t>(i)] = 1.0F;
    return out;
}

ElfFlowPipeline::ConditionState
ElfFlowPipeline::make_condition_state(const std::string& prompt, const TextGenerationConfig& cfg,
                                      int32_t eos_token_id) {
    if (!cfg.condition_latents.empty())
        return ConditionState{make_condition_latents(cfg), make_condition_mask(cfg)};
    if (prompt.empty())
        return ConditionState{make_condition_latents(cfg), make_condition_mask(cfg)};

    auto prompt_condition = make_prompt_condition(prompt, eos_token_id);
    return ConditionState{std::move(prompt_condition.latents), std::move(prompt_condition.mask)};
}

bool ElfFlowPipeline::has_active_condition(const std::vector<float>& cond_mask) const {
    return std::any_of(cond_mask.begin(), cond_mask.end(),
                       [](float value) { return value > 0.0F; });
}

int32_t ElfFlowPipeline::condition_prefix_tokens(const std::vector<float>& cond_mask) const {
    int32_t count = 0;
    for (float value : cond_mask) {
        if (value > 0.0F)
            ++count;
    }
    return count;
}

void ElfFlowPipeline::restore_condition(std::vector<float>& values,
                                        const std::vector<float>& cond_seq,
                                        const std::vector<float>& cond_mask) const {
    if (cond_mask.empty())
        return;
    for (int32_t pos = 0; pos < max_length_; ++pos) {
        if (cond_mask[static_cast<std::size_t>(pos)] <= 0.0F)
            continue;
        const std::size_t base =
            static_cast<std::size_t>(pos) * static_cast<std::size_t>(text_dim_);
        std::copy(cond_seq.begin() + static_cast<std::ptrdiff_t>(base),
                  cond_seq.begin() + static_cast<std::ptrdiff_t>(base + text_dim_),
                  values.begin() + static_cast<std::ptrdiff_t>(base));
    }
}

void ElfFlowPipeline::zero_condition(std::vector<float>& values,
                                     const std::vector<float>& cond_mask) const {
    if (cond_mask.empty())
        return;
    for (int32_t pos = 0; pos < max_length_; ++pos) {
        if (cond_mask[static_cast<std::size_t>(pos)] <= 0.0F)
            continue;
        const std::size_t base =
            static_cast<std::size_t>(pos) * static_cast<std::size_t>(text_dim_);
        std::fill(values.begin() + static_cast<std::ptrdiff_t>(base),
                  values.begin() + static_cast<std::ptrdiff_t>(base + text_dim_), 0.0F);
    }
}

ElfFlowPipeline::DenoiseOutput ElfFlowPipeline::denoise_pass(const std::vector<float>& z,
                                                             float timestep,
                                                             const std::vector<float>& x_pred_prev,
                                                             float self_cond_cfg_scale,
                                                             const std::vector<float>& cond_seq,
                                                             const std::vector<float>& cond_mask) {
    const auto model_latent = build_model_latent(z, x_pred_prev);
    const auto denoised = forward_model(model_latent, timestep, self_cond_cfg_scale, false);
    if (denoised.data.size() != z.size())
        throw std::runtime_error("ElfFlowPipeline: denoised output shape mismatch");

    DenoiseOutput out;
    out.x = denoised.data;
    out.v.resize(z.size());
    const float denom = std::max(1.0F - timestep, t_eps_);
    for (std::size_t j = 0; j < z.size(); ++j)
        out.v[j] = (out.x[j] - z[j]) / denom;

    restore_condition(out.x, cond_seq, cond_mask);
    zero_condition(out.v, cond_mask);
    return out;
}

ElfFlowPipeline::DenoiseOutput
ElfFlowPipeline::denoise_with_cfg(const std::vector<float>& z, float timestep,
                                  const std::vector<float>& x_pred_prev, float cfg_scale,
                                  float self_cond_cfg_scale, const std::vector<float>& cond_seq,
                                  const std::vector<float>& cond_mask) {
    auto cond = denoise_pass(z, timestep, x_pred_prev, self_cond_cfg_scale, cond_seq, cond_mask);
    if (cfg_scale == 1.0F)
        return cond;

    std::vector<float> z_uncond = z;
    std::vector<float> x_prev_uncond = x_pred_prev;
    zero_condition(z_uncond, cond_mask);
    zero_condition(x_prev_uncond, cond_mask);
    auto uncond = denoise_pass(z_uncond, timestep, x_prev_uncond, self_cond_cfg_scale,
                               zero_vector_like(cond_seq), cond_mask);

    DenoiseOutput out;
    out.v.resize(z.size());
    out.x.resize(z.size());
    for (std::size_t j = 0; j < z.size(); ++j) {
        out.v[j] = uncond.v[j] + cfg_scale * (cond.v[j] - uncond.v[j]);
        out.x[j] = uncond.x[j] + cfg_scale * (cond.x[j] - uncond.x[j]);
    }
    restore_condition(out.x, cond_seq, cond_mask);
    zero_condition(out.v, cond_mask);
    return out;
}

int32_t ElfFlowPipeline::resolve_max_output_tokens(const TextGenerationConfig& cfg,
                                                   bool has_condition) const {
    if (cfg.max_new_tokens > 0)
        return cfg.max_new_tokens;
    if (has_condition && max_input_length_ > 0)
        return std::max(1, max_length_ - max_input_length_);
    return max_length_;
}

int32_t ElfFlowPipeline::resolve_eos_token_id(const TextGenerationConfig& cfg) const {
    if (cfg.eos_token_id >= 0)
        return cfg.eos_token_id;
    if (!tokenizer_)
        return -1;
    for (std::string_view token :
         {"</s>", "<eos>", "<|endoftext|>", "<|end_of_text|>", "<|eot_id|>", "[SEP]"}) {
        const int32_t id = tokenizer_->id_for_token(token);
        if (id >= 0)
            return id;
    }
    return -1;
}

std::vector<int32_t> ElfFlowPipeline::decode_tokens(const std::vector<float>& latent,
                                                    float self_cond_cfg_scale, int32_t eos_token_id,
                                                    int32_t prefix_tokens_to_drop,
                                                    int32_t max_output_tokens) {
    const auto model_latent = build_model_latent(latent, zero_vector_like(latent));
    const auto logits = forward_model(model_latent, 1.0F, self_cond_cfg_scale, true);
    if (logits.row_dim <= 0 ||
        static_cast<int32_t>(logits.data.size()) < max_length_ * logits.row_dim)
        throw std::runtime_error("ElfFlowPipeline: invalid decoder logits shape");

    std::vector<int32_t> token_ids;
    token_ids.reserve(static_cast<std::size_t>(max_length_));
    const int32_t start = std::max(0, std::min(prefix_tokens_to_drop, max_length_));
    const int32_t limit =
        max_output_tokens > 0 ? std::min(max_length_, start + max_output_tokens) : max_length_;
    for (int32_t pos = start; pos < limit; ++pos) {
        const float* row = logits.data.data() +
                           static_cast<std::size_t>(pos) * static_cast<std::size_t>(logits.row_dim);
        const int32_t token_id = argmax_row(row, logits.row_dim);
        if (eos_token_id >= 0 && token_id == eos_token_id)
            break;
        token_ids.push_back(token_id);
    }
    return token_ids;
}

void ElfFlowPipeline::validate_generate_config(const TextGenerationConfig& cfg) const {
    if (!model_ || !model_->ok())
        throw std::runtime_error("ElfFlowPipeline: invalid TRT module");
    if (!tokenizer_)
        throw std::runtime_error("ElfFlowPipeline::generate requires tokenizer.json in the bundle");
    if (cfg.condition_latents.empty() != cfg.condition_mask.empty()) {
        throw std::runtime_error(
            "ElfFlowPipeline: conditional generation requires both condition_latents and "
            "condition_mask");
    }
}

ElfFlowPipeline::SamplingOptions
ElfFlowPipeline::resolve_sampling_options(const TextGenerationConfig& cfg,
                                          bool has_condition) const {
    SamplingOptions options;
    options.seed = cfg.seed >= 0 ? cfg.seed : 42;
    options.num_steps = resolve_positive_int(cfg.num_steps, elf_default_num_steps(has_condition));
    options.self_cond_cfg_scale =
        resolve_positive_float(cfg.guidance_scale, elf_default_self_cond_cfg_scale(has_condition));
    options.cfg_scale = resolve_positive_float(cfg.cfg_scale, elf_default_cfg_scale(has_condition));
    options.sde_gamma =
        resolve_nonnegative_float(cfg.sde_gamma, elf_default_sde_gamma(has_condition));
    return options;
}

ElfFlowPipeline::SamplingWorkspace
ElfFlowPipeline::make_sampling_workspace(const TextGenerationConfig& cfg,
                                         const ConditionState& cond, int32_t seed) const {
    SamplingWorkspace workspace;
    workspace.z = make_initial_latent(cfg, seed);
    workspace.x_pred.assign(workspace.z.size(), 0.0F);
    restore_condition(workspace.z, cond.latents, cond.mask);
    restore_condition(workspace.x_pred, cond.latents, cond.mask);
    return workspace;
}

void ElfFlowPipeline::validate_sde_noises(const TextGenerationConfig& cfg,
                                          const std::vector<float>& steps) const {
    const std::size_t latent_numel =
        static_cast<std::size_t>(max_length_) * static_cast<std::size_t>(text_dim_);
    const std::size_t sde_step_count = steps.size() > 1U ? steps.size() - 2U : 0U;
    if (cfg.sde_noises.empty() || cfg.sde_noises.size() == sde_step_count * latent_numel)
        return;
    throw std::runtime_error("ElfFlowPipeline: sde_noises must have "
                             "(sampling_steps.size() - 2) * elf_max_length * "
                             "elf_text_encoder_dim values");
}

void ElfFlowPipeline::apply_sde_perturbation(
    std::vector<float>& z_eval, const std::vector<float>& z, const TextGenerationConfig& cfg,
    const ConditionState& cond, float sde_gamma, float t, float t_next, std::size_t step_idx,
    float& t_eval, std::mt19937& rng, std::normal_distribution<float>& normal) const {
    if (sde_gamma <= 0.0F)
        return;
    const float h = t_next - t;
    const float alpha = std::clamp(1.0F - sde_gamma * h, 0.0F, 1.0F);
    const std::size_t noise_base = step_idx * z.size();
    t_eval = alpha * t;
    for (std::size_t j = 0; j < z_eval.size(); ++j) {
        const float eps = cfg.sde_noises.empty() ? normal(rng) : cfg.sde_noises[noise_base + j];
        z_eval[j] = alpha * z[j] + (1.0F - alpha) * eps;
    }
    restore_condition(z_eval, cond.latents, cond.mask);
}

void ElfFlowPipeline::run_intermediate_step(
    SamplingWorkspace& workspace, const TextGenerationConfig& cfg, const ConditionState& cond,
    const SamplingOptions& options, const std::vector<float>& steps, std::size_t step_idx,
    std::mt19937& rng, std::normal_distribution<float>& normal) {
    const float t = steps[step_idx];
    const float t_next = steps[step_idx + 1U];
    std::vector<float> z_eval = workspace.z;
    float t_eval = t;
    apply_sde_perturbation(z_eval, workspace.z, cfg, cond, options.sde_gamma, t, t_next, step_idx,
                           t_eval, rng, normal);

    const auto denoised = denoise_with_cfg(z_eval, t_eval, workspace.x_pred, options.cfg_scale,
                                           options.self_cond_cfg_scale, cond.latents, cond.mask);
    for (std::size_t j = 0; j < workspace.z.size(); ++j)
        workspace.z[j] = z_eval[j] + (t_next - t_eval) * denoised.v[j];
    workspace.x_pred = denoised.x;
}

void ElfFlowPipeline::run_final_step(SamplingWorkspace& workspace, const ConditionState& cond,
                                     const SamplingOptions& options,
                                     const std::vector<float>& steps) {
    const float t = steps[steps.size() - 2U];
    const float t_next = steps.back();
    const auto denoised = denoise_with_cfg(workspace.z, t, workspace.x_pred, options.cfg_scale,
                                           options.self_cond_cfg_scale, cond.latents, cond.mask);
    for (std::size_t j = 0; j < workspace.z.size(); ++j)
        workspace.z[j] = workspace.z[j] + (t_next - t) * denoised.v[j];
    workspace.x_pred = denoised.x;
}

void ElfFlowPipeline::run_sampling(SamplingWorkspace& workspace, const TextGenerationConfig& cfg,
                                   const ConditionState& cond, const SamplingOptions& options,
                                   const std::vector<float>& steps) {
    validate_sde_noises(cfg, steps);
    std::mt19937 step_rng(normalize_seed(options.seed) ^ 0x85EBCA6BU);
    std::normal_distribution<float> step_normal(0.0F, denoiser_noise_scale_);
    for (std::size_t i = 0; i + 2U < steps.size(); ++i)
        run_intermediate_step(workspace, cfg, cond, options, steps, i, step_rng, step_normal);
    if (steps.size() >= 2U)
        run_final_step(workspace, cond, options, steps);
}

bool ElfFlowPipeline::supports_device_sampling() const {
    if (!model_->has_input("latent") || !model_->has_input("timestep") ||
        !model_->has_input("decoder_mode") || !model_->has_output("denoised") ||
        !model_->has_output("decoder_logits")) {
        return false;
    }
    if (model_->tensor_dtype("latent") != DType::kFloat32 ||
        model_->tensor_dtype("denoised") != DType::kFloat32 ||
        model_->tensor_dtype("decoder_logits") != DType::kFloat32) {
        return false;
    }
    return model_->device_ptr("denoised") != nullptr &&
           model_->device_ptr("decoder_logits") != nullptr;
}

ElfFlowPipeline::DeviceSamplingWorkspace&
ElfFlowPipeline::prepare_device_sampling_workspace(const SamplingWorkspace& host_workspace,
                                                   const ConditionState& cond,
                                                   const std::vector<float>& sde_noises) {
    if (!device_sampling_workspace_) {
        device_sampling_workspace_ = std::make_unique<DeviceSamplingWorkspace>(
            max_length_, input_dim_, text_dim_, model_->stream());
    }
    auto& workspace = *device_sampling_workspace_;
    if (!workspace.ok())
        throw std::runtime_error("ElfFlowPipeline: failed to allocate device sampling workspace");
    workspace.ensure_sde_noises(sde_noises.size());

    const bool copied = workspace.z.copy_from_host(host_workspace.z.data()) &&
                        workspace.self_condition.copy_from_host(host_workspace.x_pred.data()) &&
                        workspace.condition.copy_from_host(cond.latents.data()) &&
                        workspace.condition_mask.copy_from_host(cond.mask.data());
    if (!copied)
        throw std::runtime_error("ElfFlowPipeline: failed to upload device sampling inputs");
    if (!sde_noises.empty() && !workspace.sde_noises.copy_from_host(sde_noises.data()))
        throw std::runtime_error("ElfFlowPipeline: failed to upload device SDE noise");
    return workspace;
}

std::vector<float> ElfFlowPipeline::make_device_sde_noises(const TextGenerationConfig& cfg,
                                                           const SamplingOptions& options,
                                                           const std::vector<float>& steps) const {
    if (options.sde_gamma <= 0.0F || steps.size() <= 2U)
        return {};
    if (!cfg.sde_noises.empty())
        return cfg.sde_noises;

    const std::size_t latent_numel =
        static_cast<std::size_t>(max_length_) * static_cast<std::size_t>(text_dim_);
    const std::size_t step_count = steps.size() - 2U;
    std::vector<float> noises(step_count * latent_numel);
    std::mt19937 rng(normalize_seed(options.seed) ^ 0x85EBCA6BU);
    std::normal_distribution<float> normal(0.0F, denoiser_noise_scale_);
    for (float& value : noises)
        value = normal(rng);
    return noises;
}

void ElfFlowPipeline::enqueue_device_forward(DeviceSamplingWorkspace& workspace, float timestep,
                                             float self_cond_cfg_scale, bool decoder_mode,
                                             bool zero_condition, bool zero_self_condition) {
    elf_prepare_model_latent(static_cast<float*>(workspace.model_latent.data()),
                             static_cast<const float*>(workspace.z_eval.data()),
                             static_cast<const float*>(workspace.self_condition.data()),
                             static_cast<const float*>(workspace.condition_mask.data()),
                             max_length_, text_dim_, input_dim_, zero_condition,
                             zero_self_condition, workspace.stream);
    check_cuda(cudaGetLastError(), "prepare model latent kernel");

    workspace.host_timestep = timestep;
    workspace.host_decoder_mode = decoder_mode ? 1.0F : 0.0F;
    workspace.host_self_cond_cfg_scale = self_cond_cfg_scale;
    const bool copied =
        workspace.timestep.copy_from_host(&workspace.host_timestep) &&
        workspace.decoder_mode.copy_from_host(&workspace.host_decoder_mode) &&
        workspace.self_cond_cfg_scale.copy_from_host(&workspace.host_self_cond_cfg_scale);
    if (!copied)
        throw std::runtime_error("ElfFlowPipeline: failed to upload engine scalar inputs");

    DeviceTensorMap inputs;
    inputs["latent"] = &workspace.model_latent;
    inputs["timestep"] = &workspace.timestep;
    inputs["decoder_mode"] = &workspace.decoder_mode;
    if (model_->has_input("self_cond_cfg_scale"))
        inputs["self_cond_cfg_scale"] = &workspace.self_cond_cfg_scale;
    model_->forward_device_async(inputs);
}

void ElfFlowPipeline::run_device_denoise_step(DeviceSamplingWorkspace& workspace,
                                              const SamplingOptions& options, float timestep,
                                              float next_timestep) {
    enqueue_device_forward(workspace, timestep, options.self_cond_cfg_scale, false, false, false);
    const auto* denoised = static_cast<const float*>(model_->device_ptr("denoised"));
    if (options.cfg_scale == 1.0F) {
        elf_update_latent(static_cast<float*>(workspace.z.data()),
                          static_cast<float*>(workspace.self_condition.data()),
                          static_cast<const float*>(workspace.z_eval.data()), denoised,
                          static_cast<const float*>(workspace.condition.data()),
                          static_cast<const float*>(workspace.condition_mask.data()), timestep,
                          next_timestep, t_eps_, max_length_, text_dim_, workspace.stream);
        check_cuda(cudaGetLastError(), "update latent kernel");
        return;
    }

    check_cuda(cudaMemcpyAsync(workspace.conditional_denoised.data(), denoised,
                               workspace.conditional_denoised.nbytes(), cudaMemcpyDeviceToDevice,
                               workspace.stream),
               "preserve conditional denoiser output");
    enqueue_device_forward(workspace, timestep, options.self_cond_cfg_scale, false, true, false);
    const auto* unconditional = static_cast<const float*>(model_->device_ptr("denoised"));
    elf_update_latent_cfg(static_cast<float*>(workspace.z.data()),
                          static_cast<float*>(workspace.self_condition.data()),
                          static_cast<const float*>(workspace.z_eval.data()),
                          static_cast<const float*>(workspace.conditional_denoised.data()),
                          unconditional, static_cast<const float*>(workspace.condition.data()),
                          static_cast<const float*>(workspace.condition_mask.data()),
                          options.cfg_scale, timestep, next_timestep, t_eps_, max_length_,
                          text_dim_, workspace.stream);
    check_cuda(cudaGetLastError(), "update CFG latent kernel");
}

void ElfFlowPipeline::run_device_sampling(DeviceSamplingWorkspace& workspace,
                                          const SamplingOptions& options,
                                          const std::vector<float>& steps) {
    const std::size_t latent_numel =
        static_cast<std::size_t>(max_length_) * static_cast<std::size_t>(text_dim_);
    for (std::size_t i = 0; i + 2U < steps.size(); ++i) {
        const float timestep = steps[i];
        const float next_timestep = steps[i + 1U];
        float evaluation_timestep = timestep;
        if (options.sde_gamma > 0.0F) {
            const float alpha =
                std::clamp(1.0F - options.sde_gamma * (next_timestep - timestep), 0.0F, 1.0F);
            evaluation_timestep = alpha * timestep;
            const auto* noise = static_cast<const float*>(workspace.sde_noises.data()) +
                                static_cast<std::ptrdiff_t>(i * latent_numel);
            elf_prepare_sde_latent(static_cast<float*>(workspace.z_eval.data()),
                                   static_cast<const float*>(workspace.z.data()), noise,
                                   static_cast<const float*>(workspace.condition.data()),
                                   static_cast<const float*>(workspace.condition_mask.data()),
                                   alpha, max_length_, text_dim_, workspace.stream);
            check_cuda(cudaGetLastError(), "prepare SDE latent kernel");
        } else if (!workspace.z_eval.copy_from(workspace.z)) {
            throw std::runtime_error("ElfFlowPipeline: failed to stage device latent");
        }
        run_device_denoise_step(workspace, options, evaluation_timestep, next_timestep);
    }

    if (steps.size() < 2U)
        return;
    if (!workspace.z_eval.copy_from(workspace.z))
        throw std::runtime_error("ElfFlowPipeline: failed to stage final device latent");
    run_device_denoise_step(workspace, options, steps[steps.size() - 2U], steps.back());
}

std::vector<int32_t> ElfFlowPipeline::decode_device_tokens(DeviceSamplingWorkspace& workspace,
                                                           const SamplingOptions& options,
                                                           int32_t eos_token_id,
                                                           int32_t prefix_tokens_to_drop,
                                                           int32_t max_output_tokens) {
    if (!workspace.z_eval.copy_from(workspace.z))
        throw std::runtime_error("ElfFlowPipeline: failed to stage decoder latent");
    enqueue_device_forward(workspace, 1.0F, options.self_cond_cfg_scale, true, false, true);

    const int32_t start = std::max(0, std::min(prefix_tokens_to_drop, max_length_));
    const int32_t limit =
        max_output_tokens > 0 ? std::min(max_length_, start + max_output_tokens) : max_length_;
    const int32_t row_count = limit - start;
    if (row_count <= 0)
        return {};

    const auto* logits = static_cast<const float*>(model_->device_ptr("decoder_logits"));
    elf_argmax_rows(logits, vocab_size_, start, row_count,
                    static_cast<int32_t*>(workspace.token_ids.data()), workspace.stream);
    check_cuda(cudaGetLastError(), "decoder argmax kernel");

    std::vector<int32_t> all_token_ids(static_cast<std::size_t>(row_count));
    check_cuda(cudaMemcpyAsync(all_token_ids.data(), workspace.token_ids.data(),
                               all_token_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost,
                               workspace.stream),
               "download selected token IDs");
    check_cuda(cudaStreamSynchronize(workspace.stream), "synchronize final decoder");

    std::vector<int32_t> token_ids;
    token_ids.reserve(all_token_ids.size());
    for (int32_t token_id : all_token_ids) {
        if (eos_token_id >= 0 && token_id == eos_token_id)
            break;
        token_ids.push_back(token_id);
    }
    return token_ids;
}

TextResult ElfFlowPipeline::make_device_text_result(DeviceSamplingWorkspace& workspace,
                                                    const TextGenerationConfig& cfg,
                                                    const SamplingOptions& options,
                                                    bool has_condition, int32_t eos_token_id,
                                                    int32_t cond_prefix_len, double sampling_ms) {
    const auto decode_start = std::chrono::steady_clock::now();
    TextResult result;
    result.token_ids = decode_device_tokens(workspace, options, eos_token_id, cond_prefix_len,
                                            resolve_max_output_tokens(cfg, has_condition));
    result.text = tokenizer_->decode(result.token_ids);
    const auto decode_end = std::chrono::steady_clock::now();
    result.prefill_ms = sampling_ms;
    result.decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    return result;
}

TextResult ElfFlowPipeline::make_text_result(const SamplingWorkspace& workspace,
                                             const TextGenerationConfig& cfg,
                                             const SamplingOptions& options, bool has_condition,
                                             int32_t eos_token_id, int32_t cond_prefix_len,
                                             double sampling_ms) {
    auto t_decode_start = std::chrono::steady_clock::now();
    TextResult result;
    result.token_ids =
        decode_tokens(workspace.z, options.self_cond_cfg_scale, eos_token_id, cond_prefix_len,
                      resolve_max_output_tokens(cfg, has_condition));
    result.text = tokenizer_->decode(result.token_ids);
    auto t_decode_end = std::chrono::steady_clock::now();
    result.prefill_ms = sampling_ms;
    result.decode_ms =
        std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    return result;
}

TextResult ElfFlowPipeline::generate(const std::string& prompt, const TextGenerationConfig& cfg) {
    validate_generate_config(cfg);
    auto t_start = std::chrono::steady_clock::now();
    const int32_t eos_token_id = resolve_eos_token_id(cfg);
    auto cond = make_condition_state(prompt, cfg, eos_token_id);
    const bool has_condition = has_active_condition(cond.mask);
    const auto options = resolve_sampling_options(cfg, has_condition);
    auto workspace = make_sampling_workspace(cfg, cond, options.seed);
    const auto steps = make_sampling_steps(cfg, options.num_steps, options.seed);
    if (supports_device_sampling()) {
        validate_sde_noises(cfg, steps);
        const auto sde_noises = make_device_sde_noises(cfg, options, steps);
        auto& device_workspace = prepare_device_sampling_workspace(workspace, cond, sde_noises);
        run_device_sampling(device_workspace, options, steps);
        check_cuda(cudaStreamSynchronize(device_workspace.stream), "synchronize flow sampling");
        const auto sampling_end = std::chrono::steady_clock::now();
        const double sampling_ms =
            std::chrono::duration<double, std::milli>(sampling_end - t_start).count();
        return make_device_text_result(device_workspace, cfg, options, has_condition, eos_token_id,
                                       condition_prefix_tokens(cond.mask), sampling_ms);
    }
    run_sampling(workspace, cfg, cond, options, steps);
    auto t_after_sampling = std::chrono::steady_clock::now();
    const double sampling_ms =
        std::chrono::duration<double, std::milli>(t_after_sampling - t_start).count();
    return make_text_result(workspace, cfg, options, has_condition, eos_token_id,
                            condition_prefix_tokens(cond.mask), sampling_ms);
}

EmbeddingResult ElfFlowPipeline::solve(const float* branch_input, int32_t branch_len,
                                       const float* trunk_input, int32_t trunk_len) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("ElfFlowPipeline: invalid TRT module");
    if (!branch_input || branch_len <= 0)
        throw std::runtime_error("ElfFlowPipeline::solve requires a non-empty latent input");

    const int32_t expected = max_length_ * input_dim_;
    if (branch_len != expected) {
        throw std::runtime_error("ElfFlowPipeline::solve expects branch_len == elf_max_length * "
                                 "elf_input_dim");
    }
    if (trunk_input == nullptr || trunk_len != 3)
        throw std::runtime_error("ElfFlowPipeline::solve expects exactly three control values");

    std::vector<float> latent(branch_input, branch_input + branch_len);
    const float timestep = trunk_input[0];
    const float self_cond_cfg_scale = trunk_input[1];
    const float decoder_mode_value = trunk_input[2];
    const bool decoder_mode = decoder_mode_value >= 0.5f;

    EmbeddingResult result;
    const auto output = forward_model(latent, timestep, self_cond_cfg_scale, decoder_mode);
    result.data = output.data;
    result.dim = output.row_dim;
    return result;
}

} // namespace trtmc
