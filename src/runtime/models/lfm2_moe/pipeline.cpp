/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/lfm2_moe/pipeline.h"

#include "runtime/models/lfm2_moe/chat_templates.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void normalize_eos_ids(Lfm2MoeTextGenConfig& config) {
    if (config.id_eos_ids.empty() && config.id_eos >= 0)
        config.id_eos_ids.push_back(config.id_eos);
    std::sort(config.id_eos_ids.begin(), config.id_eos_ids.end());
    config.id_eos_ids.erase(std::unique(config.id_eos_ids.begin(), config.id_eos_ids.end()),
                            config.id_eos_ids.end());
}

} // namespace

Lfm2MoeTextGenerationPipeline::Lfm2MoeTextGenerationPipeline(
    std::unique_ptr<TrtModule> decoder, std::unique_ptr<Lfm2MoeInferenceState> state,
    Lfm2MoeTextGenConfig config, cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id, std::unique_ptr<Lfm2MoeISampler> sampler)
    : decoder_(std::move(decoder)), state_(std::move(state)), config_(std::move(config)),
      stream_(stream), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id)),
      sampler_(std::move(sampler)) {
    normalize_eos_ids(config_);
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error("Lfm2MoeTextGenerationPipeline: invalid decoder module");
    if (!state_ || !state_->ok())
        throw std::runtime_error("Lfm2MoeTextGenerationPipeline: invalid inference state");
    if (config_.vocab_size <= 0)
        throw std::invalid_argument("Lfm2MoeTextGenerationPipeline: invalid vocabulary size");
    if (stream_ == nullptr || decoder_->stream() != stream_)
        throw std::invalid_argument("Lfm2MoeTextGenerationPipeline: stream mismatch");
    state_->bind_to(*decoder_);
}

std::vector<int32_t> Lfm2MoeTextGenerationPipeline::encode_prompt(const std::string& prompt,
                                                               const GenerateConfig& cfg) const {
    if (!tokenizer_)
        throw std::runtime_error("Lfm2MoeTextGenerationPipeline: no native tokenizer configured");
    std::string effective = prompt;
    if (cfg.use_chat_template && !config_.chat_template_format.empty()) {
        effective = lfm2_moe_apply_chat_template(config_.chat_template_format, prompt);
    }
    auto ids = tokenizer_->encode(effective);

    // The checkpoint tokenizer has add_bos_token=true and the official chat
    // template (including already-rendered tool prompts) begins with BOS. Keep
    // exactly one copy regardless of whether this call rendered the template.
    if (ids.size() >= 2 && config_.id_bos >= 0 && ids[0] == config_.id_bos &&
        ids[1] == config_.id_bos) {
        ids.erase(ids.begin());
    }
    if (ids.empty() && config_.id_bos >= 0)
        ids.push_back(config_.id_bos);
    return ids;
}

TextResult Lfm2MoeTextGenerationPipeline::generate(const std::string& prompt,
                                                const GenerateConfig& cfg) {
    const auto input_ids = encode_prompt(prompt, cfg);
    const int32_t max_new_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 128;
    const auto params = lfm2_moe_sampling_params_from_config(cfg, config_.id_eos_ids);
    auto generated = generate_from_ids(input_ids, max_new_tokens, params);

    std::vector<int32_t> new_tokens(generated.token_ids.begin() +
                                        static_cast<std::ptrdiff_t>(input_ids.size()),
                                    generated.token_ids.end());
    TextResult result{tokenizer_->decode(new_tokens), std::move(new_tokens), generated.prefill_ms,
                      generated.decode_ms};
    result.setup_ms = generated.setup_ms;
    return result;
}

Lfm2MoeTextGenerationPipeline::GenerationResult
Lfm2MoeTextGenerationPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                         const GenerateConfig& cfg) {
    const auto params = lfm2_moe_sampling_params_from_config(cfg, config_.id_eos_ids);
    return GenerationResult{generate_from_ids(input_ids, cfg.max_new_tokens, params).token_ids};
}

Lfm2MoeTextGenerationPipeline::TimedGenerationResult
Lfm2MoeTextGenerationPipeline::generate_from_ids(const std::vector<int32_t>& input_ids,
                                              int32_t max_new_tokens,
                                              const Lfm2MoeSamplingParams& params) {
    if (input_ids.empty() || max_new_tokens <= 0)
        return TimedGenerationResult{input_ids};
    const auto capacity = static_cast<std::size_t>(state_->max_length());
    if (input_ids.size() > capacity ||
        static_cast<std::size_t>(max_new_tokens) > capacity - input_ids.size()) {
        throw std::runtime_error(
            "LFM2 prompt and generation exceed the fixed native KV cache capacity");
    }

    Lfm2MoeISampler* active_sampler = sampler_.get();
    std::unique_ptr<Lfm2MoeISampler> local_sampler;
    if (active_sampler == nullptr) {
        local_sampler = create_lfm2_moe_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    const auto setup_start = Clock::now();
    state_->reset();
    decoder_->reset_execution_context();
    state_->bind_to(*decoder_);
    logits_device_ptr_ = nullptr;
    const auto setup_end = Clock::now();

    std::vector<float> logits;
    const auto prefill_start = Clock::now();
    // Safe v1 path: every prompt token commits native KV and rolling
    // convolution state together. A KV-only batched-prefill shortcut would
    // leave the convolution history stale.
    for (int32_t token : input_ids)
        run_step(token, logits);
    const auto prefill_end = Clock::now();

    std::vector<int32_t> output = input_ids;
    const auto decode_start = Clock::now();
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const auto sampled = active_sampler->sample(
            logits.data(), static_cast<int32_t>(logits.size()), params, output);
        output.push_back(sampled.token_id);
        if (sampled.is_eos || step + 1 >= max_new_tokens)
            break;
        run_step(sampled.token_id, logits);
    }
    const auto decode_end = Clock::now();

    return TimedGenerationResult{std::move(output), milliseconds(setup_start, setup_end),
                                 milliseconds(prefill_start, prefill_end),
                                 milliseconds(decode_start, decode_end)};
}

void Lfm2MoeTextGenerationPipeline::run_step(int32_t token_id, std::vector<float>& logits) {
    TensorMap inputs;
    inputs[config_.token_id_name] = Tensor{&token_id, {1}, DType::kInt32};
    state_->prepare_step(inputs);

    // Every persistent state output is device-sized. Never call forward(),
    // which materializes non-external outputs on the host; execute asynchronously
    // and transfer only the one-vocabulary logits row required for sampling.
    decoder_->forward_async(inputs);
    decoder_->sync();
    if (logits_device_ptr_ == nullptr) {
        if (!decoder_->has_output(config_.logits_output_name) ||
            decoder_->tensor_dtype(config_.logits_output_name) != DType::kFloat32) {
            throw std::runtime_error("Lfm2MoeTextGenerationPipeline: FP32 logits output is missing");
        }
        const auto shape = decoder_->tensor_shape(config_.logits_output_name);
        const std::vector<int64_t> vector_shape{config_.vocab_size};
        const std::vector<int64_t> row_shape{1, config_.vocab_size};
        if (shape != vector_shape && shape != row_shape) {
            throw std::runtime_error(
                "Lfm2MoeTextGenerationPipeline: logits shape must be [vocab] or [1,vocab]");
        }
        logits_device_ptr_ =
            static_cast<const float*>(decoder_->device_ptr(config_.logits_output_name));
        if (logits_device_ptr_ == nullptr)
            throw std::runtime_error("Lfm2MoeTextGenerationPipeline: logits device buffer is missing");
    }
    logits.resize(static_cast<std::size_t>(config_.vocab_size));
    if (cudaMemcpy(logits.data(), logits_device_ptr_, logits.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("Lfm2MoeTextGenerationPipeline: logits D2H copy failed");
    }

    state_->advance();
}

} // namespace trtmc
