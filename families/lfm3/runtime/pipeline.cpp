/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm3/runtime/pipeline.h"

#include "families/lfm3/runtime/chat_templates.h"

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

void normalize_eos_ids(Lfm3TextGenConfig& config) {
    if (config.id_eos_ids.empty() && config.id_eos >= 0)
        config.id_eos_ids.push_back(config.id_eos);
    std::sort(config.id_eos_ids.begin(), config.id_eos_ids.end());
    config.id_eos_ids.erase(std::unique(config.id_eos_ids.begin(), config.id_eos_ids.end()),
                            config.id_eos_ids.end());
}

} // namespace

Lfm3TextGenerationPipeline::Lfm3TextGenerationPipeline(std::unique_ptr<ITrtModule> decoder,
                                                       std::unique_ptr<Lfm3InferenceState> state,
                                                       Lfm3TextGenConfig config,
                                                       cudaStream_t stream,
                                                       std::shared_ptr<ITokenizer> tokenizer,
                                                       std::string model_id)
    : decoder_(std::move(decoder)), state_(std::move(state)), config_(std::move(config)),
      stream_(stream), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id)) {
    normalize_eos_ids(config_);
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error("Lfm3TextGenerationPipeline: invalid decoder module");
    if (!state_ || !state_->ok())
        throw std::runtime_error("Lfm3TextGenerationPipeline: invalid inference state");
    if (config_.vocab_size <= 0)
        throw std::invalid_argument("Lfm3TextGenerationPipeline: invalid vocabulary size");
    if (stream_ == nullptr || decoder_->stream() != stream_)
        throw std::invalid_argument("Lfm3TextGenerationPipeline: stream mismatch");
    state_->bind_to(*decoder_);
}

std::vector<int32_t>
Lfm3TextGenerationPipeline::encode_prompt(const std::string& prompt,
                                          const TextGenerationConfig& cfg) const {
    if (!tokenizer_)
        throw std::runtime_error("Lfm3TextGenerationPipeline: no native tokenizer configured");
    std::string effective = prompt;
    if (!cfg.system_prompt.empty())
        throw std::invalid_argument("LFM3 single-user chat does not support system_prompt");
    if (cfg.use_chat_template && config_.chat_template_format.empty())
        throw std::invalid_argument("unsupported LFM3 chat template");
    if (cfg.use_chat_template) {
        effective =
            lfm3_apply_chat_template(config_.chat_template_format, prompt, cfg.enable_thinking);
    }
    auto ids = tokenizer_->encode(effective);

    if (ids.empty())
        throw std::invalid_argument("LFM3 requires a non-empty tokenized prompt");
    return ids;
}

TextResult Lfm3TextGenerationPipeline::generate(const std::string& prompt,
                                                const TextGenerationConfig& cfg) {
    const auto input_ids = encode_prompt(prompt, cfg);
    const int32_t max_new_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 128;
    const auto params = lfm3_sampling_params_from_config(cfg, config_.id_eos_ids);
    auto generated = generate_from_ids(input_ids, max_new_tokens, params);

    std::vector<int32_t> new_tokens(generated.token_ids.begin() +
                                        static_cast<std::ptrdiff_t>(input_ids.size()),
                                    generated.token_ids.end());
    TextResult result{tokenizer_->decode(new_tokens), std::move(new_tokens), generated.prefill_ms,
                      generated.decode_ms};
    result.setup_ms = generated.setup_ms;
    return result;
}

Lfm3TextGenerationPipeline::GenerationResult
Lfm3TextGenerationPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                         const TextGenerationConfig& cfg) {
    const auto params = lfm3_sampling_params_from_config(cfg, config_.id_eos_ids);
    return GenerationResult{generate_from_ids(input_ids, cfg.max_new_tokens, params).token_ids};
}

Lfm3TextGenerationPipeline::TimedGenerationResult
Lfm3TextGenerationPipeline::generate_from_ids(const std::vector<int32_t>& input_ids,
                                              int32_t max_new_tokens,
                                              const Lfm3SamplingParams& params) {
    if (input_ids.empty() || max_new_tokens <= 0)
        return TimedGenerationResult{input_ids};
    const auto capacity = static_cast<std::size_t>(state_->max_length());
    if (input_ids.size() > capacity ||
        static_cast<std::size_t>(max_new_tokens) > capacity - input_ids.size()) {
        throw std::runtime_error(
            "LFM3 prompt and generation exceed the fixed native KV cache capacity");
    }

    auto local_sampler = create_lfm3_sampler(params);
    auto* active_sampler = local_sampler.get();
    active_sampler->reset();

    const auto setup_start = Clock::now();
    state_->reset();
    decoder_->reset_execution_context();
    state_->bind_to(*decoder_);
    logits_device_ptr_ = nullptr;
    const auto setup_end = Clock::now();

    std::vector<float> logits;
    const auto prefill_start = Clock::now();

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

void Lfm3TextGenerationPipeline::run_step(int32_t token_id, std::vector<float>& logits) {
    TensorMap inputs;
    inputs[config_.token_id_name] = Tensor{&token_id, {1}, DType::kInt32};
    state_->prepare_step(inputs);

    decoder_->forward_async(inputs);
    decoder_->sync();
    if (logits_device_ptr_ == nullptr) {
        if (!decoder_->has_output(config_.logits_output_name) ||
            decoder_->tensor_dtype(config_.logits_output_name) != DType::kFloat32) {
            throw std::runtime_error("Lfm3TextGenerationPipeline: FP32 logits output is missing");
        }
        const auto shape = decoder_->tensor_shape(config_.logits_output_name);
        const std::vector<int64_t> vector_shape{config_.vocab_size};
        const std::vector<int64_t> row_shape{1, config_.vocab_size};
        if (shape != vector_shape && shape != row_shape) {
            throw std::runtime_error(
                "Lfm3TextGenerationPipeline: logits shape must be [vocab] or [1,vocab]");
        }
        logits_device_ptr_ =
            static_cast<const float*>(decoder_->device_ptr(config_.logits_output_name));
        if (logits_device_ptr_ == nullptr)
            throw std::runtime_error("Lfm3TextGenerationPipeline: logits device buffer is missing");
    }
    logits.resize(static_cast<std::size_t>(config_.vocab_size));
    if (cudaMemcpy(logits.data(), logits_device_ptr_, logits.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("Lfm3TextGenerationPipeline: logits D2H copy failed");
    }

    state_->advance();
}

} // namespace trtmc
