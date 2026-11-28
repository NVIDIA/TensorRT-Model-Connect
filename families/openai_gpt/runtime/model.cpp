/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/openai_gpt/runtime/model.h"

#include "families/openai_gpt/runtime/sampler.h"
#include "families/openai_gpt/runtime/text_config.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

namespace trtmc::openai_gpt {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

OpenAIGPTModel::OpenAIGPTModel(std::unique_ptr<ITrtModule> prefill,
                               std::unique_ptr<ITrtModule> decode, std::unique_ptr<KvCache> cache,
                               std::unique_ptr<Tokenizer> tokenizer, ModelConfig config)
    : prefill_(std::move(prefill)), decode_(std::move(decode)), cache_(std::move(cache)),
      tokenizer_(std::move(tokenizer)), config_(std::move(config)) {
    if (!prefill_ || !prefill_->ok() || !decode_ || !decode_->ok() || !cache_ || !cache_->ok() ||
        !tokenizer_ || config_.vocab_size <= 0) {
        throw std::invalid_argument("OpenAI GPT model received an invalid runtime component");
    }
    decode_->enable_cuda_graph();
}

std::vector<internal::TaskInstance> OpenAIGPTModel::task_bindings() {
    return {internal::bind<internal::ITextContinuation>(*this, text_config_fields())};
}

std::vector<std::int32_t> OpenAIGPTModel::prompt_ids(const internal::TextSource& prefix,
                                                     std::size_t max_tokens) const {
    if (const auto* text = std::get_if<std::string_view>(&prefix))
        return tokenizer_->encode(*text, max_tokens);
    const auto tokens = std::get<Span<const std::int32_t>>(prefix);
    if (tokens.size() > max_tokens)
        throw std::invalid_argument("OpenAI GPT prefix has more than " +
                                    std::to_string(max_tokens) + " tokens");
    if (!tokens.empty() && tokens.data() == nullptr)
        throw std::invalid_argument("OpenAI GPT token prefix has no storage");
    std::vector<std::int32_t> ids(tokens.begin(), tokens.end());
    for (const auto id : ids) {
        if (id < 0 || id >= config_.vocab_size)
            throw std::invalid_argument("OpenAI GPT token ID " + std::to_string(id) +
                                        " is outside the vocabulary");
    }
    return ids;
}

void OpenAIGPTModel::step(ITrtModule& module, const std::int32_t* tokens, std::int32_t count,
                          std::vector<float>& logits) {
    TensorMap inputs;
    inputs["token_id"] = Tensor{const_cast<std::int32_t*>(tokens), {count}, DType::kInt32};
    cache_->prepare(inputs, count);
    const auto outputs = module.forward(inputs);
    const auto found = outputs.find("logits");
    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
        found->second.numel() != vocab) {
        throw std::runtime_error("OpenAI GPT engine did not return one FP32 logits row");
    }
    logits.resize(vocab);
    std::memcpy(logits.data(), found->second.data, vocab * sizeof(float));
    cache_->commit(count);
}

TextResult OpenAIGPTModel::run(const internal::TextContinuationRequest& request,
                               internal::ConfigView supplied) {
    const auto capacity = static_cast<std::int64_t>(cache_->capacity());
    const auto config = parse_text_config(supplied, cache_->capacity());
    const auto max_new_tokens = config.max_new_tokens;

    std::unique_lock<std::mutex> lock(generation_, std::try_to_lock);
    if (!lock.owns_lock())
        throw std::invalid_argument("OpenAI GPT model already has an active generation");
    const auto setup_start = Clock::now();
    const auto input =
        prompt_ids(request.prefix, static_cast<std::size_t>(capacity - max_new_tokens + 1));
    // The model has no beginning-of-sequence token to continue from.
    if (input.empty())
        throw std::invalid_argument("OpenAI GPT continuation requires a non-empty prefix");
    cache_->reset();
    Sampler sampler(config.sampling);
    std::vector<float> logits;

    const auto prefill_start = Clock::now();
    step(*prefill_, input.data(), static_cast<std::int32_t>(input.size()), logits);
    const auto decode_start = Clock::now();
    std::vector<std::int32_t> generated;
    for (std::int64_t index = 0; index < max_new_tokens; ++index) {
        const auto token = sampler.sample(logits);
        generated.push_back(token);
        const bool eos = std::find(config_.eos_token_ids.begin(), config_.eos_token_ids.end(),
                                   token) != config_.eos_token_ids.end();
        // The last requested token needs no further decoder launch.
        if (eos || index + 1 == max_new_tokens)
            break;
        step(*decode_, &token, 1, logits);
    }
    const auto decode_end = Clock::now();
    TextResult result(tokenizer_->decode(generated), generated,
                      elapsed_ms(prefill_start, decode_start),
                      elapsed_ms(decode_start, decode_end));
    result.setup_ms = elapsed_ms(setup_start, prefill_start);
    return result;
}

} // namespace trtmc::openai_gpt
