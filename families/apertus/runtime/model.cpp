/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/apertus/runtime/model.h"

#include "families/apertus/runtime/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc::apertus {
namespace {

using Clock = std::chrono::steady_clock;
using internal::ConfigField;
using internal::ConfigKind;

const ConfigField kTextFields[] = {
    {"max_new_tokens", ConfigKind::I64, std::int64_t{128}, "Maximum number of generated tokens"},
    {"temperature", ConfigKind::F64, 1.0, "Sampling temperature; 0 selects greedy decoding"},
    {"top_k", ConfigKind::I64, std::int64_t{1}, "Top-k candidates; 1 is greedy, 0 is unlimited"},
    {"top_p", ConfigKind::F64, 1.0, "Nucleus probability mass; 1 disables nucleus filtering"},
    {"min_p", ConfigKind::F64, 0.0, "Minimum probability relative to the most likely token"},
    {"seed", ConfigKind::I64, std::int64_t{-1}, "Sampling seed; negative selects a fixed default"},
};

std::int64_t integer(internal::ConfigView supplied, std::string_view name) {
    return *internal::config_get<std::int64_t>(supplied, kTextFields, name);
}

double number(internal::ConfigView supplied, std::string_view name, double low, double high) {
    const auto value = *internal::config_get<double>(supplied, kTextFields, name);
    if (!std::isfinite(value) || value < low || value > high)
        throw internal::ConfigError("Apertus config '" + std::string(name) + "' is out of range");
    return value;
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

ApertusModel::ApertusModel(std::unique_ptr<ITrtModule> prefill, std::unique_ptr<ITrtModule> decode,
                           std::unique_ptr<KvCache> cache, std::unique_ptr<Tokenizer> tokenizer,
                           ModelConfig config)
    : prefill_(std::move(prefill)), decode_(std::move(decode)), cache_(std::move(cache)),
      tokenizer_(std::move(tokenizer)), config_(std::move(config)) {
    if (!prefill_ || !prefill_->ok() || !decode_ || !decode_->ok() || !cache_ || !cache_->ok() ||
        !tokenizer_ || config_.vocab_size <= 0 || config_.prefill_chunk_tokens <= 0 ||
        config_.eos_token_ids.empty()) {
        throw std::invalid_argument("Apertus model received an invalid runtime component");
    }
    decode_->enable_cuda_graph();
}

std::vector<internal::TaskInstance> ApertusModel::task_bindings() {
    return {internal::bind<internal::ITextContinuation>(*this, kTextFields)};
}

std::vector<std::int32_t> ApertusModel::prompt_ids(const internal::TextSource& prefix) const {
    if (const auto* text = std::get_if<std::string_view>(&prefix)) {
        std::vector<std::int32_t> ids = config_.prefix_ids;
        const auto body = tokenizer_->encode(*text);
        ids.insert(ids.end(), body.begin(), body.end());
        ids.insert(ids.end(), config_.suffix_ids.begin(), config_.suffix_ids.end());
        return ids;
    }
    const auto tokens = std::get<Span<const std::int32_t>>(prefix);
    std::vector<std::int32_t> ids(tokens.begin(), tokens.end());
    for (const auto id : ids) {
        if (id < 0 || id >= config_.vocab_size)
            throw std::invalid_argument("Apertus token ID " + std::to_string(id) +
                                        " is outside the vocabulary");
    }
    return ids;
}

void ApertusModel::step(ITrtModule& module, const std::int32_t* tokens, std::int32_t count,
                        std::vector<float>& logits) {
    TensorMap inputs;
    inputs["token_id"] = Tensor{const_cast<std::int32_t*>(tokens), {count}, DType::kInt32};
    cache_->prepare(inputs, count);
    const auto outputs = module.forward(inputs);
    const auto found = outputs.find("logits");
    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
        found->second.numel() != vocab) {
        throw std::runtime_error("Apertus engine did not return one FP32 logits row");
    }
    logits.resize(vocab);
    std::memcpy(logits.data(), found->second.data, vocab * sizeof(float));
    cache_->commit(count);
}

TextResult ApertusModel::run(const internal::TextContinuationRequest& request,
                             internal::ConfigView supplied) {
    internal::validate_config(kTextFields, supplied);
    const auto max_new_tokens = integer(supplied, "max_new_tokens");
    const auto top_k = integer(supplied, "top_k");
    if (max_new_tokens <= 0 || max_new_tokens > std::numeric_limits<std::int32_t>::max() ||
        top_k < 0 || top_k > std::numeric_limits<std::int32_t>::max())
        throw internal::ConfigError("Apertus max_new_tokens or top_k is out of range");
    SamplingParams params;
    params.temperature =
        static_cast<float>(number(supplied, "temperature", 0.0, std::numeric_limits<float>::max()));
    params.top_k = static_cast<std::int32_t>(top_k);
    params.top_p = static_cast<float>(number(supplied, "top_p", 0.0, 1.0));
    params.min_p = static_cast<float>(number(supplied, "min_p", 0.0, 1.0));
    params.seed = integer(supplied, "seed");

    std::unique_lock<std::mutex> lock(generation_, std::try_to_lock);
    if (!lock.owns_lock())
        throw std::invalid_argument("Apertus model already has an active generation");
    const auto setup_start = Clock::now();
    const auto input = prompt_ids(request.prefix);
    if (input.empty())
        throw std::invalid_argument("Apertus continuation requires a non-empty prefix");
    const auto capacity = static_cast<std::int64_t>(cache_->capacity());
    if (static_cast<std::int64_t>(input.size()) > capacity ||
        max_new_tokens > capacity - static_cast<std::int64_t>(input.size())) {
        throw std::invalid_argument(
            "Apertus prefix plus max_new_tokens exceeds the bundle's KV cache capacity");
    }
    cache_->reset();
    Sampler sampler(params);
    std::vector<float> logits;

    const auto prefill_start = Clock::now();
    const auto total = static_cast<std::int32_t>(input.size());
    for (std::int32_t offset = 0; offset < total; offset += config_.prefill_chunk_tokens) {
        const auto count = std::min(config_.prefill_chunk_tokens, total - offset);
        step(*prefill_, input.data() + offset, count, logits);
    }
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

} // namespace trtmc::apertus
