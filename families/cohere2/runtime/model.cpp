/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/cohere2/runtime/model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace trtmc::cohere2 {
namespace {
using Clock = std::chrono::steady_clock;
const internal::ConfigField kFields[] = {
    {"max_new_tokens", internal::ConfigKind::I64, std::int64_t{32},
     "Maximum greedy continuation tokens"},
};
double milliseconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
} // namespace

Model::Model(std::unique_ptr<ITrtModule> engine, std::unique_ptr<Tokenizer> tokenizer,
             ModelConfig config)
    : engine_(std::move(engine)), tokenizer_(std::move(tokenizer)), config_(config) {
    if (!engine_ || !engine_->ok() || !tokenizer_ || config_.length <= 0 ||
        config_.vocabulary <= 0 || tokenizer_->vocab_size() > config_.vocabulary)
        throw std::invalid_argument("cohere2 invalid runtime components");
}

std::vector<internal::TaskInstance> Model::task_bindings() {
    return {internal::bind<internal::ITextContinuation>(*this, kFields)};
}

TextResult Model::run(const internal::TextContinuationRequest& request,
                      internal::ConfigView supplied) {
    internal::validate_config(kFields, supplied);
    const auto budget = *internal::config_get<std::int64_t>(supplied, kFields, "max_new_tokens");
    if (budget <= 0 || budget > config_.length)
        throw internal::ConfigError("cohere2 max_new_tokens exceeds the positive sequence budget");
    std::unique_lock<std::mutex> lock(generation_, std::try_to_lock);
    if (!lock.owns_lock())
        throw std::invalid_argument("cohere2 model already has an active generation");
    const auto start = Clock::now();
    std::vector<std::int32_t> prefix;
    if (const auto* text = std::get_if<std::string_view>(&request.prefix)) {
        prefix = tokenizer_->encode(*text);
        prefix.insert(prefix.begin(), config_.bos);
    } else {
        const auto ids = std::get<Span<const std::int32_t>>(request.prefix);
        prefix.assign(ids.begin(), ids.end());
    }
    if (prefix.empty() ||
        prefix.size() + static_cast<std::size_t>(budget) > static_cast<std::size_t>(config_.length))
        throw std::invalid_argument(
            "cohere2 prefix plus max_new_tokens exceeds the bundle's sequence length");
    for (const auto id : prefix) {
        if (id < 0 || id >= config_.vocabulary)
            throw std::invalid_argument("cohere2 prefix token ID is outside the vocabulary");
    }
    std::vector<std::int32_t> buffer(static_cast<std::size_t>(config_.length), config_.pad);
    std::copy(prefix.begin(), prefix.end(), buffer.begin());
    std::vector<std::int32_t> generated;
    const auto launch = Clock::now();
    auto prefill_end = launch;
    for (std::int64_t i = 0; i < budget; ++i) {
        auto last = static_cast<std::int32_t>(prefix.size() + generated.size() - 1);
        TensorMap inputs;
        inputs["token_ids"] = Tensor{buffer.data(), {config_.length}, DType::kInt32};
        inputs["last_position"] = Tensor{&last, {1}, DType::kInt32};
        const auto outputs = engine_->forward(inputs);
        const auto found = outputs.find("logits");
        if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
            found->second.numel() != static_cast<std::size_t>(config_.vocabulary))
            throw std::runtime_error("cohere2 engine did not return one FP32 vocabulary row");
        if (i == 0)
            prefill_end = Clock::now();
        const auto* logits = static_cast<const float*>(found->second.data);
        if (!std::all_of(logits, logits + config_.vocabulary,
                         [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("cohere2 engine returned non-finite logits");
        const auto token = static_cast<std::int32_t>(
            std::max_element(logits, logits + config_.vocabulary) - logits);
        generated.push_back(token);
        if (token == config_.eos)
            break;
        buffer[prefix.size() + generated.size() - 1] = token;
    }
    TextResult result(tokenizer_->decode(generated), generated, milliseconds(launch, prefill_end),
                      milliseconds(prefill_end, Clock::now()));
    result.setup_ms = milliseconds(start, launch);
    return result;
}
} // namespace trtmc::cohere2
