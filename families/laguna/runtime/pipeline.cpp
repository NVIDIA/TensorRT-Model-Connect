/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/laguna/runtime/pipeline.h"

#include "families/laguna/runtime/chat_template.h"
#include "families/laguna/runtime/tensor_names.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace trtmc::laguna {

namespace {

using internal::ConfigField;
using internal::ConfigKind;

constexpr std::int32_t kDefaultMaxNewTokens = 128;

const ConfigField kTextFields[] = {
    {"max_new_tokens", ConfigKind::I64, std::int64_t{kDefaultMaxNewTokens}, "Generation token cap"},
    {"temperature", ConfigKind::F64, 1.0, "Sampling temperature; zero selects greedy"},
    {"top_k", ConfigKind::I64, std::int64_t{1}, "Top-k sampling"},
    {"top_p", ConfigKind::F64, 1.0, "Nucleus probability"},
    {"min_p", ConfigKind::F64, 0.0, "Minimum relative probability"},
    {"seed", ConfigKind::I64, std::int64_t{-1}, "Sampling seed"},
    {"use_chat_template", ConfigKind::Bool, false, "Apply the Laguna chat template"},
    {"enable_thinking", ConfigKind::Bool, true, "Open a <think> block after the prompt"},
    {"system_prompt", ConfigKind::String, std::string_view{}, "Chat system message"},
};

TextGenerationConfig text_config(internal::ConfigView supplied) {
    internal::validate_config(kTextFields, supplied);
    const auto integer = [&](std::string_view name) {
        const auto value = *internal::config_get<std::int64_t>(supplied, kTextFields, name);
        if (value < std::numeric_limits<std::int32_t>::min() ||
            value > std::numeric_limits<std::int32_t>::max())
            throw internal::ConfigError("Laguna config integer is out of range");
        return static_cast<std::int32_t>(value);
    };
    const auto number = [&](std::string_view name) {
        const auto value = *internal::config_get<double>(supplied, kTextFields, name);
        if (!std::isfinite(value) || value < 0 || value > std::numeric_limits<float>::max())
            throw internal::ConfigError("Laguna sampling value is out of range");
        return static_cast<float>(value);
    };
    TextGenerationConfig cfg;
    cfg.max_new_tokens = integer("max_new_tokens");
    cfg.top_k = integer("top_k");
    cfg.seed = integer("seed");
    cfg.temperature = number("temperature");
    cfg.top_p = number("top_p");
    cfg.min_p = number("min_p");
    cfg.use_chat_template = *internal::config_get<bool>(supplied, kTextFields, "use_chat_template");
    cfg.enable_thinking = *internal::config_get<bool>(supplied, kTextFields, "enable_thinking");
    cfg.system_prompt = std::string(
        *internal::config_get<std::string_view>(supplied, kTextFields, "system_prompt"));
    if (cfg.max_new_tokens <= 0 || cfg.top_k < 0 || cfg.top_p > 1 || cfg.min_p > 1)
        throw internal::ConfigError("Laguna generation config is out of range");
    return cfg;
}

void reject_unsupported(const TextGenerationConfig& cfg) {
    if (cfg.repetition_penalty != 1.0F)
        throw std::invalid_argument("Laguna does not support repetition_penalty");
    if (!cfg.lora_adapter_id.empty())
        throw std::invalid_argument("Laguna does not support LoRA adapters");
    if (cfg.stop_on_boxed_answer)
        throw std::invalid_argument("Laguna does not support stop_on_boxed_answer");
    const std::string& mode = cfg.text_generation_mode;
    if (!mode.empty() && mode != "auto" && mode != "autoregressive" && mode != "ar")
        throw std::invalid_argument("Laguna supports only autoregressive generation");
    if (!cfg.use_chat_template && !cfg.system_prompt.empty())
        throw std::invalid_argument("Laguna system_prompt requires use_chat_template");
}

class Lease {
  public:
    explicit Lease(std::atomic<bool>& active) : active_(active) {
        if (active_.exchange(true))
            throw std::invalid_argument("Laguna model already has an active generation");
    }
    ~Lease() { active_.store(false); }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

  private:
    std::atomic<bool>& active_;
};

double elapsed_ms(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since)
        .count();
}

void copy_logits(const TensorMap& outputs, std::int32_t vocab, std::vector<float>& logits) {
    const auto it = outputs.find("logits");
    const auto width = static_cast<std::size_t>(vocab);
    if (it == outputs.end() || it->second.numel() != width || it->second.dtype != DType::kFloat32)
        throw std::runtime_error("Laguna engine returned no [1, vocab] float32 logits");
    logits.resize(width);
    std::memcpy(logits.data(), it->second.data, logits.size() * sizeof(float));
}

} // namespace

Pipeline::Pipeline(std::unique_ptr<ITrtModule> prefill, std::unique_ptr<ITrtModule> decode,
                   std::unique_ptr<KvCache> cache, std::unique_ptr<Tokenizer> tokenizer,
                   PipelineConfig config)
    : prefill_(std::move(prefill)), decode_(std::move(decode)), cache_(std::move(cache)),
      tokenizer_(std::move(tokenizer)), config_(std::move(config)) {
    if (!prefill_ || !prefill_->ok() || !decode_ || !decode_->ok())
        throw std::runtime_error("Laguna pipeline needs valid prefill and decode contexts");
    if (!cache_ || !cache_->ok() || !tokenizer_)
        throw std::runtime_error("Laguna pipeline needs a KV cache and a tokenizer");
    if (config_.vocab_size <= 0 || config_.num_layers <= 0 || config_.prefill_chunk_rows <= 0 ||
        config_.eos_token_ids.empty())
        throw std::runtime_error("Laguna pipeline configuration is invalid");
    decode_->enable_cuda_graph();
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {internal::bind<internal::ITextContinuation>(*this, kTextFields)};
}

TextResult Pipeline::run(const internal::TextContinuationRequest& request,
                         internal::ConfigView config) {
    const auto cfg = text_config(config);
    reject_unsupported(cfg);
    Lease lease(active_);
    if (const auto* text = std::get_if<std::string_view>(&request.prefix))
        return generate_tokens(encode_prompt(std::string(*text), cfg), cfg);
    if (cfg.use_chat_template)
        throw std::invalid_argument("Laguna token-ID input cannot apply a chat template");
    const auto tokens = std::get<Span<const std::int32_t>>(request.prefix);
    return generate_tokens({tokens.begin(), tokens.end()}, cfg);
}

std::vector<std::int32_t> Pipeline::encode_prompt(const std::string& prompt,
                                                  const TextGenerationConfig& cfg) const {
    if (!cfg.use_chat_template)
        return tokenizer_->encode(prompt, true);
    if (!config_.has_chat_template)
        throw std::invalid_argument("this Laguna bundle carries no chat template");
    // The rendered template starts with its own leading special token.
    return tokenizer_->encode(render_chat(prompt, cfg.system_prompt, cfg.enable_thinking), false);
}

SamplingParams Pipeline::sampling_params(const TextGenerationConfig& cfg) const {
    SamplingParams params;
    params.temperature = cfg.temperature;
    params.top_k = cfg.top_k;
    params.top_p = cfg.top_p;
    params.min_p = cfg.min_p;
    params.seed = cfg.seed;
    params.eos_token_ids =
        cfg.eos_token_id >= 0 ? std::vector<std::int32_t>{cfg.eos_token_id} : config_.eos_token_ids;
    return params;
}

void Pipeline::reset() {
    cache_->reset();
    decode_bound_ = false;
    decode_->reset_execution_context();
    prefill_->reset_execution_context();
}

void Pipeline::prefill(const std::vector<std::int32_t>& input_ids, std::vector<float>& logits) {
    cache_->bind_prefill(*prefill_);
    std::vector<const void*> present_k(static_cast<std::size_t>(config_.num_layers));
    std::vector<const void*> present_v(static_cast<std::size_t>(config_.num_layers));
    const auto total = static_cast<std::int32_t>(input_ids.size());
    for (std::int32_t start = 0; start < total; start += config_.prefill_chunk_rows) {
        const std::int32_t rows = std::min(config_.prefill_chunk_rows, total - start);
        TensorMap inputs;
        Tensor token;
        token.data = const_cast<std::int32_t*>(input_ids.data() + start);
        token.shape = {rows};
        token.dtype = DType::kInt32;
        inputs["token_id"] = token;
        cache_->prepare(inputs, rows);
        const TensorMap outputs = prefill_->forward(inputs);
        for (std::int32_t layer = 0; layer < config_.num_layers; ++layer) {
            const auto index = static_cast<std::size_t>(layer);
            present_k[index] = prefill_->device_ptr(layer_tensor_name("present_k", layer));
            present_v[index] = prefill_->device_ptr(layer_tensor_name("present_v", layer));
            if (present_k[index] == nullptr || present_v[index] == nullptr)
                throw std::runtime_error("Laguna prefill context has no present K/V output");
        }
        cache_->append(present_k, present_v, rows);
        if (start + rows == total)
            copy_logits(outputs, config_.vocab_size, logits);
    }
}

void Pipeline::decode_step(std::int32_t token_id, std::vector<float>& logits) {
    if (!decode_bound_) {
        cache_->bind_decode(*decode_);
        decode_bound_ = true;
    }
    TensorMap inputs;
    Tensor token;
    token.data = &token_id;
    token.shape = {1};
    token.dtype = DType::kInt32;
    inputs["token_id"] = token;
    cache_->prepare(inputs, 1);
    copy_logits(decode_->forward(inputs), config_.vocab_size, logits);
    cache_->append_decoded();
}

void Pipeline::prime_decoder(std::int32_t token_id) {
    // Capture the decode CUDA graph outside the timed decode loop. The step's
    // outputs are discarded and the cache is not advanced.
    if (!decode_bound_) {
        cache_->bind_decode(*decode_);
        decode_bound_ = true;
    }
    if (!decode_->cuda_graph_active() || decode_->cuda_graph_captured() ||
        cache_->position() >= cache_->max_length())
        return;
    TensorMap inputs;
    Tensor token;
    token.data = &token_id;
    token.shape = {1};
    token.dtype = DType::kInt32;
    inputs["token_id"] = token;
    cache_->prepare(inputs, 1);
    decode_->forward_async(inputs);
    decode_->sync();
}

TextResult Pipeline::generate_tokens(const std::vector<std::int32_t>& input_ids,
                                     const TextGenerationConfig& cfg) {
    if (input_ids.empty())
        throw std::invalid_argument("Laguna needs a non-empty prompt");
    const std::int32_t max_new = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : kDefaultMaxNewTokens;
    if (input_ids.size() + static_cast<std::size_t>(max_new) >
        static_cast<std::size_t>(cache_->max_length()))
        throw std::invalid_argument(
            "Laguna prompt plus max_new_tokens exceed the bundle's max_sequence_length (" +
            std::to_string(cache_->max_length()) + ")");
    const SamplingParams params = sampling_params(cfg);
    auto sampler = create_sampler(params);
    reset();

    const auto prefill_start = std::chrono::steady_clock::now();
    std::vector<float> logits;
    prefill(input_ids, logits);
    if (max_new > 1)
        prime_decoder(input_ids.back());
    const double prefill_ms = elapsed_ms(prefill_start);

    const auto decode_start = std::chrono::steady_clock::now();
    std::vector<std::int32_t> generated;
    for (std::int32_t step = 0; step < max_new; ++step) {
        const SampleResult next =
            sampler->sample(logits.data(), static_cast<std::int32_t>(logits.size()), params);
        generated.push_back(next.token_id);
        if (next.is_eos || step + 1 >= max_new)
            break;
        decode_step(next.token_id, logits);
    }
    const double decode_ms = elapsed_ms(decode_start);
    TextResult result(tokenizer_->decode(generated), generated, prefill_ms, decode_ms);
    return result;
}

Pipeline::LogitsTrace Pipeline::trace_logits(const std::string& prompt,
                                             const TextGenerationConfig& cfg) {
    Lease lease(active_);
    reject_unsupported(cfg);
    LogitsTrace trace;
    trace.prompt_ids = encode_prompt(prompt, cfg);
    const std::int32_t max_new = std::max(cfg.max_new_tokens, 0);
    if (trace.prompt_ids.empty() || trace.prompt_ids.size() + static_cast<std::size_t>(max_new) >
                                        static_cast<std::size_t>(cache_->max_length()))
        throw std::invalid_argument("Laguna logits trace does not fit the KV cache");
    const SamplingParams params = sampling_params(cfg);
    auto sampler = create_sampler(params);
    reset();
    prefill(trace.prompt_ids, trace.prefill_last);
    reset();
    std::vector<float> logits;
    for (const std::int32_t token : trace.prompt_ids) {
        decode_step(token, logits);
        trace.rows.push_back(logits);
    }
    for (std::int32_t step = 0; step + 1 < max_new; ++step) {
        const SampleResult next =
            sampler->sample(logits.data(), static_cast<std::int32_t>(logits.size()), params);
        if (next.is_eos)
            break;
        decode_step(next.token_id, logits);
        trace.rows.push_back(logits);
    }
    return trace;
}

} // namespace trtmc::laguna
