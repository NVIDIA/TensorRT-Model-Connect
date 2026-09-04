/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/llama/runtime/pipeline.h"

#include "families/llama/runtime/chat_templates.h"
#include "families/llama/runtime/kv_cache.h"
#include "families/llama/runtime/tensor_names.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace trtmc {

namespace {

bool contains_boxed_answer(const std::string& text) {
    const std::string marker = "\\boxed{";
    const auto start = text.find(marker);
    if (start == std::string::npos)
        return false;
    return text.find('}', start + marker.size()) != std::string::npos;
}

bool contains_final_answer(const std::string& text) {
    const std::string marker = "Final answer:";
    const auto start = text.find(marker);
    if (start == std::string::npos)
        return false;
    for (std::size_t i = start + marker.size(); i < text.size(); ++i) {
        if (!std::isspace(static_cast<unsigned char>(text[i])))
            return true;
    }
    return false;
}

LlamaTextGenConfig normalize_eos_token_ids(LlamaTextGenConfig config) {
    if (config.id_eos_ids.empty() && config.id_eos >= 0)
        config.id_eos_ids.push_back(config.id_eos);
    if (!config.id_eos_ids.empty())
        config.id_eos = config.id_eos_ids.front();
    return config;
}

std::string normalize_generation_mode(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    std::replace(mode.begin(), mode.end(), '-', '_');
    return mode;
}

} // namespace

LlamaTextGenerationPipeline::LlamaTextGenerationPipeline(std::unique_ptr<ITrtModule> decoder,
                                                         std::unique_ptr<LlamaInferenceState> state,
                                                         LlamaTextGenConfig config,
                                                         std::shared_ptr<ITokenizer> tokenizer,
                                                         std::unique_ptr<ITrtModule> prefill,
                                                         std::shared_ptr<void> distributed_owner)
    : distributed_owner_(std::move(distributed_owner)), decoder_(std::move(decoder)),
      prefill_(std::move(prefill)), state_(std::move(state)),
      config_(normalize_eos_token_ids(std::move(config))), tokenizer_(std::move(tokenizer)),
      logits_output_name_(config_.logits_output_name) {
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error("LlamaTextGenerationPipeline: invalid decoder module");
    if (!prefill_ || !prefill_->ok())
        throw std::runtime_error("LlamaTextGenerationPipeline: invalid prefill module");
    if (!tokenizer_)
        throw std::runtime_error("LlamaTextGenerationPipeline: invalid tokenizer");
    if (!state_ || !state_->ok()) {
        throw std::runtime_error("LlamaTextGenerationPipeline: invalid inference state");
    }

    decoder_->enable_cuda_graph();
}

// Encode a prompt, optionally applying a chat template first.
// Deduplicates the leading BOS token that chat templates embed but
// the tokenizer's add_special_tokens may also prepend.
static std::vector<int32_t> encode_prompt(const ITokenizer& tokenizer,
                                          const LlamaTextGenConfig& config,
                                          const std::string& prompt,
                                          const TextGenerationConfig& cfg) {
    std::string effective = prompt;
    bool templated = false;
    if (cfg.use_chat_template && !config.chat_template_format.empty()) {
        effective =
            llama_apply_chat_template(config.chat_template_format, prompt, cfg.enable_thinking);
        templated = true;
    }
    auto ids = tokenizer.encode(effective);
    if (templated && ids.size() >= 2 && config.id_bos >= 0 && ids[0] == config.id_bos &&
        ids[1] == config.id_bos) {
        ids.erase(ids.begin());
    }
    return ids;
}

TextResult LlamaTextGenerationPipeline::generate(const std::string& prompt,
                                                 const TextGenerationConfig& cfg) {

    auto input_ids = encode_prompt(*tokenizer_, config_, prompt, cfg);
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    auto sp = llama_sampling_params_from_config(cfg, config_.id_eos_ids);
    last_setup_ms_ = 0.0;
    auto timed = generate_from_ids(input_ids, max_new, sp, cfg);

    // Decode only the NEW tokens (skip input)
    std::vector<int32_t> new_tokens(timed.token_ids.begin() +
                                        static_cast<std::ptrdiff_t>(input_ids.size()),
                                    timed.token_ids.end());
    std::string text = tokenizer_->decode(new_tokens);

    auto result =
        TextResult{std::move(text), std::move(new_tokens), timed.prefill_ms, timed.decode_ms};
    result.setup_ms = last_setup_ms_;
    return result;
}

LlamaTextGenerationPipeline::GenerationResult
LlamaTextGenerationPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                          const TextGenerationConfig& cfg) {
    int32_t max_new = cfg.max_new_tokens; // honour exact value (0 = no generation)
    auto sp = llama_sampling_params_from_config(cfg, config_.id_eos_ids);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp, cfg).token_ids};
}

namespace {

void require_prefill_kv_pointers(ITrtModule& prefill, const LlamaTextGenConfig& cfg,
                                 std::vector<const void*>& pk, std::vector<const void*>& pv) {
    pk.resize(static_cast<std::size_t>(cfg.num_layers));
    pv.resize(static_cast<std::size_t>(cfg.num_layers));
    for (int32_t layer = 0; layer < cfg.num_layers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        pk[index] = prefill.device_ptr(llama_expand_layer_name(cfg.present_k_pattern, layer));
        pv[index] = prefill.device_ptr(llama_expand_layer_name(cfg.present_v_pattern, layer));
        if (pk[index] == nullptr || pv[index] == nullptr) {
            throw std::runtime_error(
                "LlamaTextGenerationPipeline: prefill module is missing K/V output for layer " +
                std::to_string(layer));
        }
    }
}

void require_batched_prefill_contract(ITrtModule& prefill, const LlamaTextGenConfig& cfg,
                                      int32_t sq, LlamaInferenceState* state) {
    if (!prefill.ok())
        throw std::runtime_error("LlamaTextGenerationPipeline: invalid prefill module");
    if (sq <= 0)
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: prefill requires a non-empty prompt");
    if (cfg.prefill_max_length <= 0 || cfg.num_layers <= 0 || cfg.vocab_size <= 0)
        throw std::runtime_error("LlamaTextGenerationPipeline: invalid prefill configuration");
    auto* kv = dynamic_cast<LlamaKvCache*>(state);
    if (kv == nullptr)
        throw std::runtime_error("LlamaTextGenerationPipeline: prefill requires LlamaKvCache");
}

int32_t resolve_prefill_chunk_limit(const LlamaKvCache& kv, const LlamaTextGenConfig& cfg,
                                    int32_t sq) {
    if (kv.needs_attention_mask()) {
        if (cfg.prefill_max_length > 0 && sq > cfg.prefill_max_length)
            throw std::runtime_error("Llama prompt exceeds the prefill profile");
        return sq;
    }
    if (cfg.prefill_max_length <= 0)
        throw std::runtime_error("Llama native KV prefill engine has no valid profile capacity");
    return cfg.prefill_max_length;
}

void validate_generation_capacity(const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
                                  LlamaInferenceState* state) {
    const auto* kv = dynamic_cast<const LlamaKvCache*>(state);
    if (kv == nullptr)
        throw std::runtime_error("Llama generation requires LlamaKvCache");

    const auto capacity = static_cast<std::size_t>(kv->max_length());
    if (input_ids.size() > capacity ||
        (max_new_tokens > 0 &&
         static_cast<std::size_t>(max_new_tokens) > capacity - input_ids.size())) {
        throw std::runtime_error(
            "Llama requested prompt and generation exceed the model's fixed KV cache capacity");
    }
}
} // namespace

void LlamaTextGenerationPipeline::run_prefill_batched(const std::vector<int32_t>& input_ids,
                                                      std::vector<float>& logits) {
    const auto sq = static_cast<int32_t>(input_ids.size());
    require_batched_prefill_contract(*prefill_, config_, sq, state_.get());
    auto* kv = static_cast<LlamaKvCache*>(state_.get());

    // The prefill module shares the same external KV cache buffers as the
    // decode module(s), so we rebind the cache_k/cache_v inputs onto the
    // prefill execution context before running.
    kv->bind_cache_inputs(*prefill_);
    if (sq > kv->max_length()) {
        throw std::runtime_error("Llama sequence exceeds the model's fixed KV cache capacity");
    }

    std::vector<const void*> pk, pv;
    require_prefill_kv_pointers(*prefill_, config_, pk, pv);

    const int32_t chunk_limit = resolve_prefill_chunk_limit(*kv, config_, sq);
    int32_t launches = 0;
    int32_t max_chunk = 0;
    for (int32_t start = 0; start < sq;) {
        const int32_t chunk_size = std::min(chunk_limit, sq - start);
        run_prefill_chunk(input_ids.data() + start, chunk_size, pk, pv, *kv, logits);
        ++launches;
        max_chunk = std::max(max_chunk, chunk_size);
        start += chunk_size;
    }
    std::cerr << "[trtmc.prefill] tokens=" << sq << " launches=" << launches
              << " max_chunk=" << max_chunk << '\n';
}

void LlamaTextGenerationPipeline::run_prefill_chunk(const int32_t* token_ids, int32_t chunk_size,
                                                    const std::vector<const void*>& present_k,
                                                    const std::vector<const void*>& present_v,
                                                    LlamaKvCache& kv, std::vector<float>& logits) {
    TensorMap inputs;
    Tensor token_tensor;
    token_tensor.data = const_cast<int32_t*>(token_ids);
    token_tensor.shape = {static_cast<int64_t>(chunk_size)};
    token_tensor.dtype = DType::kInt32;
    inputs[config_.token_id_name] = token_tensor;
    state_->prepare_step(inputs, chunk_size);

    TensorMap outputs = prefill_->forward(inputs);
    const auto logits_it = outputs.find(config_.logits_output_name);
    if (logits_it == outputs.end()) {
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: prefill module has no logits output");
    }

    const auto& logits_tensor = logits_it->second;
    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    if (static_cast<std::size_t>(logits_tensor.numel()) < vocab) {
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: prefill logits are smaller than vocabulary");
    }

    logits.resize(vocab);
    const auto logits_offset = static_cast<std::size_t>(logits_tensor.numel()) - vocab;
    std::memcpy(logits.data(), static_cast<const float*>(logits_tensor.data) + logits_offset,
                vocab * sizeof(float));
    kv.append_prefill_kv(present_k, present_v, chunk_size);
}

void LlamaTextGenerationPipeline::prime_decoder_after_batched_prefill(
    const std::vector<int32_t>& input_ids) {
    if (input_ids.empty())
        return;

    ITrtModule& decoder = bind_decoder_for_step();
    if (!decoder.cuda_graph_active())
        return;

    int32_t token_id = input_ids.back();
    TensorMap inputs;
    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;
    inputs[config_.token_id_name] = token_tensor;

    state_->prepare_step(inputs);
    decoder.forward_async(inputs);
    decoder.sync();
}

void LlamaTextGenerationPipeline::run_prefill(const std::vector<int32_t>& input_ids,
                                              std::vector<float>& logits) {
    run_prefill_batched(input_ids, logits);
    prime_decoder_after_batched_prefill(input_ids);
    state_->mark_prefill_complete();
}

std::string
LlamaTextGenerationPipeline::resolve_generation_mode(const TextGenerationConfig& cfg) const {
    std::string mode = normalize_generation_mode(cfg.text_generation_mode);
    if (mode.empty() || mode == "auto" || mode == "autoregressive")
        return "ar";
    return mode;
}

void LlamaTextGenerationPipeline::reset_generation_context() {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    state_->reset();
    state_bound_ = false;
    decoder_->reset_execution_context();
    prefill_->reset_execution_context();
    last_setup_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

LlamaTextGenerationPipeline::TimedGenResult LlamaTextGenerationPipeline::generate_from_ids(
    const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
    const LlamaSamplingParams& params, const TextGenerationConfig& cfg) {
    using Clock = std::chrono::steady_clock;
    if (max_new_tokens == 0 || input_ids.empty())
        return TimedGenResult{input_ids, 0.0, 0.0};
    validate_generation_capacity(input_ids, max_new_tokens, state_.get());

    const std::string mode = resolve_generation_mode(cfg);
    if (mode != "ar")
        throw std::runtime_error("LlamaTextGenerationPipeline: unsupported generation mode '" +
                                 mode + "'");
    auto active_sampler = create_llama_sampler(params);
    active_sampler->reset();

    reset_generation_context();
    state_->set_prompt_length(static_cast<int32_t>(input_ids.size()));

    std::vector<float> logits;
    const auto t0 = Clock::now();
    run_prefill(input_ids, logits);
    const auto t1 = Clock::now();

    std::vector<int32_t> output = input_ids;
    run_decode_loop(active_sampler.get(), params, output, logits, max_new_tokens, cfg,
                    static_cast<int32_t>(input_ids.size()));
    const auto t2 = Clock::now();

    const double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    return TimedGenResult{std::move(output), prefill_ms, decode_ms};
}

bool LlamaTextGenerationPipeline::should_stop_on_answer(const std::vector<int32_t>& output,
                                                        int32_t prompt_token_count,
                                                        const TextGenerationConfig& cfg,
                                                        int32_t steps, int32_t stop_interval,
                                                        bool is_eos) const {
    if (!cfg.stop_on_boxed_answer || !tokenizer_)
        return false;
    if ((steps % stop_interval) != 0 && !is_eos)
        return false;
    std::vector<int32_t> new_tokens(output.begin() + prompt_token_count, output.end());
    const std::string decoded = tokenizer_->decode(new_tokens);
    return contains_boxed_answer(decoded) || contains_final_answer(decoded);
}

int32_t LlamaTextGenerationPipeline::run_decode_loop(
    LlamaISampler* sampler, const LlamaSamplingParams& params, std::vector<int32_t>& output,
    std::vector<float>& logits, int32_t max_new_tokens, const TextGenerationConfig& cfg,
    int32_t prompt_token_count) {
    const int32_t vocab_size = static_cast<int32_t>(logits.size());
    const int32_t stop_interval = std::max(cfg.stop_check_interval, 1);
    int32_t steps = 0;
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const LlamaSampleResult result = sampler->sample(logits.data(), vocab_size, params);
        const bool is_eos = result.is_eos || llama_is_eos_token(params, result.token_id);
        output.push_back(result.token_id);
        ++steps;
        if (should_stop_on_answer(output, prompt_token_count, cfg, steps, stop_interval, is_eos))
            break;
        if (is_eos)
            break;
        run_step(result.token_id, logits);
    }
    return steps;
}

ITrtModule& LlamaTextGenerationPipeline::bind_decoder_for_step() {
    if (!state_bound_) {
        state_->bind_to(*decoder_);
        state_bound_ = true;
    }
    return *decoder_;
}

void LlamaTextGenerationPipeline::run_step(int32_t token_id, std::vector<float>& logits) {
    TensorMap inputs;

    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;
    inputs[config_.token_id_name] = token_tensor;

    ITrtModule& decoder = bind_decoder_for_step();
    state_->prepare_step(inputs);

    TensorMap outputs = decoder.forward(inputs);

    auto it = outputs.find(logits_output_name_);
    if (it == outputs.end()) {
        throw std::runtime_error("LlamaTextGenerationPipeline: no '" + logits_output_name_ +
                                 "' output");
    }

    const auto& logits_tensor = it->second;
    auto num_logits = logits_tensor.numel();
    logits.resize(static_cast<std::size_t>(num_logits));
    std::memcpy(logits.data(), logits_tensor.data, num_logits * sizeof(float));

    state_->advance();
}

} // namespace trtmc
