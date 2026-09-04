/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/gemma/runtime/pipeline.h"

#include "families/gemma/runtime/chat_templates.h"
#include "families/gemma/runtime/kv_cache.h"
#include "families/gemma/runtime/tensor_names.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

GemmaTextGenConfig normalize_eos_token_ids(GemmaTextGenConfig config) {
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

GemmaTextGenerationPipeline::GemmaTextGenerationPipeline(std::unique_ptr<ITrtModule> decoder,
                                                         std::unique_ptr<GemmaInferenceState> state,
                                                         GemmaTextGenConfig config,
                                                         std::shared_ptr<ITokenizer> tokenizer,
                                                         std::unique_ptr<ITrtModule> prefill,
                                                         std::shared_ptr<void> distributed_owner)
    : distributed_owner_(std::move(distributed_owner)), decoder_(std::move(decoder)),
      prefill_(std::move(prefill)), state_(std::move(state)),
      config_(normalize_eos_token_ids(std::move(config))), tokenizer_(std::move(tokenizer)),
      logits_output_name_(config_.logits_output_name) {
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error("GemmaTextGenerationPipeline: invalid decoder module");
    if (!prefill_ || !prefill_->ok())
        throw std::runtime_error("GemmaTextGenerationPipeline: invalid prefill module");
    if (!tokenizer_)
        throw std::runtime_error("GemmaTextGenerationPipeline: invalid tokenizer");
    if (!state_ || !state_->ok()) {
        throw std::runtime_error("GemmaTextGenerationPipeline: invalid inference state");
    }

    decoder_->enable_cuda_graph();
}

// Encode a prompt, optionally applying a chat template first.
// Deduplicates the leading BOS token that chat templates embed but
// the tokenizer's add_special_tokens may also prepend.
static std::vector<int32_t> encode_prompt(const ITokenizer& tokenizer,
                                          const GemmaTextGenConfig& config,
                                          const std::string& prompt,
                                          const TextGenerationConfig& cfg) {
    std::string effective = prompt;
    bool templated = false;
    if (cfg.use_chat_template) {
        if (config.chat_template_format.empty())
            throw std::runtime_error(
                "GemmaTextGenerationPipeline: checkpoint has no supported chat template");
        effective =
            gemma_apply_chat_template(config.chat_template_format, prompt, cfg.enable_thinking);
        templated = true;
    }
    auto ids = tokenizer.encode(effective);
    if (templated && ids.size() >= 2 && config.id_bos >= 0 && ids[0] == config.id_bos &&
        ids[1] == config.id_bos) {
        ids.erase(ids.begin());
    }
    return ids;
}

TextResult GemmaTextGenerationPipeline::generate(const std::string& prompt,
                                                 const TextGenerationConfig& cfg) {

    auto input_ids = encode_prompt(*tokenizer_, config_, prompt, cfg);
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    auto sp = gemma_sampling_params_from_config(cfg, config_.id_eos_ids);
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

GemmaTextGenerationPipeline::GenerationResult
GemmaTextGenerationPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                          const TextGenerationConfig& cfg) {
    int32_t max_new = cfg.max_new_tokens; // honour exact value (0 = no generation)
    auto sp = gemma_sampling_params_from_config(cfg, config_.id_eos_ids);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp, cfg).token_ids};
}

namespace {

void require_prefill_kv_pointers(ITrtModule& prefill, const GemmaTextGenConfig& cfg,
                                 std::vector<const void*>& pk, std::vector<const void*>& pv) {
    pk.resize(static_cast<std::size_t>(cfg.num_layers));
    pv.resize(static_cast<std::size_t>(cfg.num_layers));
    for (int32_t layer = 0; layer < cfg.num_layers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        pk[index] = prefill.device_ptr(gemma_expand_layer_name(cfg.present_k_pattern, layer));
        pv[index] = prefill.device_ptr(gemma_expand_layer_name(cfg.present_v_pattern, layer));
        if (pk[index] == nullptr || pv[index] == nullptr) {
            throw std::runtime_error(
                "GemmaTextGenerationPipeline: prefill module is missing K/V output for layer " +
                std::to_string(layer));
        }
    }
}

void require_batched_prefill_contract(ITrtModule& prefill, const GemmaTextGenConfig& cfg,
                                      int32_t sq, GemmaInferenceState* state) {
    if (!prefill.ok())
        throw std::runtime_error("GemmaTextGenerationPipeline: invalid prefill module");
    if (sq <= 0)
        throw std::runtime_error(
            "GemmaTextGenerationPipeline: prefill requires a non-empty prompt");
    if (cfg.prefill_max_length <= 0 || cfg.num_layers <= 0 || cfg.vocab_size <= 0)
        throw std::runtime_error("GemmaTextGenerationPipeline: invalid prefill configuration");
    if (sq > cfg.prefill_max_length)
        throw std::runtime_error("GemmaTextGenerationPipeline: prompt exceeds the prefill profile");
    auto* kv = dynamic_cast<GemmaKvCache*>(state);
    if (kv == nullptr)
        throw std::runtime_error("GemmaTextGenerationPipeline: prefill requires GemmaKvCache");
}

} // namespace

void GemmaTextGenerationPipeline::run_prefill_batched(const std::vector<int32_t>& input_ids,
                                                      std::vector<float>& logits) {
    const auto sq = static_cast<int32_t>(input_ids.size());
    require_batched_prefill_contract(*prefill_, config_, sq, state_.get());
    auto* kv = static_cast<GemmaKvCache*>(state_.get());

    // The prefill module shares the same external KV cache buffers as the
    // decode module(s), so we rebind the cache_k/cache_v inputs onto the
    // prefill execution context before running.
    kv->bind_cache_inputs(*prefill_);

    TensorMap inputs;
    Tensor tok_t;
    tok_t.data = const_cast<int32_t*>(input_ids.data());
    tok_t.shape = {static_cast<int64_t>(sq)};
    tok_t.dtype = DType::kInt32;
    inputs[config_.token_id_name] = tok_t;
    state_->prepare_step(inputs, sq);

    TensorMap outputs = prefill_->forward(inputs);
    auto logits_it = outputs.find(config_.logits_output_name);
    if (logits_it == outputs.end())
        throw std::runtime_error(
            "GemmaTextGenerationPipeline: prefill module has no logits output");

    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    const auto& lt = logits_it->second;
    if (static_cast<std::size_t>(lt.numel()) < vocab)
        throw std::runtime_error(
            "GemmaTextGenerationPipeline: prefill logits are smaller than vocabulary");
    logits.resize(vocab);
    const auto offset = static_cast<std::size_t>(lt.numel()) - vocab;
    std::memcpy(logits.data(), static_cast<const float*>(lt.data) + offset, vocab * sizeof(float));

    std::vector<const void*> pk, pv;
    require_prefill_kv_pointers(*prefill_, config_, pk, pv);
    kv->write_prefill_kv(pk, pv, sq);
}

void GemmaTextGenerationPipeline::run_prefill(const std::vector<int32_t>& input_ids,
                                              std::vector<float>& logits) {
    run_prefill_batched(input_ids, logits);
    state_->mark_prefill_complete();
}

std::string
GemmaTextGenerationPipeline::resolve_generation_mode(const TextGenerationConfig& cfg) const {
    std::string mode = normalize_generation_mode(cfg.text_generation_mode);
    if (mode.empty() || mode == "auto" || mode == "autoregressive")
        return "ar";
    return mode;
}

void GemmaTextGenerationPipeline::reset_generation_context() {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    state_->reset();
    state_bound_ = false;
    decoder_->reset_execution_context();
    prefill_->reset_execution_context();
    last_setup_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

GemmaTextGenerationPipeline::TimedGenResult GemmaTextGenerationPipeline::generate_from_ids(
    const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
    const GemmaSamplingParams& params, const TextGenerationConfig& cfg) {
    using Clock = std::chrono::steady_clock;
    if (max_new_tokens == 0 || input_ids.empty())
        return TimedGenResult{input_ids, 0.0, 0.0};

    const std::string mode = resolve_generation_mode(cfg);
    if (mode != "ar")
        throw std::runtime_error("GemmaTextGenerationPipeline: unsupported generation mode '" +
                                 mode + "'");
    auto active_sampler = create_gemma_sampler(params);
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

bool GemmaTextGenerationPipeline::should_stop_on_answer(const std::vector<int32_t>& output,
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

int32_t GemmaTextGenerationPipeline::run_decode_loop(
    GemmaISampler* sampler, const GemmaSamplingParams& params, std::vector<int32_t>& output,
    std::vector<float>& logits, int32_t max_new_tokens, const TextGenerationConfig& cfg,
    int32_t prompt_token_count) {
    const int32_t vocab_size = static_cast<int32_t>(logits.size());
    const int32_t stop_interval = std::max(cfg.stop_check_interval, 1);
    int32_t steps = 0;
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const GemmaSampleResult result = sampler->sample(logits.data(), vocab_size, params);
        output.push_back(result.token_id);
        ++steps;
        if (should_stop_on_answer(output, prompt_token_count, cfg, steps, stop_interval,
                                  result.is_eos))
            break;
        if (result.is_eos)
            break;
        // The sampled token is already the final requested output. Do not
        // execute a decoder step whose logits cannot be consumed.
        if (step + 1 >= max_new_tokens)
            break;
        run_step(result.token_id, logits);
    }
    return steps;
}

ITrtModule& GemmaTextGenerationPipeline::bind_decoder_for_step() {
    if (!state_bound_) {
        state_->bind_to(*decoder_);
        state_bound_ = true;
    }
    return *decoder_;
}

void GemmaTextGenerationPipeline::run_step(int32_t token_id, std::vector<float>& logits) {
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
        throw std::runtime_error("GemmaTextGenerationPipeline: no '" + logits_output_name_ +
                                 "' output");
    }

    const auto& logits_tensor = it->second;
    auto num_logits = logits_tensor.numel();
    logits.resize(static_cast<std::size_t>(num_logits));
    std::memcpy(logits.data(), logits_tensor.data, num_logits * sizeof(float));

    state_->advance();
}

} // namespace trtmc
