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
#include <limits>
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

GemmaTextGenerationPipeline::GemmaTextGenerationPipeline(
    std::unique_ptr<ITrtModule> decoder, std::unique_ptr<GemmaInferenceState> state,
    GemmaTextGenConfig config, std::shared_ptr<ITokenizer> tokenizer,
    std::unique_ptr<ITrtModule> prefill, std::shared_ptr<void> distributed_owner,
    std::unique_ptr<GemmaVisionEncoder> vision, std::unique_ptr<GemmaAudioEncoder> audio,
    std::unique_ptr<ITrtModule> assistant)
    : distributed_owner_(std::move(distributed_owner)), decoder_(std::move(decoder)),
      prefill_(std::move(prefill)), assistant_(std::move(assistant)), state_(std::move(state)),
      config_(normalize_eos_token_ids(std::move(config))), tokenizer_(std::move(tokenizer)),
      logits_output_name_(config_.logits_output_name), vision_(std::move(vision)),
      audio_(std::move(audio)) {
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error("GemmaTextGenerationPipeline: invalid decoder module");
    if (config_.single_context) {
        if (prefill_ || decoder_->profile_idx() != 0 ||
            !decoder_->input_is_dynamic(config_.token_id_name))
            throw std::runtime_error("GemmaTextGenerationPipeline: invalid shared dynamic module");
    } else if (!prefill_ || !prefill_->ok()) {
        throw std::runtime_error("GemmaTextGenerationPipeline: invalid prefill module");
    }
    if (!tokenizer_)
        throw std::runtime_error("GemmaTextGenerationPipeline: invalid tokenizer");
    if (!state_ || !state_->ok()) {
        throw std::runtime_error("GemmaTextGenerationPipeline: invalid inference state");
    }

    if (assistant_ && config_.draft_block_size > 0 &&
        (!assistant_->ok() || config_.draft_feature_width <= 0 || config_.draft_mask_token_id < 0 ||
         config_.draft_mask_token_id >= config_.vocab_size ||
         !prefill_module().has_input("logits_indices") ||
         !prefill_module().has_output("target_features") ||
         !assistant_->has_input("target_features") || !assistant_->has_output("draft_logits") ||
         ((config_.draft_markov || config_.draft_eagle) &&
          !assistant_->has_output("proposal_ids")) ||
         (config_.draft_markov && !assistant_->has_output("proposal_confidence"))))
        throw std::runtime_error("Gemma4 native block draft has an invalid paired contract");
    if (assistant_ && config_.draft_block_size == 0 &&
        (!assistant_->ok() || config_.mtp_hidden_size <= 0 || config_.mtp_target_layers.empty() ||
         !prefill_module().has_input("logits_indices") ||
         !prefill_module().has_output("target_hidden")))
        throw std::runtime_error(
            "Gemma4 native MTP bundle has an invalid target/assistant contract");
    // A shared context changes between prompt and token shapes. Keep the
    // ordinary dynamic enqueue path rather than retaining a phase capture.
    if (!config_.single_context)
        decoder_->enable_cuda_graph();
}

// Encode a prompt, optionally applying a chat template first.
// Deduplicates the leading BOS token that chat templates embed but
// the tokenizer's add_special_tokens may also prepend.
static std::vector<int32_t> encode_prompt(const ITokenizer& tokenizer,
                                          const GemmaTextGenConfig& config,
                                          const std::string& prompt,
                                          const TextGenerationConfig& cfg) {
    if (!cfg.use_chat_template && !cfg.system_prompt.empty())
        throw std::invalid_argument("Gemma system_prompt requires use_chat_template");
    std::string effective = prompt;
    bool templated = false;
    if (cfg.use_chat_template) {
        if (config.chat_template_format.empty())
            throw std::runtime_error(
                "GemmaTextGenerationPipeline: checkpoint has no supported chat template");
        effective = gemma_apply_chat_template(config.chat_template_format, prompt,
                                              cfg.enable_thinking, cfg.system_prompt);
        templated = true;
    }
    auto ids = tokenizer.encode(effective);
    if (templated && ids.size() >= 2 && config.id_bos >= 0 && ids[0] == config.id_bos &&
        ids[1] == config.id_bos) {
        ids.erase(ids.begin());
    }
    return ids;
}

namespace {

Span<const internal::ConfigField> continuation_fields(bool media = false) {
    using internal::ConfigField;
    using internal::ConfigKind;
    static const std::vector<ConfigField> fields{
        {"max_new_tokens", ConfigKind::I64, std::nullopt, "Generated token budget"},
        {"temperature", ConfigKind::F64, 1.0, "Sampling temperature"},
        {"top_k", ConfigKind::I64, std::int64_t{1}, "Top-k sampling"},
        {"top_p", ConfigKind::F64, 1.0, "Nucleus sampling"},
        {"seed", ConfigKind::I64, std::int64_t{-1}, "Sampling seed; -1 is unspecified"},
        {"use_chat_template", ConfigKind::Bool, false, "Apply the checkpoint chat template"},
        {"enable_thinking", ConfigKind::Bool, true, "Checkpoint thinking control"},
        {"system_prompt", ConfigKind::String, std::string_view{}, "Optional system message"},
    };
    static const auto image_fields = [] {
        auto copy = fields;
        copy[5].default_value = true;
        return copy;
    }();
    const auto& selected = media ? image_fields : fields;
    return {selected.data(), selected.size()};
}

} // namespace

std::vector<internal::TaskInstance> GemmaTextGenerationPipeline::task_bindings() {
    std::vector<internal::TaskInstance> bindings{
        internal::bind<internal::ITextContinuation>(*this, continuation_fields())};
    if (vision_)
        bindings.push_back(
            internal::bind<internal::IImagesTextToText>(*this, continuation_fields(true)));
    if (audio_)
        bindings.push_back(
            internal::bind<internal::IAudioTextToText>(*this, continuation_fields(true)));
    if (vision_ && audio_)
        bindings.push_back(
            internal::bind<internal::IImageAudioTextToText>(*this, continuation_fields(true)));
    return bindings;
}

TextResult GemmaTextGenerationPipeline::run(const internal::TextContinuationRequest& input,
                                            internal::ConfigView supplied) {
    const auto* text = std::get_if<std::string_view>(&input.prefix);
    if (!text)
        throw std::invalid_argument("Gemma native continuation accepts UTF-8 text prefixes only");
    return generate(std::string(*text), request_config(supplied, false));
}

TextGenerationConfig GemmaTextGenerationPipeline::request_config(internal::ConfigView supplied,
                                                                 bool media) const {
    const auto fields = continuation_fields(media);
    internal::validate_config(fields, supplied);
    const auto integer = [&](const char* name, int fallback) {
        const auto value =
            internal::config_get<std::int64_t>(supplied, fields, name).value_or(fallback);
        if (value < std::numeric_limits<std::int32_t>::min() ||
            value > std::numeric_limits<std::int32_t>::max())
            throw std::invalid_argument(std::string("Gemma option outside int32 range: ") + name);
        return static_cast<std::int32_t>(value);
    };
    TextGenerationConfig config;
    config.max_new_tokens = integer("max_new_tokens", default_max_new_tokens());
    if (config.max_new_tokens <= 0)
        throw std::invalid_argument("Gemma max_new_tokens must be positive");
    config.top_k = integer("top_k", 1);
    config.seed = integer("seed", -1);
    config.temperature = *internal::config_get<double>(supplied, fields, "temperature");
    config.top_p = *internal::config_get<double>(supplied, fields, "top_p");
    config.use_chat_template = *internal::config_get<bool>(supplied, fields, "use_chat_template");
    config.enable_thinking = *internal::config_get<bool>(supplied, fields, "enable_thinking");
    config.system_prompt =
        *internal::config_get<std::string_view>(supplied, fields, "system_prompt");
    if (media && !config.use_chat_template)
        throw std::invalid_argument("Gemma4 media requires its provider chat template");
    return config;
}

TextResult GemmaTextGenerationPipeline::generate(const std::string& prompt,
                                                 const TextGenerationConfig& cfg) {

    media_features_.clear();
    media_mask_.clear();
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
    media_features_.clear();
    media_mask_.clear();
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
    require_batched_prefill_contract(prefill_module(), config_, sq, state_.get());
    auto* kv = static_cast<GemmaKvCache*>(state_.get());

    // The prefill module shares the same external KV cache buffers as the
    // decode module(s), so we rebind the cache_k/cache_v inputs onto the
    // prefill execution context before running.
    kv->bind_cache_inputs(prefill_module());

    TensorMap inputs;
    Tensor tok_t;
    tok_t.data = const_cast<int32_t*>(input_ids.data());
    tok_t.shape = {static_cast<int64_t>(sq)};
    tok_t.dtype = DType::kInt32;
    inputs[config_.token_id_name] = tok_t;
    state_->prepare_step(inputs, sq);

    int32_t last_row = sq - 1;
    if (assistant_)
        inputs["logits_indices"] = Tensor{&last_row, {1}, DType::kInt32};
    add_media_inputs(inputs, sq, true);
    TensorMap outputs = prefill_module().forward(inputs);
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

    if (assistant_ && config_.draft_block_size > 0) {
        const auto it = outputs.find("target_features");
        const auto count = static_cast<std::size_t>(sq) * config_.draft_feature_width;
        if (it == outputs.end() || it->second.dtype != DType::kFloat32 ||
            it->second.numel() != count || !it->second.data)
            throw std::runtime_error("Gemma4 block target feature output is invalid");
        const auto* features = static_cast<const float*>(it->second.data);
        draft_context_.assign(features, features + count);
    } else if (assistant_) {
        const auto it = outputs.find("target_hidden");
        if (it == outputs.end() || it->second.dtype != DType::kFloat32 ||
            it->second.numel() != static_cast<std::size_t>(config_.mtp_hidden_size) ||
            it->second.data == nullptr)
            throw std::runtime_error("Gemma4 MTP target hidden output is invalid");
        const auto* hidden = static_cast<const float*>(it->second.data);
        target_hidden_.assign(hidden, hidden + config_.mtp_hidden_size);
    }
    std::vector<const void*> pk, pv;
    require_prefill_kv_pointers(prefill_module(), config_, pk, pv);
    kv->write_prefill_kv(pk, pv, sq);
    if (assistant_)
        decoder_->sync(); // The assistant has a different stream from the KV writer.
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
    mtp_stats_ = {};
    block_stats_ = {};
    draft_context_.clear();
    target_hidden_.clear();
    if (assistant_)
        assistant_->reset_execution_context();
    state_bound_ = false;
    decoder_->reset_execution_context();
    if (!config_.single_context)
        prefill_->reset_execution_context();
    last_setup_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

GemmaTextGenerationPipeline::TimedGenResult GemmaTextGenerationPipeline::generate_from_ids(
    const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
    const GemmaSamplingParams& params, const TextGenerationConfig& cfg) {
    using Clock = std::chrono::steady_clock;
    if (max_new_tokens == 0 || input_ids.empty())
        return TimedGenResult{input_ids, 0.0, 0.0};

    // Prefill predicts the first generated token; only the remaining tokens
    // are consumed by the decoder. Never wrap positions beyond the built RoPE.
    if (max_new_tokens < 0 || static_cast<std::int64_t>(input_ids.size()) + max_new_tokens - 1 >
                                  config_.prefill_max_length)
        throw std::invalid_argument("Gemma prompt and generation exceed the engine capacity");

    const std::string mode = resolve_generation_mode(cfg);
    if (mode != "ar")
        throw std::runtime_error("GemmaTextGenerationPipeline: unsupported generation mode '" +
                                 mode + "'");
    const bool greedy =
        params.temperature <= 0 || params.top_p <= 0 || (params.top_k == 1 && params.top_p >= 1) ||
        (params.seed < 0 && params.top_k <= 1 && params.top_p >= 1 && params.min_p <= 0);
    if (assistant_ && config_.draft_block_size == 0 && !greedy)
        throw std::invalid_argument("Gemma4 native MTP currently requires greedy sampling");
    auto active_sampler = create_gemma_sampler(params);
    active_sampler->reset();

    reset_generation_context();
    state_->set_prompt_length(static_cast<int32_t>(input_ids.size()));

    std::vector<float> logits;
    const auto t0 = Clock::now();
    run_prefill(input_ids, logits);
    const auto t1 = Clock::now();

    std::vector<int32_t> output = input_ids;
    if (assistant_ && config_.draft_block_size > 0)
        run_block_loop(active_sampler.get(), params, output, logits, max_new_tokens, cfg,
                       static_cast<int32_t>(input_ids.size()));
    else if (assistant_)
        run_mtp_loop(active_sampler.get(), params, output, logits, max_new_tokens, cfg,
                     static_cast<int32_t>(input_ids.size()));
    else
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
        if (config_.single_context) {
            auto* cache = dynamic_cast<GemmaKvCache*>(state_.get());
            if (!cache)
                throw std::runtime_error("Gemma shared context requires GemmaKvCache");
            cache->bind_cache_inputs(*decoder_);
        } else {
            state_->bind_to(*decoder_);
        }
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

    add_media_inputs(inputs, 1, false);
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

    if (config_.single_context) {
        // Dynamic output allocations remain module-owned across prompt/token
        // shape changes; never replace them with a one-row external buffer.
        std::vector<const void*> keys, values;
        require_prefill_kv_pointers(decoder, config_, keys, values);
        static_cast<GemmaKvCache*>(state_.get())->append_prefill_kv(keys, values, 1);
    } else {
        state_->advance();
    }
}

} // namespace trtmc
