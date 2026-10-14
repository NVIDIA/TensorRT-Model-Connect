/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2_moe/runtime/pipeline.h"

#include "families/lfm2_moe/runtime/chat_templates.h"
#include "families/lfm2_moe/runtime/conversation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
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

using internal::ConfigField;
using internal::ConfigKind;
const std::vector<ConfigField>& text_fields(bool conversation) {
    static const std::vector<ConfigField> common = {
        {"max_new_tokens", ConfigKind::I64, std::int64_t{128}, "Generated token cap"},
        {"temperature", ConfigKind::F64, 1.0, "Sampling temperature; zero is greedy"},
        {"top_k", ConfigKind::I64, std::int64_t{1}, "Top-k sampling; zero is unrestricted"},
        {"top_p", ConfigKind::F64, 1.0, "Nucleus probability"},
        {"min_p", ConfigKind::F64, 0.0, "Minimum relative probability"},
        {"repetition_penalty", ConfigKind::F64, 1.0, "Prompt and continuation repetition penalty"},
        {"seed", ConfigKind::I64, std::int64_t{-1}, "Native sampler seed"},
        {"eos_token_id", ConfigKind::I64, std::int64_t{-1},
         "EOS override; negative uses checkpoint"},
    };
    static const auto continuation = [] {
        auto fields = common;
        fields.push_back(
            {"use_chat_template", ConfigKind::Bool, false, "Apply single-user chat template"});
        return fields;
    }();
    static const auto dialogue = [] {
        auto fields = common;
        fields.push_back({"preserve_thinking", ConfigKind::Bool, false,
                          "Preserve reasoning before the last user turn"});
        return fields;
    }();
    return conversation ? dialogue : continuation;
}

Span<const ConfigField> field_span(bool conversation) {
    const auto& fields = text_fields(conversation);
    return {fields.data(), fields.size()};
}

TextGenerationConfig sdk_config(internal::ConfigView values, bool conversation) {
    const auto fields = field_span(conversation);
    internal::validate_config(fields, values);
    const auto integer = [&](const char* name) {
        const auto value = *internal::config_get<std::int64_t>(values, fields, name);
        if (value < std::numeric_limits<int32_t>::min() ||
            value > std::numeric_limits<int32_t>::max())
            throw internal::ConfigError("LFM2-MoE generation integer is out of range");
        return static_cast<int32_t>(value);
    };
    const auto number = [&](const char* name) {
        const auto value = *internal::config_get<double>(values, fields, name);
        if (!std::isfinite(value) || value < 0 || value > std::numeric_limits<float>::max())
            throw internal::ConfigError("LFM2-MoE generation number is out of range");
        return static_cast<float>(value);
    };
    TextGenerationConfig config;
    config.max_new_tokens = integer("max_new_tokens");
    config.top_k = integer("top_k");
    config.seed = integer("seed");
    config.eos_token_id = integer("eos_token_id");
    config.temperature = number("temperature");
    config.top_p = number("top_p");
    config.min_p = number("min_p");
    config.repetition_penalty = number("repetition_penalty");
    if (config.max_new_tokens <= 0 || config.top_k < 0 || config.top_p > 1 || config.min_p > 1 ||
        config.repetition_penalty <= 0)
        throw internal::ConfigError("LFM2-MoE generation config is out of range");
    if (!conversation)
        config.use_chat_template = *internal::config_get<bool>(values, fields, "use_chat_template");
    return config;
}

std::string decode_conversation(const ITokenizer& tokenizer, const std::vector<int32_t>& ids) {
    std::string decoded;
    std::vector<int32_t> text;
    const auto flush = [&] {
        decoded += tokenizer.decode(text);
        text.clear();
    };
    for (int32_t id : ids) {
        const auto token = tokenizer.token_for_id(id);
        if (token == "<think>" || token == "</think>" || token == "<|tool_call_start|>" ||
            token == "<|tool_call_end|>") {
            flush();
            decoded += token;
        } else {
            text.push_back(id);
        }
    }
    flush();
    return decoded;
}

} // namespace

Lfm2MoeTextGenerationPipeline::Lfm2MoeTextGenerationPipeline(
    std::unique_ptr<ITrtModule> decoder, std::unique_ptr<Lfm2MoeInferenceState> state,
    Lfm2MoeTextGenConfig config, cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id)
    : decoder_(std::move(decoder)), state_(std::move(state)), config_(std::move(config)),
      stream_(stream), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id)) {
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

std::vector<int32_t>
Lfm2MoeTextGenerationPipeline::encode_prompt(const std::string& prompt,
                                             const TextGenerationConfig& cfg) const {
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
                                                   const TextGenerationConfig& cfg) {
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

std::vector<internal::TaskInstance> Lfm2MoeTextGenerationPipeline::task_bindings() {
    std::vector<internal::TaskInstance> bindings{
        internal::bind<internal::ITextContinuation>(*this, field_span(false))};
    if (config_.chat_template_format == "chatml")
        bindings.push_back(internal::bind<internal::ITextConversation>(*this, field_span(true)));
    return bindings;
}

TextResult Lfm2MoeTextGenerationPipeline::run(const internal::TextContinuationRequest& request,
                                              internal::ConfigView values) {
    const auto config = sdk_config(values, false);
    if (!tokenizer_)
        throw std::invalid_argument("LFM2-MoE continuation requires a tokenizer");
    std::vector<int32_t> input;
    if (const auto* prefix = std::get_if<std::string_view>(&request.prefix))
        input = encode_prompt(std::string(*prefix), config);
    else {
        if (config.use_chat_template)
            throw std::invalid_argument("LFM2-MoE token input cannot apply a chat template");
        const auto tokens = std::get<Span<const int32_t>>(request.prefix);
        input.assign(tokens.begin(), tokens.end());
    }
    if (input.empty() || input.size() + static_cast<std::size_t>(config.max_new_tokens) >
                             static_cast<std::size_t>(state_->max_length()))
        throw std::invalid_argument(
            "LFM2-MoE continuation exceeds native cache capacity or has no input");
    if (std::any_of(input.begin(), input.end(),
                    [&](int32_t id) { return id < 0 || id >= config_.vocab_size; }))
        throw std::invalid_argument("LFM2-MoE continuation token ID is outside the vocabulary");
    auto generated =
        generate_from_ids(input, config.max_new_tokens,
                          lfm2_moe_sampling_params_from_config(config, config_.id_eos_ids));
    std::vector<int32_t> ids(generated.token_ids.begin() +
                                 static_cast<std::ptrdiff_t>(input.size()),
                             generated.token_ids.end());
    TextResult result{tokenizer_->decode(ids), std::move(ids), generated.prefill_ms,
                      generated.decode_ms};
    result.setup_ms = generated.setup_ms;
    return result;
}

internal::ConversationResult
Lfm2MoeTextGenerationPipeline::run(const internal::TextConversationRequest& request,
                                   internal::ConfigView values) {
    const auto config = sdk_config(values, true);
    if (config_.chat_template_format != "chatml")
        throw internal::UnsupportedTask(
            "LFM2-MoE conversation requires the LFM2-MoE.5 reasoning template");
    const auto preserve =
        *internal::config_get<bool>(values, field_span(true), "preserve_thinking");
    const auto prompt = lfm2_moe_render_conversation(request, preserve);
    const auto input = encode_prompt(prompt, config);
    if (input.size() + static_cast<std::size_t>(config.max_new_tokens) >
        static_cast<std::size_t>(state_->max_length()))
        throw std::invalid_argument("LFM2-MoE conversation exceeds native cache capacity");
    const auto params = lfm2_moe_sampling_params_from_config(config, config_.id_eos_ids);
    auto generated = generate_from_ids(input, config.max_new_tokens, params);
    std::vector<int32_t> ids(generated.token_ids.begin() +
                                 static_cast<std::ptrdiff_t>(input.size()),
                             generated.token_ids.end());
    auto result = lfm2_moe_parse_conversation(decode_conversation(*tokenizer_, ids), false,
                                              !request.tools.empty());
    lfm2_moe_assign_tool_call_ids(result, request);
    result.setup_ms = generated.setup_ms;
    result.prefill_ms = generated.prefill_ms;
    result.decode_ms = generated.decode_ms;
    const bool stopped = !ids.empty() && lfm2_moe_is_eos_token(params, ids.back());
    result.finish_reason = stopped ? internal::FinishReason::Stop : internal::FinishReason::Length;
    if (stopped && std::any_of(result.parts.begin(), result.parts.end(), [](const auto& part) {
            const auto* call = std::get_if<internal::ToolCall>(&part);
            return call && call->state == internal::ToolCallState::Complete;
        }))
        result.finish_reason = internal::FinishReason::ToolCalls;
    result.usage.input_tokens = input.size();
    result.usage.output_tokens = ids.size();
    result.usage.total_tokens = input.size() + ids.size();
    result.token_ids = std::move(ids);
    return result;
}

Lfm2MoeTextGenerationPipeline::GenerationResult
Lfm2MoeTextGenerationPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                            const TextGenerationConfig& cfg) {
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
            "LFM2-MoE prompt and generation exceed the fixed native KV cache capacity");
    }

    auto local_sampler = create_lfm2_moe_sampler(params);
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
        run_step(sampled.token_id, logits, true);
    }
    const auto decode_end = Clock::now();

    return TimedGenerationResult{std::move(output), milliseconds(setup_start, setup_end),
                                 milliseconds(prefill_start, prefill_end),
                                 milliseconds(decode_start, decode_end)};
}

void Lfm2MoeTextGenerationPipeline::run_step(int32_t token_id, std::vector<float>& logits,
                                             bool decode_phase) {
    TensorMap inputs;
    inputs[config_.token_id_name] = Tensor{&token_id, {1}, DType::kInt32};
    int32_t phase = decode_phase ? 1 : 0;
    if (decoder_->has_input("decode_phase"))
        inputs["decode_phase"] = Tensor{&phase, {1, 1}, DType::kInt32};
    state_->prepare_step(inputs);

    // Every persistent state output is device-sized. Never call forward(),
    // which materializes non-external outputs on the host; execute asynchronously
    // and transfer only the one-vocabulary logits row required for sampling.
    decoder_->forward_async(inputs);
    decoder_->sync();
    if (logits_device_ptr_ == nullptr) {
        if (!decoder_->has_output(config_.logits_output_name) ||
            decoder_->tensor_dtype(config_.logits_output_name) != DType::kFloat32) {
            throw std::runtime_error(
                "Lfm2MoeTextGenerationPipeline: FP32 logits output is missing");
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
            throw std::runtime_error(
                "Lfm2MoeTextGenerationPipeline: logits device buffer is missing");
    }
    logits.resize(static_cast<std::size_t>(config_.vocab_size));
    if (cudaMemcpy(logits.data(), logits_device_ptr_, logits.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("Lfm2MoeTextGenerationPipeline: logits D2H copy failed");
    }

    state_->advance();
}

} // namespace trtmc
