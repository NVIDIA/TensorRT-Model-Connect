/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/glm/pipeline.h"

#include "runtime/models/glm/chat_templates.h"
#include "runtime/models/glm/tensor_names.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

struct StepTraceConfig {
    bool enabled{false};
    std::string path;
    int32_t start_position{0};
    int32_t end_position{std::numeric_limits<int32_t>::max()};
};

StepTraceConfig& mutable_step_trace_config() {
    static StepTraceConfig config;
    return config;
}

const StepTraceConfig& step_trace_config() {
    return mutable_step_trace_config();
}

bool step_trace_enabled() {
    return step_trace_config().enabled;
}

void append_logits_trace(const char* phase, int32_t position, int32_t token_id,
                         const std::vector<float>& logits) {
    const auto& config = step_trace_config();
    if (!config.enabled || position < config.start_position || position > config.end_position ||
        logits.empty()) {
        return;
    }

    std::ofstream output(config.path, std::ios::app);
    if (!output)
        throw std::runtime_error("GLM native C++ logits trace cannot open output path");
    output << "{\"phase\":\"" << phase << "\",\"position\":" << position
           << ",\"token_id\":" << token_id << ",\"logits\":[";
    output << std::setprecision(9);
    for (std::size_t index = 0; index < logits.size(); ++index) {
        if (index != 0)
            output << ',';
        output << logits[index];
    }
    output << "]}\n";
}

std::string normalize_generation_mode(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    std::replace(mode.begin(), mode.end(), '-', '_');
    if (mode.empty())
        return "auto";
    if (mode == "autoregressive")
        return "ar";
    return mode;
}

bool contains_boxed_answer(const std::string& text) {
    const std::string marker = "\\boxed{";
    const auto start = text.find(marker);
    return start != std::string::npos && text.find('}', start + marker.size()) != std::string::npos;
}

bool contains_final_answer(const std::string& text) {
    const std::string marker = "Final answer:";
    const auto start = text.find(marker);
    if (start == std::string::npos)
        return false;
    for (std::size_t index = start + marker.size(); index < text.size(); ++index) {
        if (!std::isspace(static_cast<unsigned char>(text[index])))
            return true;
    }
    return false;
}

std::vector<int32_t> encode_prompt(const ITokenizer& tokenizer, const GlmTextGenConfig& config,
                                   const std::string& prompt, const GenerateConfig& generation) {
    std::string effective_prompt = prompt;
    bool templated = false;
    if (generation.use_chat_template && !config.chat_template_format.empty()) {
        effective_prompt = glm_apply_chat_template(config.chat_template_format, prompt,
                                                   generation.enable_thinking);
        templated = true;
    }
    auto ids = tokenizer.encode(effective_prompt);
    if (templated && ids.size() >= 2 && config.id_bos >= 0 && ids[0] == config.id_bos &&
        ids[1] == config.id_bos) {
        ids.erase(ids.begin());
    }
    return ids;
}

void validate_generation_capacity(const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
                                  const GlmKvCache& cache) {
    const auto prompt_tokens = input_ids.size();
    const auto decode_writes =
        max_new_tokens > 0
            ? static_cast<std::size_t>(max_new_tokens - (step_trace_enabled() ? 0 : 1))
            : std::size_t{0};
    const auto capacity = static_cast<std::size_t>(cache.max_length());
    if (prompt_tokens > capacity || decode_writes > capacity - prompt_tokens) {
        throw std::runtime_error(
            "GLM prompt and generated-token cache writes exceed the model context");
    }
}

void gather_present_pointers(TrtModule& prefill, const GlmTextGenConfig& config,
                             std::vector<const void*>& present_k,
                             std::vector<const void*>& present_v) {
    present_k.resize(static_cast<std::size_t>(config.num_layers));
    present_v.resize(static_cast<std::size_t>(config.num_layers));
    for (int32_t layer = 0; layer < config.num_layers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        present_k[index] =
            prefill.device_ptr(glm_expand_layer_name(config.present_k_pattern, layer));
        present_v[index] =
            prefill.device_ptr(glm_expand_layer_name(config.present_v_pattern, layer));
        if (present_k[index] == nullptr || present_v[index] == nullptr) {
            throw std::runtime_error(
                "GLM native prefill engine is missing aliased present outputs");
        }
    }
}

template <typename T>
void require_valid_state(const std::unique_ptr<T>& state, const char* message) {
    if (!state || !state->ok())
        throw std::runtime_error(message);
}

void validate_generation_metadata(const GlmTextGenConfig& config) {
    if (config.vocab_size <= 0 || config.num_layers <= 0 || config.prefill_max_length <= 0) {
        throw std::runtime_error("GlmTextGenerationPipeline: invalid native engine metadata");
    }
}

} // namespace

void apply_text_trace_config_from_registry(const std::string& path, int32_t start_position,
                                           int32_t end_position, int32_t top_k) {
    (void)top_k;
    auto& config = mutable_step_trace_config();
    config.path = path;
    config.enabled = !path.empty();
    config.start_position = start_position;
    config.end_position = end_position;
    if (config.enabled)
        std::ofstream(config.path, std::ios::trunc);
}

GlmTextGenerationPipeline::GlmTextGenerationPipeline(std::unique_ptr<TrtModule> decoder,
                                                     std::unique_ptr<TrtModule> prefill,
                                                     std::unique_ptr<GlmKvCache> state,
                                                     GlmTextGenConfig config, cudaStream_t stream,
                                                     std::shared_ptr<ITokenizer> tokenizer,
                                                     std::string model_id,
                                                     std::unique_ptr<GlmISampler> sampler)
    : decoder_(std::move(decoder)), prefill_(std::move(prefill)), state_(std::move(state)),
      config_(std::move(config)), stream_(stream), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id)), sampler_(std::move(sampler)),
      prefer_gpu_greedy_(config_.prefer_gpu_greedy) {
    require_valid_state(decoder_, "GlmTextGenerationPipeline: invalid decode engine");
    require_valid_state(prefill_, "GlmTextGenerationPipeline: split prefill engine is required");
    require_valid_state(state_, "GlmTextGenerationPipeline: invalid native KV state");
    validate_generation_metadata(config_);
    if (!config_.disable_cuda_graph)
        decoder_->enable_cuda_graph();
}

TextResult GlmTextGenerationPipeline::generate(const std::string& prompt,
                                               const GenerateConfig& config) {
    if (!tokenizer_)
        throw std::runtime_error("GlmTextGenerationPipeline: no tokenizer configured");

    const auto input_ids = encode_prompt(*tokenizer_, config_, prompt, config);
    const int32_t max_new_tokens = config.max_new_tokens > 0 ? config.max_new_tokens : 128;
    const int32_t eos_token_id = config.eos_token_id >= 0 ? config.eos_token_id : config_.id_eos;
    const auto sampling = glm_sampling_params_from_config(config, eos_token_id);

    last_setup_ms_ = 0.0;
    auto timed = generate_from_ids(input_ids, max_new_tokens, sampling, config);
    std::vector<int32_t> generated(timed.token_ids.begin() +
                                       static_cast<std::ptrdiff_t>(input_ids.size()),
                                   timed.token_ids.end());
    auto result = TextResult{tokenizer_->decode(generated), std::move(generated), timed.prefill_ms,
                             timed.decode_ms};
    result.setup_ms = last_setup_ms_;
    return result;
}

GlmTextGenerationPipeline::GenerationResult
GlmTextGenerationPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                        const GenerateConfig& config) {
    const int32_t eos_token_id = config.eos_token_id >= 0 ? config.eos_token_id : config_.id_eos;
    const auto sampling = glm_sampling_params_from_config(config, eos_token_id);
    return GenerationResult{
        generate_from_ids(input_ids, config.max_new_tokens, sampling, config).token_ids};
}

std::unique_ptr<GlmISampler>
GlmTextGenerationPipeline::make_step_sampler(const GlmSamplingParams& params) {
    const bool greedy = params.temperature < 1e-6F || (params.top_k <= 1 && params.top_p >= 1.0F &&
                                                       params.min_p <= 0.0F && params.seed < 0);
    if (!step_trace_enabled() && prefer_gpu_greedy_ && greedy) {
        if (auto sampler = create_glm_gpu_greedy_sampler(stream_))
            return sampler;
    }
    return create_glm_sampler(params);
}

void GlmTextGenerationPipeline::reset_generation_context() {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    state_->reset();
    decoder_->reset_execution_context();
    prefill_->reset_execution_context();
    decoder_bound_ = false;
    device_logits_ = nullptr;
    last_setup_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

GlmTextGenerationPipeline::TimedGenResult GlmTextGenerationPipeline::generate_from_ids(
    const std::vector<int32_t>& input_ids, int32_t max_new_tokens, const GlmSamplingParams& params,
    const GenerateConfig& config) {
    using Clock = std::chrono::steady_clock;
    const std::string mode = normalize_generation_mode(config.text_generation_mode);
    if (mode != "auto" && mode != "ar") {
        throw std::runtime_error("GlmTextGenerationPipeline: unsupported generation mode '" + mode +
                                 "'");
    }
    if (max_new_tokens <= 0 || input_ids.empty())
        return TimedGenResult{input_ids, 0.0, 0.0};

    auto& cache = *state_;
    validate_generation_capacity(input_ids, max_new_tokens, cache);

    GlmISampler* active_sampler = sampler_.get();
    std::unique_ptr<GlmISampler> local_sampler;
    if (active_sampler == nullptr) {
        local_sampler = make_step_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    reset_generation_context();

    std::vector<float> logits;
    const bool gpu_sampling = active_sampler->logits_location() == GlmLogitsLocation::DEVICE;
    const auto prefill_start = Clock::now();
    run_prefill(input_ids, logits, gpu_sampling);
    const auto decode_start = Clock::now();

    std::vector<int32_t> output = input_ids;
    run_decode_loop(active_sampler, params, output, logits, max_new_tokens, gpu_sampling, config,
                    static_cast<int32_t>(input_ids.size()));
    const auto end = Clock::now();

    return TimedGenResult{
        std::move(output),
        std::chrono::duration<double, std::milli>(decode_start - prefill_start).count(),
        std::chrono::duration<double, std::milli>(end - decode_start).count()};
}

void GlmTextGenerationPipeline::run_prefill_chunk(const int32_t* token_ids, int32_t chunk_size,
                                                  GlmKvCache& cache,
                                                  const std::vector<const void*>& present_k,
                                                  const std::vector<const void*>& present_v,
                                                  std::vector<float>& logits,
                                                  bool retain_device_logits) {
    TensorMap inputs;
    inputs[config_.token_id_name] =
        Tensor{const_cast<int32_t*>(token_ids), {static_cast<int64_t>(chunk_size)}, DType::kInt32};
    cache.prepare_step(inputs, chunk_size);

    TensorMap outputs = prefill_->forward(inputs);
    const auto logits_it = outputs.find(config_.logits_output_name);
    if (logits_it == outputs.end())
        throw std::runtime_error("GLM native prefill engine has no logits output");
    const auto& logits_tensor = logits_it->second;
    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    if (static_cast<std::size_t>(logits_tensor.numel()) < vocab)
        throw std::runtime_error("GLM native prefill logits are smaller than vocabulary");

    const auto logits_offset = static_cast<std::size_t>(logits_tensor.numel()) - vocab;
    logits.resize(vocab);
    std::memcpy(logits.data(), static_cast<const float*>(logits_tensor.data) + logits_offset,
                vocab * sizeof(float));
    if (retain_device_logits) {
        const auto* device =
            static_cast<const float*>(prefill_->device_ptr(config_.logits_output_name));
        if (device == nullptr)
            throw std::runtime_error("GLM native prefill logits have no device buffer");
        device_logits_ = device + logits_offset;
    }

    cache.append_prefill_kv(present_k, present_v, chunk_size);
}

void GlmTextGenerationPipeline::run_prefill(const std::vector<int32_t>& input_ids,
                                            std::vector<float>& logits, bool retain_device_logits) {
    // E2E parity tracing intentionally exercises the same production native
    // decode engine one token at a time so it can retain every HF-comparable
    // prompt row. The normal runtime path below remains split/chunked prefill.
    if (step_trace_enabled()) {
        for (const int32_t token_id : input_ids)
            run_step(token_id, logits, "prefill");
        return;
    }

    auto& cache = *state_;
    cache.bind_to(*prefill_);
    if (input_ids.size() > static_cast<std::size_t>(cache.max_length()))
        throw std::runtime_error("GLM prompt exceeds the model context");

    std::vector<const void*> present_k;
    std::vector<const void*> present_v;
    gather_present_pointers(*prefill_, config_, present_k, present_v);

    int32_t chunks = 0;
    const int32_t token_count = static_cast<int32_t>(input_ids.size());
    for (int32_t start = 0; start < token_count;) {
        const int32_t chunk_size = std::min(config_.prefill_max_length, token_count - start);
        run_prefill_chunk(input_ids.data() + start, chunk_size, cache, present_k, present_v, logits,
                          retain_device_logits);
        start += chunk_size;
        ++chunks;
    }
    log_prefill(token_count, chunks);
    prime_decoder_after_prefill(input_ids);
}

void GlmTextGenerationPipeline::prime_decoder_after_prefill(const std::vector<int32_t>& input_ids) {
    if (input_ids.empty() || !decoder_->cuda_graph_active())
        return;

    int32_t token_id = input_ids.back();
    TensorMap inputs;
    inputs[config_.token_id_name] = Tensor{&token_id, {1}, DType::kInt32};
    TrtModule& decoder = bind_decoder();
    state_->prepare_step(inputs);
    decoder.forward_async(inputs);
    decoder.sync();
}

TrtModule& GlmTextGenerationPipeline::bind_decoder() {
    if (!decoder_bound_) {
        state_->bind_to(*decoder_);
        decoder_bound_ = true;
    }
    return *decoder_;
}

void GlmTextGenerationPipeline::run_step(int32_t token_id, std::vector<float>& logits,
                                         const char* trace_phase) {
    const int32_t position = state_->position();
    TensorMap inputs;
    inputs[config_.token_id_name] = Tensor{&token_id, {1}, DType::kInt32};
    TrtModule& decoder = bind_decoder();
    state_->prepare_step(inputs);

    TensorMap outputs = decoder.forward(inputs);
    const auto logits_it = outputs.find(config_.logits_output_name);
    if (logits_it == outputs.end())
        throw std::runtime_error("GLM native decode engine has no logits output");
    const auto& logits_tensor = logits_it->second;
    logits.resize(static_cast<std::size_t>(logits_tensor.numel()));
    std::memcpy(logits.data(), logits_tensor.data, logits.size() * sizeof(float));
    append_logits_trace(trace_phase, position, token_id, logits);
    state_->advance();
}

void GlmTextGenerationPipeline::run_step_device(int32_t token_id) {
    TensorMap inputs;
    inputs[config_.token_id_name] = Tensor{&token_id, {1}, DType::kInt32};
    TrtModule& decoder = bind_decoder();
    state_->prepare_step(inputs);
    decoder.forward_async(inputs);
    decoder.sync();
    device_logits_ = static_cast<const float*>(decoder.device_ptr(config_.logits_output_name));
    if (device_logits_ == nullptr)
        throw std::runtime_error("GLM native decode logits have no device buffer");
    state_->advance();
}

int32_t GlmTextGenerationPipeline::run_decode_loop(
    GlmISampler* sampler, const GlmSamplingParams& params, std::vector<int32_t>& output,
    std::vector<float>& logits, int32_t max_new_tokens, bool gpu_sampling,
    const GenerateConfig& config, int32_t prompt_token_count) {
    const int32_t vocab_size =
        gpu_sampling ? config_.vocab_size : static_cast<int32_t>(logits.size());
    const int32_t stop_interval = std::max(config.stop_check_interval, 1);
    const auto start = std::chrono::steady_clock::now();
    int32_t steps = 0;
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const float* sample_logits = gpu_sampling ? device_logits_ : logits.data();
        const auto result = sampler->sample(sample_logits, vocab_size, params);
        output.push_back(result.token_id);
        ++steps;
        if (should_stop_on_answer(output, prompt_token_count, config, steps, stop_interval,
                                  result.is_eos) ||
            result.is_eos || (!step_trace_enabled() && step + 1 == max_new_tokens)) {
            break;
        }
        if (gpu_sampling)
            run_step_device(result.token_id);
        else
            run_step(result.token_id, logits);
    }
    const auto end = std::chrono::steady_clock::now();
    const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    log_decode_summary(steps, milliseconds);
    return steps;
}

bool GlmTextGenerationPipeline::should_stop_on_answer(const std::vector<int32_t>& output,
                                                      int32_t prompt_token_count,
                                                      const GenerateConfig& config, int32_t steps,
                                                      int32_t stop_interval, bool is_eos) const {
    if (!config.stop_on_boxed_answer || !tokenizer_)
        return false;
    if (steps % stop_interval != 0 && !is_eos)
        return false;
    const std::vector<int32_t> generated(output.begin() + prompt_token_count, output.end());
    const std::string text = tokenizer_->decode(generated);
    return contains_boxed_answer(text) || contains_final_answer(text);
}

void GlmTextGenerationPipeline::log_prefill(int32_t token_count, int32_t chunk_count) const {
    if (!config_.log_runtime_stats)
        return;
    std::cerr << "[trtmc] Batched prefill ("
              << (config_.prefill_log_label.empty() ? "native prefill engine"
                                                    : config_.prefill_log_label)
              << "): " << token_count << " tokens in " << chunk_count << " call"
              << (chunk_count == 1 ? "" : "s") << " (max chunk=" << config_.prefill_max_length
              << ")\n";
}

void GlmTextGenerationPipeline::log_decode_summary(int32_t steps, double milliseconds) const {
    if (steps <= 0 || !config_.log_runtime_stats)
        return;
    const double tokens_per_second = steps * 1000.0 / milliseconds;
    std::cerr << "[trtmc] Decode: " << steps << " tokens, " << milliseconds << " ms, "
              << tokens_per_second << " tok/s"
              << (decoder_->cuda_graph_active() ? " [CUDA Graph ON]" : "") << '\n';
}

int32_t GlmTextGenerationPipeline::argmax(const std::vector<float>& logits) {
    if (logits.empty())
        return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace trtmc
