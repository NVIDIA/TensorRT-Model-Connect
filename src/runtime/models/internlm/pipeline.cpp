/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/internlm/pipeline.h"

#include "runtime/models/internlm/chat_templates.h"
#include "runtime/models/internlm/kv_cache.h"
#include "runtime/models/internlm/tensor_names.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace trtmc {

namespace {

struct StepTraceConfig {
    bool enabled{false};
    std::string path;
    int32_t start_position{0};
    int32_t end_position{std::numeric_limits<int32_t>::max()};
    int32_t top_k{8};
};

// Process-wide step-trace state. Populated once from the resolved
// ConfigBundle by `apply_text_trace_config_from_registry` (below), called
// from the decoder plugin before pipeline construction. Replaces the
// TRTMC_TEXT_STEP_TRACE_* environment variables, which are now deleted.
StepTraceConfig& mutable_step_trace_config() {
    static StepTraceConfig cfg;
    return cfg;
}

const StepTraceConfig& step_trace_config() {
    return mutable_step_trace_config();
}

bool step_trace_enabled() {
    return step_trace_config().enabled;
}

} // namespace

// Called from decoder_plugin::create() with values resolved from
// ctx.runtime_config for the "text_trace" namespace. An empty path keeps
// tracing disabled. When a non-empty path is supplied, this truncates the
// target file so repeated runs don't concatenate. Not re-entrant; the
// caller serializes creation.
void apply_text_trace_config_from_registry(const std::string& path, int32_t start_position,
                                           int32_t end_position, int32_t top_k) {
    StepTraceConfig& cfg = mutable_step_trace_config();
    cfg.path = path;
    cfg.enabled = !path.empty();
    cfg.start_position = start_position;
    cfg.end_position = end_position;
    cfg.top_k = std::max(int32_t{1}, top_k);
    if (cfg.enabled) {
        std::ofstream clear(cfg.path, std::ios::trunc);
    }
}

namespace {

std::vector<int32_t> top_logit_indices(const std::vector<float>& logits, int32_t top_n) {
    std::vector<int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
        order.begin(), order.begin() + top_n, order.end(), [&logits](int32_t lhs, int32_t rhs) {
            if (logits[static_cast<std::size_t>(lhs)] != logits[static_cast<std::size_t>(rhs)]) {
                return logits[static_cast<std::size_t>(lhs)] >
                       logits[static_cast<std::size_t>(rhs)];
            }
            return lhs < rhs;
        });
    return order;
}

void write_step_trace_line(std::ostream& out, const char* phase, int32_t position_before,
                           int32_t token_id, int32_t decoder_idx, int32_t rows_before,
                           int32_t rows_after, const std::vector<float>& logits,
                           const std::vector<int32_t>& order, int32_t top_n) {
    out << std::setprecision(std::numeric_limits<float>::max_digits10) << "{\"phase\":\"" << phase
        << "\",\"position_before\":" << position_before << ",\"token_id\":" << token_id
        << ",\"decoder_idx\":" << decoder_idx << ",\"rows_before\":" << rows_before
        << ",\"rows_after\":" << rows_after << ",\"argmax_token\":" << order.front()
        << ",\"argmax_logit\":" << logits[static_cast<std::size_t>(order.front())]
        << ",\"top_ids\":[";
    for (int32_t i = 0; i < top_n; ++i) {
        if (i > 0)
            out << ',';
        out << order[static_cast<std::size_t>(i)];
    }
    out << "],\"top_logits\":[";
    for (int32_t i = 0; i < top_n; ++i) {
        if (i > 0)
            out << ',';
        out << logits[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
    }
    out << "]}\n";
}

void maybe_append_step_trace(const char* phase, int32_t position_before, int32_t token_id,
                             int32_t decoder_idx, int32_t rows_before, int32_t rows_after,
                             const std::vector<float>& logits) {
    const auto& cfg = step_trace_config();
    if (!cfg.enabled || position_before < cfg.start_position || position_before > cfg.end_position)
        return;
    if (logits.empty())
        return;
    const int32_t top_n = std::min<int32_t>(cfg.top_k, static_cast<int32_t>(logits.size()));
    const auto order = top_logit_indices(logits, top_n);
    std::ofstream out(cfg.path, std::ios::app);
    if (!out)
        return;
    write_step_trace_line(out, phase, position_before, token_id, decoder_idx, rows_before,
                          rows_after, logits, order, top_n);
}

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

std::string normalize_generation_mode(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    std::replace(mode.begin(), mode.end(), '-', '_');
    return mode;
}

} // namespace

InternlmTextGenerationPipeline::InternlmTextGenerationPipeline(
    std::vector<DecoderContext> decoders, std::unique_ptr<InternlmKvCache> state,
    InternlmTextGenConfig config, cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str, std::unique_ptr<InternlmISampler> sampler,
    std::unique_ptr<TrtModule> prefill)
    : decoders_(std::move(decoders)), prefill_(std::move(prefill)), state_(std::move(state)),
      config_(std::move(config)), stream_(stream), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id_str)), sampler_(std::move(sampler)),
      logits_output_name_(config_.logits_output_name) {
    if (decoders_.size() != 1 || !decoders_.front().module || !decoders_.front().module->ok())
        throw std::runtime_error(
            "InternlmTextGenerationPipeline requires exactly one native decode module");
    if (!prefill_ || !prefill_->ok())
        throw std::runtime_error(
            "InternlmTextGenerationPipeline requires a native split prefill module");
    if (!state_ || !state_->ok()) {
        throw std::runtime_error("InternlmTextGenerationPipeline: invalid inference state");
    }

    // CUDA Graphs: capture TRT kernels on first step, replay on subsequent
    // steps. Disabled via --set runtime.disable_cuda_graph=true (replaces
    // the deleted TRTMC_DISABLE_CUDA_GRAPH env var).
    if (!config_.disable_cuda_graph) {
        for (auto& decoder_ctx : decoders_)
            decoder_ctx.module->enable_cuda_graph();
    }

    // GPU-side argmax is only valid for truly greedy decoding. Populated
    // from runtime.prefer_gpu_greedy (replaces the deleted TRTMC_GPU_ARGMAX
    // env var). We record the preference here and instantiate per-call
    // when the requested sampling parameters are actually greedy.
    prefer_gpu_greedy_ = config_.prefer_gpu_greedy;
}

// Encode a prompt, optionally applying a chat template first.
// Deduplicates the leading BOS token that chat templates embed but
// the tokenizer's add_special_tokens may also prepend.
static std::vector<int32_t> encode_prompt(const ITokenizer& tokenizer,
                                          const InternlmTextGenConfig& config,
                                          const std::string& prompt, const GenerateConfig& cfg) {
    std::string effective = prompt;
    bool templated = false;
    if (cfg.use_chat_template && !config.chat_template_format.empty()) {
        effective =
            internlm_apply_chat_template(config.chat_template_format, prompt, cfg.enable_thinking);
        templated = true;
    }
    auto ids = tokenizer.encode(effective);
    if (templated && ids.size() >= 2 && config.id_bos >= 0 && ids[0] == config.id_bos &&
        ids[1] == config.id_bos) {
        ids.erase(ids.begin());
    }
    return ids;
}

TextResult InternlmTextGenerationPipeline::generate(const std::string& prompt,
                                                    const GenerateConfig& cfg) {
    if (!tokenizer_) {
        throw std::runtime_error("InternlmTextGenerationPipeline: no tokenizer configured");
    }

    auto input_ids = encode_prompt(*tokenizer_, config_, prompt, cfg);
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;

    auto sp = internlm_sampling_params_from_config(cfg, eos);
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

InternlmTextGenerationPipeline::GenerationResult
InternlmTextGenerationPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                             const GenerateConfig& cfg) {
    int32_t max_new = cfg.max_new_tokens; // honour exact value (0 = no generation)
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    auto sp = internlm_sampling_params_from_config(cfg, eos);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp, cfg).token_ids};
}

std::unique_ptr<InternlmISampler>
InternlmTextGenerationPipeline::make_step_sampler(const InternlmSamplingParams& params) {
    const bool greedy_params =
        (params.temperature < 1e-6F) ||
        (params.top_k <= 1 && params.top_p >= 1.0F && params.min_p <= 0.0F && params.seed < 0);
    if (!step_trace_enabled() && prefer_gpu_greedy_ && greedy_params) {
        if (auto gpu = create_internlm_gpu_greedy_sampler(stream_))
            return gpu;
    }
    return create_internlm_sampler(params);
}

namespace {
void validate_generation_capacity(const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
                                  const InternlmKvCache* state) {
    const auto capacity = static_cast<std::size_t>(state->max_length());
    if (input_ids.size() > capacity ||
        (max_new_tokens > 0 &&
         static_cast<std::size_t>(max_new_tokens) > capacity - input_ids.size())) {
        throw std::runtime_error(
            "Internlm requested prompt and generation exceed the model's fixed KV cache capacity");
    }
}
} // namespace

void InternlmTextGenerationPipeline::run_prefill_chunk(const int32_t* token_ids, int32_t chunk_size,
                                                       std::vector<float>& logits,
                                                       bool retain_device_logits) {
    TensorMap inputs;
    Tensor token_tensor;
    token_tensor.data = const_cast<int32_t*>(token_ids);
    token_tensor.shape = {static_cast<int64_t>(chunk_size)};
    token_tensor.dtype = DType::kInt32;
    inputs[config_.token_id_name] = token_tensor;
    state_->prepare_step(inputs, chunk_size);

    TensorMap outputs = prefill_->forward(inputs);
    auto logits_it = outputs.find(config_.logits_output_name);
    if (logits_it == outputs.end()) {
        throw std::runtime_error(
            "InternlmTextGenerationPipeline: prefill module has no logits output");
    }

    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    const auto& logits_tensor = logits_it->second;
    if (static_cast<std::size_t>(logits_tensor.numel()) < vocab) {
        throw std::runtime_error(
            "InternlmTextGenerationPipeline: prefill logits are smaller than vocabulary");
    }
    logits.resize(vocab);
    const auto logits_offset = static_cast<std::size_t>(logits_tensor.numel()) - vocab;
    std::memcpy(logits.data(), static_cast<const float*>(logits_tensor.data) + logits_offset,
                vocab * sizeof(float));

    if (retain_device_logits) {
        const auto* device_logits =
            static_cast<const float*>(prefill_->device_ptr(config_.logits_output_name));
        if (device_logits == nullptr) {
            throw std::runtime_error(
                "InternlmTextGenerationPipeline: prefill logits have no device buffer");
        }
        d_logits_ptr_ = device_logits + logits_offset;
    }
    state_->advance(chunk_size);
}

void InternlmTextGenerationPipeline::log_batched_prefill(int32_t token_count, int32_t chunk_count,
                                                         int32_t chunk_limit) const {
    if (!config_.log_runtime_stats)
        return;

    std::cerr << "[trtmc] Batched prefill (";
    std::cerr << (config_.prefill_log_label.empty() ? "prefill engine" : config_.prefill_log_label);
    std::cerr << "): " << token_count << " tokens in " << chunk_count << " call";
    if (chunk_count != 1)
        std::cerr << 's';
    std::cerr << " (max chunk=" << chunk_limit << ")\n";
}

void InternlmTextGenerationPipeline::run_prefill_batched(const std::vector<int32_t>& input_ids,
                                                         std::vector<float>& logits,
                                                         bool retain_device_logits) {
    const auto sq = static_cast<int32_t>(input_ids.size());
    if (sq <= 0)
        throw std::invalid_argument("Internlm native prefill requires at least one token");
    if (config_.prefill_max_length <= 0)
        throw std::runtime_error("Internlm native prefill engine has no valid profile capacity");

    state_->bind_cache_inputs(*prefill_);
    if (sq > state_->max_length()) {
        throw std::runtime_error("Internlm sequence exceeds the model's fixed KV cache capacity");
    }

    int32_t chunk_count = 0;
    for (int32_t start = 0; start < sq;) {
        const int32_t chunk_size = std::min(config_.prefill_max_length, sq - start);
        run_prefill_chunk(input_ids.data() + start, chunk_size, logits, retain_device_logits);
        ++chunk_count;
        start += chunk_size;
    }

    log_batched_prefill(sq, chunk_count, config_.prefill_max_length);
}

void InternlmTextGenerationPipeline::prime_decoder_after_batched_prefill(
    const std::vector<int32_t>& input_ids) {
    if (input_ids.empty())
        return;

    TrtModule& decoder = bind_decoder_for_step();
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

void InternlmTextGenerationPipeline::run_prefill(const std::vector<int32_t>& input_ids,
                                                 std::vector<float>& logits, bool gpu_sampling) {
    if (step_trace_enabled()) {
        if (gpu_sampling) {
            throw std::runtime_error(
                "Internlm text trace requires host logits and cannot use GPU-only sampling");
        }
        for (const int32_t token_id : input_ids)
            run_step(token_id, logits, "prefill");
        return;
    }

    run_prefill_batched(input_ids, logits, gpu_sampling);
    prime_decoder_after_batched_prefill(input_ids);
}

std::string
InternlmTextGenerationPipeline::resolve_generation_mode(const GenerateConfig& cfg) const {
    std::string mode = normalize_generation_mode(cfg.text_generation_mode);
    if (mode.empty() || mode == "autoregressive")
        mode = mode.empty() ? "auto" : "ar";
    if (mode != "auto" && mode != "ar") {
        throw std::runtime_error(
            "InternLM native KV runtime supports autoregressive generation only");
    }
    return mode;
}

void InternlmTextGenerationPipeline::reset_generation_context() {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    state_->reset();
    d_logits_ptr_ = nullptr;
    state_bound_ = false;
    for (auto& decoder_ctx : decoders_)
        decoder_ctx.module->reset_execution_context();
    if (prefill_)
        prefill_->reset_execution_context();
    last_setup_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

InternlmTextGenerationPipeline::TimedGenResult InternlmTextGenerationPipeline::generate_from_ids(
    const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
    const InternlmSamplingParams& params, const GenerateConfig& cfg) {
    using Clock = std::chrono::steady_clock;
    resolve_generation_mode(cfg);
    if (max_new_tokens == 0 || input_ids.empty())
        return TimedGenResult{input_ids, 0.0, 0.0};
    validate_generation_capacity(input_ids, max_new_tokens, state_.get());

    InternlmISampler* active_sampler = sampler_.get();
    std::unique_ptr<InternlmISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = make_step_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    reset_generation_context();

    std::vector<float> logits;
    const bool gpu_sampling = (active_sampler->logits_location() == InternlmLogitsLocation::DEVICE);
    const auto t0 = Clock::now();
    run_prefill(input_ids, logits, gpu_sampling);
    const auto t1 = Clock::now();

    std::vector<int32_t> output = input_ids;
    run_decode_loop(active_sampler, params, output, logits, max_new_tokens, gpu_sampling, cfg,
                    static_cast<int32_t>(input_ids.size()));
    const auto t2 = Clock::now();

    const double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    return TimedGenResult{std::move(output), prefill_ms, decode_ms};
}

bool InternlmTextGenerationPipeline::should_stop_on_answer(const std::vector<int32_t>& output,
                                                           int32_t prompt_token_count,
                                                           const GenerateConfig& cfg, int32_t steps,
                                                           int32_t stop_interval,
                                                           bool is_eos) const {
    if (!cfg.stop_on_boxed_answer || !tokenizer_)
        return false;
    if ((steps % stop_interval) != 0 && !is_eos)
        return false;
    std::vector<int32_t> new_tokens(output.begin() + prompt_token_count, output.end());
    const std::string decoded = tokenizer_->decode(new_tokens);
    return contains_boxed_answer(decoded) || contains_final_answer(decoded);
}

void InternlmTextGenerationPipeline::log_decode_summary(int32_t steps, double ms) const {
    if (steps <= 0 || !config_.log_runtime_stats)
        return;
    const double tps = steps * 1000.0 / ms;
    const bool cuda_graph_on = decoders_.front().module->cuda_graph_active();
    std::cerr << "[trtmc] Decode: " << steps << " tokens, " << ms << " ms, " << tps << " tok/s"
              << (cuda_graph_on ? " [CUDA Graph ON]" : "") << '\n';
}

int32_t InternlmTextGenerationPipeline::run_decode_loop(
    InternlmISampler* sampler, const InternlmSamplingParams& params, std::vector<int32_t>& output,
    std::vector<float>& logits, int32_t max_new_tokens, bool gpu_sampling,
    const GenerateConfig& cfg, int32_t prompt_token_count) {
    const int32_t vocab_size =
        gpu_sampling ? config_.vocab_size : static_cast<int32_t>(logits.size());
    const int32_t stop_interval = std::max(cfg.stop_check_interval, 1);
    const auto decode_start = std::chrono::steady_clock::now();
    int32_t steps = 0;
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const float* sample_ptr = gpu_sampling ? d_logits_ptr_ : logits.data();
        const InternlmSampleResult result = sampler->sample(sample_ptr, vocab_size, params);
        output.push_back(result.token_id);
        ++steps;
        if (!step_trace_enabled()) {
            if (should_stop_on_answer(output, prompt_token_count, cfg, steps, stop_interval,
                                      result.is_eos))
                break;
            if (result.is_eos)
                break;
        }
        if (gpu_sampling)
            run_step_device(result.token_id);
        else
            run_step(result.token_id, logits, "decode");
    }
    const auto decode_end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    log_decode_summary(steps, ms);
    return steps;
}

TrtModule& InternlmTextGenerationPipeline::bind_decoder_for_step() {
    if (!state_bound_) {
        state_->bind_to(*decoders_.front().module);
        state_bound_ = true;
    }
    return *decoders_.front().module;
}

void InternlmTextGenerationPipeline::run_step(int32_t token_id, std::vector<float>& logits,
                                              const char* phase) {
    TensorMap inputs;
    const int32_t position_before = state_->position();
    const int32_t cache_capacity = state_->max_length();

    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;
    inputs[config_.token_id_name] = token_tensor;

    TrtModule& decoder = bind_decoder_for_step();
    state_->prepare_step(inputs);

    TensorMap outputs = decoder.forward(inputs);

    auto it = outputs.find(logits_output_name_);
    if (it == outputs.end()) {
        throw std::runtime_error("InternlmTextGenerationPipeline: no '" + logits_output_name_ +
                                 "' output");
    }

    const auto& logits_tensor = it->second;
    auto num_logits = logits_tensor.numel();
    logits.resize(static_cast<std::size_t>(num_logits));
    std::memcpy(logits.data(), logits_tensor.data, num_logits * sizeof(float));

    state_->advance();
    maybe_append_step_trace(phase, position_before, token_id, 0, cache_capacity, cache_capacity,
                            logits);
}

void InternlmTextGenerationPipeline::run_step_device(int32_t token_id) {
    TensorMap inputs;

    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;
    inputs[config_.token_id_name] = token_tensor;

    TrtModule& decoder = bind_decoder_for_step();
    state_->prepare_step(inputs);

    // Use forward_async + sync instead of forward() to skip the D2H output copy.
    // The GPU argmax kernel reads logits directly from the device buffer.
    decoder.forward_async(inputs);
    decoder.sync();

    // Get device pointer to logits output buffer (still on GPU).
    d_logits_ptr_ = static_cast<const float*>(decoder.device_ptr(logits_output_name_));

    state_->advance();
}

int32_t InternlmTextGenerationPipeline::argmax(const std::vector<float>& logits) {
    if (logits.empty())
        return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace trtmc
