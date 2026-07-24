/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/llama/pipeline.h"

#include "runtime/backend/runtime_memory_backend.h"
#include "runtime/domains/text/dynamic_memory/kv_cache_budget.h"
#include "runtime/models/llama/chat_templates.h"
#include "runtime/models/llama/kv_cache.h"
#include "runtime/models/llama/tensor_names.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

void write_step_trace_line(std::ostream& out, int32_t position_before, int32_t token_id,
                           int32_t decoder_idx, int32_t rows_before, int32_t rows_after,
                           const std::vector<float>& logits, const std::vector<int32_t>& order,
                           int32_t top_n) {
    out << "{\"position_before\":" << position_before << ",\"token_id\":" << token_id
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

void maybe_append_step_trace(int32_t position_before, int32_t token_id, int32_t decoder_idx,
                             int32_t rows_before, int32_t rows_after,
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
    write_step_trace_line(out, position_before, token_id, decoder_idx, rows_before, rows_after,
                          logits, order, top_n);
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

std::vector<LlamaTextGenerationPipeline::DecoderContext>
single_decoder_context(std::unique_ptr<TrtModule> decoder) {
    std::vector<LlamaTextGenerationPipeline::DecoderContext> decoders;
    decoders.push_back(LlamaTextGenerationPipeline::DecoderContext{0, std::move(decoder)});
    return decoders;
}

std::string normalize_generation_mode(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    std::replace(mode.begin(), mode.end(), '-', '_');
    return mode;
}

bool greedy_text_diffusion_params(const LlamaSamplingParams& params) {
    return params.seed < 0 &&
           (params.temperature <= 1e-6F ||
            (params.top_k <= 1 && params.top_p >= 1.0F - 1e-6F && params.min_p <= 1e-6F));
}

struct TokenConfidence {
    int32_t pos{0};
    int32_t token_id{0};
    float confidence{0.0F};
};

TokenConfidence argmax_with_confidence(const float* logits, int32_t vocab, int32_t pos) {
    TokenConfidence out;
    out.pos = pos;
    if (logits == nullptr || vocab <= 0)
        return out;
    int32_t best = 0;
    float max_logit = logits[0];
    for (int32_t i = 1; i < vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            best = i;
        }
    }
    double denom = 0.0;
    for (int32_t i = 0; i < vocab; ++i)
        denom += std::exp(static_cast<double>(logits[i] - max_logit));
    out.token_id = best;
    out.confidence = denom > 0.0 ? static_cast<float>(1.0 / denom) : 0.0F;
    return out;
}

std::vector<int32_t> transfer_quota_schedule(int32_t masked, int32_t steps) {
    steps = std::max(steps, 1);
    std::vector<int32_t> quota(static_cast<std::size_t>(steps), 0);
    const int32_t base = masked / steps;
    const int32_t rem = masked % steps;
    for (int32_t i = 0; i < steps; ++i)
        quota[static_cast<std::size_t>(i)] = base + (i < rem ? 1 : 0);
    return quota;
}

std::vector<TokenConfidence> masked_predictions(const std::vector<float>& logits,
                                                const std::vector<int32_t>& block,
                                                int32_t mask_token_id, int32_t vocab_size) {
    std::vector<TokenConfidence> preds;
    if (vocab_size <= 0)
        return preds;
    const auto rows = static_cast<int32_t>(logits.size() / static_cast<std::size_t>(vocab_size));
    const int32_t usable = std::min<int32_t>(rows, static_cast<int32_t>(block.size()));
    preds.reserve(static_cast<std::size_t>(usable));
    for (int32_t i = 0; i < usable; ++i) {
        if (block[static_cast<std::size_t>(i)] != mask_token_id)
            continue;
        preds.push_back(argmax_with_confidence(
            logits.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(vocab_size),
            vocab_size, i));
    }
    std::sort(preds.begin(), preds.end(),
              [](const TokenConfidence& lhs, const TokenConfidence& rhs) {
                  if (lhs.confidence != rhs.confidence)
                      return lhs.confidence > rhs.confidence;
                  return lhs.pos < rhs.pos;
              });
    return preds;
}

void apply_diffusion_transfer(std::vector<int32_t>& block,
                              const std::vector<TokenConfidence>& preds, int32_t quota,
                              bool use_threshold, float threshold) {
    if (preds.empty())
        return;
    if (use_threshold) {
        block[static_cast<std::size_t>(preds.front().pos)] = preds.front().token_id;
        for (std::size_t i = 1; i < preds.size(); ++i) {
            if (preds[i].confidence >= threshold)
                block[static_cast<std::size_t>(preds[i].pos)] = preds[i].token_id;
        }
        return;
    }
    quota = std::max(0, std::min<int32_t>(quota, static_cast<int32_t>(preds.size())));
    for (int32_t i = 0; i < quota; ++i)
        block[static_cast<std::size_t>(preds[static_cast<std::size_t>(i)].pos)] =
            preds[static_cast<std::size_t>(i)].token_id;
}

void apply_linear_spec_transfer(std::vector<int32_t>& block,
                                const std::vector<TokenConfidence>& preds, bool threshold_enabled,
                                float threshold) {
    if (preds.empty())
        return;
    if (!threshold_enabled) {
        for (const auto& pred : preds)
            block[static_cast<std::size_t>(pred.pos)] = pred.token_id;
        return;
    }

    bool changed = false;
    for (const auto& pred : preds) {
        if (pred.confidence >= threshold) {
            block[static_cast<std::size_t>(pred.pos)] = pred.token_id;
            changed = true;
        }
    }
    if (!changed)
        block[static_cast<std::size_t>(preds.front().pos)] = preds.front().token_id;
}

bool has_mask_token(const std::vector<int32_t>& block, int32_t mask_token_id) {
    return std::find(block.begin(), block.end(), mask_token_id) != block.end();
}

} // namespace

LlamaTextGenerationPipeline::LlamaTextGenerationPipeline(
    std::unique_ptr<TrtModule> decoder, std::unique_ptr<LlamaInferenceState> state,
    LlamaTextGenConfig config, cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str, std::unique_ptr<LlamaISampler> sampler,
    std::shared_ptr<void> distributed_owner)
    : LlamaTextGenerationPipeline(single_decoder_context(std::move(decoder)), std::move(state),
                                  std::move(config), stream, std::move(tokenizer),
                                  std::move(model_id_str), std::move(sampler),
                                  /*prefill=*/nullptr, /*linear_spec_lora_prefill=*/nullptr,
                                  std::move(distributed_owner)) {}

LlamaTextGenerationPipeline::LlamaTextGenerationPipeline(
    std::vector<DecoderContext> decoders, std::unique_ptr<LlamaInferenceState> state,
    LlamaTextGenConfig config, cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str, std::unique_ptr<LlamaISampler> sampler,
    std::unique_ptr<TrtModule> prefill, std::unique_ptr<TrtModule> linear_spec_lora_prefill,
    std::shared_ptr<void> distributed_owner)
    : distributed_owner_(std::move(distributed_owner)), decoders_(std::move(decoders)),
      prefill_(std::move(prefill)), linear_spec_lora_prefill_(std::move(linear_spec_lora_prefill)),
      state_(std::move(state)), config_(std::move(config)), stream_(stream),
      tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)),
      sampler_(std::move(sampler)), logits_output_name_(config_.logits_output_name) {
    if (decoders_.empty()) {
        throw std::runtime_error("LlamaTextGenerationPipeline: no decoder modules");
    }
    for (const auto& decoder_ctx : decoders_) {
        if (!decoder_ctx.module || !decoder_ctx.module->ok()) {
            throw std::runtime_error("LlamaTextGenerationPipeline: invalid decoder module");
        }
    }
    if (!state_ || !state_->ok()) {
        throw std::runtime_error("LlamaTextGenerationPipeline: invalid inference state");
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
                                          const LlamaTextGenConfig& config,
                                          const std::string& prompt, const GenerateConfig& cfg) {
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
                                                 const GenerateConfig& cfg) {
    if (!tokenizer_) {
        throw std::runtime_error("LlamaTextGenerationPipeline: no tokenizer configured");
    }

    auto input_ids = encode_prompt(*tokenizer_, config_, prompt, cfg);
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;

    auto sp = llama_sampling_params_from_config(cfg, eos);
    last_setup_ms_ = 0.0;
    auto timed = generate_from_ids(input_ids, max_new, sp, cfg);
    state_->finalize_runtime_memory();
    state_->sample_runtime_memory_high_water();

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
                                          const GenerateConfig& cfg) {
    int32_t max_new = cfg.max_new_tokens; // honour exact value (0 = no generation)
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    auto sp = llama_sampling_params_from_config(cfg, eos);
    auto timed = generate_from_ids(input_ids, max_new, sp, cfg);
    state_->finalize_runtime_memory();
    state_->sample_runtime_memory_high_water();
    return GenerationResult{std::move(timed.token_ids)};
}

RuntimeMemoryQualificationResultV1 LlamaTextGenerationPipeline::qualify_runtime_memory(
    const RuntimeMemoryQualificationRequestV1& request) {
    if (!state_->runtime_owned_kv()) {
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: qualification requires a runtime-memory bundle");
    }
    if (request.input_ids.empty()) {
        throw RuntimeMemoryQualificationAdmissionError(
            "LlamaTextGenerationPipeline: qualification input_ids must not be empty");
    }
    for (std::size_t index = 0; index < request.input_ids.size(); ++index) {
        const auto token = request.input_ids[index];
        if (token < 0 || token >= config_.vocab_size) {
            throw RuntimeMemoryQualificationAdmissionError(
                "LlamaTextGenerationPipeline: token ID at index " + std::to_string(index) +
                " is outside [0, " + std::to_string(config_.vocab_size) + ")");
        }
    }

    const int32_t logical_limit =
        config_.max_sequence_length > 0 ? config_.max_sequence_length : state_->max_length();
    try {
        validate_sequence_admission_with_runtime_memory(
            request.input_ids.size(), request.max_new_tokens, logical_limit,
            config_.runtime_sequence_admission, "LlamaTextGenerationPipeline");
    } catch (const std::exception& error) {
        throw RuntimeMemoryQualificationAdmissionError(error.what());
    }

    reset_generation_context();
    state_->set_prompt_length(static_cast<int32_t>(request.input_ids.size()));

    RuntimeMemoryQualificationResultV1 result;
    result.prompt_tokens = request.input_ids.size();
    result.runtime_kv_capacity_tokens = state_->runtime_kv_capacity_tokens();
    result.effective_request_limit = static_cast<std::uint64_t>(logical_limit);
    result.prefill_chunk_limit = static_cast<std::uint32_t>(state_->prefill_chunk_limit());

    std::vector<float> logits;
    active_qualification_ = &result;
    qualification_invocation_index_ = 0;
    try {
        run_prefill(request.input_ids, logits, /*gpu_sampling=*/false);
        if (logits.size() != static_cast<std::size_t>(config_.vocab_size)) {
            throw std::runtime_error(
                "LlamaTextGenerationPipeline: qualification prefill returned an unexpected "
                "logit count");
        }
        result.step_logits.push_back(logits);

        result.selected_token_ids.reserve(static_cast<std::size_t>(request.max_new_tokens));
        result.step_logits.reserve(static_cast<std::size_t>(request.max_new_tokens) + 1U);
        for (int32_t step = 0; step < request.max_new_tokens; ++step) {
            const int32_t token = argmax(logits);
            result.selected_token_ids.push_back(token);
            run_step(token, logits);
            if (logits.size() != static_cast<std::size_t>(config_.vocab_size)) {
                throw std::runtime_error(
                    "LlamaTextGenerationPipeline: qualification decode returned an unexpected "
                    "logit count");
            }
            result.step_logits.push_back(logits);
            ++result.decode_launches;
        }
    } catch (...) {
        active_qualification_ = nullptr;
        throw;
    }
    active_qualification_ = nullptr;

    result.prefill_launches = last_prefill_launches_;
    result.final_kv_position = static_cast<std::uint64_t>(state_->position());
    state_->finalize_runtime_memory();
    state_->sample_runtime_memory_high_water();
    result.runtime_memory_receipt_json = state_->runtime_memory_receipt_json();
    finalize_runtime_memory_invocation_traces(result);
    return result;
}

std::unique_ptr<LlamaISampler>
LlamaTextGenerationPipeline::make_step_sampler(const LlamaSamplingParams& params) {
    const bool greedy_params =
        (params.temperature < 1e-6F) ||
        (params.top_k <= 1 && params.top_p >= 1.0F && params.min_p <= 0.0F && params.seed < 0);
    if (prefer_gpu_greedy_ && greedy_params) {
        if (auto gpu = create_llama_gpu_greedy_sampler(stream_))
            return gpu;
    }
    return create_llama_sampler(params);
}

// Helper: gather per-layer present_k/present_v device pointers from the
// prefill TrtModule. Returns false if any layer's tensor is missing — in
// that case the caller falls back to the per-token decode loop.
namespace {
bool gather_prefill_kv_pointers(TrtModule& prefill, const LlamaTextGenConfig& cfg,
                                std::vector<const void*>& pk, std::vector<const void*>& pv) {
    pk.resize(static_cast<std::size_t>(cfg.num_layers));
    pv.resize(static_cast<std::size_t>(cfg.num_layers));
    for (int32_t i = 0; i < cfg.num_layers; ++i) {
        const auto li = static_cast<std::size_t>(i);
        pk[li] = prefill.device_ptr(llama_expand_layer_name(cfg.present_k_pattern, i));
        pv[li] = prefill.device_ptr(llama_expand_layer_name(cfg.present_v_pattern, i));
        if (pk[li] == nullptr || pv[li] == nullptr)
            return false;
    }
    return true;
}

bool batched_prefill_supported(const TrtModule* prefill, const LlamaTextGenConfig& cfg, int32_t sq,
                               LlamaInferenceState* state) {
    if (prefill == nullptr || sq <= 0)
        return false;
    if (cfg.prefill_max_length > 0 && sq > cfg.prefill_max_length)
        return false;
    if (cfg.num_layers <= 0 || cfg.vocab_size <= 0)
        return false;
    return dynamic_cast<LlamaKvCache*>(state) != nullptr;
}

TensorMap forward_selected_outputs(TrtModule& module, const TensorMap& inputs,
                                   const std::vector<std::string>& names) {
    if (auto* runtime_module = dynamic_cast<IRuntimeMemoryModuleV1*>(&module))
        return runtime_module->forward_selected(inputs, names);
    return module.forward(inputs);
}
} // namespace

bool LlamaTextGenerationPipeline::run_prefill_batched(const std::vector<int32_t>& input_ids,
                                                      std::vector<float>& logits) {
    const auto sq = static_cast<int32_t>(input_ids.size());
    if (!batched_prefill_supported(prefill_.get(), config_, sq, state_.get()))
        return false;
    auto* kv = static_cast<LlamaKvCache*>(state_.get());

    // The prefill module shares the same external KV cache buffers as the
    // decode module(s), so we rebind the cache_k/cache_v inputs onto the
    // prefill execution context before running.
    kv->bind_cache_inputs(*prefill_);
    state_bound_ = false;

    TensorMap inputs;
    Tensor tok_t;
    tok_t.data = const_cast<int32_t*>(input_ids.data());
    tok_t.shape = {static_cast<int64_t>(sq)};
    tok_t.dtype = DType::kInt32;
    inputs[config_.token_id_name] = tok_t;
    state_->prepare_step(inputs, sq);

    TensorMap outputs = forward_selected_outputs(*prefill_, inputs, {config_.logits_output_name});
    auto logits_it = outputs.find(config_.logits_output_name);
    if (logits_it == outputs.end())
        return false;

    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    const auto& lt = logits_it->second;
    if (static_cast<std::size_t>(lt.numel()) < vocab)
        return false;
    logits.resize(vocab);
    const auto offset = static_cast<std::size_t>(lt.numel()) - vocab;
    std::memcpy(logits.data(), static_cast<const float*>(lt.data) + offset, vocab * sizeof(float));

    std::vector<const void*> pk, pv;
    if (!gather_prefill_kv_pointers(*prefill_, config_, pk, pv))
        return false;
    kv->write_prefill_kv(pk, pv, sq);
    if (config_.log_runtime_stats) {
        std::cerr << "[trtmc] Batched prefill (";
        if (!config_.prefill_log_label.empty()) {
            std::cerr << config_.prefill_log_label;
        } else {
            std::cerr << "profile " << config_.prefill_profile_index;
        }
        std::cerr << "): " << sq << " tokens in one call\n";
    }
    return true;
}

bool LlamaTextGenerationPipeline::run_prefill_runtime_chunks(const std::vector<int32_t>& input_ids,
                                                             std::vector<float>& logits,
                                                             bool gpu_sampling) {
    if (!state_->runtime_owned_kv())
        return false;
    if (prefill_ == nullptr)
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: runtime-memory bundle is missing its prefill role");

    const int32_t chunk_limit = state_->prefill_chunk_limit();
    if (chunk_limit <= 0)
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: invalid runtime prefill chunk limit");
    if (config_.prefill_max_length > 0 && chunk_limit > config_.prefill_max_length) {
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: runtime prefill chunk exceeds engine profile");
    }

    state_->bind_to(*prefill_);
    state_bound_ = false;
    std::size_t offset = 0;
    int32_t launches = 0;
    while (offset < input_ids.size()) {
        const auto remaining = input_ids.size() - offset;
        const int32_t sq = static_cast<int32_t>(std::min<std::size_t>(remaining, chunk_limit));
        const bool last = offset + static_cast<std::size_t>(sq) == input_ids.size();
        const auto history_tokens = static_cast<std::uint64_t>(state_->position());
        const auto active_tokens = history_tokens + static_cast<std::uint64_t>(sq);
        const auto bound_tokens = qualification_bound_tokens(history_tokens);
        const auto transfer_before = qualification_transfer_snapshot(*prefill_);
        const auto commit_before = qualification_commit_snapshot();

        TensorMap inputs;
        Tensor tokens;
        tokens.data = const_cast<int32_t*>(input_ids.data() + offset);
        tokens.shape = {static_cast<int64_t>(sq)};
        tokens.dtype = DType::kInt32;
        inputs[config_.token_id_name] = tokens;
        state_->prepare_step(inputs, sq);

        if (last && !gpu_sampling) {
            const auto outputs =
                forward_selected_outputs(*prefill_, inputs, {config_.logits_output_name});
            const auto found = outputs.find(config_.logits_output_name);
            if (found == outputs.end())
                throw std::runtime_error(
                    "LlamaTextGenerationPipeline: prefill role did not return logits");
            const auto vocab = static_cast<std::size_t>(config_.vocab_size);
            if (static_cast<std::size_t>(found->second.numel()) < vocab)
                throw std::runtime_error(
                    "LlamaTextGenerationPipeline: prefill logits are truncated");
            logits.resize(vocab);
            const auto row_offset = static_cast<std::size_t>(found->second.numel()) - vocab;
            std::memcpy(logits.data(), static_cast<const float*>(found->second.data) + row_offset,
                        vocab * sizeof(float));
        } else {
            prefill_->forward_async(inputs);
            prefill_->sync();
            if (last) {
                d_logits_ptr_ =
                    static_cast<const float*>(prefill_->device_ptr(config_.logits_output_name));
            }
        }

        state_->advance(sq);
        append_qualification_invocation("prefill", "engine_plan:prefill", *prefill_, offset,
                                        offset + static_cast<std::size_t>(sq), history_tokens,
                                        active_tokens, bound_tokens, transfer_before,
                                        commit_before);
        offset += static_cast<std::size_t>(sq);
        ++launches;
    }

    const auto expected =
        static_cast<int32_t>((input_ids.size() + static_cast<std::size_t>(chunk_limit) - 1) /
                             static_cast<std::size_t>(chunk_limit));
    if (launches != expected)
        throw std::logic_error("LlamaTextGenerationPipeline: incorrect prefill launch count");
    last_prefill_launches_ = static_cast<std::uint32_t>(launches);
    if (config_.log_runtime_stats) {
        std::cerr << "[trtmc] Chunked prefill: tokens=" << input_ids.size()
                  << " chunk_limit=" << chunk_limit << " launches=" << launches << '\n';
    }
    state_->mark_prefill_complete();
    return true;
}

void LlamaTextGenerationPipeline::prime_decoder_after_batched_prefill(
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

void LlamaTextGenerationPipeline::run_prefill(const std::vector<int32_t>& input_ids,
                                              std::vector<float>& logits, bool gpu_sampling) {
    if (!config_.kv_cache_compaction &&
        input_ids.size() > static_cast<std::size_t>(state_->max_length())) {
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: prompt length exceeds runtime KV cache capacity");
    }
    if (run_prefill_runtime_chunks(input_ids, logits, gpu_sampling))
        return;
    // Fast path: batched prefill engine writes K/V for the whole prompt in
    // one forward and returns last-token logits on host.
    if (!gpu_sampling && run_prefill_batched(input_ids, logits)) {
        prime_decoder_after_batched_prefill(input_ids);
        state_->mark_prefill_complete();
        return;
    }
    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i) {
        if (gpu_sampling)
            run_step_device(input_ids[i]);
        else
            run_step(input_ids[i], logits);
    }
    const int32_t last_token = input_ids.back();
    if (gpu_sampling)
        run_step_device(last_token);
    else
        run_step(last_token, logits);
    state_->mark_prefill_complete();
}

TrtModule& LlamaTextGenerationPipeline::require_block_prefill(int32_t sq,
                                                              TrtModule* prefill_override) {
    TrtModule* prefill = prefill_override != nullptr ? prefill_override : prefill_.get();
    if (prefill == nullptr)
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: block generation requires prefill module");
    if (sq <= 0)
        throw std::runtime_error("LlamaTextGenerationPipeline: empty block");
    if (config_.prefill_max_length > 0 && sq > config_.prefill_max_length) {
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: block length exceeds prefill profile");
    }
    return *prefill;
}

LlamaKvCache& LlamaTextGenerationPipeline::require_block_kv_cache() {
    auto* kv = dynamic_cast<LlamaKvCache*>(state_.get());
    if (kv == nullptr)
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: block generation requires LlamaKvCache");
    return *kv;
}

void LlamaTextGenerationPipeline::copy_block_logits(const TensorMap& outputs,
                                                    std::vector<float>& logits) const {
    auto logits_it = outputs.find(config_.logits_output_name);
    if (logits_it == outputs.end())
        throw std::runtime_error("LlamaTextGenerationPipeline: prefill module has no '" +
                                 config_.logits_output_name + "' output");

    const auto& lt = logits_it->second;
    const auto num_logits = static_cast<std::size_t>(lt.numel());
    logits.resize(num_logits);
    std::memcpy(logits.data(), lt.data, num_logits * sizeof(float));
}

void LlamaTextGenerationPipeline::append_prefill_kv(LlamaKvCache& kv, TrtModule& prefill,
                                                    int32_t sq) {
    std::vector<const void*> pk, pv;
    if (!gather_prefill_kv_pointers(prefill, config_, pk, pv)) {
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: prefill module is missing present_k/present_v outputs");
    }
    kv.append_prefill_kv(pk, pv, sq);
}

void LlamaTextGenerationPipeline::run_prefill_block(const std::vector<int32_t>& input_ids,
                                                    bool bidirectional, bool append_kv,
                                                    std::vector<float>& logits,
                                                    TrtModule* prefill_override) {
    const auto sq = static_cast<int32_t>(input_ids.size());
    TrtModule& prefill = require_block_prefill(sq, prefill_override);
    LlamaKvCache& kv = require_block_kv_cache();
    if (append_kv && kv.position() + sq > kv.max_length()) {
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: prefill block exceeds runtime KV cache capacity");
    }

    kv.bind_cache_inputs(prefill);
    state_bound_ = false;

    TensorMap inputs;
    Tensor tok_t;
    tok_t.data = const_cast<int32_t*>(input_ids.data());
    tok_t.shape = {static_cast<int64_t>(sq)};
    tok_t.dtype = DType::kInt32;
    inputs[config_.token_id_name] = tok_t;
    if (bidirectional)
        kv.prepare_bidirectional_step(inputs, sq);
    else
        kv.prepare_step(inputs, sq);

    copy_block_logits(forward_selected_outputs(prefill, inputs, {config_.logits_output_name}),
                      logits);
    if (append_kv)
        append_prefill_kv(kv, prefill, sq);
}

std::string LlamaTextGenerationPipeline::resolve_generation_mode(const GenerateConfig& cfg) const {
    std::string mode = normalize_generation_mode(cfg.text_generation_mode);
    if (mode.empty())
        mode = "auto";
    if (mode == "auto" && config_.supports_text_diffusion)
        mode = "diffusion";
    if (mode == "autoregressive")
        mode = "ar";
    if (mode == "linear_speculation")
        mode = "linear_spec";
    if (mode == "linear_speculation_lora" || mode == "linear_spec_adapter")
        mode = "linear_spec_lora";
    return mode;
}

void LlamaTextGenerationPipeline::reset_generation_context() {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    state_->reset();
    last_prefill_launches_ = 0;
    state_bound_ = false;
    for (auto& decoder_ctx : decoders_)
        decoder_ctx.module->reset_execution_context();
    if (prefill_)
        prefill_->reset_execution_context();
    if (linear_spec_lora_prefill_)
        linear_spec_lora_prefill_->reset_execution_context();
    last_setup_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int32_t LlamaTextGenerationPipeline::resolve_text_diffusion_block_length(
    const GenerateConfig& cfg, int32_t max_new_tokens, bool require_divisible) const {
    if (!config_.supports_text_diffusion || config_.mask_token_id < 0)
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: bundle does not support text diffusion");
    const int32_t block_len =
        cfg.block_length > 0 ? cfg.block_length : std::max(config_.diffusion_block_length, 1);
    if (require_divisible && max_new_tokens % block_len != 0) {
        throw std::runtime_error("LlamaTextGenerationPipeline: diffusion mode requires "
                                 "max_new_tokens % block_length == 0");
    }
    return block_len;
}

int32_t LlamaTextGenerationPipeline::seed_next_token_from_prefill(
    const std::vector<int32_t>& input_ids, std::vector<float>& logits, int32_t vocab) {
    run_prefill_block(input_ids, /*bidirectional=*/false, /*append_kv=*/true, logits);
    if (static_cast<int32_t>(logits.size()) < vocab)
        throw std::runtime_error("LlamaTextGenerationPipeline: missing prefill logits");
    return argmax_with_confidence(logits.data() + logits.size() - static_cast<std::size_t>(vocab),
                                  vocab, 0)
        .token_id;
}

void LlamaTextGenerationPipeline::fill_diffusion_block(std::vector<int32_t>& block,
                                                       std::vector<float>& logits,
                                                       int32_t block_len, int32_t vocab,
                                                       bool use_threshold, float threshold) {
    const int32_t initial_masked = block_len - 1;
    const auto quotas = transfer_quota_schedule(initial_masked, block_len);
    for (int32_t step = 0; step < block_len && has_mask_token(block, config_.mask_token_id);
         ++step) {
        run_prefill_block(block, /*bidirectional=*/true, /*append_kv=*/false, logits);
        if (static_cast<int32_t>(logits.size()) < block_len * vocab) {
            throw std::runtime_error(
                "LlamaTextGenerationPipeline: diffusion engine must output full block logits");
        }
        const auto preds = masked_predictions(logits, block, config_.mask_token_id, vocab);
        apply_diffusion_transfer(block, preds, quotas[static_cast<std::size_t>(step)],
                                 use_threshold, threshold);
    }
}

int32_t LlamaTextGenerationPipeline::verify_diffusion_block(const std::vector<int32_t>& block,
                                                            std::vector<float>& logits,
                                                            int32_t block_len, int32_t vocab) {
    run_prefill_block(block, /*bidirectional=*/false, /*append_kv=*/true, logits);
    if (static_cast<int32_t>(logits.size()) < block_len * vocab) {
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: diffusion engine must output full verify logits");
    }
    return argmax_with_confidence(logits.data() + (static_cast<std::size_t>(block_len - 1) *
                                                   static_cast<std::size_t>(vocab)),
                                  vocab, block_len - 1)
        .token_id;
}

bool LlamaTextGenerationPipeline::append_tokens_until_eos(const std::vector<int32_t>& tokens,
                                                          std::vector<int32_t>& output,
                                                          const LlamaSamplingParams& params) const {
    for (int32_t token : tokens) {
        output.push_back(token);
        if (params.eos_token_id >= 0 && token == params.eos_token_id)
            return true;
    }
    return false;
}

void LlamaTextGenerationPipeline::fill_linear_spec_block(std::vector<int32_t>& block,
                                                         std::vector<float>& logits,
                                                         int32_t block_len, int32_t vocab,
                                                         bool threshold_enabled, float threshold,
                                                         bool use_lora_draft) {
    while (has_mask_token(block, config_.mask_token_id)) {
        TrtModule* draft_prefill = use_lora_draft ? linear_spec_lora_prefill_.get() : nullptr;
        run_prefill_block(block, /*bidirectional=*/true, /*append_kv=*/false, logits,
                          draft_prefill);
        if (static_cast<int32_t>(logits.size()) < block_len * vocab) {
            throw std::runtime_error(
                "LlamaTextGenerationPipeline: linear_spec engine must output full block logits");
        }
        const auto preds = masked_predictions(logits, block, config_.mask_token_id, vocab);
        apply_linear_spec_transfer(block, preds, threshold_enabled, threshold);
    }
}

std::vector<int32_t>
LlamaTextGenerationPipeline::verify_linear_spec_block(const std::vector<int32_t>& block,
                                                      std::vector<float>& logits, int32_t block_len,
                                                      int32_t vocab) {
    run_prefill_block(block, /*bidirectional=*/false, /*append_kv=*/true, logits);
    if (static_cast<int32_t>(logits.size()) < block_len * vocab) {
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: linear_spec engine must output full verify logits");
    }

    std::vector<int32_t> ar_tokens;
    ar_tokens.reserve(static_cast<std::size_t>(block_len));
    for (int32_t i = 0; i < block_len; ++i) {
        ar_tokens.push_back(
            argmax_with_confidence(
                logits.data() + (static_cast<std::size_t>(i) * static_cast<std::size_t>(vocab)),
                vocab, i)
                .token_id);
    }
    return ar_tokens;
}

int32_t
LlamaTextGenerationPipeline::count_linear_spec_accepts(const std::vector<int32_t>& ar_tokens,
                                                       const std::vector<int32_t>& block) {
    if (ar_tokens.empty())
        return 0;
    if (block.size() < 2)
        return 1;
    int32_t accepted = 0;
    const auto limit = static_cast<int32_t>(std::min(ar_tokens.size(), block.size() - 1));
    for (int32_t i = 0; i < limit; ++i) {
        if (ar_tokens[static_cast<std::size_t>(i)] != block[static_cast<std::size_t>(i + 1)])
            break;
        ++accepted;
    }
    return accepted + 1;
}

bool LlamaTextGenerationPipeline::append_linear_spec_tokens(
    const std::vector<int32_t>& ar_tokens, int32_t emit_count, std::vector<int32_t>& output,
    int32_t& generated, const LlamaSamplingParams& params) const {
    for (int32_t i = 0; i < emit_count; ++i) {
        const int32_t token = ar_tokens[static_cast<std::size_t>(i)];
        output.push_back(token);
        ++generated;
        if (params.eos_token_id >= 0 && token == params.eos_token_id)
            return true;
    }
    return false;
}

LlamaTextGenerationPipeline::TimedGenResult LlamaTextGenerationPipeline::generate_from_ids(
    const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
    const LlamaSamplingParams& params, const GenerateConfig& cfg) {
    using Clock = std::chrono::steady_clock;
    const int32_t logical_max_sequence =
        config_.max_sequence_length > 0 ? config_.max_sequence_length : state_->max_length();
    validate_sequence_admission_with_runtime_memory(
        input_ids.size(), max_new_tokens, logical_max_sequence, config_.runtime_sequence_admission,
        "LlamaTextGenerationPipeline");
    if (max_new_tokens == 0 || input_ids.empty())
        return TimedGenResult{input_ids, 0.0, 0.0};

    const std::string mode = resolve_generation_mode(cfg);
    if (mode == "diffusion" || mode == "dlm")
        return generate_diffusion_from_ids(input_ids, max_new_tokens, params, cfg);
    if (mode == "linear_spec" || mode == "linear_spec_lora")
        return generate_linear_spec_from_ids(input_ids, max_new_tokens, params, cfg,
                                             mode == "linear_spec_lora");
    if (mode != "auto" && mode != "ar")
        throw std::runtime_error("LlamaTextGenerationPipeline: unsupported generation mode '" +
                                 mode + "'");

    LlamaISampler* active_sampler = sampler_.get();
    std::unique_ptr<LlamaISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = make_step_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    reset_generation_context();
    state_->set_prompt_length(static_cast<int32_t>(input_ids.size()));

    std::vector<float> logits;
    const bool gpu_sampling = (active_sampler->logits_location() == LlamaLogitsLocation::DEVICE);
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

LlamaTextGenerationPipeline::TimedGenResult
LlamaTextGenerationPipeline::generate_diffusion_from_ids(const std::vector<int32_t>& input_ids,
                                                         int32_t max_new_tokens,
                                                         const LlamaSamplingParams& params,
                                                         const GenerateConfig& cfg) {
    using Clock = std::chrono::steady_clock;
    if (!greedy_text_diffusion_params(params)) {
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: diffusion mode currently supports greedy temperature=0 "
            "generation");
    }
    const int32_t block_len =
        resolve_text_diffusion_block_length(cfg, max_new_tokens, /*require_divisible=*/true);
    const bool use_threshold = cfg.confidence_threshold >= 0.0F;
    const float threshold = cfg.confidence_threshold;
    const int32_t vocab = config_.vocab_size;

    reset_generation_context();
    state_->set_prompt_length(static_cast<int32_t>(input_ids.size()));

    std::vector<float> logits;
    const auto t0 = Clock::now();
    int32_t next_token = seed_next_token_from_prefill(input_ids, logits, vocab);
    const auto t1 = Clock::now();

    std::vector<int32_t> output = input_ids;
    const int32_t num_blocks = max_new_tokens / block_len;
    const auto decode_start = Clock::now();
    for (int32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        std::vector<int32_t> block(static_cast<std::size_t>(block_len), config_.mask_token_id);
        block[0] = next_token;
        fill_diffusion_block(block, logits, block_len, vocab, use_threshold, threshold);
        next_token = verify_diffusion_block(block, logits, block_len, vocab);

        if (append_tokens_until_eos(block, output, params)) {
            const auto t2 = Clock::now();
            return TimedGenResult{
                std::move(output), std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - decode_start).count()};
        }
    }

    const auto t2 = Clock::now();
    return TimedGenResult{std::move(output),
                          std::chrono::duration<double, std::milli>(t1 - t0).count(),
                          std::chrono::duration<double, std::milli>(t2 - decode_start).count()};
}

LlamaTextGenerationPipeline::TimedGenResult
LlamaTextGenerationPipeline::generate_linear_spec_from_ids(const std::vector<int32_t>& input_ids,
                                                           int32_t max_new_tokens,
                                                           const LlamaSamplingParams& params,
                                                           const GenerateConfig& cfg,
                                                           bool use_lora_draft) {
    using Clock = std::chrono::steady_clock;
    if (!greedy_text_diffusion_params(params)) {
        throw std::runtime_error(
            "LlamaTextGenerationPipeline: linear_spec mode currently supports greedy temperature=0 "
            "generation");
    }
    if (use_lora_draft && linear_spec_lora_prefill_ == nullptr) {
        throw std::runtime_error("LlamaTextGenerationPipeline: linear_spec_lora mode requires a "
                                 "linear-spec LoRA engine");
    }
    const int32_t block_len =
        resolve_text_diffusion_block_length(cfg, max_new_tokens, /*require_divisible=*/false);
    const bool threshold_enabled = cfg.confidence_threshold > 0.0F;
    const float threshold = cfg.confidence_threshold;
    const int32_t vocab = config_.vocab_size;

    reset_generation_context();
    state_->set_prompt_length(static_cast<int32_t>(input_ids.size()));

    std::vector<float> logits;
    const auto t0 = Clock::now();
    int32_t next_token = seed_next_token_from_prefill(input_ids, logits, vocab);
    const auto t1 = Clock::now();

    std::vector<int32_t> output = input_ids;
    output.push_back(next_token);
    if (params.eos_token_id >= 0 && next_token == params.eos_token_id) {
        return TimedGenResult{std::move(output),
                              std::chrono::duration<double, std::milli>(t1 - t0).count(), 0.0};
    }

    auto* kv = dynamic_cast<LlamaKvCache*>(state_.get());
    if (kv == nullptr)
        throw std::runtime_error("LlamaTextGenerationPipeline: linear_spec requires LlamaKvCache");

    int32_t generated = 1;
    const auto decode_start = Clock::now();
    while (generated < max_new_tokens) {
        const int32_t cache_len = kv->position();
        std::vector<int32_t> block(static_cast<std::size_t>(block_len), config_.mask_token_id);
        block[0] = next_token;

        fill_linear_spec_block(block, logits, block_len, vocab, threshold_enabled, threshold,
                               use_lora_draft);
        const auto ar_tokens = verify_linear_spec_block(block, logits, block_len, vocab);
        const int32_t accepted = count_linear_spec_accepts(ar_tokens, block);
        const int32_t emit_count = std::min(accepted, max_new_tokens - generated);
        kv->set_position(cache_len + emit_count);
        next_token = ar_tokens[static_cast<std::size_t>(emit_count - 1)];

        if (append_linear_spec_tokens(ar_tokens, emit_count, output, generated, params)) {
            const auto t2 = Clock::now();
            return TimedGenResult{
                std::move(output), std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - decode_start).count()};
        }
    }

    const auto t2 = Clock::now();
    return TimedGenResult{std::move(output),
                          std::chrono::duration<double, std::milli>(t1 - t0).count(),
                          std::chrono::duration<double, std::milli>(t2 - decode_start).count()};
}

bool LlamaTextGenerationPipeline::should_stop_on_answer(const std::vector<int32_t>& output,
                                                        int32_t prompt_token_count,
                                                        const GenerateConfig& cfg, int32_t steps,
                                                        int32_t stop_interval, bool is_eos) const {
    if (!cfg.stop_on_boxed_answer || !tokenizer_)
        return false;
    if ((steps % stop_interval) != 0 && !is_eos)
        return false;
    std::vector<int32_t> new_tokens(output.begin() + prompt_token_count, output.end());
    const std::string decoded = tokenizer_->decode(new_tokens);
    return contains_boxed_answer(decoded) || contains_final_answer(decoded);
}

void LlamaTextGenerationPipeline::log_decode_summary(int32_t steps, double ms) const {
    if (steps <= 0 || !config_.log_runtime_stats)
        return;
    const double tps = steps * 1000.0 / ms;
    const bool cuda_graph_on =
        active_decoder_index_ >= 0 &&
        decoders_[static_cast<std::size_t>(active_decoder_index_)].module->cuda_graph_active();
    std::cerr << "[trtmc] Decode: " << steps << " tokens, " << ms << " ms, " << tps << " tok/s"
              << (cuda_graph_on ? " [CUDA Graph ON]" : "") << '\n';
}

int32_t LlamaTextGenerationPipeline::run_decode_loop(
    LlamaISampler* sampler, const LlamaSamplingParams& params, std::vector<int32_t>& output,
    std::vector<float>& logits, int32_t max_new_tokens, bool gpu_sampling,
    const GenerateConfig& cfg, int32_t prompt_token_count) {
    const int32_t vocab_size =
        gpu_sampling ? config_.vocab_size : static_cast<int32_t>(logits.size());
    const int32_t stop_interval = std::max(cfg.stop_check_interval, 1);
    const auto decode_start = std::chrono::steady_clock::now();
    int32_t steps = 0;
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const float* sample_ptr = gpu_sampling ? d_logits_ptr_ : logits.data();
        const LlamaSampleResult result = sampler->sample(sample_ptr, vocab_size, params);
        output.push_back(result.token_id);
        ++steps;
        if (should_stop_on_answer(output, prompt_token_count, cfg, steps, stop_interval,
                                  result.is_eos))
            break;
        if (result.is_eos)
            break;
        if (gpu_sampling)
            run_step_device(result.token_id);
        else
            run_step(result.token_id, logits);
    }
    const auto decode_end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    log_decode_summary(steps, ms);
    return steps;
}

int32_t LlamaTextGenerationPipeline::select_decoder_index(int32_t desired_rows) const {
    if (decoders_.size() == 1)
        return 0;

    int32_t fallback_idx = 0;
    int32_t fallback_rows = std::numeric_limits<int32_t>::max();
    for (std::size_t i = 0; i < decoders_.size(); ++i) {
        const int32_t kv_rows = decoders_[i].kv_rows;
        if (kv_rows == desired_rows)
            return static_cast<int32_t>(i);
        if (kv_rows > 0 && kv_rows >= desired_rows && kv_rows < fallback_rows) {
            fallback_rows = kv_rows;
            fallback_idx = static_cast<int32_t>(i);
        }
    }
    return fallback_idx;
}

std::uint64_t
LlamaTextGenerationPipeline::qualification_bound_tokens(std::uint64_t history_tokens) const {
    const auto selected = state_->runtime_kv_bound_tokens(history_tokens);
    if (selected == 0)
        throw std::logic_error(
            "LlamaTextGenerationPipeline: runtime state did not report a T bound");
    return selected;
}

void LlamaTextGenerationPipeline::append_qualification_invocation(
    const char* role, const char* plan_id, const TrtModule& module, std::uint64_t chunk_begin,
    std::uint64_t chunk_end, std::uint64_t history_tokens, std::uint64_t active_tokens,
    std::uint64_t bound_tokens, const RuntimeMemoryTransferSnapshotV1& transfer_before,
    const RuntimeKvCommitSnapshot& commit_before) {
    if (active_qualification_ == nullptr)
        return;
    const auto* ledger = dynamic_cast<const IRuntimeMemoryTransferLedgerV1*>(&module);
    if (ledger == nullptr) {
        throw std::logic_error(
            "LlamaTextGenerationPipeline: runtime backend has no transfer ledger");
    }
    const auto transfer_delta =
        runtime_memory_transfer_delta(transfer_before, ledger->runtime_memory_transfer_snapshot());
    const auto commit_after = qualification_commit_snapshot();
    if (commit_after.device_to_device_bytes < commit_before.device_to_device_bytes ||
        commit_after.device_to_device_events < commit_before.device_to_device_events) {
        throw std::logic_error("LlamaTextGenerationPipeline: runtime commit counters regressed");
    }
    RuntimeMemoryInvocationTraceV1 trace;
    trace.invocation_index = qualification_invocation_index_++;
    trace.role = role;
    trace.plan_id = plan_id;
    trace.profile_id = module.profile_idx();
    trace.chunk_begin = chunk_begin;
    trace.chunk_end = chunk_end;
    trace.kv_base_address = state_->runtime_kv_base_address();
    trace.history_tokens = history_tokens;
    trace.active_tokens = active_tokens;
    trace.bound_tokens = bound_tokens;
    trace.context_device_memory_bytes = state_->runtime_context_device_memory_bytes();
    trace.cuda_graph_status = module.cuda_graph_active() ? "active" : "uncaptured";
    trace.kv_device_to_host_bytes = transfer_delta.runtime_kv_device_to_host_bytes;
    trace.kv_append_bytes =
        commit_after.device_to_device_bytes - commit_before.device_to_device_bytes;
    trace.kv_append_events =
        commit_after.device_to_device_events - commit_before.device_to_device_events;
    trace.full_history_device_to_device_bytes = transfer_delta.runtime_kv_device_to_device_bytes;
    active_qualification_->invocations.push_back(std::move(trace));
}

RuntimeMemoryTransferSnapshotV1
LlamaTextGenerationPipeline::qualification_transfer_snapshot(const TrtModule& module) const {
    if (active_qualification_ == nullptr)
        return {};
    const auto* ledger = dynamic_cast<const IRuntimeMemoryTransferLedgerV1*>(&module);
    if (ledger == nullptr) {
        throw std::logic_error(
            "LlamaTextGenerationPipeline: runtime backend has no transfer ledger");
    }
    return ledger->runtime_memory_transfer_snapshot();
}

RuntimeKvCommitSnapshot LlamaTextGenerationPipeline::qualification_commit_snapshot() const {
    if (active_qualification_ == nullptr)
        return {};
    const auto* cache = dynamic_cast<const LlamaKvCache*>(state_.get());
    if (cache == nullptr || !cache->runtime_owned_kv()) {
        throw std::logic_error("LlamaTextGenerationPipeline: runtime qualification has no "
                               "contiguous KV commit ledger");
    }
    return cache->runtime_kv_commit_snapshot();
}

TrtModule& LlamaTextGenerationPipeline::bind_decoder_for_step() {
    const int32_t desired_rows = std::max(state_->preferred_cache_rows(), 1);
    const int32_t next_idx = select_decoder_index(desired_rows);
    if (!state_bound_ || next_idx != active_decoder_index_) {
        active_decoder_index_ = next_idx;
        state_->bind_to(*decoders_[static_cast<std::size_t>(active_decoder_index_)].module);
        state_bound_ = true;
    }
    return *decoders_[static_cast<std::size_t>(active_decoder_index_)].module;
}

void LlamaTextGenerationPipeline::run_step(int32_t token_id, std::vector<float>& logits) {
    TensorMap inputs;
    const int32_t position_before = state_->position();
    const int32_t rows_before = std::max(state_->preferred_cache_rows(), 1);

    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;
    inputs[config_.token_id_name] = token_tensor;

    TrtModule& decoder = bind_decoder_for_step();
    const auto transfer_before = qualification_transfer_snapshot(decoder);
    const auto commit_before = qualification_commit_snapshot();
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
    append_qualification_invocation(
        "decode", "engine_plan:decode", decoder, static_cast<std::uint64_t>(position_before),
        static_cast<std::uint64_t>(position_before + 1),
        static_cast<std::uint64_t>(position_before),
        static_cast<std::uint64_t>(position_before + 1), static_cast<std::uint64_t>(rows_before),
        transfer_before, commit_before);
    // At the exact model boundary there is no next invocation. Do not ask a
    // runtime-owned state to size A=position+1 merely to populate optional
    // observability; that would turn a successfully executed Mth token into
    // an artificial M+1 admission failure.
    const int32_t rows_after = resolve_runtime_memory_post_step_trace_rows(
        state_->position(), state_->max_length(), rows_before,
        [&] { return std::max(state_->preferred_cache_rows(), 1); });
    maybe_append_step_trace(position_before, token_id, active_decoder_index_, rows_before,
                            rows_after, logits);
}

void LlamaTextGenerationPipeline::run_step_device(int32_t token_id) {
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

int32_t LlamaTextGenerationPipeline::argmax(const std::vector<float>& logits) {
    if (logits.empty())
        return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace trtmc
