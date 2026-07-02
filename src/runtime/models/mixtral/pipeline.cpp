/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/mixtral/pipeline.h"

#include "runtime/models/mixtral/chat_templates.h"
#include "runtime/models/mixtral/kv_cache.h"
#include "runtime/models/mixtral/tensor_names.h"

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

std::vector<MixtralTextGenerationPipeline::DecoderContext>
single_decoder_context(std::unique_ptr<TrtModule> decoder) {
    std::vector<MixtralTextGenerationPipeline::DecoderContext> decoders;
    decoders.push_back(MixtralTextGenerationPipeline::DecoderContext{0, std::move(decoder)});
    return decoders;
}

std::string normalize_generation_mode(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    std::replace(mode.begin(), mode.end(), '-', '_');
    return mode;
}

bool greedy_text_diffusion_params(const MixtralSamplingParams& params) {
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

MixtralTextGenerationPipeline::MixtralTextGenerationPipeline(
    std::unique_ptr<TrtModule> decoder, std::unique_ptr<MixtralInferenceState> state,
    MixtralTextGenConfig config, cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str, std::unique_ptr<MixtralISampler> sampler,
    std::shared_ptr<void> distributed_owner)
    : MixtralTextGenerationPipeline(single_decoder_context(std::move(decoder)), std::move(state),
                                    std::move(config), stream, std::move(tokenizer),
                                    std::move(model_id_str), std::move(sampler),
                                    /*prefill=*/nullptr, /*linear_spec_lora_prefill=*/nullptr,
                                    std::move(distributed_owner)) {}

MixtralTextGenerationPipeline::MixtralTextGenerationPipeline(
    std::vector<DecoderContext> decoders, std::unique_ptr<MixtralInferenceState> state,
    MixtralTextGenConfig config, cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str, std::unique_ptr<MixtralISampler> sampler,
    std::unique_ptr<TrtModule> prefill, std::unique_ptr<TrtModule> linear_spec_lora_prefill,
    std::shared_ptr<void> distributed_owner)
    : distributed_owner_(std::move(distributed_owner)), decoders_(std::move(decoders)),
      prefill_(std::move(prefill)), linear_spec_lora_prefill_(std::move(linear_spec_lora_prefill)),
      state_(std::move(state)), config_(std::move(config)), stream_(stream),
      tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)),
      sampler_(std::move(sampler)), logits_output_name_(config_.logits_output_name) {
    if (decoders_.empty()) {
        throw std::runtime_error("MixtralTextGenerationPipeline: no decoder modules");
    }
    for (const auto& decoder_ctx : decoders_) {
        if (!decoder_ctx.module || !decoder_ctx.module->ok()) {
            throw std::runtime_error("MixtralTextGenerationPipeline: invalid decoder module");
        }
    }
    if (!state_ || !state_->ok()) {
        throw std::runtime_error("MixtralTextGenerationPipeline: invalid inference state");
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
                                          const MixtralTextGenConfig& config,
                                          const std::string& prompt, const GenerateConfig& cfg) {
    std::string effective = prompt;
    bool templated = false;
    if (cfg.use_chat_template && !config.chat_template_format.empty()) {
        effective =
            mixtral_apply_chat_template(config.chat_template_format, prompt, cfg.enable_thinking);
        templated = true;
    }
    auto ids = tokenizer.encode(effective);
    if (templated && ids.size() >= 2 && config.id_bos >= 0 && ids[0] == config.id_bos &&
        ids[1] == config.id_bos) {
        ids.erase(ids.begin());
    }
    return ids;
}

TextResult MixtralTextGenerationPipeline::generate(const std::string& prompt,
                                                   const GenerateConfig& cfg) {
    if (!tokenizer_) {
        throw std::runtime_error("MixtralTextGenerationPipeline: no tokenizer configured");
    }

    auto input_ids = encode_prompt(*tokenizer_, config_, prompt, cfg);
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;

    auto sp = mixtral_sampling_params_from_config(cfg, eos);
    auto timed = generate_from_ids(input_ids, max_new, sp, cfg);

    // Decode only the NEW tokens (skip input)
    std::vector<int32_t> new_tokens(timed.token_ids.begin() +
                                        static_cast<std::ptrdiff_t>(input_ids.size()),
                                    timed.token_ids.end());
    std::string text = tokenizer_->decode(new_tokens);

    return TextResult{std::move(text), std::move(new_tokens), timed.prefill_ms, timed.decode_ms};
}

MixtralTextGenerationPipeline::GenerationResult
MixtralTextGenerationPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                            const GenerateConfig& cfg) {
    int32_t max_new = cfg.max_new_tokens; // honour exact value (0 = no generation)
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    auto sp = mixtral_sampling_params_from_config(cfg, eos);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp, cfg).token_ids};
}

std::unique_ptr<MixtralISampler>
MixtralTextGenerationPipeline::make_step_sampler(const MixtralSamplingParams& params) {
    const bool greedy_params =
        (params.temperature < 1e-6F) ||
        (params.top_k <= 1 && params.top_p >= 1.0F && params.min_p <= 0.0F && params.seed < 0);
    if (prefer_gpu_greedy_ && greedy_params) {
        if (auto gpu = create_mixtral_gpu_greedy_sampler(stream_))
            return gpu;
    }
    return create_mixtral_sampler(params);
}

// Helper: gather per-layer present_k/present_v device pointers from the
// prefill TrtModule. Returns false if any layer's tensor is missing — in
// that case the caller falls back to the per-token decode loop.
namespace {
bool gather_prefill_kv_pointers(TrtModule& prefill, const MixtralTextGenConfig& cfg,
                                std::vector<const void*>& pk, std::vector<const void*>& pv) {
    pk.resize(static_cast<std::size_t>(cfg.num_layers));
    pv.resize(static_cast<std::size_t>(cfg.num_layers));
    for (int32_t i = 0; i < cfg.num_layers; ++i) {
        const auto li = static_cast<std::size_t>(i);
        pk[li] = prefill.device_ptr(mixtral_expand_layer_name(cfg.present_k_pattern, i));
        pv[li] = prefill.device_ptr(mixtral_expand_layer_name(cfg.present_v_pattern, i));
        if (pk[li] == nullptr || pv[li] == nullptr)
            return false;
    }
    return true;
}

bool batched_prefill_supported(const TrtModule* prefill, const MixtralTextGenConfig& cfg,
                               int32_t sq, MixtralInferenceState* state) {
    if (prefill == nullptr || sq <= 0)
        return false;
    if (cfg.prefill_max_length > 0 && sq > cfg.prefill_max_length)
        return false;
    if (cfg.num_layers <= 0 || cfg.vocab_size <= 0)
        return false;
    return dynamic_cast<MixtralKvCache*>(state) != nullptr;
}
} // namespace

bool MixtralTextGenerationPipeline::run_prefill_batched(const std::vector<int32_t>& input_ids,
                                                        std::vector<float>& logits) {
    const auto sq = static_cast<int32_t>(input_ids.size());
    if (!batched_prefill_supported(prefill_.get(), config_, sq, state_.get()))
        return false;
    auto* kv = static_cast<MixtralKvCache*>(state_.get());

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

void MixtralTextGenerationPipeline::prime_decoder_after_batched_prefill(
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

void MixtralTextGenerationPipeline::run_prefill(const std::vector<int32_t>& input_ids,
                                                std::vector<float>& logits, bool gpu_sampling) {
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

TrtModule& MixtralTextGenerationPipeline::require_block_prefill(int32_t sq,
                                                                TrtModule* prefill_override) {
    TrtModule* prefill = prefill_override != nullptr ? prefill_override : prefill_.get();
    if (prefill == nullptr)
        throw std::runtime_error(
            "MixtralTextGenerationPipeline: block generation requires prefill module");
    if (sq <= 0)
        throw std::runtime_error("MixtralTextGenerationPipeline: empty block");
    if (config_.prefill_max_length > 0 && sq > config_.prefill_max_length) {
        throw std::runtime_error(
            "MixtralTextGenerationPipeline: block length exceeds prefill profile");
    }
    return *prefill;
}

MixtralKvCache& MixtralTextGenerationPipeline::require_block_kv_cache() {
    auto* kv = dynamic_cast<MixtralKvCache*>(state_.get());
    if (kv == nullptr)
        throw std::runtime_error(
            "MixtralTextGenerationPipeline: block generation requires MixtralKvCache");
    return *kv;
}

void MixtralTextGenerationPipeline::copy_block_logits(const TensorMap& outputs,
                                                      std::vector<float>& logits) const {
    auto logits_it = outputs.find(config_.logits_output_name);
    if (logits_it == outputs.end())
        throw std::runtime_error("MixtralTextGenerationPipeline: prefill module has no '" +
                                 config_.logits_output_name + "' output");

    const auto& lt = logits_it->second;
    const auto num_logits = static_cast<std::size_t>(lt.numel());
    logits.resize(num_logits);
    std::memcpy(logits.data(), lt.data, num_logits * sizeof(float));
}

void MixtralTextGenerationPipeline::append_prefill_kv(MixtralKvCache& kv, TrtModule& prefill,
                                                      int32_t sq) {
    std::vector<const void*> pk, pv;
    if (!gather_prefill_kv_pointers(prefill, config_, pk, pv)) {
        throw std::runtime_error(
            "MixtralTextGenerationPipeline: prefill module is missing present_k/present_v outputs");
    }
    kv.append_prefill_kv(pk, pv, sq);
}

void MixtralTextGenerationPipeline::run_prefill_block(const std::vector<int32_t>& input_ids,
                                                      bool bidirectional, bool append_kv,
                                                      std::vector<float>& logits,
                                                      TrtModule* prefill_override) {
    const auto sq = static_cast<int32_t>(input_ids.size());
    TrtModule& prefill = require_block_prefill(sq, prefill_override);
    MixtralKvCache& kv = require_block_kv_cache();

    kv.bind_cache_inputs(prefill);

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

    copy_block_logits(prefill.forward(inputs), logits);
    if (append_kv)
        append_prefill_kv(kv, prefill, sq);
}

std::string
MixtralTextGenerationPipeline::resolve_generation_mode(const GenerateConfig& cfg) const {
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

void MixtralTextGenerationPipeline::reset_generation_context() {
    state_->reset();
    state_bound_ = false;
    for (auto& decoder_ctx : decoders_)
        decoder_ctx.module->reset_execution_context();
    if (prefill_)
        prefill_->reset_execution_context();
    if (linear_spec_lora_prefill_)
        linear_spec_lora_prefill_->reset_execution_context();
}

int32_t MixtralTextGenerationPipeline::resolve_text_diffusion_block_length(
    const GenerateConfig& cfg, int32_t max_new_tokens, bool require_divisible) const {
    if (!config_.supports_text_diffusion || config_.mask_token_id < 0)
        throw std::runtime_error(
            "MixtralTextGenerationPipeline: bundle does not support text diffusion");
    const int32_t block_len =
        cfg.block_length > 0 ? cfg.block_length : std::max(config_.diffusion_block_length, 1);
    if (require_divisible && max_new_tokens % block_len != 0) {
        throw std::runtime_error("MixtralTextGenerationPipeline: diffusion mode requires "
                                 "max_new_tokens % block_length == 0");
    }
    return block_len;
}

int32_t MixtralTextGenerationPipeline::seed_next_token_from_prefill(
    const std::vector<int32_t>& input_ids, std::vector<float>& logits, int32_t vocab) {
    run_prefill_block(input_ids, /*bidirectional=*/false, /*append_kv=*/true, logits);
    if (static_cast<int32_t>(logits.size()) < vocab)
        throw std::runtime_error("MixtralTextGenerationPipeline: missing prefill logits");
    return argmax_with_confidence(logits.data() + logits.size() - static_cast<std::size_t>(vocab),
                                  vocab, 0)
        .token_id;
}

void MixtralTextGenerationPipeline::fill_diffusion_block(std::vector<int32_t>& block,
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
                "MixtralTextGenerationPipeline: diffusion engine must output full block logits");
        }
        const auto preds = masked_predictions(logits, block, config_.mask_token_id, vocab);
        apply_diffusion_transfer(block, preds, quotas[static_cast<std::size_t>(step)],
                                 use_threshold, threshold);
    }
}

int32_t MixtralTextGenerationPipeline::verify_diffusion_block(const std::vector<int32_t>& block,
                                                              std::vector<float>& logits,
                                                              int32_t block_len, int32_t vocab) {
    run_prefill_block(block, /*bidirectional=*/false, /*append_kv=*/true, logits);
    if (static_cast<int32_t>(logits.size()) < block_len * vocab) {
        throw std::runtime_error(
            "MixtralTextGenerationPipeline: diffusion engine must output full verify logits");
    }
    return argmax_with_confidence(logits.data() + (static_cast<std::size_t>(block_len - 1) *
                                                   static_cast<std::size_t>(vocab)),
                                  vocab, block_len - 1)
        .token_id;
}

bool MixtralTextGenerationPipeline::append_tokens_until_eos(
    const std::vector<int32_t>& tokens, std::vector<int32_t>& output,
    const MixtralSamplingParams& params) const {
    for (int32_t token : tokens) {
        output.push_back(token);
        if (params.eos_token_id >= 0 && token == params.eos_token_id)
            return true;
    }
    return false;
}

void MixtralTextGenerationPipeline::fill_linear_spec_block(std::vector<int32_t>& block,
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
                "MixtralTextGenerationPipeline: linear_spec engine must output full block logits");
        }
        const auto preds = masked_predictions(logits, block, config_.mask_token_id, vocab);
        apply_linear_spec_transfer(block, preds, threshold_enabled, threshold);
    }
}

std::vector<int32_t>
MixtralTextGenerationPipeline::verify_linear_spec_block(const std::vector<int32_t>& block,
                                                        std::vector<float>& logits,
                                                        int32_t block_len, int32_t vocab) {
    run_prefill_block(block, /*bidirectional=*/false, /*append_kv=*/true, logits);
    if (static_cast<int32_t>(logits.size()) < block_len * vocab) {
        throw std::runtime_error(
            "MixtralTextGenerationPipeline: linear_spec engine must output full verify logits");
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
MixtralTextGenerationPipeline::count_linear_spec_accepts(const std::vector<int32_t>& ar_tokens,
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

bool MixtralTextGenerationPipeline::append_linear_spec_tokens(
    const std::vector<int32_t>& ar_tokens, int32_t emit_count, std::vector<int32_t>& output,
    int32_t& generated, const MixtralSamplingParams& params) const {
    for (int32_t i = 0; i < emit_count; ++i) {
        const int32_t token = ar_tokens[static_cast<std::size_t>(i)];
        output.push_back(token);
        ++generated;
        if (params.eos_token_id >= 0 && token == params.eos_token_id)
            return true;
    }
    return false;
}

MixtralTextGenerationPipeline::TimedGenResult MixtralTextGenerationPipeline::generate_from_ids(
    const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
    const MixtralSamplingParams& params, const GenerateConfig& cfg) {
    using Clock = std::chrono::steady_clock;
    if (max_new_tokens == 0 || input_ids.empty())
        return TimedGenResult{input_ids, 0.0, 0.0};

    const std::string mode = resolve_generation_mode(cfg);
    if (mode == "diffusion" || mode == "dlm")
        return generate_diffusion_from_ids(input_ids, max_new_tokens, params, cfg);
    if (mode == "linear_spec" || mode == "linear_spec_lora")
        return generate_linear_spec_from_ids(input_ids, max_new_tokens, params, cfg,
                                             mode == "linear_spec_lora");
    if (mode != "auto" && mode != "ar")
        throw std::runtime_error("MixtralTextGenerationPipeline: unsupported generation mode '" +
                                 mode + "'");

    MixtralISampler* active_sampler = sampler_.get();
    std::unique_ptr<MixtralISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = make_step_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    reset_generation_context();
    state_->set_prompt_length(static_cast<int32_t>(input_ids.size()));

    std::vector<float> logits;
    const bool gpu_sampling = (active_sampler->logits_location() == MixtralLogitsLocation::DEVICE);
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

MixtralTextGenerationPipeline::TimedGenResult
MixtralTextGenerationPipeline::generate_diffusion_from_ids(const std::vector<int32_t>& input_ids,
                                                           int32_t max_new_tokens,
                                                           const MixtralSamplingParams& params,
                                                           const GenerateConfig& cfg) {
    using Clock = std::chrono::steady_clock;
    if (!greedy_text_diffusion_params(params)) {
        throw std::runtime_error(
            "MixtralTextGenerationPipeline: diffusion mode currently supports greedy temperature=0 "
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

MixtralTextGenerationPipeline::TimedGenResult
MixtralTextGenerationPipeline::generate_linear_spec_from_ids(const std::vector<int32_t>& input_ids,
                                                             int32_t max_new_tokens,
                                                             const MixtralSamplingParams& params,
                                                             const GenerateConfig& cfg,
                                                             bool use_lora_draft) {
    using Clock = std::chrono::steady_clock;
    if (!greedy_text_diffusion_params(params)) {
        throw std::runtime_error("MixtralTextGenerationPipeline: linear_spec mode currently "
                                 "supports greedy temperature=0 "
                                 "generation");
    }
    if (use_lora_draft && linear_spec_lora_prefill_ == nullptr) {
        throw std::runtime_error("MixtralTextGenerationPipeline: linear_spec_lora mode requires a "
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

    auto* kv = dynamic_cast<MixtralKvCache*>(state_.get());
    if (kv == nullptr)
        throw std::runtime_error(
            "MixtralTextGenerationPipeline: linear_spec requires MixtralKvCache");

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

bool MixtralTextGenerationPipeline::should_stop_on_answer(const std::vector<int32_t>& output,
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

void MixtralTextGenerationPipeline::log_decode_summary(int32_t steps, double ms) const {
    if (steps <= 0 || !config_.log_runtime_stats)
        return;
    const double tps = steps * 1000.0 / ms;
    const bool cuda_graph_on =
        active_decoder_index_ >= 0 &&
        decoders_[static_cast<std::size_t>(active_decoder_index_)].module->cuda_graph_active();
    std::cerr << "[trtmc] Decode: " << steps << " tokens, " << ms << " ms, " << tps << " tok/s"
              << (cuda_graph_on ? " [CUDA Graph ON]" : "") << '\n';
}

int32_t MixtralTextGenerationPipeline::run_decode_loop(
    MixtralISampler* sampler, const MixtralSamplingParams& params, std::vector<int32_t>& output,
    std::vector<float>& logits, int32_t max_new_tokens, bool gpu_sampling,
    const GenerateConfig& cfg, int32_t prompt_token_count) {
    const int32_t vocab_size =
        gpu_sampling ? config_.vocab_size : static_cast<int32_t>(logits.size());
    const int32_t stop_interval = std::max(cfg.stop_check_interval, 1);
    const auto decode_start = std::chrono::steady_clock::now();
    int32_t steps = 0;
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const float* sample_ptr = gpu_sampling ? d_logits_ptr_ : logits.data();
        const MixtralSampleResult result = sampler->sample(sample_ptr, vocab_size, params);
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

int32_t MixtralTextGenerationPipeline::select_decoder_index(int32_t desired_rows) const {
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

TrtModule& MixtralTextGenerationPipeline::bind_decoder_for_step() {
    const int32_t desired_rows = std::max(state_->preferred_cache_rows(), 1);
    const int32_t next_idx = select_decoder_index(desired_rows);
    if (!state_bound_ || next_idx != active_decoder_index_) {
        active_decoder_index_ = next_idx;
        state_->bind_to(*decoders_[static_cast<std::size_t>(active_decoder_index_)].module);
        state_bound_ = true;
    }
    return *decoders_[static_cast<std::size_t>(active_decoder_index_)].module;
}

void MixtralTextGenerationPipeline::run_step(int32_t token_id, std::vector<float>& logits) {
    TensorMap inputs;
    const int32_t position_before = state_->position();
    const int32_t rows_before = std::max(state_->preferred_cache_rows(), 1);

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
        throw std::runtime_error("MixtralTextGenerationPipeline: no '" + logits_output_name_ +
                                 "' output");
    }

    const auto& logits_tensor = it->second;
    auto num_logits = logits_tensor.numel();
    logits.resize(static_cast<std::size_t>(num_logits));
    std::memcpy(logits.data(), logits_tensor.data, num_logits * sizeof(float));

    state_->advance();
    maybe_append_step_trace(position_before, token_id, active_decoder_index_, rows_before,
                            std::max(state_->preferred_cache_rows(), 1), logits);
}

void MixtralTextGenerationPipeline::run_step_device(int32_t token_id) {
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

int32_t MixtralTextGenerationPipeline::argmax(const std::vector<float>& logits) {
    if (logits.empty())
        return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace trtmc
