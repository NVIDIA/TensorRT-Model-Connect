/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_labs_diffusion/runtime/pipeline.h"

#include "families/nemotron_labs_diffusion/runtime/chat_templates.h"
#include "families/nemotron_labs_diffusion/runtime/kv_cache.h"
#include "families/nemotron_labs_diffusion/runtime/tensor_names.h"

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

std::string normalize_generation_mode(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    std::replace(mode.begin(), mode.end(), '-', '_');
    return mode;
}

bool greedy_text_diffusion_params(const NemotronLabsDiffusionSamplingParams& params) {
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

NemotronLabsDiffusionTextGenerationPipeline::NemotronLabsDiffusionTextGenerationPipeline(
    std::unique_ptr<ITrtModule> decoder, std::unique_ptr<NemotronLabsDiffusionInferenceState> state,
    NemotronLabsDiffusionTextGenConfig config, std::shared_ptr<ITokenizer> tokenizer,
    std::unique_ptr<ITrtModule> prefill, std::unique_ptr<ITrtModule> linear_spec_lora_prefill,
    std::shared_ptr<void> distributed_owner)
    : distributed_owner_(std::move(distributed_owner)), decoder_(std::move(decoder)),
      prefill_(std::move(prefill)), linear_spec_lora_prefill_(std::move(linear_spec_lora_prefill)),
      state_(std::move(state)), config_(std::move(config)), tokenizer_(std::move(tokenizer)),
      logits_output_name_(config_.logits_output_name) {
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error(
            "NemotronLabsDiffusionTextGenerationPipeline: invalid decoder module");
    if (!prefill_ || !prefill_->ok())
        throw std::runtime_error(
            "NemotronLabsDiffusionTextGenerationPipeline: invalid prefill module");
    if (!tokenizer_)
        throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: invalid tokenizer");
    if (!state_ || !state_->ok()) {
        throw std::runtime_error(
            "NemotronLabsDiffusionTextGenerationPipeline: invalid inference state");
    }

    decoder_->enable_cuda_graph();
}

// Encode a prompt, optionally applying a chat template first.
// Deduplicates the leading BOS token that chat templates embed but
// the tokenizer's add_special_tokens may also prepend.
static std::vector<int32_t> encode_prompt(const ITokenizer& tokenizer,
                                          const NemotronLabsDiffusionTextGenConfig& config,
                                          const std::string& prompt,
                                          const TextGenerationConfig& cfg) {
    std::string effective = prompt;
    bool templated = false;
    if (cfg.use_chat_template && !config.chat_template_format.empty()) {
        effective = nemotron_labs_diffusion_apply_chat_template(config.chat_template_format, prompt,
                                                                cfg.enable_thinking);
        templated = true;
    }
    auto ids = tokenizer.encode(effective);
    if (templated && ids.size() >= 2 && config.id_bos >= 0 && ids[0] == config.id_bos &&
        ids[1] == config.id_bos) {
        ids.erase(ids.begin());
    }
    return ids;
}

TextResult NemotronLabsDiffusionTextGenerationPipeline::generate(const std::string& prompt,
                                                                 const TextGenerationConfig& cfg) {

    auto input_ids = encode_prompt(*tokenizer_, config_, prompt, cfg);
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;

    auto sp = nemotron_labs_diffusion_sampling_params_from_config(cfg, eos);
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

NemotronLabsDiffusionTextGenerationPipeline::GenerationResult
NemotronLabsDiffusionTextGenerationPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                                          const TextGenerationConfig& cfg) {
    int32_t max_new = cfg.max_new_tokens; // honour exact value (0 = no generation)
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    auto sp = nemotron_labs_diffusion_sampling_params_from_config(cfg, eos);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp, cfg).token_ids};
}

namespace {

void require_prefill_kv_pointers(ITrtModule& prefill, const NemotronLabsDiffusionTextGenConfig& cfg,
                                 std::vector<const void*>& pk, std::vector<const void*>& pv) {
    pk.resize(static_cast<std::size_t>(cfg.num_layers));
    pv.resize(static_cast<std::size_t>(cfg.num_layers));
    for (int32_t layer = 0; layer < cfg.num_layers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        pk[index] = prefill.device_ptr(
            nemotron_labs_diffusion_expand_layer_name(cfg.present_k_pattern, layer));
        pv[index] = prefill.device_ptr(
            nemotron_labs_diffusion_expand_layer_name(cfg.present_v_pattern, layer));
        if (pk[index] == nullptr || pv[index] == nullptr) {
            throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: prefill module "
                                     "is missing K/V output for layer " +
                                     std::to_string(layer));
        }
    }
}

void require_batched_prefill_contract(ITrtModule& prefill,
                                      const NemotronLabsDiffusionTextGenConfig& cfg, int32_t sq,
                                      NemotronLabsDiffusionInferenceState* state) {
    if (!prefill.ok())
        throw std::runtime_error(
            "NemotronLabsDiffusionTextGenerationPipeline: invalid prefill module");
    if (sq <= 0)
        throw std::runtime_error(
            "NemotronLabsDiffusionTextGenerationPipeline: prefill requires a non-empty prompt");
    if (cfg.prefill_max_length <= 0 || cfg.num_layers <= 0 || cfg.vocab_size <= 0)
        throw std::runtime_error(
            "NemotronLabsDiffusionTextGenerationPipeline: invalid prefill configuration");
    if (sq > cfg.prefill_max_length)
        throw std::runtime_error(
            "NemotronLabsDiffusionTextGenerationPipeline: prompt exceeds the prefill profile");
    auto* kv = dynamic_cast<NemotronLabsDiffusionKvCache*>(state);
    if (kv == nullptr)
        throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: prefill requires "
                                 "NemotronLabsDiffusionKvCache");
}

} // namespace

void NemotronLabsDiffusionTextGenerationPipeline::run_prefill_batched(
    const std::vector<int32_t>& input_ids, std::vector<float>& logits) {
    const auto sq = static_cast<int32_t>(input_ids.size());
    require_batched_prefill_contract(*prefill_, config_, sq, state_.get());
    auto* kv = static_cast<NemotronLabsDiffusionKvCache*>(state_.get());

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
            "NemotronLabsDiffusionTextGenerationPipeline: prefill module has no logits output");

    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    const auto& lt = logits_it->second;
    if (static_cast<std::size_t>(lt.numel()) < vocab)
        throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: prefill logits are "
                                 "smaller than vocabulary");
    logits.resize(vocab);
    const auto offset = static_cast<std::size_t>(lt.numel()) - vocab;
    std::memcpy(logits.data(), static_cast<const float*>(lt.data) + offset, vocab * sizeof(float));

    std::vector<const void*> pk, pv;
    require_prefill_kv_pointers(*prefill_, config_, pk, pv);
    kv->write_prefill_kv(pk, pv, sq);
}

void NemotronLabsDiffusionTextGenerationPipeline::prime_decoder_after_batched_prefill(
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

void NemotronLabsDiffusionTextGenerationPipeline::run_prefill(const std::vector<int32_t>& input_ids,
                                                              std::vector<float>& logits) {
    run_prefill_batched(input_ids, logits);
    prime_decoder_after_batched_prefill(input_ids);
    state_->mark_prefill_complete();
}

ITrtModule&
NemotronLabsDiffusionTextGenerationPipeline::require_block_prefill(int32_t sq,
                                                                   ITrtModule* prefill_override) {
    ITrtModule* prefill = prefill_override != nullptr ? prefill_override : prefill_.get();
    if (prefill == nullptr)
        throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: block generation "
                                 "requires prefill module");
    if (sq <= 0)
        throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: empty block");
    if (config_.prefill_max_length > 0 && sq > config_.prefill_max_length) {
        throw std::runtime_error(
            "NemotronLabsDiffusionTextGenerationPipeline: block length exceeds prefill profile");
    }
    return *prefill;
}

NemotronLabsDiffusionKvCache&
NemotronLabsDiffusionTextGenerationPipeline::require_block_kv_cache() {
    auto* kv = dynamic_cast<NemotronLabsDiffusionKvCache*>(state_.get());
    if (kv == nullptr)
        throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: block generation "
                                 "requires NemotronLabsDiffusionKvCache");
    return *kv;
}

void NemotronLabsDiffusionTextGenerationPipeline::copy_block_logits(
    const TensorMap& outputs, std::vector<float>& logits) const {
    auto logits_it = outputs.find(config_.logits_output_name);
    if (logits_it == outputs.end())
        throw std::runtime_error(
            "NemotronLabsDiffusionTextGenerationPipeline: prefill module has no '" +
            config_.logits_output_name + "' output");

    const auto& lt = logits_it->second;
    const auto num_logits = static_cast<std::size_t>(lt.numel());
    logits.resize(num_logits);
    std::memcpy(logits.data(), lt.data, num_logits * sizeof(float));
}

void NemotronLabsDiffusionTextGenerationPipeline::append_prefill_kv(
    NemotronLabsDiffusionKvCache& kv, ITrtModule& prefill, int32_t sq) {
    std::vector<const void*> pk, pv;
    require_prefill_kv_pointers(prefill, config_, pk, pv);
    kv.append_prefill_kv(pk, pv, sq);
}

void NemotronLabsDiffusionTextGenerationPipeline::run_prefill_block(
    const std::vector<int32_t>& input_ids, bool bidirectional, bool append_kv,
    std::vector<float>& logits, ITrtModule* prefill_override) {
    const auto sq = static_cast<int32_t>(input_ids.size());
    ITrtModule& prefill = require_block_prefill(sq, prefill_override);
    NemotronLabsDiffusionKvCache& kv = require_block_kv_cache();

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

std::string NemotronLabsDiffusionTextGenerationPipeline::resolve_generation_mode(
    const TextGenerationConfig& cfg) const {
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

void NemotronLabsDiffusionTextGenerationPipeline::reset_generation_context() {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    state_->reset();
    state_bound_ = false;
    decoder_->reset_execution_context();
    prefill_->reset_execution_context();
    if (linear_spec_lora_prefill_)
        linear_spec_lora_prefill_->reset_execution_context();
    last_setup_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int32_t NemotronLabsDiffusionTextGenerationPipeline::resolve_text_diffusion_block_length(
    const TextGenerationConfig& cfg, int32_t max_new_tokens, bool require_divisible) const {
    if (!config_.supports_text_diffusion || config_.mask_token_id < 0)
        throw std::runtime_error(
            "NemotronLabsDiffusionTextGenerationPipeline: bundle does not support text diffusion");
    const int32_t block_len =
        cfg.block_length > 0 ? cfg.block_length : std::max(config_.diffusion_block_length, 1);
    if (require_divisible && max_new_tokens % block_len != 0) {
        throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: diffusion mode "
                                 "requires max_new_tokens % block_length == 0");
    }
    return block_len;
}

int32_t NemotronLabsDiffusionTextGenerationPipeline::seed_next_token_from_prefill(
    const std::vector<int32_t>& input_ids, std::vector<float>& logits, int32_t vocab) {
    run_prefill_block(input_ids, /*bidirectional=*/false, /*append_kv=*/true, logits);
    if (static_cast<int32_t>(logits.size()) < vocab)
        throw std::runtime_error(
            "NemotronLabsDiffusionTextGenerationPipeline: missing prefill logits");
    return argmax_with_confidence(logits.data() + logits.size() - static_cast<std::size_t>(vocab),
                                  vocab, 0)
        .token_id;
}

void NemotronLabsDiffusionTextGenerationPipeline::fill_diffusion_block(
    std::vector<int32_t>& block, std::vector<float>& logits, int32_t block_len, int32_t vocab,
    bool use_threshold, float threshold) {
    const int32_t initial_masked = block_len - 1;
    const auto quotas = transfer_quota_schedule(initial_masked, block_len);
    for (int32_t step = 0; step < block_len && has_mask_token(block, config_.mask_token_id);
         ++step) {
        run_prefill_block(block, /*bidirectional=*/true, /*append_kv=*/false, logits);
        if (static_cast<int32_t>(logits.size()) < block_len * vocab) {
            throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: diffusion "
                                     "engine must output full block logits");
        }
        const auto preds = masked_predictions(logits, block, config_.mask_token_id, vocab);
        apply_diffusion_transfer(block, preds, quotas[static_cast<std::size_t>(step)],
                                 use_threshold, threshold);
    }
}

int32_t NemotronLabsDiffusionTextGenerationPipeline::verify_diffusion_block(
    const std::vector<int32_t>& block, std::vector<float>& logits, int32_t block_len,
    int32_t vocab) {
    run_prefill_block(block, /*bidirectional=*/false, /*append_kv=*/true, logits);
    if (static_cast<int32_t>(logits.size()) < block_len * vocab) {
        throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: diffusion engine "
                                 "must output full verify logits");
    }
    return argmax_with_confidence(logits.data() + (static_cast<std::size_t>(block_len - 1) *
                                                   static_cast<std::size_t>(vocab)),
                                  vocab, block_len - 1)
        .token_id;
}

bool NemotronLabsDiffusionTextGenerationPipeline::append_tokens_until_eos(
    const std::vector<int32_t>& tokens, std::vector<int32_t>& output,
    const NemotronLabsDiffusionSamplingParams& params) const {
    for (int32_t token : tokens) {
        output.push_back(token);
        if (params.eos_token_id >= 0 && token == params.eos_token_id)
            return true;
    }
    return false;
}

void NemotronLabsDiffusionTextGenerationPipeline::fill_linear_spec_block(
    std::vector<int32_t>& block, std::vector<float>& logits, int32_t block_len, int32_t vocab,
    bool threshold_enabled, float threshold, bool use_lora_draft) {
    while (has_mask_token(block, config_.mask_token_id)) {
        ITrtModule* draft_prefill = use_lora_draft ? linear_spec_lora_prefill_.get() : nullptr;
        run_prefill_block(block, /*bidirectional=*/true, /*append_kv=*/false, logits,
                          draft_prefill);
        if (static_cast<int32_t>(logits.size()) < block_len * vocab) {
            throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: linear_spec "
                                     "engine must output full block logits");
        }
        const auto preds = masked_predictions(logits, block, config_.mask_token_id, vocab);
        apply_linear_spec_transfer(block, preds, threshold_enabled, threshold);
    }
}

std::vector<int32_t> NemotronLabsDiffusionTextGenerationPipeline::verify_linear_spec_block(
    const std::vector<int32_t>& block, std::vector<float>& logits, int32_t block_len,
    int32_t vocab) {
    run_prefill_block(block, /*bidirectional=*/false, /*append_kv=*/true, logits);
    if (static_cast<int32_t>(logits.size()) < block_len * vocab) {
        throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: linear_spec engine "
                                 "must output full verify logits");
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

int32_t NemotronLabsDiffusionTextGenerationPipeline::count_linear_spec_accepts(
    const std::vector<int32_t>& ar_tokens, const std::vector<int32_t>& block) {
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

bool NemotronLabsDiffusionTextGenerationPipeline::append_linear_spec_tokens(
    const std::vector<int32_t>& ar_tokens, int32_t emit_count, std::vector<int32_t>& output,
    int32_t& generated, const NemotronLabsDiffusionSamplingParams& params) const {
    for (int32_t i = 0; i < emit_count; ++i) {
        const int32_t token = ar_tokens[static_cast<std::size_t>(i)];
        output.push_back(token);
        ++generated;
        if (params.eos_token_id >= 0 && token == params.eos_token_id)
            return true;
    }
    return false;
}

NemotronLabsDiffusionTextGenerationPipeline::TimedGenResult
NemotronLabsDiffusionTextGenerationPipeline::generate_from_ids(
    const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
    const NemotronLabsDiffusionSamplingParams& params, const TextGenerationConfig& cfg) {
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
        throw std::runtime_error(
            "NemotronLabsDiffusionTextGenerationPipeline: unsupported generation mode '" + mode +
            "'");
    auto active_sampler = create_nemotron_labs_diffusion_sampler(params);
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

NemotronLabsDiffusionTextGenerationPipeline::TimedGenResult
NemotronLabsDiffusionTextGenerationPipeline::generate_diffusion_from_ids(
    const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
    const NemotronLabsDiffusionSamplingParams& params, const TextGenerationConfig& cfg) {
    using Clock = std::chrono::steady_clock;
    if (!greedy_text_diffusion_params(params)) {
        throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: diffusion mode "
                                 "currently supports greedy temperature=0 "
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

NemotronLabsDiffusionTextGenerationPipeline::TimedGenResult
NemotronLabsDiffusionTextGenerationPipeline::generate_linear_spec_from_ids(
    const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
    const NemotronLabsDiffusionSamplingParams& params, const TextGenerationConfig& cfg,
    bool use_lora_draft) {
    using Clock = std::chrono::steady_clock;
    if (!greedy_text_diffusion_params(params)) {
        throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: linear_spec mode "
                                 "currently supports greedy temperature=0 "
                                 "generation");
    }
    if (use_lora_draft && linear_spec_lora_prefill_ == nullptr) {
        throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: linear_spec_lora "
                                 "mode requires a linear-spec LoRA engine");
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

    auto* kv = dynamic_cast<NemotronLabsDiffusionKvCache*>(state_.get());
    if (kv == nullptr)
        throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: linear_spec "
                                 "requires NemotronLabsDiffusionKvCache");

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

bool NemotronLabsDiffusionTextGenerationPipeline::should_stop_on_answer(
    const std::vector<int32_t>& output, int32_t prompt_token_count, const TextGenerationConfig& cfg,
    int32_t steps, int32_t stop_interval, bool is_eos) const {
    if (!cfg.stop_on_boxed_answer || !tokenizer_)
        return false;
    if ((steps % stop_interval) != 0 && !is_eos)
        return false;
    std::vector<int32_t> new_tokens(output.begin() + prompt_token_count, output.end());
    const std::string decoded = tokenizer_->decode(new_tokens);
    return contains_boxed_answer(decoded) || contains_final_answer(decoded);
}

int32_t NemotronLabsDiffusionTextGenerationPipeline::run_decode_loop(
    NemotronLabsDiffusionISampler* sampler, const NemotronLabsDiffusionSamplingParams& params,
    std::vector<int32_t>& output, std::vector<float>& logits, int32_t max_new_tokens,
    const TextGenerationConfig& cfg, int32_t prompt_token_count) {
    const int32_t vocab_size = static_cast<int32_t>(logits.size());
    const int32_t stop_interval = std::max(cfg.stop_check_interval, 1);
    int32_t steps = 0;
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const NemotronLabsDiffusionSampleResult result =
            sampler->sample(logits.data(), vocab_size, params);
        output.push_back(result.token_id);
        ++steps;
        if (should_stop_on_answer(output, prompt_token_count, cfg, steps, stop_interval,
                                  result.is_eos))
            break;
        if (result.is_eos)
            break;
        run_step(result.token_id, logits);
    }
    return steps;
}

ITrtModule& NemotronLabsDiffusionTextGenerationPipeline::bind_decoder_for_step() {
    if (!state_bound_) {
        state_->bind_to(*decoder_);
        state_bound_ = true;
    }
    return *decoder_;
}

void NemotronLabsDiffusionTextGenerationPipeline::run_step(int32_t token_id,
                                                           std::vector<float>& logits) {
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
        throw std::runtime_error("NemotronLabsDiffusionTextGenerationPipeline: no '" +
                                 logits_output_name_ + "' output");
    }

    const auto& logits_tensor = it->second;
    auto num_logits = logits_tensor.numel();
    logits.resize(static_cast<std::size_t>(num_logits));
    std::memcpy(logits.data(), logits_tensor.data, num_logits * sizeof(float));

    state_->advance();
}

} // namespace trtmc
