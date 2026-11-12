/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/gemma/runtime/kv_cache.h"
#include "families/gemma/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace trtmc {
namespace {

std::vector<float> block_values(const TensorMap& outputs, const char* name, std::size_t count) {
    const auto it = outputs.find(name);
    if (it == outputs.end() || it->second.dtype != DType::kFloat32 || it->second.numel() != count ||
        !it->second.data)
        throw std::runtime_error(std::string("Gemma4 block draft invalid output: ") + name);
    const auto* data = static_cast<const float*>(it->second.data);
    std::vector<float> values(data, data + count);
    if (!std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error(std::string("Gemma4 block draft nonfinite output: ") + name);
    return values;
}

} // namespace

std::vector<int32_t>
GemmaTextGenerationPipeline::propose_block(int32_t pending, const std::vector<int32_t>& output) {
    const int32_t past = state_->position();
    const int32_t count = std::min(config_.draft_block_size, state_->max_length() - past);
    if (past <= 0 || count <= 0 ||
        draft_context_.size() != static_cast<std::size_t>(past) * config_.draft_feature_width)
        throw std::runtime_error("Gemma4 block draft context does not match committed target");
    // z-lab's block includes the pending anchor. A one-row block has
    // no proposal to compute, but the target still consumes the pending token.
    if (config_.draft_skip_first && count == 1)
        return {};
    std::vector<int32_t> ids(count, config_.draft_mask_token_id);
    ids[0] = pending;
    std::vector<int32_t> positions(count), context_positions(past);
    std::iota(positions.begin(), positions.end(), past);
    std::iota(context_positions.begin(), context_positions.end(), 0);
    TensorMap inputs{
        {"draft_tokens", {ids.data(), {count}, DType::kInt32}},
        {"draft_positions", {positions.data(), {count}, DType::kInt32}},
        {"context_positions", {context_positions.data(), {past}, DType::kInt32}},
        {"target_features",
         {draft_context_.data(), {past, config_.draft_feature_width}, DType::kFloat32}},
    };
    if (config_.draft_eagle) {
        if (output.size() != static_cast<std::size_t>(past + 1) || output.back() != pending)
            throw std::runtime_error("Gemma4 EAGLE3 shifted tokens do not match target features");
        ids.assign(output.begin() + 1, output.end());
        positions.resize(past);
        std::iota(positions.begin(), positions.end(), 0);
        inputs["draft_tokens"] = {ids.data(), {past}, DType::kInt32};
        inputs["draft_positions"] = {positions.data(), {past}, DType::kInt32};
        inputs.erase("context_positions");
    }
    const auto outputs = assistant_->forward(inputs);
    const int32_t rows = config_.draft_eagle ? config_.draft_block_size : count;
    const auto values =
        block_values(outputs, "draft_logits", static_cast<std::size_t>(rows) * config_.vocab_size);
    std::vector<int32_t> proposals;
    if (config_.draft_markov || config_.draft_eagle) {
        const auto ids = outputs.find("proposal_ids");
        if (ids == outputs.end() || ids->second.dtype != DType::kInt32 ||
            ids->second.numel() != static_cast<std::size_t>(rows) || !ids->second.data)
            throw std::runtime_error("Gemma4 autoregressive draft invalid proposal IDs");
        const auto* data = static_cast<const int32_t*>(ids->second.data);
        proposals.assign(data, data + count);
        if (!std::all_of(proposals.begin(), proposals.end(),
                         [&](int32_t id) { return id >= 0 && id < config_.vocab_size; }))
            throw std::runtime_error(
                "Gemma4 autoregressive draft proposal outside target vocabulary");
        if (config_.draft_markov) {
            // Both heads execute. Keep the paired Edge scheduler default off;
            // confidence is not an implicit quality gate.
            block_values(outputs, "proposal_confidence", count);
        }
    } else {
        for (int32_t row = config_.draft_skip_first ? 1 : 0; row < count; ++row) {
            const auto start = values.begin() + row * config_.vocab_size;
            proposals.push_back(
                static_cast<int32_t>(std::max_element(start, start + config_.vocab_size) - start));
        }
    }
    block_stats_.proposed += proposals.size();
    return proposals;
}

void GemmaTextGenerationPipeline::verify_block(const std::vector<int32_t>& tokens,
                                               std::vector<float>& logits,
                                               std::vector<float>& features) {
    auto* cache = static_cast<GemmaKvCache*>(state_.get());
    const auto count = static_cast<int32_t>(tokens.size());
    if (count < 1 || count > config_.draft_block_size + (config_.draft_skip_first ? 0 : 1) ||
        cache->position() + count > cache->max_length())
        throw std::runtime_error("Gemma4 block verification exceeds target capacity");
    cache->bind_cache_inputs(prefill_module());
    std::vector<int32_t> indices(tokens.size());
    std::iota(indices.begin(), indices.end(), 0);
    TensorMap inputs{
        {config_.token_id_name, {const_cast<int32_t*>(tokens.data()), {count}, DType::kInt32}},
        {"logits_indices", {indices.data(), {count}, DType::kInt32}},
    };
    state_->prepare_step(inputs, count);
    const auto outputs = prefill_module().forward(inputs);
    logits = block_values(outputs, config_.logits_output_name.c_str(),
                          tokens.size() * config_.vocab_size);
    features =
        block_values(outputs, "target_features", tokens.size() * config_.draft_feature_width);
}

void GemmaTextGenerationPipeline::run_block_loop(GemmaISampler* sampler,
                                                 const GemmaSamplingParams& params,
                                                 std::vector<int32_t>& output,
                                                 std::vector<float>& logits, int32_t maximum,
                                                 const TextGenerationConfig& cfg,
                                                 int32_t prompt_count) {
    int32_t emitted = 0;
    const int32_t interval = std::max(cfg.stop_check_interval, 1);
    auto emit = [&](int32_t token) {
        output.push_back(token);
        ++emitted;
        const bool eos = gemma_is_eos_token(params, token);
        return eos || emitted >= maximum ||
               should_stop_on_answer(output, prompt_count, cfg, emitted, interval, eos);
    };
    int32_t sampled_next = -1;
    while (emitted < maximum) {
        // Sample each target distribution exactly once. A rejected proposal
        // must retain the already sampled target token, not draw a replacement
        // at the next block boundary. This preserves ordinary target sampling
        // (including its seeded RNG sequence) with deterministic draft blocks.
        const int32_t pending =
            sampled_next >= 0 ? sampled_next
                              : sampler->sample(logits.data(), config_.vocab_size, params).token_id;
        sampled_next = -1;
        if (emit(pending))
            break;
        const auto proposals = propose_block(pending, output);
        const int32_t count = std::min<int32_t>(1 + static_cast<int32_t>(proposals.size()),
                                                state_->max_length() - state_->position());
        // FP16 target GEMMs can produce different K/V and logits for a
        // multi-row verification shape, even inside the same serialized plan.
        // Keep the ordinary one-row target shape and commit only its verified
        // causal prefix. The draft still proposes real blocks; no speedup is
        // claimed for this conservative native verifier.
        auto consume = [&](int32_t token) {
            std::vector<float> features;
            verify_block({token}, logits, features);
            commit_verified_prefix(1);
            draft_context_.insert(draft_context_.end(), features.begin(), features.end());
        };
        ++block_stats_.verified;
        int32_t current = pending;
        bool stop = false;
        for (int32_t row = 0; row < count; ++row) {
            consume(current);
            if (row + 1 == count)
                break;
            const auto expected =
                sampler->sample(logits.data(), config_.vocab_size, params).token_id;
            if (proposals[row] != expected) {
                sampled_next = expected;
                break;
            }
            ++block_stats_.accepted;
            current = expected;
            if (emit(expected)) {
                // This accepted row was part of the original verification
                // window and fits the cache. Never commit any later proposal.
                consume(current);
                stop = true;
                break;
            }
        }
        if (stop)
            break;
    }
    std::clog << "Gemma4 native "
              << (config_.draft_eagle ? "EAGLE3" : (config_.draft_markov ? "DSpark" : "DFlash"))
              << ": proposed=" << block_stats_.proposed << " accepted=" << block_stats_.accepted
              << " verified=" << block_stats_.verified << '\n';
}

} // namespace trtmc
