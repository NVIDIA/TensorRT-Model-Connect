/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/gemma/runtime/kv_cache.h"
#include "families/gemma/runtime/pipeline.h"
#include "families/gemma/runtime/tensor_names.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace trtmc {
namespace {

std::vector<float> require_values(const TensorMap& outputs, const char* name, std::size_t count) {
    const auto it = outputs.find(name);
    if (it == outputs.end() || it->second.dtype != DType::kFloat32 || it->second.numel() != count ||
        it->second.data == nullptr)
        throw std::runtime_error(std::string("Gemma4 MTP invalid output: ") + name);
    const auto* values = static_cast<const float*>(it->second.data);
    std::vector<float> result(values, values + count);
    if (!std::all_of(result.begin(), result.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error(std::string("Gemma4 MTP nonfinite output: ") + name);
    return result;
}

} // namespace

int32_t GemmaTextGenerationPipeline::propose_mtp(int32_t token) {
    auto* cache = dynamic_cast<GemmaKvCache*>(state_.get());
    if (!cache || cache->position() <= 0 || cache->position() >= cache->max_length() ||
        target_hidden_.size() != static_cast<std::size_t>(config_.mtp_hidden_size))
        throw std::runtime_error("Gemma4 MTP has invalid target state");
    for (const auto& [kind, layer] : config_.mtp_target_layers) {
        if (layer < 0 || layer >= cache->num_layers())
            throw std::runtime_error("Gemma4 MTP shared layer is outside the target cache");
        for (const auto* label : {"k", "v"}) {
            auto& tensor = *label == 'k' ? cache->cache_k(layer) : cache->cache_v(layer);
            const auto name = std::string("shared_") + label + "_" + kind;
            if (!assistant_->has_input(name) || assistant_->tensor_shape(name) != tensor.shape() ||
                assistant_->tensor_dtype(name) != tensor.dtype())
                throw std::runtime_error("Gemma4 MTP shared cache geometry or precision differs");
            assistant_->bind_external(name, tensor.data());
        }
    }
    int32_t position = cache->position();
    TensorMap inputs{
        {"token_id", Tensor{&token, {1}, DType::kInt32}},
        {"target_hidden",
         Tensor{target_hidden_.data(), {1, config_.mtp_hidden_size}, DType::kFloat32}},
        {"position_id", Tensor{&position, {1}, DType::kInt32}},
        {"shared_length", Tensor{&position, {1}, DType::kInt32}},
    };
    const auto outputs = assistant_->forward(inputs);
    const auto logits = require_values(outputs, "logits", config_.vocab_size);
    ++mtp_stats_.proposed;
    return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

void GemmaTextGenerationPipeline::verify_mtp(const std::vector<int32_t>& tokens,
                                             std::vector<float>& logits,
                                             std::vector<float>& hidden) {
    auto* cache = static_cast<GemmaKvCache*>(state_.get());
    const auto count = static_cast<int32_t>(tokens.size());
    if (count < 1 || count > 2 || cache->position() + count > cache->max_length())
        throw std::runtime_error("Gemma4 MTP verification exceeds target capacity");
    cache->bind_cache_inputs(prefill_module());
    std::vector<int32_t> indices(tokens.size());
    std::iota(indices.begin(), indices.end(), 0);
    TensorMap inputs{
        {config_.token_id_name,
         Tensor{const_cast<int32_t*>(tokens.data()), {count}, DType::kInt32}},
        {"logits_indices", Tensor{indices.data(), {count}, DType::kInt32}},
    };
    state_->prepare_step(inputs, count);
    const auto outputs = prefill_module().forward(inputs);
    logits = require_values(outputs, config_.logits_output_name.c_str(),
                            tokens.size() * config_.vocab_size);
    hidden = require_values(outputs, "target_hidden", tokens.size() * config_.mtp_hidden_size);
}

void GemmaTextGenerationPipeline::commit_verified_prefix(int32_t count) {
    auto* cache = static_cast<GemmaKvCache*>(state_.get());
    std::vector<const void*> keys, values;
    for (int32_t layer = 0; layer < config_.num_layers; ++layer) {
        keys.push_back(
            prefill_module().device_ptr(gemma_expand_layer_name(config_.present_k_pattern, layer)));
        values.push_back(
            prefill_module().device_ptr(gemma_expand_layer_name(config_.present_v_pattern, layer)));
        if (!keys.back() || !values.back())
            throw std::runtime_error("Gemma4 MTP target omitted present K/V");
    }
    // Copy only the accepted causal prefix. Rejected proposal rows never enter
    // persistent cache; a later verification overwrites the temporary outputs.
    cache->append_prefill_kv(keys, values, count);
    decoder_->sync(); // Complete cache writes before the assistant's separate stream reads.
}

void GemmaTextGenerationPipeline::run_mtp_loop(GemmaISampler* sampler,
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
    std::optional<int32_t> correction;
    while (emitted < maximum) {
        const auto pending =
            correction ? *correction
                       : sampler->sample(logits.data(), config_.vocab_size, params).token_id;
        correction.reset();
        if (emit(pending))
            break;
        const auto proposal = propose_mtp(pending);
        const int32_t room = state_->max_length() - state_->position();
        // Keep the ordinary one-token target shape. A batched accepted row
        // can produce different FP16 K/V and logits, changing later target
        // decisions even when the first row is exact. The assistant still
        // proposes and is accepted/rejected; this verifier claims no speedup.
        auto consume = [&](int32_t token) {
            verify_mtp({token}, logits, target_hidden_);
            commit_verified_prefix(1);
        };
        consume(pending);
        ++mtp_stats_.verified;
        const auto expected = sampler->sample(logits.data(), config_.vocab_size, params).token_id;
        correction = expected;
        if (proposal == expected) {
            ++mtp_stats_.accepted;
            // The final emitted token need not be cached. At the capacity
            // boundary, the verified pending row predicts it next iteration.
            if (room >= 2) {
                correction.reset();
                consume(proposal);
                if (emit(proposal))
                    break;
            }
        }
        // Reuse the verified correction after rejection or at the capacity boundary.
    }
    std::clog << "Gemma4 native MTP: proposed=" << mtp_stats_.proposed
              << " accepted=" << mtp_stats_.accepted << " verified=" << mtp_stats_.verified << '\n';
}

} // namespace trtmc
