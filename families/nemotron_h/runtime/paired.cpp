/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_h/runtime/paired.h"

#include "families/nemotron_h/runtime/plugin_helpers.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {
void require_tensor(ITrtModule& module, const char* name, DType dtype,
                    const std::vector<int64_t>& shape, bool input) {
    if (!(input ? module.has_input(name) : module.has_output(name)) ||
        module.tensor_dtype(name) != dtype || module.tensor_shape(name) != shape)
        throw std::runtime_error(std::string("Nemotron paired tensor contract mismatch: ") + name);
}

void copy_from_device(void* destination, ITrtModule& module, const char* name, std::size_t bytes) {
    void* source = module.device_ptr(name);
    if (!source)
        throw std::runtime_error(std::string("Nemotron paired missing output: ") + name);
    const auto status = cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Nemotron paired output transfer: ") +
                                 cudaGetErrorString(status));
}

Tensor host_tensor(void* data, std::vector<int64_t> shape, DType dtype) {
    Tensor tensor;
    tensor.data = data;
    tensor.shape = std::move(shape);
    tensor.dtype = dtype;
    return tensor;
}

struct Candidate {
    int32_t token;
    double log_probability;
};

// DDTree scores the full-vocabulary distribution, not a top-K-renormalized one.
std::vector<Candidate> tree_candidates(const std::vector<float>& logits, int32_t top_k) {
    std::vector<int32_t> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    const auto higher = [&](int32_t a, int32_t b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    };
    const auto count = std::min<std::size_t>(top_k, ids.size());
    std::partial_sort(ids.begin(), ids.begin() + count, ids.end(), higher);
    const double maximum = logits[ids.front()];
    if (!std::isfinite(maximum))
        throw std::runtime_error("Nemotron paired nonfinite draft logits");
    double mass = 0;
    for (const float value : logits) {
        if (!std::isfinite(value))
            throw std::runtime_error("Nemotron paired nonfinite draft logits");
        mass += std::exp(value - maximum);
    }
    const double log_normalizer = maximum + std::log(mass);
    std::vector<Candidate> result;
    for (std::size_t slot = 0; slot < count; ++slot)
        result.push_back({ids[slot], logits[ids[slot]] - log_normalizer});
    return result;
}
} // namespace

NemotronHPairedDraft::NemotronHPairedDraft(NemotronHPairedConfig config,
                                           std::unique_ptr<ITrtModule> draft,
                                           std::unique_ptr<ITrtModule> markov)
    : config_(std::move(config)), draft_(std::move(draft)), markov_(std::move(markov)),
      feature_width_(static_cast<std::size_t>(config_.hidden_size) *
                     config_.feature_layers.size()) {
    if (!draft_ || !draft_->ok())
        throw std::runtime_error("Nemotron paired draft engine failed to load");
    const int64_t rows = config_.rows, capacity = config_.capacity, vocab = config_.vocab_size;
    require_tensor(*draft_, "draft_token_ids", DType::kInt32, {rows}, true);
    require_tensor(*draft_, "draft_positions", DType::kInt32, {rows}, true);
    require_tensor(*draft_, "target_hidden", DType::kFloat16,
                   {capacity, static_cast<int64_t>(feature_width_)}, true);
    require_tensor(*draft_, "draft_attention_mask", DType::kFloat32, {rows, capacity + rows}, true);
    require_tensor(*draft_, "draft_logits", DType::kFloat32, {rows, vocab}, false);
    if (config_.variant != "dflash") {
        if (!markov_ || !markov_->ok())
            throw std::runtime_error("Nemotron DSpark Markov engine failed to load");
        require_tensor(*markov_, "previous_token", DType::kInt32, {1}, true);
        require_tensor(*markov_, "markov_bias", DType::kFloat32, {1, vocab}, false);
    }
    features_.resize(static_cast<std::size_t>(capacity) * feature_width_);
}

void NemotronHPairedDraft::reset(ITrtModule& target, std::size_t prompt_length,
                                 int32_t max_new_tokens, const NemotronHSamplingParams& sampling) {
    if (prompt_length > static_cast<std::size_t>(config_.max_input_length) || max_new_tokens < 0 ||
        prompt_length + static_cast<std::size_t>(max_new_tokens) >
            static_cast<std::size_t>(config_.capacity))
        throw std::invalid_argument("Nemotron paired prompt and generation exceed bundle capacity");
    if ((tree() || config_.variant == "dflash") && !nemotron_h_is_greedy(sampling))
        throw std::invalid_argument("Nemotron DFlash and DSpark tree support greedy decoding only");
    require_tensor(target, "target_hidden", DType::kFloat16,
                   {1, static_cast<int64_t>(feature_width_)}, false);
    seen_ = 0;
    std::fill(features_.begin(), features_.end(), 0);
}

void NemotronHPairedDraft::observe(ITrtModule& target, int32_t position) {
    if (position != seen_ || position >= config_.capacity)
        throw std::runtime_error("Nemotron paired target feature position mismatch");
    copy_from_device(features_.data() + static_cast<std::size_t>(position) * feature_width_, target,
                     "target_hidden", feature_width_ * sizeof(std::uint16_t));
    ++seen_;
}

std::vector<NemotronHProposal>
NemotronHPairedDraft::propose(int32_t anchor, const NemotronHSamplingParams& sampling,
                              std::uint64_t& rng) {
    if (anchor < 0 || anchor >= config_.vocab_size || seen_ <= 0 || seen_ >= config_.capacity)
        throw std::invalid_argument("Nemotron paired invalid anchor or feature history");
    const int32_t rows = config_.rows, capacity = config_.capacity, vocab = config_.vocab_size;
    std::vector<int32_t> ids(rows, config_.mask_token), positions(rows);
    ids[0] = anchor;
    std::iota(positions.begin(), positions.end(), seen_);
    std::vector<float> mask(static_cast<std::size_t>(rows) * (capacity + rows),
                            -std::numeric_limits<float>::infinity());
    for (int32_t row = 0; row < rows; ++row) {
        const int32_t earliest =
            config_.sliding_window ? std::max(0, positions[row] - config_.sliding_window + 1) : 0;
        for (int32_t column = earliest; column < seen_; ++column)
            mask[static_cast<std::size_t>(row) * (capacity + rows) + column] = 0;
        for (int32_t column = 0; column < rows; ++column) {
            if (!config_.sliding_window || (column <= row && positions[column] >= earliest))
                mask[static_cast<std::size_t>(row) * (capacity + rows) + capacity + column] = 0;
        }
    }
    TensorMap inputs{
        {"draft_token_ids", host_tensor(ids.data(), {rows}, DType::kInt32)},
        {"draft_positions", host_tensor(positions.data(), {rows}, DType::kInt32)},
        {"target_hidden",
         host_tensor(features_.data(), {capacity, static_cast<int64_t>(feature_width_)},
                     DType::kFloat16)},
        {"draft_attention_mask",
         host_tensor(mask.data(), {rows, capacity + rows}, DType::kFloat32)},
    };
    draft_->forward_async(inputs);
    draft_->sync();
    std::vector<float> backbone(static_cast<std::size_t>(rows) * vocab);
    copy_from_device(backbone.data(), *draft_, "draft_logits", backbone.size() * sizeof(float));

    std::vector<NemotronHProposal> result{{anchor, -1, 0, {}}};
    std::vector<std::vector<Candidate>> candidates;
    int32_t previous = anchor;
    for (int32_t row = 1; row < rows; ++row) {
        std::vector<float> logits(backbone.begin() + static_cast<std::size_t>(row) * vocab,
                                  backbone.begin() + static_cast<std::size_t>(row + 1) * vocab);
        if (markov_) {
            markov_->forward_async(
                {{"previous_token", host_tensor(&previous, {1}, DType::kInt32)}});
            markov_->sync();
            std::vector<float> bias(vocab);
            copy_from_device(bias.data(), *markov_, "markov_bias", bias.size() * sizeof(float));
            for (int32_t token = 0; token < vocab; ++token)
                logits[token] += bias[token];
        }
        if (tree()) {
            // Edge 0.11 DDTree uses the Markov-corrected greedy CHAIN logits at
            // each depth, including on siblings. Recomputing per-parent Markov
            // logits would implement a different proposal algorithm.
            candidates.push_back(tree_candidates(logits, config_.top_k));
            previous = candidates.back().front().token;
        } else {
            auto probabilities = nemotron_h_probabilities(logits.data(), vocab, sampling);
            previous = nemotron_h_sample_probabilities(probabilities, rng);
            result.push_back({previous, row - 1, row, std::move(probabilities)});
        }
    }
    if (!tree())
        return result;

    std::vector<double> scores{0};
    std::vector<std::size_t> next_slot{0};
    while (result.size() < static_cast<std::size_t>(config_.verify_size)) {
        int32_t best_parent = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t parent = 0; parent < result.size(); ++parent) {
            const auto depth = static_cast<std::size_t>(result[parent].depth);
            if (depth >= candidates.size() || next_slot[parent] >= candidates[depth].size())
                continue;
            const double score =
                scores[parent] + candidates[depth][next_slot[parent]].log_probability;
            // Parent order implements the pinned source's deterministic tie break.
            if (score > best_score) {
                best_score = score;
                best_parent = static_cast<int32_t>(parent);
            }
        }
        if (best_parent < 0)
            break;
        const auto depth = result[best_parent].depth;
        const auto candidate = candidates[depth][next_slot[best_parent]++];
        result.push_back({candidate.token, best_parent, depth + 1, {}});
        scores.push_back(best_score);
        next_slot.push_back(0);
    }
    return result;
}

std::unique_ptr<NemotronHPairedDraft> create_nemotron_h_paired(const FamilyContext& context,
                                                               int32_t hidden, int32_t layers,
                                                               int32_t vocab, int32_t capacity,
                                                               const std::string& precision,
                                                               int32_t tensor_parallel_size) {
    if (!context.reader.find_section("native_paired.json"))
        return nullptr;
    const auto json = nlohmann::json::parse(
        nemotron_h::require_text_section(context.reader, "native_paired.json"));
    if (!json.is_object() || json.size() != 13 || json.at("schema_version") != 1)
        throw std::runtime_error("Nemotron paired invalid metadata schema");
    NemotronHPairedConfig config{
        json.at("execution_variant").get<std::string>(),
        json.at("hidden_size").get<int32_t>(),
        json.at("vocab_size").get<int32_t>(),
        json.at("max_cache_length").get<int32_t>(),
        json.at("max_input_length").get<int32_t>(),
        json.at("feature_layers").get<std::vector<int32_t>>(),
        json.at("draft_rows").get<int32_t>(),
        json.at("mask_token_id").get<int32_t>(),
        json.at("sliding_window").get<int32_t>(),
        json.at("draft_top_k").get<int32_t>(),
        json.at("verify_size").get<int32_t>(),
        json.at("markov_rank").get<int32_t>(),
    };
    auto layers_sorted = config.feature_layers;
    std::sort(layers_sorted.begin(), layers_sorted.end());
    const bool dflash = config.variant == "dflash";
    const bool tree = config.variant == "dspark_tree";
    if ((!dflash && !tree && config.variant != "dspark") || hidden != config.hidden_size ||
        vocab != config.vocab_size || capacity != config.capacity ||
        config.max_input_length != std::min(capacity, 1024) || precision != "fp16" ||
        tensor_parallel_size != 1 || layers_sorted.empty() || layers_sorted.front() < 0 ||
        layers_sorted.back() >= layers ||
        std::adjacent_find(layers_sorted.begin(), layers_sorted.end()) != layers_sorted.end() ||
        config.rows != (dflash ? 16 : 9) || config.mask_token < 0 || config.mask_token >= vocab ||
        config.sliding_window != (dflash ? 0 : 1024) || config.top_k != (tree ? 4 : 1) ||
        config.verify_size != (dflash || tree ? 16 : 9) ||
        (dflash ? config.markov_rank != 0 : config.markov_rank <= 0))
        throw std::runtime_error("Nemotron paired metadata does not match the target and draft");
    const auto load = [&](const char* section) {
        const auto plan = nemotron_h::require_section(context.reader, section);
        return context.backend.create_module(plan.data(), plan.size(), {});
    };
    auto draft = load("native_draft.plan");
    auto markov = dflash ? nullptr : load("native_markov.plan");
    return std::make_unique<NemotronHPairedDraft>(std::move(config), std::move(draft),
                                                  std::move(markov));
}

} // namespace trtmc
