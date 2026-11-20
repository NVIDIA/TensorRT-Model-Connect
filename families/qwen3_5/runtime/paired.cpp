/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_5/runtime/paired.h"

#include "families/qwen3_5/runtime/plugin_helpers.h"
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
        throw std::runtime_error(std::string("Qwen3.5 paired tensor contract mismatch: ") + name);
}

void copy_from_device(void* destination, ITrtModule& module, const char* name, std::size_t bytes) {
    void* source = module.device_ptr(name);
    if (!source)
        throw std::runtime_error(std::string("Qwen3.5 paired missing output: ") + name);
    const auto status = cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Qwen3.5 paired output transfer: ") +
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
        throw std::runtime_error("Qwen3.5 paired nonfinite draft logits");
    double mass = 0;
    for (const float value : logits) {
        if (!std::isfinite(value))
            throw std::runtime_error("Qwen3.5 paired nonfinite draft logits");
        mass += std::exp(value - maximum);
    }
    const double log_normalizer = maximum + std::log(mass);
    std::vector<Candidate> result;
    for (std::size_t slot = 0; slot < count; ++slot)
        result.push_back({ids[slot], logits[ids[slot]] - log_normalizer});
    return result;
}
} // namespace

Qwen35PairedDraft::Qwen35PairedDraft(Qwen35PairedConfig config, std::unique_ptr<ITrtModule> draft)
    : config_(std::move(config)), draft_(std::move(draft)),
      feature_width_(static_cast<std::size_t>(config_.hidden_size) *
                     config_.feature_layers.size()) {
    if (!draft_ || !draft_->ok())
        throw std::runtime_error("Qwen3.5 paired draft engine failed to load");
    const int64_t rows = config_.rows, capacity = config_.capacity, vocab = config_.vocab_size;
    if (capacity <= 16 || config_.hidden_size <= 0 || vocab <= 0 || rows != 16 ||
        config_.feature_layers.empty())
        throw std::runtime_error("Qwen3.5 paired invalid draft geometry");
    require_tensor(*draft_, "draft_token_ids", DType::kInt32, {rows}, true);
    require_tensor(*draft_, "draft_positions", DType::kInt32, {rows}, true);
    require_tensor(*draft_, "target_hidden", DType::kFloat16,
                   {capacity, static_cast<int64_t>(feature_width_)}, true);
    require_tensor(*draft_, "draft_attention_mask", DType::kFloat32, {rows, capacity + rows}, true);
    require_tensor(*draft_, "draft_logits", DType::kFloat32, {rows, vocab}, false);
    if (feature_width_ > std::numeric_limits<std::size_t>::max() /
                             (static_cast<std::size_t>(capacity) * sizeof(std::uint16_t)))
        throw std::runtime_error("Qwen3.5 paired feature buffer size overflow");
    features_.resize(static_cast<std::size_t>(capacity) * feature_width_);
}

void Qwen35PairedDraft::reset(ITrtModule& target, std::size_t prompt_length, int32_t max_new_tokens,
                              const Qwen35SamplingParams& sampling) {
    if (prompt_length > static_cast<std::size_t>(config_.max_input_length) || max_new_tokens < 0 ||
        prompt_length + static_cast<std::size_t>(max_new_tokens) >
            static_cast<std::size_t>(config_.capacity))
        throw std::invalid_argument("Qwen3.5 paired prompt and generation exceed bundle capacity");
    if (tree() && !qwen35_is_greedy(sampling))
        throw std::invalid_argument("Qwen3.5 DFlash DDTree supports greedy decoding only");
    require_tensor(target, "target_hidden", DType::kFloat16,
                   {1, static_cast<int64_t>(feature_width_)}, false);
    seen_ = 0;
    std::fill(features_.begin(), features_.end(), 0);
}

void Qwen35PairedDraft::observe(ITrtModule& target, int32_t position) {
    if (position != seen_ || position >= config_.capacity)
        throw std::runtime_error("Qwen3.5 paired target feature position mismatch");
    copy_from_device(features_.data() + static_cast<std::size_t>(position) * feature_width_, target,
                     "target_hidden", feature_width_ * sizeof(std::uint16_t));
    ++seen_;
}

std::vector<Qwen35Proposal> Qwen35PairedDraft::propose(int32_t anchor,
                                                       const Qwen35SamplingParams& sampling,
                                                       int32_t budget, std::uint64_t& rng) {
    if (budget <= 0 || anchor < 0 || anchor >= config_.vocab_size || seen_ <= 0 ||
        seen_ >= config_.capacity)
        throw std::invalid_argument("Qwen3.5 paired invalid anchor or feature history");
    const int32_t rows = config_.rows, capacity = config_.capacity, vocab = config_.vocab_size;
    std::vector<int32_t> ids(rows, config_.mask_token), positions(rows);
    ids[0] = anchor;
    std::iota(positions.begin(), positions.end(), seen_);
    std::vector<float> mask(static_cast<std::size_t>(rows) * (capacity + rows),
                            -std::numeric_limits<float>::infinity());
    for (int32_t row = 0; row < rows; ++row) {
        // Pinned Qwen DFlash is bidirectional over all confirmed target
        // features and all proposal rows, irrespective of generic SWA labels.
        for (int32_t column = 0; column < seen_; ++column)
            mask[static_cast<std::size_t>(row) * (capacity + rows) + column] = 0;
        for (int32_t column = 0; column < rows; ++column)
            mask[static_cast<std::size_t>(row) * (capacity + rows) + capacity + column] = 0;
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
    try {
        draft_->forward_async(inputs);
        draft_->sync();
    } catch (...) {
        // Host inputs must outlive any queued upload even on failure.
        try {
            draft_->sync();
        } catch (...) {
        }
        throw;
    }
    std::vector<float> backbone(static_cast<std::size_t>(rows) * vocab);
    copy_from_device(backbone.data(), *draft_, "draft_logits", backbone.size() * sizeof(float));

    std::vector<Qwen35Proposal> result{{anchor, -1, 0, {}}};
    std::vector<std::vector<Candidate>> candidates;
    for (int32_t row = 1; row < rows && (tree() || row <= budget); ++row) {
        std::vector<float> logits(backbone.begin() + static_cast<std::size_t>(row) * vocab,
                                  backbone.begin() + static_cast<std::size_t>(row + 1) * vocab);
        if (tree()) {
            // Independent per-depth logits define all siblings at that depth.
            candidates.push_back(tree_candidates(logits, config_.top_k));
        } else {
            auto probabilities = qwen35_probabilities(logits.data(), vocab, sampling);
            const auto token = qwen35_draw_probabilities(probabilities, rng);
            result.push_back({token, row - 1, row, std::move(probabilities)});
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

std::unique_ptr<Qwen35PairedDraft>
create_qwen35_paired(const FamilyContext& context, int32_t hidden, int32_t layers, int32_t vocab,
                     int32_t capacity, const std::string& precision, int32_t tensor_parallel_size,
                     bool has_mtp) {
    if (!context.reader.find_section("native_paired.json")) {
        if (context.reader.find_section("native_draft.plan"))
            throw std::runtime_error("Qwen3.5 draft engine requires paired metadata");
        return nullptr;
    }
    if (has_mtp)
        throw std::runtime_error("Qwen3.5 cannot combine embedded MTP and DFlash");
    const auto json =
        nlohmann::json::parse(qwen3_5::require_text_section(context.reader, "native_paired.json"));
    if (!json.is_object() || json.size() != 11 || json.at("schema_version") != 1)
        throw std::runtime_error("Qwen3.5 paired invalid metadata schema");
    Qwen35PairedConfig config{
        json.at("execution_variant").get<std::string>(),
        json.at("hidden_size").get<int32_t>(),
        json.at("vocab_size").get<int32_t>(),
        json.at("max_cache_length").get<int32_t>(),
        json.at("max_input_length").get<int32_t>(),
        json.at("feature_layers").get<std::vector<int32_t>>(),
        json.at("draft_rows").get<int32_t>(),
        json.at("mask_token_id").get<int32_t>(),
        json.at("draft_top_k").get<int32_t>(),
        json.at("verify_size").get<int32_t>(),
    };
    auto layers_sorted = config.feature_layers;
    std::sort(layers_sorted.begin(), layers_sorted.end());
    const bool dflash = config.variant == "dflash";
    const bool tree = config.variant == "dflash_ddtree";
    if ((!dflash && !tree) || hidden != config.hidden_size || vocab != config.vocab_size ||
        capacity != config.capacity || config.max_input_length != capacity || capacity <= 16 ||
        precision != "fp16" || tensor_parallel_size != 1 || layers_sorted.empty() ||
        layers_sorted.front() < 0 || layers_sorted.back() >= layers ||
        std::adjacent_find(layers_sorted.begin(), layers_sorted.end()) != layers_sorted.end() ||
        config.rows != 16 || config.mask_token < 0 || config.mask_token >= vocab ||
        config.top_k != (tree ? 4 : 1) || config.verify_size != 16)
        throw std::runtime_error("Qwen3.5 paired metadata does not match the target and draft");
    const auto load = [&](const char* section) {
        const auto plan = qwen3_5::require_section(context.reader, section);
        return context.backend.create_module(plan.data(), plan.size(), {});
    };
    auto draft = load("native_draft.plan");
    return std::make_unique<Qwen35PairedDraft>(std::move(config), std::move(draft));
}

} // namespace trtmc
