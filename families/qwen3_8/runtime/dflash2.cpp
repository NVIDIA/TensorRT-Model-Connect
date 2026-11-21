/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_8/runtime/dflash2.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {
namespace {
void require_tensor(ITrtModule& module, const char* name, DType dtype,
                    const std::vector<int64_t>& shape, bool input) {
    if (!(input ? module.has_input(name) : module.has_output(name)) ||
        module.tensor_dtype(name) != dtype || module.tensor_shape(name) != shape)
        throw std::runtime_error(std::string("Qwen3.8 DFlash2 tensor mismatch: ") + name);
}
void download(void* destination, ITrtModule& module, const char* name, std::size_t bytes) {
    auto* source = module.device_ptr(name);
    if (!source)
        throw std::runtime_error(std::string("Qwen3.8 DFlash2 missing output: ") + name);
    const auto status = cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Qwen3.8 DFlash2 transfer failed: ") +
                                 cudaGetErrorString(status));
}
Tensor host(void* data, std::vector<int64_t> shape, DType dtype) {
    Tensor tensor;
    tensor.data = data;
    tensor.shape = std::move(shape);
    tensor.dtype = dtype;
    return tensor;
}
} // namespace

std::vector<float> qwen38_dflash2_probabilities(const std::vector<float>& scores,
                                                const std::vector<int32_t>& ids, int32_t vocab,
                                                const Qwen38SamplingParams& sampling) {
    if (scores.empty() || scores.size() != ids.size() || vocab <= 0)
        throw std::invalid_argument("Invalid DFlash2 selector support");
    const float temperature =
        std::isfinite(sampling.temperature) ? std::max(0.0F, sampling.temperature) : 1.0F;
    // Edge 0.11 shouldUseNonGreedySampling, with its default automatic proposal policy.
    const bool non_greedy = temperature >= 1e-3F && sampling.top_k != 1 &&
                            (sampling.top_k > 1 || sampling.top_p < 1.0F - 1e-6F ||
                             (temperature > 1e-3F && std::fabs(temperature - 1.0F) > 1e-3F));
    std::vector<float> result(static_cast<std::size_t>(vocab), 0);
    std::size_t best = 0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (!std::isfinite(scores[i]) || ids[i] < 0 || ids[i] >= vocab)
            throw std::runtime_error("Invalid DFlash2 selector score/token");
        if (scores[i] > scores[best] || (scores[i] == scores[best] && ids[i] < ids[best]))
            best = i;
    }
    if (!non_greedy) {
        result[ids[best]] = 1;
        return result;
    }
    double mass = 0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        const float value = std::exp((scores[i] - scores[best]) / temperature);
        result[ids[i]] = value;
        mass += value;
    }
    for (const auto token : ids)
        result[token] = static_cast<float>(result[token] / mass);
    return result;
}

Qwen38DFlash2::Qwen38DFlash2(Qwen38DFlash2Config config, std::unique_ptr<ITrtModule> draft,
                             std::unique_ptr<ITrtModule> selector, std::vector<char> embeddings)
    : config_(config), draft_(std::move(draft)), selector_(std::move(selector)),
      embeddings_(std::move(embeddings)),
      feature_width_(static_cast<std::size_t>(config.hidden) * config.feature_count) {
    if (config.hidden <= 0 || config.vocab < 16 || config.capacity <= 8 ||
        config.feature_count <= 0 || config.mask_token < 0 || config.mask_token >= config.vocab ||
        config.window <= 0 || config.rank <= 0 || config.top_k != 16 || !draft_ || !draft_->ok() ||
        !selector_ || !selector_->ok())
        throw std::invalid_argument("Invalid native Qwen3.8 DFlash2 bundle geometry/engines");
    if (embeddings_.size() !=
        static_cast<std::size_t>(config.vocab) * config.hidden * sizeof(uint16_t))
        throw std::invalid_argument("Qwen3.8 DFlash2 target embedding size mismatch");
    const int64_t h = config.hidden, cap = config.capacity, k = config.top_k, rank = config.rank;
    require_tensor(*draft_, "inputs_embeds", DType::kFloat16, {8, h}, true);
    require_tensor(*draft_, "draft_positions", DType::kInt32, {8}, true);
    require_tensor(*draft_, "target_hidden", DType::kFloat16,
                   {cap, static_cast<int64_t>(feature_width_)}, true);
    require_tensor(*draft_, "draft_attention_mask", DType::kFloat32, {8, cap + 8}, true);
    require_tensor(*draft_, "candidate_ids", DType::kInt32, {7, k}, false);
    require_tensor(*draft_, "unary_values", DType::kFloat32, {7, k}, false);
    require_tensor(*draft_, "projected_hidden", DType::kFloat16, {7, rank}, false);
    require_tensor(*selector_, "previous_token", DType::kInt32, {1}, true);
    require_tensor(*selector_, "candidate_ids", DType::kInt32, {k}, true);
    require_tensor(*selector_, "unary_values", DType::kFloat32, {1, k}, true);
    require_tensor(*selector_, "projected_hidden", DType::kFloat16, {1, rank}, true);
    require_tensor(*selector_, "selector_scores", DType::kFloat32, {1, k}, false);
    features_.resize(static_cast<std::size_t>(cap) * feature_width_);
}

void Qwen38DFlash2::reset(ITrtModule& target, std::size_t prompt_length, int32_t max_new_tokens) {
    if (prompt_length == 0 || max_new_tokens < 0 ||
        prompt_length > static_cast<std::size_t>(config_.capacity) ||
        static_cast<std::size_t>(max_new_tokens) >
            static_cast<std::size_t>(config_.capacity) - prompt_length)
        throw std::invalid_argument("Qwen3.8 DFlash2 prompt and generation exceed bundle capacity");
    require_tensor(target, "draft_features", DType::kFloat16,
                   {1, static_cast<int64_t>(feature_width_)}, false);
    seen_ = 0;
    std::fill(features_.begin(), features_.end(), 0);
}
void Qwen38DFlash2::observe(ITrtModule& target) {
    if (seen_ >= config_.capacity)
        throw std::runtime_error("Qwen3.8 DFlash2 feature capacity exceeded");
    target.sync();
    download(features_.data() + static_cast<std::size_t>(seen_) * feature_width_, target,
             "draft_features", feature_width_ * sizeof(uint16_t));
    ++seen_;
}

std::vector<Qwen38DraftProposal> Qwen38DFlash2::propose(int32_t anchor,
                                                        const Qwen38SamplingParams& sampling,
                                                        const std::vector<int32_t>& /*history*/,
                                                        int32_t budget, uint64_t& rng) {
    if (anchor < 0 || anchor >= config_.vocab || seen_ <= 0 || seen_ >= config_.capacity ||
        budget <= 0)
        throw std::invalid_argument("Qwen3.8 DFlash2 invalid proposal state");
    const int32_t rows = 8, cap = config_.capacity, h = config_.hidden, v = config_.vocab;
    std::vector<uint16_t> embeddings(static_cast<std::size_t>(rows) * h);
    for (int32_t i = 0; i < rows; ++i) {
        const auto token = i == 0 ? anchor : config_.mask_token;
        std::memcpy(embeddings.data() + static_cast<std::size_t>(i) * h,
                    embeddings_.data() + static_cast<std::size_t>(token) * h * sizeof(uint16_t),
                    static_cast<std::size_t>(h) * sizeof(uint16_t));
    }
    std::vector<int32_t> positions(rows);
    std::iota(positions.begin(), positions.end(), seen_);
    std::vector<float> mask(static_cast<std::size_t>(rows) * (cap + rows),
                            -std::numeric_limits<float>::infinity());
    for (int32_t i = 0; i < rows; ++i) {
        auto* row = mask.data() + static_cast<std::size_t>(i) * (cap + rows);
        const auto earliest = std::max(0, positions[i] - config_.window + 1);
        if (earliest < seen_)
            std::fill(row + earliest, row + seen_, 0);
        std::fill(row + cap, row + cap + rows, 0);
    }
    draft_->forward_async({
        {"inputs_embeds", host(embeddings.data(), {rows, h}, DType::kFloat16)},
        {"draft_positions", host(positions.data(), {rows}, DType::kInt32)},
        {"target_hidden",
         host(features_.data(), {cap, static_cast<int64_t>(feature_width_)}, DType::kFloat16)},
        {"draft_attention_mask", host(mask.data(), {rows, cap + rows}, DType::kFloat32)},
    });
    draft_->sync();
    const int32_t k = config_.top_k, rank = config_.rank;
    std::vector<int32_t> ids(7 * k);
    std::vector<float> unary(7 * k);
    std::vector<uint16_t> projected(7 * rank);
    download(ids.data(), *draft_, "candidate_ids", ids.size() * sizeof(int32_t));
    download(unary.data(), *draft_, "unary_values", unary.size() * sizeof(float));
    download(projected.data(), *draft_, "projected_hidden", projected.size() * sizeof(uint16_t));
    std::vector<Qwen38DraftProposal> proposals;
    int32_t previous = anchor;
    for (int32_t row = 0; row < std::min(7, budget); ++row) {
        std::vector<int32_t> candidates(ids.begin() + row * k, ids.begin() + (row + 1) * k);
        // Check before the selector's GPU gathers to avoid out-of-range reads.
        auto unique = candidates;
        std::sort(unique.begin(), unique.end());
        if (unique.front() < 0 || unique.back() >= v ||
            std::adjacent_find(unique.begin(), unique.end()) != unique.end())
            throw std::runtime_error("Invalid Qwen3.8 DFlash2 candidate IDs");
        selector_->forward_async({
            {"previous_token", host(&previous, {1}, DType::kInt32)},
            {"candidate_ids", host(candidates.data(), {k}, DType::kInt32)},
            {"unary_values", host(unary.data() + row * k, {1, k}, DType::kFloat32)},
            {"projected_hidden", host(projected.data() + row * rank, {1, rank}, DType::kFloat16)},
        });
        selector_->sync();
        std::vector<float> scores(k);
        download(scores.data(), *selector_, "selector_scores", scores.size() * sizeof(float));
        auto probabilities = qwen38_dflash2_probabilities(scores, candidates, v, sampling);
        previous = qwen38_draw_probabilities(probabilities, rng);
        proposals.push_back({previous, std::move(probabilities)});
    }
    return proposals;
}
} // namespace trtmc
