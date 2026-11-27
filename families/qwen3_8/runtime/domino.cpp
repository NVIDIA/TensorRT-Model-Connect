/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_8/runtime/domino.h"

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
        throw std::runtime_error(std::string("Qwen3.6 Domino tensor mismatch: ") + name);
}
void download(void* destination, ITrtModule& module, const char* name, std::size_t bytes) {
    auto* source = module.device_ptr(name);
    if (!source)
        throw std::runtime_error(std::string("Qwen3.6 Domino missing output: ") + name);
    const auto status = cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Qwen3.6 Domino transfer failed: ") +
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

Qwen38Domino::Qwen38Domino(Qwen38DominoConfig config, std::unique_ptr<ITrtModule> draft,
                           std::unique_ptr<ITrtModule> correction, std::vector<char> embeddings)
    : config_(config), draft_(std::move(draft)), correction_(std::move(correction)),
      embeddings_(std::move(embeddings)),
      feature_width_(static_cast<std::size_t>(config.hidden) * config.feature_count) {
    if (config.hidden <= 0 || config.vocab < 16 || config.capacity < 16 ||
        config.feature_count <= 0 || config.mask_token < 0 || config.mask_token >= config.vocab ||
        config.pool != std::min(config.vocab, 2048) || config.rank <= 0 || config.state <= 0 ||
        (config.top_k != 1 && config.top_k != 4) || !draft_ || !draft_->ok() || !correction_ ||
        !correction_->ok())
        throw std::invalid_argument("Invalid native Qwen3.6 Domino bundle geometry/engines");
    if (embeddings_.size() !=
        static_cast<std::size_t>(config.vocab) * config.hidden * sizeof(uint16_t))
        throw std::invalid_argument("Qwen3.6 Domino target embedding size mismatch");
    const int64_t h = config.hidden, cap = config.capacity, k = config.pool, r = config.rank;
    require_tensor(*draft_, "inputs_embeds", DType::kFloat16, {16, h}, true);
    require_tensor(*draft_, "draft_positions", DType::kInt32, {16}, true);
    require_tensor(*draft_, "target_hidden", DType::kFloat16,
                   {cap, static_cast<int64_t>(feature_width_)}, true);
    require_tensor(*draft_, "draft_attention_mask", DType::kFloat32, {16, cap + 16}, true);
    require_tensor(*draft_, "pool_ids", DType::kInt32, {k}, false);
    require_tensor(*draft_, "pool_base_logits", DType::kFloat32, {15, k}, false);
    require_tensor(*draft_, "pool_codebook", DType::kFloat16, {k, r}, false);
    require_tensor(*draft_, "proposal_zproj", DType::kFloat16, {14, r}, false);
    require_tensor(*correction_, "token_embedding", DType::kFloat16, {1, h}, true);
    require_tensor(*correction_, "previous_state", DType::kFloat16, {1, config.state}, true);
    require_tensor(*correction_, "next_state", DType::kFloat16, {1, config.state}, false);
    require_tensor(*correction_, "pool_codebook", DType::kFloat16, {k, r}, true);
    require_tensor(*correction_, "z_projection", DType::kFloat16, {1, r}, true);
    require_tensor(*correction_, "base_scores", DType::kFloat32, {1, k}, true);
    require_tensor(*correction_, "corrected_pool_scores", DType::kFloat32, {1, k}, false);
    features_.resize(static_cast<std::size_t>(cap) * feature_width_);
}

void Qwen38Domino::validate_sampling(const Qwen38SamplingParams& sampling) const {
    if (!qwen38_greedy_distribution(sampling))
        throw std::invalid_argument("Qwen3.6 Domino supports greedy decoding only");
}

void Qwen38Domino::reset(ITrtModule& target, std::size_t prompt_length, int32_t max_new_tokens) {
    if (prompt_length == 0 || max_new_tokens < 0 ||
        prompt_length > static_cast<std::size_t>(config_.capacity) ||
        static_cast<std::size_t>(max_new_tokens) >
            static_cast<std::size_t>(config_.capacity) - prompt_length)
        throw std::invalid_argument("Qwen3.6 Domino prompt and generation exceed bundle capacity");
    require_tensor(target, "draft_features", DType::kFloat16,
                   {1, static_cast<int64_t>(feature_width_)}, false);
    seen_ = 0;
    std::fill(features_.begin(), features_.end(), 0);
}
void Qwen38Domino::observe(ITrtModule& target) {
    if (seen_ >= config_.capacity)
        throw std::runtime_error("Qwen3.6 Domino feature capacity exceeded");
    target.sync();
    download(features_.data() + static_cast<std::size_t>(seen_) * feature_width_, target,
             "draft_features", feature_width_ * sizeof(uint16_t));
    ++seen_;
}

Qwen38DraftRound Qwen38Domino::propose(int32_t anchor, const Qwen38SamplingParams& sampling,
                                       const std::vector<int32_t>& /*history*/, int32_t budget,
                                       uint64_t& /*rng*/) {
    validate_sampling(sampling);
    if (anchor < 0 || anchor >= config_.vocab || seen_ <= 0 || seen_ >= config_.capacity ||
        budget <= 0)
        throw std::invalid_argument("Qwen3.6 Domino invalid proposal state");
    const int32_t rows = 16, cap = config_.capacity, h = config_.hidden, v = config_.vocab;
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
        std::fill(row, row + seen_, 0);
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
    const int32_t k = config_.pool, r = config_.rank, s = config_.state;
    std::vector<int32_t> ids(k);
    std::vector<float> scores(15 * k);
    std::vector<uint16_t> book(static_cast<std::size_t>(k) * r), z(14 * r), state(s, 0);
    download(ids.data(), *draft_, "pool_ids", k * sizeof(int32_t));
    download(scores.data(), *draft_, "pool_base_logits", scores.size() * sizeof(float));
    download(book.data(), *draft_, "pool_codebook", book.size() * sizeof(uint16_t));
    download(z.data(), *draft_, "proposal_zproj", z.size() * sizeof(uint16_t));
    auto sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    if (sorted.front() < 0 || sorted.back() >= v ||
        std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
        throw std::runtime_error("Invalid Qwen3.6 Domino candidate IDs");

    auto best = [&](int row) {
        const auto first = scores.begin() + row * k;
        if (!std::all_of(first, first + k, [](float f) { return std::isfinite(f); }))
            throw std::runtime_error("Non-finite Qwen3.6 Domino pool score");
        // A tied score selects the first pool slot, not the smallest token ID.
        return ids[std::max_element(first, first + k) - first];
    };
    auto advance = [&](int32_t token, int row, bool emit_scores) {
        auto* emb = embeddings_.data() + static_cast<std::size_t>(token) * h * sizeof(uint16_t);
        auto* logits = scores.data() + row * k;
        correction_->forward_async({
            {"token_embedding", host(emb, {1, h}, DType::kFloat16)},
            {"previous_state", host(state.data(), {1, s}, DType::kFloat16)},
            {"pool_codebook", host(book.data(), {k, r}, DType::kFloat16)},
            {"z_projection", host(z.data() + (row - 1) * r, {1, r}, DType::kFloat16)},
            {"base_scores", host(logits, {1, k}, DType::kFloat32)},
        });
        correction_->sync();
        download(state.data(), *correction_, "next_state", s * sizeof(uint16_t));
        if (emit_scores)
            download(logits, *correction_, "corrected_pool_scores", k * sizeof(float));
    };
    // The GRU is seeded once per round, not carried through verification.
    auto token = best(0);
    advance(anchor, 1, false);
    constexpr int depth_limit = 15;
    for (int row = 1; row < depth_limit; ++row) {
        advance(token, row, true);
        token = best(row);
    }

    // DDTree expands depth-wise distributions produced by the greedy GRU
    // rollout. It does not rerun that GRU separately for every alternative.
    struct Candidate {
        int32_t token;
        float logp;
    };
    std::vector<std::vector<Candidate>> depths;
    for (int row = 0; row < depth_limit; ++row) {
        const auto* base = scores.data() + row * k;
        std::vector<int32_t> slots(k);
        std::iota(slots.begin(), slots.end(), 0);
        std::partial_sort(
            slots.begin(), slots.begin() + config_.top_k, slots.end(),
            [&](auto a, auto b) { return base[a] == base[b] ? a < b : base[a] > base[b]; });
        const float max = base[slots[0]];
        float mass = 0;
        for (int col = 0; col < k; ++col)
            mass += std::exp(base[col] - max);
        const float logmass = std::log(mass);
        auto& level = depths.emplace_back();
        for (int slot = 0; slot < config_.top_k; ++slot) {
            const auto col = slots[slot];
            level.push_back({ids[col], (base[col] - max) - logmass});
        }
    }
    Qwen38VerifyTree tree{{anchor, -1}};
    struct Frontier {
        int depth;
        int next;
        float logp;
    };
    std::vector<Frontier> nodes{{0, 0, 0}};
    const int verify = config_.top_k == 1 ? 16 : 60;
    while (static_cast<int>(tree.size()) < verify) {
        int parent = -1;
        float highest = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            const auto& n = nodes[i];
            if (n.depth == depth_limit || n.next == config_.top_k)
                continue;
            const auto value = n.logp + depths[n.depth][n.next].logp;
            if (value > highest) {
                highest = value;
                parent = i;
            }
        }
        if (parent < 0)
            break;
        auto& n = nodes[parent];
        tree.push_back({depths[n.depth][n.next++].token, parent});
        const int child_depth = n.depth + 1;
        nodes.push_back({child_depth, 0, highest});
    }
    return tree;
}
} // namespace trtmc
