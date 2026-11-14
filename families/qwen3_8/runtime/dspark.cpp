/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_8/runtime/dspark.h"

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
        throw std::runtime_error(std::string("Qwen3.8 DSpark tensor mismatch: ") + name);
}
void download(void* destination, ITrtModule& module, const char* name, std::size_t bytes) {
    auto* source = module.device_ptr(name);
    if (!source)
        throw std::runtime_error(std::string("Qwen3.8 DSpark missing output: ") + name);
    const auto status = cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Qwen3.8 DSpark transfer failed: ") +
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

Qwen38DSpark::Qwen38DSpark(Qwen38DSparkConfig config, std::unique_ptr<ITrtModule> draft,
                           std::unique_ptr<ITrtModule> heads, std::vector<char> embeddings)
    : config_(config), draft_(std::move(draft)), heads_(std::move(heads)),
      embeddings_(std::move(embeddings)),
      feature_width_(static_cast<std::size_t>(config.hidden) * config.feature_count) {
    if (config.hidden <= 0 || config.vocab <= 0 || config.capacity <= 8 ||
        config.feature_count <= 0 || config.mask_token < 0 || config.mask_token >= config.vocab ||
        config.rows != 7 + !config.sample_from_anchor || !draft_ || !draft_->ok() || !heads_ ||
        !heads_->ok())
        throw std::invalid_argument("Invalid native Qwen3.8 DSpark bundle geometry/engines");
    if (embeddings_.size() !=
        static_cast<std::size_t>(config.vocab) * config.hidden * sizeof(uint16_t))
        throw std::invalid_argument("Qwen3.8 DSpark target embedding size mismatch");
    const int64_t rows = config.rows, h = config.hidden, v = config.vocab, cap = config.capacity;
    require_tensor(*draft_, "inputs_embeds", DType::kFloat16, {rows, h}, true);
    require_tensor(*draft_, "draft_positions", DType::kInt32, {rows}, true);
    require_tensor(*draft_, "target_hidden", DType::kFloat16,
                   {cap, static_cast<int64_t>(feature_width_)}, true);
    require_tensor(*draft_, "draft_attention_mask", DType::kFloat32, {rows, cap + rows}, true);
    require_tensor(*draft_, "draft_logits", DType::kFloat32, {rows, v}, false);
    require_tensor(*draft_, "draft_hidden", DType::kFloat16, {rows, h}, false);
    require_tensor(*heads_, "previous_token", DType::kInt32, {1}, true);
    require_tensor(*heads_, "markov_bias", DType::kFloat32, {1, v}, false);
    if (config.confidence) {
        require_tensor(*heads_, "draft_hidden", DType::kFloat16, {1, h}, true);
        require_tensor(*heads_, "confidence", DType::kFloat32, {1, 1}, false);
    }
    features_.resize(static_cast<std::size_t>(cap) * feature_width_);
}

void Qwen38DSpark::reset(ITrtModule& target, std::size_t prompt_length, int32_t max_new_tokens) {
    if (prompt_length == 0 || max_new_tokens < 0 ||
        prompt_length > static_cast<std::size_t>(config_.capacity) ||
        static_cast<std::size_t>(max_new_tokens) >
            static_cast<std::size_t>(config_.capacity) - prompt_length)
        throw std::invalid_argument("Qwen3.8 DSpark prompt and generation exceed bundle capacity");
    require_tensor(target, "draft_features", DType::kFloat16,
                   {1, static_cast<int64_t>(feature_width_)}, false);
    seen_ = 0;
    std::fill(features_.begin(), features_.end(), 0);
}
void Qwen38DSpark::observe(ITrtModule& target) {
    if (seen_ >= config_.capacity)
        throw std::runtime_error("Qwen3.8 DSpark feature capacity exceeded");
    target.sync();
    download(features_.data() + static_cast<std::size_t>(seen_) * feature_width_, target,
             "draft_features", feature_width_ * sizeof(uint16_t));
    ++seen_;
}

std::vector<Qwen38DSparkProposal> Qwen38DSpark::propose(int32_t anchor,
                                                        const Qwen38SamplingParams& sampling,
                                                        const std::vector<int32_t>& history,
                                                        int32_t budget, uint64_t& rng) {
    if (anchor < 0 || anchor >= config_.vocab || seen_ <= 0 || seen_ >= config_.capacity ||
        budget <= 0)
        throw std::invalid_argument("Qwen3.8 DSpark invalid proposal state");
    const int32_t rows = config_.rows, cap = config_.capacity, h = config_.hidden,
                  v = config_.vocab;
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
        std::fill(row + cap, row + cap + (config_.causal ? i + 1 : rows), 0);
    }
    draft_->forward_async({
        {"inputs_embeds", host(embeddings.data(), {rows, h}, DType::kFloat16)},
        {"draft_positions", host(positions.data(), {rows}, DType::kInt32)},
        {"target_hidden",
         host(features_.data(), {cap, static_cast<int64_t>(feature_width_)}, DType::kFloat16)},
        {"draft_attention_mask", host(mask.data(), {rows, cap + rows}, DType::kFloat32)},
    });
    draft_->sync();
    std::vector<float> backbone(static_cast<std::size_t>(rows) * v);
    std::vector<uint16_t> hidden(static_cast<std::size_t>(rows) * h);
    download(backbone.data(), *draft_, "draft_logits", backbone.size() * sizeof(float));
    if (config_.confidence)
        download(hidden.data(), *draft_, "draft_hidden", hidden.size() * sizeof(uint16_t));
    std::vector<Qwen38DSparkProposal> proposals;
    auto proposal_history = history;
    int32_t previous = anchor;
    const int32_t offset = config_.sample_from_anchor ? 0 : 1;
    for (int32_t i = 0; i < std::min(7, budget); ++i) {
        const int32_t row = i + offset;
        TensorMap inputs{{"previous_token", host(&previous, {1}, DType::kInt32)}};
        if (config_.confidence)
            inputs["draft_hidden"] =
                host(hidden.data() + static_cast<std::size_t>(row) * h, {1, h}, DType::kFloat16);
        heads_->forward_async(inputs);
        heads_->sync();
        std::vector<float> logits(v);
        download(logits.data(), *heads_, "markov_bias", logits.size() * sizeof(float));
        for (int32_t token = 0; token < v; ++token)
            logits[token] += backbone[static_cast<std::size_t>(row) * v + token];
        qwen38_apply_repetition_penalty(logits, sampling.repetition_penalty, proposal_history);
        auto probabilities = qwen38_probabilities(logits.data(), v, sampling);
        previous = qwen38_draw_probabilities(probabilities, rng);
        proposals.push_back({previous, std::move(probabilities)});
        proposal_history.push_back(previous);
        // The admitted MC profile uses fixed block length (Edge scheduler=off).
        // Confidence is still computed, but never changes target probabilities.
    }
    return proposals;
}
} // namespace trtmc
