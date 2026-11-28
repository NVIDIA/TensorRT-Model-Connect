/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "native_lilicorr.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace trtmc::nemotron_bolt {
namespace {
constexpr int H = 4096, V = 131072, N = 7, K = 8;
template <class T>
std::vector<T> values(const TensorMap& out, const std::string& name, DType type,
                      std::vector<int64_t> shape) {
    const auto& t = out.at(name);
    if (!t.data || t.dtype != type || t.shape != shape)
        throw std::runtime_error("Invalid Bolt draft tensor " + name);
    std::vector<T> result(t.nbytes() / sizeof(T));
    std::memcpy(result.data(), t.data, t.nbytes());
    return result;
}
uint16_t fp16(float x) {
    const auto h = __float2half_rn(x);
    uint16_t val;
    std::memcpy(&val, &h, sizeof(val));
    return val;
}
float unpack(uint16_t v) {
    __half h;
    std::memcpy(&h, &v, sizeof(h));
    return __half2float(h);
}
void fetch(const std::vector<char>& table, int id, float* out) {
    if (id < 0 || id >= V || table.size() != size_t(V) * H * sizeof(uint16_t))
        throw std::runtime_error("Invalid Bolt draft token embedding");
    for (int i = 0; i < H; ++i) {
        __nv_bfloat16 b;
        std::memcpy(&b, table.data() + (size_t(id) * H + i) * sizeof(b), sizeof(b));
        // Released draft embedding table is converted to FP16 before the residual stream.
        out[i] = unpack(fp16(__bfloat162float(b)));
    }
}
} // namespace
NativeLiLiCorr::NativeLiLiCorr(IBackend& backend, const NativeLanguage::PlanReader& read)
    : proposal_(backend, read) {
    auto load = [&](const char* name) {
        const auto bytes = read(name);
        auto m = backend.create_module(bytes.data(), bytes.size(), {});
        if (!m || !m->ok())
            throw std::runtime_error(std::string("Cannot load native Bolt LiLiCorr ") + name);
        return m;
    };
    project_ = load("project");
    vocab_ = load("vocab");
    lattice_ = load("lattice");
}
void NativeLiLiCorr::reset(int first) {
    proposal_.reset(first);
    anchor_.clear();
}
void NativeLiLiCorr::accepted(const std::vector<uint16_t>& features) {
    if (features.size() != 6 * H)
        throw std::invalid_argument("Bolt requires all six target feature rows");
    auto x = values<uint16_t>(
        project_->forward(
            {{"target_features",
              {const_cast<uint16_t*>(features.data()), {1, 6 * H}, DType::kBFloat16}}}),
        "accepted_hidden", DType::kFloat16, {1, H});
    proposal_.append(x);
    anchor_.resize(H);
    std::transform(x.begin(), x.end(), anchor_.begin(), unpack);
}
std::array<BoltDraftChoice, N> NativeLiLiCorr::propose(int root, const std::vector<char>& table,
                                                       BoltVerificationSampler& sampler) {
    std::vector<float> block((N + 1) * H);
    fetch(table, root, block.data());
    fetch(table, 990, block.data() + H);
    for (int j = 2; j <= N; ++j)
        std::copy_n(block.data() + H, H, block.data() + j * H);
    auto hidden = proposal_.predict(block);
    std::array<int32_t, N * K> ids{};
    std::array<float, N * K> lp{};
    std::vector<uint16_t> emb(N * K * H);
    std::vector<float> token(H);
    for (int i = 0; i < N; ++i) {
        const auto out = vocab_->forward(
            {{"prediction_hidden", {hidden.data() + i * H, {1, H}, DType::kFloat16}}});
        const auto v = values<int32_t>(out, "candidate_ids", DType::kInt32, {1, K});
        const auto w = values<float>(out, "candidate_log_probs", DType::kFloat32, {1, K});
        for (int j = 0; j < K; ++j) {
            int ix = i * K + j;
            ids[ix] = v[j];
            lp[ix] = w[j];
            fetch(table, v[j], token.data());
            std::transform(token.begin(), token.end(), emb.data() + ix * H, fp16);
        }
    }
    int valid = !anchor_.empty();
    std::vector<float> empty;
    if (!valid)
        empty.assign(H, 0.f);
    auto scores = values<float>(
        lattice_->forward(
            {{"candidate_embeddings", {emb.data(), {N * K, H}, DType::kFloat16}},
             {"candidate_log_probs", {lp.data(), {N, K}, DType::kFloat32}},
             {"pass_hidden", {hidden.data(), {N, H}, DType::kFloat16}},
             {"anchor_hidden", {valid ? anchor_.data() : empty.data(), {1, H}, DType::kFloat32}},
             {"anchor_valid", {&valid, {1, 1}, DType::kInt32}}}),
        "transition_scores", DType::kFloat32, {N, K, K});
    std::array<BoltDraftChoice, N> out;
    int previous = 0;
    for (int i = 0; i < N; ++i) {
        out[i] = sampler.propose(ids.data() + i * K, scores.data() + (i * K + previous) * K);
        previous = out[i].column;
    }
    return out;
}
} // namespace trtmc::nemotron_bolt
