/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/cache_policy.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using trtmc::hstu::CachePolicy;
using trtmc::hstu::CacheReuseKind;
using trtmc::hstu::CacheReuseReason;
using trtmc::hstu::HistorySignature;
using trtmc::hstu::plan_cache_reuse;

std::size_t checks = 0;

void expect(const HistorySignature* cached, const HistorySignature& incoming,
            const CachePolicy& policy, CacheReuseKind kind, CacheReuseReason reason,
            std::size_t reused_tokens, const char* name) {
    const auto result = plan_cache_reuse(cached, incoming, policy);
    if (result.kind != kind || result.reason != reason || result.reused_tokens != reused_tokens)
        throw std::runtime_error(std::string(name) + ": got " +
                                 trtmc::hstu::cache_reuse_reason_name(result.reason));
    ++checks;
}

void reject(const HistorySignature& cached, const HistorySignature& incoming,
            CacheReuseReason reason, const char* name, const CachePolicy& policy = {}) {
    expect(&cached, incoming, policy, CacheReuseKind::kRecompute, reason, 0, name);
}

HistorySignature history() {
    HistorySignature result;
    result.model_namespace = "model-artifact-v1";
    result.feature_version = "item-action-v2";
    result.history_revision = "history-lineage-v3";
    // Two item/action pairs use distinct internal table offsets.
    result.token_rows = {11, 101, 12, 102};
    result.position_ids = {0, 1, 2, 3};
    result.effective_scale = 512.0F;
    return result;
}

HistorySignature appended(HistorySignature result) {
    result.token_rows.insert(result.token_rows.end(), {13, 103});
    if (!result.position_ids.empty())
        result.position_ids.insert(result.position_ids.end(), {4, 5});
    if (!result.time_ids.empty())
        result.time_ids.insert(result.time_ids.end(), {0, 0});
    return result;
}

void test_hits_and_identity() {
    const auto base = history();
    expect(nullptr, base, {}, CacheReuseKind::kRecompute, CacheReuseReason::kCold, 0, "cold");
    expect(&base, base, {}, CacheReuseKind::kExactHistory, CacheReuseReason::kExactHistory, 4,
           "exact history permits different causal candidates");
    expect(&base, appended(base), {}, CacheReuseKind::kAppendHistory,
           CacheReuseReason::kAppendHistory, 4, "append complete item-action pair");

    auto next = base;
    next.model_namespace += "-new";
    reject(base, next, CacheReuseReason::kModelNamespaceChanged, "model rollout");
    next = base;
    next.feature_version += "-new";
    reject(base, next, CacheReuseReason::kFeatureVersionChanged, "feature definition update");
    next = appended(base);
    next.history_revision += "-new";
    reject(base, next, CacheReuseReason::kHistoryRevisionChanged, "explicit history invalidation");
    next = base;
    next.token_rows[2] = 14;
    reject(base, next, CacheReuseReason::kHistoryChanged, "corrected history item");
    next = base;
    next.token_rows[3] = 104;
    reject(base, next, CacheReuseReason::kHistoryChanged, "corrected history action");
    next = base;
    std::swap(next.token_rows[0], next.token_rows[2]);
    reject(base, next, CacheReuseReason::kHistoryChanged, "reordered history");
    next = base;
    next.token_rows.resize(2);
    next.position_ids.resize(2);
    reject(base, next, CacheReuseReason::kHistoryTruncated, "shortened history");
    next = base;
    next.token_rows = {12, 102, 13, 103};
    reject(base, next, CacheReuseReason::kHistoryChanged, "moved history window");
}

void test_attention_dependencies() {
    const auto base = history();
    CachePolicy policy;
    policy.is_causal = false;
    reject(base, base, CacheReuseReason::kCandidateDependentAttention,
           "noncausal ranking depends on request candidates", policy);
    policy.is_ranking = false;
    expect(&base, base, policy, CacheReuseKind::kExactHistory, CacheReuseReason::kExactHistory, 4,
           "noncausal retrieval candidates are outside encoder");
    reject(base, appended(base), CacheReuseReason::kNonCausalAppend,
           "noncausal retrieval append changes previous layers", policy);

    auto contextual = base;
    contextual.contextual_length = 1;
    expect(&contextual, contextual, {}, CacheReuseKind::kExactHistory,
           CacheReuseReason::kExactHistory, 4, "context exact hit");
    reject(contextual, appended(contextual), CacheReuseReason::kContextualAppend,
           "bidirectional context propagates appended history");
    policy = {};
    policy.disable_contextual_mask = true;
    expect(&contextual, appended(contextual), policy, CacheReuseKind::kAppendHistory,
           CacheReuseReason::kAppendHistory, 4, "ordinary causal context allows append");
    auto next = contextual;
    next.contextual_length = 2;
    reject(contextual, next, CacheReuseReason::kContextLengthChanged,
           "same rows with different context boundary");
    next = contextual;
    next.token_rows[0] = 15;
    reject(contextual, next, CacheReuseReason::kHistoryChanged, "context ID changed");
}

void test_derived_embedding_dependencies() {
    auto base = history();
    auto next = base;
    next.position_ids[1] = 7;
    reject(base, next, CacheReuseReason::kPositionChanged, "position definition changed");
    next = base;
    next.position_ids.clear();
    reject(base, next, CacheReuseReason::kPositionChanged, "position encoding disabled");
    reject(next, base, CacheReuseReason::kPositionChanged, "position encoding enabled");
    base.time_ids = {4, 3, 2, 1};
    next = base;
    next.time_ids[0] = 5;
    reject(base, next, CacheReuseReason::kTimeChanged, "new query timestamp shifts bucket");
    next = base;
    next.time_ids.clear();
    reject(base, next, CacheReuseReason::kTimeChanged, "time encoding disabled");
    reject(next, base, CacheReuseReason::kTimeChanged, "time encoding enabled");
    next = appended(base);
    next.position_ids = {6, 5, 4, 3, 2, 1};
    reject(base, next, CacheReuseReason::kPositionChanged, "reversed positions shift on append");
    next = appended(base);
    next.time_ids[1] = 4;
    reject(base, next, CacheReuseReason::kTimeChanged, "append changes prior time buckets");
    next = appended(base);
    expect(&base, next, {}, CacheReuseKind::kAppendHistory, CacheReuseReason::kAppendHistory, 4,
           "stable derived IDs allow timestamp-aware append");
    base.position_ids.clear();
    base.time_ids.clear();
    expect(&base, appended(base), {}, CacheReuseKind::kAppendHistory,
           CacheReuseReason::kAppendHistory, 4, "no positional encoder");
}

void test_scaling_and_validation() {
    const auto base = history();
    auto next = base;
    next.effective_scale = 260.0F;
    reject(base, next, CacheReuseReason::kScaleChanged, "batch maximum or candidate count changed");
    next = appended(base);
    next.effective_scale = 2.0F;
    reject(base, next, CacheReuseReason::kScaleChanged,
           "suffix length cannot replace full divisor");
    next = base;
    next.effective_scale = std::nextafter(base.effective_scale, 0.0F);
    reject(base, next, CacheReuseReason::kScaleChanged, "scale comparison is exact");
    for (float invalid : {0.0F, -1.0F, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        next = base;
        next.effective_scale = invalid;
        reject(base, next, CacheReuseReason::kInvalidSignature, "invalid incoming scale");
        reject(next, base, CacheReuseReason::kInvalidSignature, "invalid cached scale");
    }
    next = base;
    next.position_ids.pop_back();
    reject(base, next, CacheReuseReason::kInvalidSignature, "incomplete position metadata");
    next = base;
    next.time_ids = {0};
    reject(base, next, CacheReuseReason::kInvalidSignature, "incomplete time metadata");
    next = base;
    next.contextual_length = -1;
    reject(base, next, CacheReuseReason::kInvalidSignature, "negative context length");
    next.contextual_length = 5;
    reject(base, next, CacheReuseReason::kInvalidSignature, "context exceeds prefix");
    next = base;
    next.token_rows[0] = -1;
    reject(base, next, CacheReuseReason::kInvalidSignature, "negative internal row");
    next = base;
    next.position_ids[0] = -1;
    reject(base, next, CacheReuseReason::kInvalidSignature, "negative position ID");
    next = base;
    next.token_rows.clear();
    next.position_ids.clear();
    reject(next, base, CacheReuseReason::kNoCachedTokens, "empty entry is not a KV hit");
}

} // namespace

int main() {
    try {
        test_hits_and_identity();
        test_attention_dependencies();
        test_derived_embedding_dependencies();
        test_scaling_and_validation();
        std::cout << "HSTU cache policy: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "HSTU cache policy failed: " << error.what() << '\n';
        return 1;
    }
}
