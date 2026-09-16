/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cache_policy.h"

#include <algorithm>
#include <cmath>

namespace trtmc::hstu {
namespace {

bool valid_ids(const std::vector<std::int32_t>& ids) {
    return std::all_of(ids.begin(), ids.end(), [](std::int32_t id) { return id >= 0; });
}

bool valid_signature(const HistorySignature& signature) {
    const auto length = signature.token_rows.size();
    return std::isfinite(signature.effective_scale) && signature.effective_scale > 0.0F &&
           signature.contextual_length >= 0 &&
           static_cast<std::size_t>(signature.contextual_length) <= length &&
           (signature.position_ids.empty() || signature.position_ids.size() == length) &&
           (signature.time_ids.empty() || signature.time_ids.size() == length) &&
           valid_ids(signature.token_rows) && valid_ids(signature.position_ids) &&
           valid_ids(signature.time_ids);
}

bool same_prefix(const std::vector<std::int32_t>& prefix,
                 const std::vector<std::int32_t>& sequence) {
    return prefix.size() <= sequence.size() &&
           std::equal(prefix.begin(), prefix.end(), sequence.begin());
}

bool same_derived_prefix(const std::vector<std::int32_t>& prefix,
                         const std::vector<std::int32_t>& sequence) {
    return prefix.empty() == sequence.empty() && same_prefix(prefix, sequence);
}

CacheReuse recompute(CacheReuseReason reason) {
    return {CacheReuseKind::kRecompute, reason, 0};
}

} // namespace

CacheReuse plan_cache_reuse(const HistorySignature* cached, const HistorySignature& incoming,
                            const CachePolicy& policy) {
    if (!valid_signature(incoming) || (cached != nullptr && !valid_signature(*cached)))
        return recompute(CacheReuseReason::kInvalidSignature);
    if (cached == nullptr)
        return recompute(CacheReuseReason::kCold);
    if (cached->token_rows.empty())
        return recompute(CacheReuseReason::kNoCachedTokens);
    if (cached->model_namespace != incoming.model_namespace)
        return recompute(CacheReuseReason::kModelNamespaceChanged);
    if (cached->feature_version != incoming.feature_version)
        return recompute(CacheReuseReason::kFeatureVersionChanged);
    if (cached->history_revision != incoming.history_revision)
        return recompute(CacheReuseReason::kHistoryRevisionChanged);
    if (cached->contextual_length != incoming.contextual_length)
        return recompute(CacheReuseReason::kContextLengthChanged);
    // Noncausal ranking history depends on request-local candidate inputs. The
    // history-only signature deliberately cannot certify those dependencies.
    if (!policy.is_causal && policy.is_ranking)
        return recompute(CacheReuseReason::kCandidateDependentAttention);
    if (cached->effective_scale != incoming.effective_scale)
        return recompute(CacheReuseReason::kScaleChanged);
    if (cached->token_rows.size() > incoming.token_rows.size())
        return recompute(CacheReuseReason::kHistoryTruncated);
    if (!same_prefix(cached->token_rows, incoming.token_rows))
        return recompute(CacheReuseReason::kHistoryChanged);
    if (!same_derived_prefix(cached->position_ids, incoming.position_ids))
        return recompute(CacheReuseReason::kPositionChanged);
    if (!same_derived_prefix(cached->time_ids, incoming.time_ids))
        return recompute(CacheReuseReason::kTimeChanged);
    if (cached->token_rows.size() == incoming.token_rows.size())
        return {CacheReuseKind::kExactHistory, CacheReuseReason::kExactHistory,
                cached->token_rows.size()};
    if (!policy.is_causal)
        return recompute(CacheReuseReason::kNonCausalAppend);
    // Context queries see the entire history. An append changes those outputs,
    // which changes previously computed K/V in subsequent HSTU layers.
    if (incoming.contextual_length > 0 && !policy.disable_contextual_mask)
        return recompute(CacheReuseReason::kContextualAppend);
    return {CacheReuseKind::kAppendHistory, CacheReuseReason::kAppendHistory,
            cached->token_rows.size()};
}

const char* cache_reuse_reason_name(CacheReuseReason reason) {
    switch (reason) {
    case CacheReuseReason::kCold:
        return "cold";
    case CacheReuseReason::kInvalidSignature:
        return "invalid_signature";
    case CacheReuseReason::kNoCachedTokens:
        return "no_cached_tokens";
    case CacheReuseReason::kModelNamespaceChanged:
        return "model_namespace_changed";
    case CacheReuseReason::kFeatureVersionChanged:
        return "feature_version_changed";
    case CacheReuseReason::kHistoryRevisionChanged:
        return "history_revision_changed";
    case CacheReuseReason::kContextLengthChanged:
        return "context_length_changed";
    case CacheReuseReason::kCandidateDependentAttention:
        return "candidate_dependent_attention";
    case CacheReuseReason::kScaleChanged:
        return "scale_changed";
    case CacheReuseReason::kHistoryTruncated:
        return "history_truncated";
    case CacheReuseReason::kHistoryChanged:
        return "history_changed";
    case CacheReuseReason::kPositionChanged:
        return "position_changed";
    case CacheReuseReason::kTimeChanged:
        return "time_changed";
    case CacheReuseReason::kNonCausalAppend:
        return "noncausal_append";
    case CacheReuseReason::kContextualAppend:
        return "contextual_append";
    case CacheReuseReason::kExactHistory:
        return "exact_history";
    case CacheReuseReason::kAppendHistory:
        return "append_history";
    }
    return "unknown";
}

} // namespace trtmc::hstu
