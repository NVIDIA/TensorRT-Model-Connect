/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::hstu {

// Describes only the persistent context/history prefix. Candidates never enter
// token_rows. Rows include table offsets, and item/action tokens are interleaved
// in exactly the same order as the model input. The owner supplies namespaces
// that bind the model artifact and feature definitions to these internal rows.
struct HistorySignature {
    std::string model_namespace;
    std::string feature_version;
    std::string history_revision;
    std::vector<std::int32_t> token_rows;
    // Empty when the corresponding input is disabled; otherwise one ID per row.
    std::vector<std::int32_t> position_ids;
    std::vector<std::int32_t> time_ids;
    // Use the full request's divisor, including its original batch padding.
    float effective_scale{0.0F};
    std::int32_t contextual_length{0};
};

struct CachePolicy {
    bool is_causal{true};
    bool is_ranking{true};
    bool disable_contextual_mask{false};
};

enum class CacheReuseKind { kRecompute, kExactHistory, kAppendHistory };

enum class CacheReuseReason {
    kCold,
    kInvalidSignature,
    kNoCachedTokens,
    kModelNamespaceChanged,
    kFeatureVersionChanged,
    kHistoryRevisionChanged,
    kContextLengthChanged,
    kCandidateDependentAttention,
    kScaleChanged,
    kHistoryTruncated,
    kHistoryChanged,
    kPositionChanged,
    kTimeChanged,
    kNonCausalAppend,
    kContextualAppend,
    kExactHistory,
    kAppendHistory,
};

struct CacheReuse {
    CacheReuseKind kind{CacheReuseKind::kRecompute};
    CacheReuseReason reason{CacheReuseReason::kCold};
    std::size_t reused_tokens{0};
};

// A null cached signature is a cache miss. Every rejected reuse recomputes the
// complete request; none of these decisions permit approximate/stale KV reuse.
CacheReuse plan_cache_reuse(const HistorySignature* cached, const HistorySignature& incoming,
                            const CachePolicy& policy);

const char* cache_reuse_reason_name(CacheReuseReason reason);

} // namespace trtmc::hstu
