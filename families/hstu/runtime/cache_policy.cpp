/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cache_policy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

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

constexpr std::array<std::uint8_t, 8> kSignatureMagic{'H', 'S', 'T', 'U', 'S', 'I', 'G', 0};
constexpr std::uint32_t kSignatureVersion = 2;
constexpr std::size_t kSignatureHeaderBytes = kSignatureMagic.size() + 3 * sizeof(std::uint32_t);
static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559,
              "HSTU history signatures require IEEE binary32 floats");

void require_codec(bool condition) {
    if (!condition)
        throw std::invalid_argument("invalid HSTU binary history signature");
}

std::size_t add_field_size(std::size_t total, std::size_t count, std::size_t width) {
    const auto maximum = std::numeric_limits<std::size_t>::max();
    require_codec(count <= std::numeric_limits<std::uint32_t>::max());
    require_codec(count <= (maximum - sizeof(std::uint32_t)) / width);
    const auto field = sizeof(std::uint32_t) + count * width;
    require_codec(field <= maximum - total);
    return total + field;
}

std::size_t encoded_size(const HistorySignature& signature) {
    auto size = kSignatureHeaderBytes;
    for (const auto* value :
         {&signature.model_namespace, &signature.feature_version, &signature.history_revision})
        size = add_field_size(size, value->size(), 1);
    for (const auto* value : {&signature.token_rows, &signature.position_ids, &signature.time_ids})
        size = add_field_size(size, value->size(), sizeof(std::uint32_t));
    return size;
}

void append_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_string(std::vector<std::uint8_t>& bytes, const std::string& value) {
    append_u32(bytes, static_cast<std::uint32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

void append_ids(std::vector<std::uint8_t>& bytes, const std::vector<std::int32_t>& values) {
    append_u32(bytes, static_cast<std::uint32_t>(values.size()));
    for (const auto value : values) {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        append_u32(bytes, bits);
    }
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& bytes, std::size_t& cursor) {
    require_codec(cursor <= bytes.size() && bytes.size() - cursor >= sizeof(std::uint32_t));
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
        value |= static_cast<std::uint32_t>(bytes[cursor++]) << shift;
    return value;
}

std::int32_t read_i32(const std::vector<std::uint8_t>& bytes, std::size_t& cursor) {
    const auto bits = read_u32(bytes, cursor);
    std::int32_t result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::string read_string(const std::vector<std::uint8_t>& bytes, std::size_t& cursor) {
    const auto count = read_u32(bytes, cursor);
    require_codec(count <= bytes.size() - cursor);
    std::string result(reinterpret_cast<const char*>(bytes.data() + cursor), count);
    cursor += count;
    return result;
}

std::vector<std::int32_t> read_ids(const std::vector<std::uint8_t>& bytes, std::size_t& cursor,
                                   std::uint32_t count, std::size_t limit) {
    require_codec(count <= limit && count <= (bytes.size() - cursor) / sizeof(std::uint32_t));
    std::vector<std::int32_t> result(count);
    for (auto& value : result)
        value = read_i32(bytes, cursor);
    return result;
}

std::vector<std::int32_t> read_optional_ids(const std::vector<std::uint8_t>& bytes,
                                            std::size_t& cursor, std::size_t token_count) {
    const auto count = read_u32(bytes, cursor);
    require_codec(count == 0 || count == token_count);
    return read_ids(bytes, cursor, count, token_count);
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

std::vector<std::uint8_t> encode_history_signature(const HistorySignature& signature,
                                                   std::size_t max_token_count) {
    require_codec(signature.token_rows.size() <= max_token_count && valid_signature(signature));
    std::vector<std::uint8_t> bytes;
    bytes.reserve(encoded_size(signature));
    bytes.insert(bytes.end(), kSignatureMagic.begin(), kSignatureMagic.end());
    append_u32(bytes, kSignatureVersion);
    std::uint32_t context_bits, scale_bits;
    std::memcpy(&context_bits, &signature.contextual_length, sizeof(context_bits));
    std::memcpy(&scale_bits, &signature.effective_scale, sizeof(scale_bits));
    append_u32(bytes, context_bits);
    append_u32(bytes, scale_bits);
    append_string(bytes, signature.model_namespace);
    append_string(bytes, signature.feature_version);
    append_string(bytes, signature.history_revision);
    append_ids(bytes, signature.token_rows);
    append_ids(bytes, signature.position_ids);
    append_ids(bytes, signature.time_ids);
    return bytes;
}

HistorySignature decode_history_signature(const std::vector<std::uint8_t>& bytes,
                                          std::size_t max_token_count) {
    require_codec(bytes.size() >= kSignatureHeaderBytes);
    require_codec(std::equal(kSignatureMagic.begin(), kSignatureMagic.end(), bytes.begin()));
    std::size_t cursor = kSignatureMagic.size();
    require_codec(read_u32(bytes, cursor) == kSignatureVersion);
    HistorySignature result;
    result.contextual_length = read_i32(bytes, cursor);
    const auto scale_bits = read_u32(bytes, cursor);
    std::memcpy(&result.effective_scale, &scale_bits, sizeof(scale_bits));
    result.model_namespace = read_string(bytes, cursor);
    result.feature_version = read_string(bytes, cursor);
    result.history_revision = read_string(bytes, cursor);
    const auto token_count = read_u32(bytes, cursor);
    result.token_rows = read_ids(bytes, cursor, token_count, max_token_count);
    result.position_ids = read_optional_ids(bytes, cursor, token_count);
    result.time_ids = read_optional_ids(bytes, cursor, token_count);
    require_codec(cursor == bytes.size() && valid_signature(result));
    return result;
}

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
