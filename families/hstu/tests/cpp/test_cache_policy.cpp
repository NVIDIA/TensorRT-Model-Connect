/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/cache_policy.h"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using trtmc::hstu::CachePolicy;
using trtmc::hstu::CacheReuseKind;
using trtmc::hstu::CacheReuseReason;
using trtmc::hstu::decode_history_signature;
using trtmc::hstu::encode_history_signature;
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

void check(bool condition, const char* name) {
    if (!condition)
        throw std::runtime_error(name);
    ++checks;
}

template <class Action>
void invalid_codec(Action&& action, const char* name) {
    bool rejected = false;
    try {
        action();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, name);
}

std::uint32_t float_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float from_float_bits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void same_signature(const HistorySignature& expected, const HistorySignature& actual) {
    check(expected.model_namespace == actual.model_namespace &&
              expected.feature_version == actual.feature_version &&
              expected.history_revision == actual.history_revision &&
              expected.token_rows == actual.token_rows &&
              expected.position_ids == actual.position_ids &&
              expected.time_ids == actual.time_ids &&
              expected.contextual_length == actual.contextual_length &&
              float_bits(expected.effective_scale) == float_bits(actual.effective_scale),
          "binary signature preserves every field and float bit");
}

std::uint32_t wire_u32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return static_cast<std::uint32_t>(bytes.at(offset)) |
           (static_cast<std::uint32_t>(bytes.at(offset + 1)) << 8) |
           (static_cast<std::uint32_t>(bytes.at(offset + 2)) << 16) |
           (static_cast<std::uint32_t>(bytes.at(offset + 3)) << 24);
}

void set_wire_u32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes.at(offset++) = static_cast<std::uint8_t>(value >> shift);
}

std::array<std::size_t, 6> count_offsets(const std::vector<std::uint8_t>& bytes) {
    std::array<std::size_t, 6> offsets{};
    std::size_t cursor = 20; // magic, version, context, exact FP32 bits
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        offsets[index] = cursor;
        cursor += 4 + wire_u32(bytes, cursor) * (index < 3 ? 1 : 4);
    }
    check(cursor == bytes.size(), "wire fields consume the exact encoded size");
    return offsets;
}

void test_binary_roundtrip() {
    auto base = history();
    base.model_namespace = std::string("a\0b\xff", 4);
    base.feature_version.clear();
    base.history_revision = std::string("\x80\0\x7f", 3);
    base.token_rows[2] = std::numeric_limits<std::int32_t>::max();
    base.contextual_length = 2;
    base.time_ids = {13, 12, 11, 10};
    auto encoded = encode_history_signature(base, 4);
    same_signature(base, decode_history_signature(encoded, 4));
    const std::vector<std::uint8_t> header{'H', 'S', 'T', 'U', 'S', 'I', 'G', 0, 2, 0,
                                           0,   0,   2,   0,   0,   0,   0,   0, 0, 0x44};
    check(std::vector<std::uint8_t>(encoded.begin(), encoded.begin() + 20) == header,
          "header uses specified magic, version and little-endian FP32/context bits");
    const auto offsets = count_offsets(encoded);
    check(wire_u32(encoded, offsets[3] + 4 + 2 * 4) == 0x7fffffffU,
          "INT32 maximum retains its exact fixed-width bits");
    for (const auto bits : {0x00000001U, 0x00800000U, 0x3f800001U, 0x7f7fffffU}) {
        base.effective_scale = from_float_bits(bits);
        encoded = encode_history_signature(base, 4);
        check(wire_u32(encoded, 16) == bits, "FP32 scale bit pattern is serialized verbatim");
        same_signature(base, decode_history_signature(encoded, 4));
    }
    base.position_ids.clear();
    same_signature(base, decode_history_signature(encode_history_signature(base, 4), 4));
    base.time_ids.clear();
    same_signature(base, decode_history_signature(encode_history_signature(base, 4), 4));
    base.token_rows.clear();
    base.contextual_length = 0;
    same_signature(base, decode_history_signature(encode_history_signature(base, 0), 0));
}

void test_binary_corruption() {
    auto base = history();
    base.time_ids = {4, 3, 2, 1};
    const auto encoded = encode_history_signature(base, 4);
    for (std::size_t size = 0; size < encoded.size(); ++size) {
        const std::vector<std::uint8_t> truncated(encoded.begin(), encoded.begin() + size);
        invalid_codec([&] { decode_history_signature(truncated, 4); },
                      "every truncated payload is rejected");
    }
    auto damaged = encoded;
    damaged.push_back(0);
    invalid_codec([&] { decode_history_signature(damaged, 4); }, "trailing bytes are rejected");
    for (std::size_t index = 0; index < 8; ++index) {
        damaged = encoded;
        damaged[index] ^= 1;
        invalid_codec([&] { decode_history_signature(damaged, 4); }, "all magic bytes are checked");
    }
    damaged = encoded;
    set_wire_u32(damaged, 8, 3);
    invalid_codec([&] { decode_history_signature(damaged, 4); },
                  "unsupported private version is rejected");
    const auto offsets = count_offsets(encoded);
    for (const auto offset : offsets) {
        for (const auto count : {0xffffffffU, 0x40000000U}) {
            damaged = encoded;
            set_wire_u32(damaged, offset, count);
            invalid_codec([&] { decode_history_signature(damaged, 4); },
                          "malicious string/vector counts rejected before allocation");
            invalid_codec(
                [&] { decode_history_signature(damaged, std::numeric_limits<std::size_t>::max()); },
                "remaining byte bounds reject large counts independently of the token cap");
        }
    }
    for (const auto offset : {offsets[4], offsets[5]}) {
        damaged = encoded;
        set_wire_u32(damaged, offset, 3);
        invalid_codec([&] { decode_history_signature(damaged, 4); },
                      "derived vector length must equal N or zero");
    }
    for (const auto offset : {offsets[3], offsets[4], offsets[5]}) {
        damaged = encoded;
        set_wire_u32(damaged, offset + 4, 0xffffffffU);
        invalid_codec([&] { decode_history_signature(damaged, 4); },
                      "negative ID remains an invalid signature");
    }
    invalid_codec([&] { decode_history_signature(encoded, 3); }, "decoder token budget enforced");
    invalid_codec([&] { encode_history_signature(base, 3); }, "encoder token budget enforced");
}

void test_binary_invalid_values() {
    const auto base = history();
    const auto encoded = encode_history_signature(base, 4);
    for (const auto bits :
         {0U, 0x80000000U, 0xbf800000U, 0x7f800000U, 0xff800000U, 0x7fc00001U, 0x7f800001U}) {
        auto invalid = base;
        invalid.effective_scale = from_float_bits(bits);
        invalid_codec([&] { encode_history_signature(invalid, 4); },
                      "encoder rejects invalid scale");
        auto damaged = encoded;
        set_wire_u32(damaged, 16, bits);
        invalid_codec([&] { decode_history_signature(damaged, 4); },
                      "decoder rejects invalid scale bits");
    }
    for (const auto context : {0xffffffffU, 5U}) {
        auto damaged = encoded;
        set_wire_u32(damaged, 12, context);
        invalid_codec([&] { decode_history_signature(damaged, 4); },
                      "decoder validates context boundary");
    }
    auto invalid = base;
    invalid.contextual_length = -1;
    invalid_codec([&] { encode_history_signature(invalid, 4); },
                  "encoder rejects negative context");
    invalid = base;
    invalid.position_ids.pop_back();
    invalid_codec([&] { encode_history_signature(invalid, 4); },
                  "encoder rejects incomplete positions");
    invalid = base;
    invalid.time_ids = {1};
    invalid_codec([&] { encode_history_signature(invalid, 4); },
                  "encoder rejects incomplete timestamps");
}

} // namespace

int main() {
    try {
        test_hits_and_identity();
        test_attention_dependencies();
        test_derived_embedding_dependencies();
        test_scaling_and_validation();
        test_binary_roundtrip();
        test_binary_corruption();
        test_binary_invalid_values();
        std::cout << "HSTU cache policy: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "HSTU cache policy failed: " << error.what() << '\n';
        return 1;
    }
}
