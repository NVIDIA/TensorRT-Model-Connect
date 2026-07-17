/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace trtmc::wan22::cudnn_sdpa {

inline constexpr char kPluginName[] = "Wan22DitCudnnSdpa";
inline constexpr char kPluginVersion[] = "1";

// TensorRT plugin-field ABI.  attention_kind is intentionally explicit: the
// two qualified Wan2.2 attention contracts have the same Q shape but different
// KV sequence lengths.
inline constexpr char kAttentionKindField[] = "attention_kind";
inline constexpr char kBatchField[] = "batch";
inline constexpr char kHeadsField[] = "heads";
inline constexpr char kQSequenceField[] = "q_sequence";
inline constexpr char kKvSequenceField[] = "kv_sequence";
inline constexpr char kHeadDimensionField[] = "head_dimension";

enum class AttentionKind : int32_t {
    kSelf = 0,
    kCross = 1,
};

struct Config {
    int32_t attention_kind{static_cast<int32_t>(AttentionKind::kSelf)};
    int32_t batch{1};
    int32_t heads{24};
    int32_t q_sequence{27'280};
    int32_t kv_sequence{27'280};
    int32_t head_dimension{128};
};

inline constexpr int32_t kProductionBatch = 1;
inline constexpr int32_t kProductionHeads = 24;
inline constexpr int32_t kProductionQSequence = 27'280;
inline constexpr int32_t kSelfKvSequence = 27'280;
inline constexpr int32_t kCrossKvSequence = 512;
inline constexpr int32_t kProductionHeadDimension = 128;

constexpr bool is_qualified_config(const Config& config) noexcept {
    const bool common = config.batch == kProductionBatch && config.heads == kProductionHeads &&
                        config.q_sequence == kProductionQSequence &&
                        config.head_dimension == kProductionHeadDimension;
    const bool is_self = config.attention_kind == static_cast<int32_t>(AttentionKind::kSelf) &&
                         config.kv_sequence == kSelfKvSequence;
    const bool is_cross = config.attention_kind == static_cast<int32_t>(AttentionKind::kCross) &&
                          config.kv_sequence == kCrossKvSequence;
    return common && (is_self || is_cross);
}

inline constexpr uint32_t kSerializationMagic = 0x57415344U; // "WASD"
inline constexpr uint32_t kSerializationVersion = 1U;
inline constexpr uint32_t kSerializedConfigBytes = 12U * sizeof(uint32_t);

// Only the semantic shape contract is serialized.  In particular, no cuDNN
// engine/config identifier is persisted: every target re-runs HeurMode A and
// selects its own first supported execution plan.
struct SerializedConfig {
    uint32_t magic{kSerializationMagic};
    uint32_t version{kSerializationVersion};
    uint32_t byte_size{kSerializedConfigBytes};
    uint32_t reserved0{0};
    Config config{};
    uint32_t reserved1{0};
    uint32_t reserved2{0};
};

static_assert(std::is_trivially_copyable_v<Config>);
static_assert(std::is_trivially_copyable_v<SerializedConfig>);
static_assert(sizeof(Config) == 6U * sizeof(int32_t));
static_assert(sizeof(SerializedConfig) == kSerializedConfigBytes);

constexpr SerializedConfig make_serialized_config(const Config& config) noexcept {
    SerializedConfig value{};
    value.config = config;
    return value;
}

constexpr bool is_valid_serialized_config(const SerializedConfig& value) noexcept {
    return value.magic == kSerializationMagic && value.version == kSerializationVersion &&
           value.byte_size == kSerializedConfigBytes && value.reserved0 == 0 &&
           value.reserved1 == 0 && value.reserved2 == 0 && is_qualified_config(value.config);
}

} // namespace trtmc::wan22::cudnn_sdpa
