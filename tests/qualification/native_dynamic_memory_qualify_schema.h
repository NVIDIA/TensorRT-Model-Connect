/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::qualification {

inline nlohmann::json make_sequential_request_samples() {
    return nlohmann::json::array();
}

inline nlohmann::json make_runtime_phase_memory_samples() {
    return nlohmann::json::array();
}

inline nlohmann::json make_runtime_phase_memory_sample(std::string phase, std::uint32_t device,
                                                       nlohmann::json sample) {
    if (!sample.is_object())
        throw std::invalid_argument("runtime phase memory sample must be an object");
    sample["phase"] = std::move(phase);
    sample["device"] = device;
    return sample;
}

inline void attach_runtime_phase_memory_samples(nlohmann::json& lifetime,
                                                const nlohmann::json& samples) {
    lifetime["runtime_phase_memory_samples"] = samples;
}

inline void validate_single_warmup_arguments(bool requested, std::uint32_t repeat,
                                             std::uint32_t load_cycles,
                                             std::uint64_t second_max_sequence_length,
                                             std::uint64_t controlled_reservation_target_tokens) {
    if (!requested)
        return;
    if (repeat != 1 || load_cycles != 1 || second_max_sequence_length != 0 ||
        controlled_reservation_target_tokens != 0) {
        throw std::invalid_argument(
            "--warmup-load-cycle requires the ordinary repeat=1, load-cycles=1 mode");
    }
}

inline nlohmann::json make_single_warmup_lifetime_protocol() {
    return {
        {"schema_version", 1},
        {"execution_order", {"warmup", "measured"}},
        {"warmup_count", 1},
        {"measured_count", 1},
    };
}

inline nlohmann::json make_auto_policy() {
    return {{"kind", "auto"}};
}

inline nlohmann::json make_fraction_policy(double fraction) {
    if (!(std::isfinite(fraction) && fraction > 0.0 && fraction <= 1.0))
        throw std::invalid_argument("fraction policy requires a value in (0, 1]");
    return {
        {"kind", "fraction"},
        {"requested_fraction", fraction},
    };
}

inline nlohmann::json make_bytes_policy(std::uint64_t bytes) {
    if (bytes == 0)
        throw std::invalid_argument("bytes policy requires a positive byte count");
    return {
        {"kind", "bytes"},
        {"requested_bytes", bytes},
    };
}

inline nlohmann::json make_max_sequence_length_policy(std::uint64_t tokens) {
    if (tokens == 0)
        throw std::invalid_argument("max-sequence policy requires a positive token count");
    return {
        {"kind", "max_sequence_length"},
        {"requested_tokens", tokens},
    };
}

inline constexpr const char* kExecutionAttemptEvidenceSource =
    "runtime_memory_transfer_snapshot_v1.execution_attempt_events";

inline nlohmann::json make_attention_execution_ledger(const std::string& source, bool available,
                                                      std::uint64_t module_count,
                                                      std::uint64_t before, std::uint64_t after,
                                                      std::uint64_t delta) {
    if (source != kExecutionAttemptEvidenceSource || !available || module_count == 0 ||
        after < before || delta != after - before) {
        throw std::invalid_argument(
            "attention execution ledger requires complete monotonic evidence");
    }
    return {
        {"source", source}, {"available", available}, {"module_count", module_count},
        {"before", before}, {"after", after},         {"delta", delta},
    };
}

inline bool attention_execution_ledger_proves_before_attention(const nlohmann::json& ledger) {
    if (!ledger.is_object() || ledger.size() != 6 || !ledger.contains("source") ||
        !ledger.contains("available") || !ledger.contains("module_count") ||
        !ledger.contains("before") || !ledger.contains("after") || !ledger.contains("delta")) {
        return false;
    }
    try {
        const auto validated = make_attention_execution_ledger(
            ledger.at("source").get<std::string>(), ledger.at("available").get<bool>(),
            ledger.at("module_count").get<std::uint64_t>(),
            ledger.at("before").get<std::uint64_t>(), ledger.at("after").get<std::uint64_t>(),
            ledger.at("delta").get<std::uint64_t>());
        return validated.at("delta").get<std::uint64_t>() == 0;
    } catch (const std::exception&) {
        return false;
    }
}

inline void attach_lifetime_execution_evidence(nlohmann::json& lifetime,
                                               std::uint32_t execution_ordinal, const char* role,
                                               bool measured) {
    if (!lifetime.is_object())
        throw std::invalid_argument("lifetime execution evidence requires an object");
    const std::string role_name = role != nullptr ? role : "";
    if (role_name != "warmup" && role_name != "measured")
        throw std::invalid_argument("lifetime role must be warmup or measured");
    if ((role_name == "warmup" && measured) || (role_name == "measured" && !measured)) {
        throw std::invalid_argument("lifetime role and measured flag disagree");
    }
    lifetime["execution_ordinal"] = execution_ordinal;
    lifetime["role"] = role_name;
    lifetime["measured"] = measured;
}

inline bool float32_logits_bitwise_equal(const std::vector<std::vector<float>>& lhs,
                                         const std::vector<std::vector<float>>& rhs) {
    if (lhs.size() != rhs.size())
        return false;
    for (std::size_t row = 0; row < lhs.size(); ++row) {
        if (lhs[row].size() != rhs[row].size())
            return false;
        const auto bytes = lhs[row].size() * sizeof(float);
        if (bytes != 0 && std::memcmp(lhs[row].data(), rhs[row].data(), bytes) != 0)
            return false;
    }
    return true;
}

inline nlohmann::json
make_cold_warm_output_equivalence(bool prompt_tokens_equal, bool prefill_launches_equal,
                                  bool decode_launches_equal, bool final_kv_position_equal,
                                  bool selected_token_ids_equal, bool step_top1_token_ids_equal,
                                  bool full_float32_logits_bitwise_equal) {
    const auto passed = prompt_tokens_equal && prefill_launches_equal && decode_launches_equal &&
                        final_kv_position_equal && selected_token_ids_equal &&
                        step_top1_token_ids_equal && full_float32_logits_bitwise_equal;
    return {
        {"schema_version", 1},
        {"warmup_execution_ordinal", 0},
        {"measured_execution_ordinal", 1},
        {"prompt_tokens_equal", prompt_tokens_equal},
        {"prefill_launches_equal", prefill_launches_equal},
        {"decode_launches_equal", decode_launches_equal},
        {"final_kv_position_equal", final_kv_position_equal},
        {"selected_token_ids_equal", selected_token_ids_equal},
        {"step_top1_token_ids_equal", step_top1_token_ids_equal},
        {"full_float32_logits_bitwise_equal", full_float32_logits_bitwise_equal},
        {"passed", passed},
    };
}

inline constexpr std::uint64_t kControlledReservationAlignmentBytes = 2ULL * 1024ULL * 1024ULL;
inline constexpr std::uint32_t kMaxControlledBulkCorrectionAttempts = 64;
inline constexpr std::uint64_t kControlledPreplanningHeadroomBytes =
    32ULL * kControlledReservationAlignmentBytes;
inline constexpr std::uint64_t kControlledInitialBulkChunkBytes =
    64ULL * kControlledReservationAlignmentBytes;
inline constexpr std::uint64_t kControlledTargetToleranceRows = 19;

inline std::uint64_t controlled_bulk_correction_bytes(std::uint64_t visible_free_bytes,
                                                      std::uint64_t visible_free_upper_bound,
                                                      std::uint64_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        throw std::invalid_argument("bulk correction alignment must be a power of two");
    if (visible_free_bytes < visible_free_upper_bound)
        return 0;
    const auto excess = visible_free_bytes - visible_free_upper_bound + 1U;
    const auto remainder = excess % alignment;
    if (remainder == 0)
        return excess;
    const auto increment = alignment - remainder;
    if (excess > std::numeric_limits<std::uint64_t>::max() - increment)
        throw std::overflow_error("bulk correction bytes overflow uint64");
    return excess + increment;
}

enum class ControlledFreeWindowActionKind {
    kInWindow,
    kAllocate,
    kRelease,
};

struct ControlledFreeWindowAction {
    ControlledFreeWindowActionKind kind{ControlledFreeWindowActionKind::kInWindow};
    std::uint64_t bytes{0};
    std::uint64_t deficit_bytes{0};
    std::uint64_t excess_bytes{0};
};

inline ControlledFreeWindowAction
decide_controlled_free_window_action(std::uint64_t visible_free_bytes,
                                     std::uint64_t visible_free_lower_bound,
                                     std::uint64_t visible_free_upper_bound,
                                     std::uint64_t alignment, std::uint64_t releasable_tail_bytes) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        throw std::invalid_argument("free-window alignment must be a power of two");
    if (visible_free_lower_bound >= visible_free_upper_bound ||
        visible_free_upper_bound - visible_free_lower_bound != alignment) {
        throw std::invalid_argument("free-window bounds must describe one alignment-wide window");
    }
    if (visible_free_bytes >= visible_free_lower_bound &&
        visible_free_bytes < visible_free_upper_bound) {
        return {};
    }
    if (visible_free_bytes >= visible_free_upper_bound) {
        const auto bytes = controlled_bulk_correction_bytes(visible_free_bytes,
                                                            visible_free_upper_bound, alignment);
        return {
            ControlledFreeWindowActionKind::kAllocate,
            bytes,
            0,
            visible_free_bytes - visible_free_upper_bound + 1U,
        };
    }
    if (releasable_tail_bytes == 0 || releasable_tail_bytes % alignment != 0)
        throw std::invalid_argument(
            "low free-window recovery requires an aligned releasable tail allocation");
    return {
        ControlledFreeWindowActionKind::kRelease,
        releasable_tail_bytes,
        visible_free_lower_bound - visible_free_bytes,
        0,
    };
}

inline std::uint64_t controlled_auto_capacity_from_final_free(std::uint64_t final_free_bytes,
                                                              std::uint64_t safety_reserve_bytes,
                                                              double fraction,
                                                              std::uint64_t bytes_per_token) {
    if (!(std::isfinite(fraction) && fraction > 0.0 && fraction <= 1.0))
        throw std::invalid_argument("controlled auto fraction must be in (0, 1]");
    if (bytes_per_token == 0)
        throw std::invalid_argument("controlled KV bytes per token must be positive");
    if (final_free_bytes <= safety_reserve_bytes)
        return 0;
    const auto safe_bytes = final_free_bytes - safety_reserve_bytes;
    const long double budget =
        static_cast<long double>(safe_bytes) * static_cast<long double>(fraction);
    return static_cast<std::uint64_t>(std::min(
               budget, static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))) /
           bytes_per_token;
}

} // namespace trtmc::qualification
