/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Shared text-generation KV budget arithmetic.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc {

enum class KvCacheBudgetPolicy {
    kAuto,
    kFraction,
    kBytes,
};

struct KvCacheBudgetRequest {
    std::uint64_t free_bytes{0};
    std::uint64_t row_bytes{0};
    std::uint64_t explicit_bytes{0};
    double explicit_fraction{0.0};
    std::uint64_t max_sequence_length{0};
    int32_t engine_max_rows{0};
    double automatic_fraction{0.90};
    // Space kept available for one-row present tensors, sampling buffers,
    // CUDA graph capture, and allocator fragmentation. Model runtimes choose
    // the concrete reserve; pure arithmetic callers may leave it at zero.
    std::uint64_t minimum_reserve_bytes{0};
};

struct KvCacheBudget {
    KvCacheBudgetPolicy policy{KvCacheBudgetPolicy::kAuto};
    double fraction{0.0};
    std::uint64_t requested_bytes{0};
    std::uint64_t budget_bytes{0};
    std::uint64_t memory_capacity_rows{0};
    int32_t runtime_rows{0};
    std::uint64_t allocated_bytes{0};
    std::uint64_t reserved_bytes{0};
    std::uint64_t usable_free_bytes{0};
    bool clamped_to_engine_max{false};
};

inline KvCacheBudget resolve_kv_cache_budget(const KvCacheBudgetRequest& request) {
    if (request.free_bytes == 0)
        throw std::invalid_argument("KV cache budgeting requires non-zero free GPU memory");
    if (request.row_bytes == 0)
        throw std::invalid_argument("KV cache row size must be positive");
    if (request.engine_max_rows <= 0)
        throw std::invalid_argument("KV cache engine capability must be positive");
    if (request.explicit_bytes != 0 && request.explicit_fraction != 0.0) {
        throw std::invalid_argument(
            "KV cache bytes and percentage policies are mutually exclusive");
    }
    if (request.minimum_reserve_bytes >= request.free_bytes) {
        throw std::runtime_error(
            "Free GPU memory is not larger than the required runtime safety reserve");
    }

    KvCacheBudget result;
    result.reserved_bytes = request.minimum_reserve_bytes;
    result.usable_free_bytes = request.free_bytes - result.reserved_bytes;
    if (request.explicit_bytes != 0) {
        result.policy = KvCacheBudgetPolicy::kBytes;
        result.requested_bytes = request.explicit_bytes;
        result.budget_bytes = request.explicit_bytes;
    } else {
        const bool explicit_fraction = request.explicit_fraction != 0.0;
        const double fraction =
            explicit_fraction ? request.explicit_fraction : request.automatic_fraction;
        if (!(fraction > 0.0 && fraction <= 1.0))
            throw std::invalid_argument("KV cache memory fraction must be in (0, 1]");
        result.policy =
            explicit_fraction ? KvCacheBudgetPolicy::kFraction : KvCacheBudgetPolicy::kAuto;
        result.fraction = fraction;
        const std::uint64_t percentage_bytes = static_cast<std::uint64_t>(
            static_cast<long double>(request.free_bytes) * static_cast<long double>(fraction));
        result.budget_bytes = std::min(percentage_bytes, result.usable_free_bytes);
        result.requested_bytes = result.budget_bytes;
    }

    result.memory_capacity_rows = result.budget_bytes / request.row_bytes;
    if (result.memory_capacity_rows == 0) {
        throw std::runtime_error("KV cache memory budget is smaller than one token row (" +
                                 std::to_string(request.row_bytes) + " bytes)");
    }

    if (request.max_sequence_length > static_cast<std::uint64_t>(request.engine_max_rows)) {
        throw std::runtime_error(
            "Requested max sequence length " + std::to_string(request.max_sequence_length) +
            " exceeds bundle engine capability " + std::to_string(request.engine_max_rows));
    }
    if (request.max_sequence_length != 0 &&
        request.max_sequence_length > result.memory_capacity_rows) {
        throw std::runtime_error("Requested max sequence length " +
                                 std::to_string(request.max_sequence_length) +
                                 " does not fit the KV memory budget; maximum fitting length is " +
                                 std::to_string(result.memory_capacity_rows));
    }

    const std::uint64_t policy_rows = request.max_sequence_length != 0
                                          ? request.max_sequence_length
                                          : result.memory_capacity_rows;
    const std::uint64_t runtime_rows =
        std::min(policy_rows, static_cast<std::uint64_t>(request.engine_max_rows));
    if (runtime_rows > static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("Resolved KV cache rows exceed int32 runtime limits");
    if (runtime_rows > std::numeric_limits<std::uint64_t>::max() / request.row_bytes)
        throw std::runtime_error("Resolved KV cache allocation exceeds uint64 byte limits");

    result.runtime_rows = static_cast<int32_t>(runtime_rows);
    result.allocated_bytes = runtime_rows * request.row_bytes;
    result.clamped_to_engine_max =
        request.max_sequence_length == 0 &&
        result.memory_capacity_rows > static_cast<std::uint64_t>(request.engine_max_rows);

    if (result.allocated_bytes > result.usable_free_bytes) {
        throw std::runtime_error(
            "Resolved KV cache allocation exceeds safely usable GPU memory after the runtime "
            "reserve");
    }
    return result;
}

inline const char* kv_cache_budget_policy_name(KvCacheBudgetPolicy policy) {
    switch (policy) {
    case KvCacheBudgetPolicy::kAuto:
        return "auto";
    case KvCacheBudgetPolicy::kFraction:
        return "fraction";
    case KvCacheBudgetPolicy::kBytes:
        return "bytes";
    }
    return "unknown";
}

inline void validate_sequence_admission(std::size_t prompt_tokens, int32_t max_new_tokens,
                                        int32_t state_max_length, const std::string& runtime_name) {
    if (max_new_tokens < 0) {
        throw std::invalid_argument(runtime_name +
                                    ": max_new_tokens must be greater than or equal to zero");
    }
    if (state_max_length <= 0) {
        throw std::runtime_error(runtime_name + ": runtime KV cache capacity must be positive");
    }

    const auto capacity = static_cast<std::size_t>(state_max_length);
    const auto requested_new_tokens = static_cast<std::size_t>(max_new_tokens);
    if (prompt_tokens > capacity || requested_new_tokens > capacity - prompt_tokens) {
        throw std::runtime_error(runtime_name + ": prompt length " + std::to_string(prompt_tokens) +
                                 " plus max_new_tokens " + std::to_string(max_new_tokens) +
                                 " exceeds runtime max sequence length " +
                                 std::to_string(state_max_length));
    }
}

// Runtime-memory bundles keep these authorities distinct:
//   M: model_context_limit (semantic model capability)
//   R: runtime_kv_capacity_tokens (physical allocation)
//   U: request_context_limit (optional user policy)
//   B: kv_bytes_per_token (resource conversion)
// The receipt byte fields make resource failures actionable without changing
// the legacy admission message for static bundles.
struct RuntimeSequenceAdmissionContext {
    std::uint64_t model_context_limit{0};
    std::uint64_t runtime_kv_capacity_tokens{0};
    std::uint64_t request_context_limit{0};
    std::uint64_t kv_bytes_per_token{0};
    std::uint64_t kv_budget_bytes{0};
    std::uint64_t kv_reserved_bytes{0};

    bool enabled() const noexcept { return model_context_limit != 0; }
};

inline void validate_runtime_sequence_admission(std::size_t prompt_tokens, int32_t max_new_tokens,
                                                const RuntimeSequenceAdmissionContext& limits,
                                                const std::string& runtime_name) {
    if (max_new_tokens < 0) {
        throw std::invalid_argument(runtime_name +
                                    ": max_new_tokens must be greater than or equal to zero");
    }
    if (limits.model_context_limit == 0 || limits.runtime_kv_capacity_tokens == 0 ||
        limits.kv_bytes_per_token == 0) {
        throw std::runtime_error(runtime_name +
                                 ": runtime-memory admission metadata is incomplete");
    }
    if (limits.runtime_kv_capacity_tokens > limits.model_context_limit ||
        limits.request_context_limit > limits.model_context_limit) {
        throw std::runtime_error(runtime_name +
                                 ": runtime-memory admission metadata is inconsistent");
    }

    const auto requested_new_tokens = static_cast<std::uint64_t>(max_new_tokens);
    const bool prompt_fits_u64 =
        prompt_tokens <= static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max());
    const auto prompt_u64 = prompt_fits_u64 ? static_cast<std::uint64_t>(prompt_tokens)
                                            : std::numeric_limits<std::uint64_t>::max();
    const bool total_overflows =
        !prompt_fits_u64 ||
        prompt_u64 > std::numeric_limits<std::uint64_t>::max() - requested_new_tokens;
    const auto requested_total = total_overflows ? std::numeric_limits<std::uint64_t>::max()
                                                 : prompt_u64 + requested_new_tokens;
    const auto total_text =
        total_overflows ? std::string("overflow") : std::to_string(requested_total);
    const auto common_detail = "prompt_tokens=" + std::to_string(prompt_tokens) +
                               ", max_new_tokens=" + std::to_string(max_new_tokens) +
                               ", requested_total=" + total_text;
    const auto exceeds = [&](std::uint64_t limit) {
        return total_overflows || requested_total > limit;
    };

    if (exceeds(limits.model_context_limit)) {
        throw std::runtime_error(
            runtime_name + ": semantic model context limit exceeded (" + common_detail +
            ", model_context_limit=" + std::to_string(limits.model_context_limit) + ")");
    }

    // Report the first effective runtime bound. U is a policy error when it is
    // at least as restrictive as R; otherwise the physical allocation is the
    // limiting resource and must be reported as such.
    if (limits.request_context_limit != 0 &&
        limits.request_context_limit <= limits.runtime_kv_capacity_tokens &&
        exceeds(limits.request_context_limit)) {
        throw std::runtime_error(
            runtime_name + ": runtime max-sequence policy exceeded (" + common_detail +
            ", request_context_limit=" + std::to_string(limits.request_context_limit) +
            ", model_context_limit=" + std::to_string(limits.model_context_limit) +
            ", runtime_kv_capacity_tokens=" + std::to_string(limits.runtime_kv_capacity_tokens) +
            ")");
    }

    if (exceeds(limits.runtime_kv_capacity_tokens)) {
        std::string required_bytes = "overflow";
        if (!total_overflows && requested_total <= std::numeric_limits<std::uint64_t>::max() /
                                                       limits.kv_bytes_per_token) {
            required_bytes = std::to_string(requested_total * limits.kv_bytes_per_token);
        }
        throw std::runtime_error(
            runtime_name + ": runtime KV resource capacity exceeded (" + common_detail +
            ", runtime_kv_capacity_tokens=" + std::to_string(limits.runtime_kv_capacity_tokens) +
            ", required_kv_bytes=" + required_bytes +
            ", kv_bytes_per_token=" + std::to_string(limits.kv_bytes_per_token) +
            ", kv_budget_bytes=" + std::to_string(limits.kv_budget_bytes) +
            ", kv_reserved_bytes=" + std::to_string(limits.kv_reserved_bytes) +
            ", model_context_limit=" + std::to_string(limits.model_context_limit) +
            ", request_context_limit=" + std::to_string(limits.request_context_limit) + ")");
    }
}

inline void validate_sequence_admission_with_runtime_memory(
    std::size_t prompt_tokens, int32_t max_new_tokens, int32_t legacy_max_length,
    const RuntimeSequenceAdmissionContext& runtime_limits, const std::string& runtime_name) {
    if (runtime_limits.enabled()) {
        validate_runtime_sequence_admission(prompt_tokens, max_new_tokens, runtime_limits,
                                            runtime_name);
        return;
    }
    validate_sequence_admission(prompt_tokens, max_new_tokens, legacy_max_length, runtime_name);
}

} // namespace trtmc
