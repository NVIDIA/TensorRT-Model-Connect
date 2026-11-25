/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/config.h"
#include "trtmc/internal/matrix.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/scores.h"

namespace trtmc::internal {

enum class RankingFeatureType : std::uint32_t { Float32 = 1, Int32 = 2, Int64 = 3 };
struct RankingFeatureView {
    std::string_view name;
    RankingFeatureType type;
    Span<const std::int64_t> shape;
    Span<const std::uint8_t> bytes; // Contiguous host storage, borrowed through run.
};
struct StructuredRankingRequest {
    // One native scoring operation. The family owns required feature names,
    // tensor axes, batch/profile constraints, padding and preprocessing.
    Span<const RankingFeatureView> features;
};
struct CandidateScoresResult {
    FloatMatrix scores; // [request row,candidate], preserving input order and padding.
    ScoreKind kind{ScoreKind::Unbounded};
};
struct RankingCacheStats {
    bool supported{false};
    std::uint64_t budget_bytes{}, reserved_bytes{}, used_bytes{}, peak_bytes{};
    std::uint64_t active_sequences{}, allocation_failures{};
};
class IStructuredRanking {
  public:
    using TaskInterface = IStructuredRanking;
    static constexpr std::string_view kTask = "structured_ranking";
    virtual ~IStructuredRanking() = default;
    virtual CandidateScoresResult run(const StructuredRankingRequest&, ConfigView) = 0;
    virtual RankingCacheStats ranking_cache_stats() const { return {}; }
    // Exactly one nonzero cap, or both zero to use the family default. The
    // family owns ratio accounting, range checks, admission and lifecycle.
    virtual void configure_cache_budget(std::uint64_t, double) {
        throw UnsupportedTask("ranking cache configuration is unsupported");
    }
};
} // namespace trtmc::internal
