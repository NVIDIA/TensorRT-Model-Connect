/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/core.hpp"
#include "trtmc/ranking.h"

namespace trtmc {
enum class RankingFeatureType : std::uint32_t {
    Float32 = TRTMC_RANKING_FLOAT32,
    Int32 = TRTMC_RANKING_INT32,
    Int64 = TRTMC_RANKING_INT64
};
struct RankingFeatureView {
    std::string_view name;
    RankingFeatureType type;
    Span<const std::int64_t> shape;
    Span<const std::uint8_t> bytes;
};
struct StructuredRankingRequest {
    Span<const RankingFeatureView> features;
};
class CandidateScoresResult : public detail::ViewResult<trtmc_candidate_scores_view_v1> {
  public:
    using ViewResult::ViewResult;
    trtmc_f32_matrix_view_v1 scores() const { return view().scores; }
    std::uint32_t kind() const { return view().kind; }
};
using RankingCacheStats = trtmc_ranking_cache_stats_v1;
class StructuredRanking {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_STRUCTURED_RANKING;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != kMajor || table->minor != kMinor ||
            table->byte_size < sizeof(trtmc_structured_ranking_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible structured ranking Task table");
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, kMajor, kMinor);
    }
    CandidateScoresResult run(const StructuredRankingRequest& input,
                              const Config& config = {}) const {
        std::vector<trtmc_ranking_feature_v1> features;
        for (const auto& feature : input.features)
            features.push_back({detail::c_string(feature.name),
                                static_cast<std::uint32_t>(feature.type), feature.shape.data(),
                                feature.shape.size(), feature.bytes.data(), feature.bytes.size()});
        const trtmc_structured_ranking_request_v1 request{features.data(), features.size()};
        const auto entries = config.c_entries();
        const auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run(model_->handle, &request, &options, &raw, &error);
        detail::ResultOwner owner(model_, raw);
        detail::check(model_->api, status, error);
        return CandidateScoresResult(std::move(owner), api_->result_view);
    }
    RankingCacheStats cache_stats() const {
        RankingCacheStats out{};
        trtmc_error* error = nullptr;
        const auto status = api_->cache_stats(model_->handle, &out, &error);
        detail::check(model_->api, status, error);
        return out;
    }
    void configure_cache(std::uint64_t bytes = 0, double memory_ratio = 0) const {
        trtmc_error* error = nullptr;
        const auto status = api_->configure_cache(model_->handle, bytes, memory_ratio, &error);
        detail::check(model_->api, status, error);
    }

  private:
    friend class Model;
    StructuredRanking(std::shared_ptr<detail::ModelState> model,
                      const trtmc_api_header* table) noexcept
        : model_(std::move(model)),
          api_(reinterpret_cast<const trtmc_structured_ranking_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> model_;
    const trtmc_structured_ranking_api_v1* api_;
};
} // namespace trtmc
