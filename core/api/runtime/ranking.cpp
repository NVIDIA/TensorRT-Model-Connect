/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/ranking.h"

#include "api_internal.h"
#include "trtmc/ranking.h"

#include <cmath>
#include <limits>
#include <set>

namespace trtmc::api {
namespace {
internal::RankingFeatureView feature_view(const trtmc_ranking_feature_v1& input) {
    const auto name = string_view(input.name);
    require(!name.empty(), "ranking feature name is empty");
    std::uint64_t size = input.type == TRTMC_RANKING_INT64 ? 8 : 4;
    require(input.type == TRTMC_RANKING_FLOAT32 || input.type == TRTMC_RANKING_INT32 ||
                input.type == TRTMC_RANKING_INT64,
            "unknown ranking feature type");
    const auto shape = checked_span(input.shape, input.rank);
    require(!shape.empty(), "ranking feature rank is zero");
    for (const auto extent : shape) {
        require(extent >= 0, "ranking feature extent is negative");
        require(!extent || size <= std::numeric_limits<std::uint64_t>::max() /
                                       static_cast<std::uint64_t>(extent),
                "ranking feature shape overflows");
        size *= static_cast<std::uint64_t>(extent);
    }
    require(size == input.byte_size, "ranking feature byte size differs from shape");
    const auto bytes = checked_span(static_cast<const std::uint8_t*>(input.data), input.byte_size);
    return {name, static_cast<internal::RankingFeatureType>(input.type), shape, bytes};
}

struct RankingStorage final : ResultStorage {
    explicit RankingStorage(internal::CandidateScoresResult result) : value(std::move(result)) {
        view.scores = matrix_result_view(value.scores);
        require_valid_scores();
        view.kind = static_cast<std::uint32_t>(value.kind);
    }
    void require_valid_scores() const {
        if (value.kind != internal::ScoreKind::Logit &&
            value.kind != internal::ScoreKind::Probability &&
            value.kind != internal::ScoreKind::Unbounded)
            throw ApiFailure{TRTMC_INTERNAL_ERROR, "family returned an invalid ranking score kind"};
        for (float score : value.scores.values)
            if (!std::isfinite(score) ||
                (value.kind == internal::ScoreKind::Probability && (score < 0 || score > 1)))
                throw ApiFailure{TRTMC_INTERNAL_ERROR, "family returned an invalid ranking score"};
    }
    internal::CandidateScoresResult value;
    trtmc_candidate_scores_view_v1 view{};
};

trtmc_status TRTMC_CALL run_ranking(trtmc_model* model,
                                    const trtmc_structured_ranking_request_v1* request,
                                    const trtmc_config_view_v1* config, trtmc_result** out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(request && out, "ranking request or result output is null");
        const std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<internal::IStructuredRanking>(
            model, internal::IStructuredRanking::kTask);
        const ConvertedConfig options(config);
        validate_task_config(model_owner(model),
                             internal::contract_key<internal::IStructuredRanking>(),
                             options.view());
        const auto inputs = checked_span(request->features, request->feature_count);
        require(!inputs.empty(), "ranking features are required");
        std::vector<internal::RankingFeatureView> features;
        std::set<std::string_view> names;
        features.reserve(inputs.size());
        for (const auto& input : inputs) {
            auto feature = feature_view(input);
            require(names.insert(feature.name).second, "duplicate ranking feature name");
            features.push_back(feature);
        }
        *out = make_result<RankingStorage>(
            family.run({{features.data(), features.size()}}, options.view()));
    });
}
trtmc_status TRTMC_CALL ranking_view(const trtmc_result* result,
                                     trtmc_candidate_scores_view_v1* out,
                                     trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out, "ranking result view output is null");
        *out = require_result<RankingStorage>(result).view;
    });
}
trtmc_status TRTMC_CALL cache_stats(trtmc_model* model, trtmc_ranking_cache_stats_v1* out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out, "ranking cache statistics output is null");
        const std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<internal::IStructuredRanking>(
            model, internal::IStructuredRanking::kTask);
        const auto stats = family.ranking_cache_stats();
        *out = {stats.supported ? 1U : 0U, stats.budget_bytes, stats.reserved_bytes,
                stats.used_bytes,          stats.peak_bytes,   stats.active_sequences,
                stats.allocation_failures};
    });
}
trtmc_status TRTMC_CALL configure_cache(trtmc_model* model, std::uint64_t bytes, double ratio,
                                        trtmc_error** error) noexcept {
    return guarded(error, [&] {
        const std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<internal::IStructuredRanking>(
            model, internal::IStructuredRanking::kTask);
        family.configure_cache_budget(bytes, ratio);
    });
}
const trtmc_structured_ranking_api_v1 ranking_api = {
    {1, 0, sizeof(trtmc_structured_ranking_api_v1)},
    run_ranking,
    ranking_view,
    cache_stats,
    configure_cache};
} // namespace
Span<const TaskBinding> ranking_task_bindings() noexcept {
    static const TaskBinding bindings[] = {
        {internal::IStructuredRanking::kTask, 1, 0, &ranking_api.header}};
    return {bindings};
}
} // namespace trtmc::api
