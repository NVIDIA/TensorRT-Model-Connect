/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/hstu/runtime/staged_pipeline.h"

#include <cstring>

namespace trtmc::hstu {
std::vector<internal::TaskInstance> StagedPipeline::task_bindings() {
    return {internal::bind<internal::IStructuredRanking>(*this)};
}
internal::CandidateScoresResult
StagedPipeline::run(const internal::StructuredRankingRequest& request,
                    internal::ConfigView config) {
    internal::validate_config({}, config);
    std::vector<FeatureTensor> features;
    for (const auto& input : request.features) {
        DType dtype;
        switch (input.type) {
        case internal::RankingFeatureType::Float32:
            dtype = DType::kFloat32;
            break;
        case internal::RankingFeatureType::Int32:
            dtype = DType::kInt32;
            break;
        case internal::RankingFeatureType::Int64:
            dtype = DType::kInt64;
            break;
        default:
            throw std::invalid_argument("unsupported ranking feature type");
        }
        FeatureTensor feature{
            std::string(input.name), dtype, {input.shape.begin(), input.shape.end()}, {}};
        if (!input.bytes.empty())
            feature.bytes.assign(input.bytes.begin(), input.bytes.end());
        features.push_back(std::move(feature));
    }
    const auto outputs = score(features);
    if (outputs.size() != 1 || outputs[0].name != "scores" || outputs[0].shape.size() != 2 ||
        outputs[0].dtype != DType::kFloat32)
        throw std::runtime_error("unexpected DLRMv3 score output");
    const auto& output = outputs[0];
    if (output.bytes.size() != checked_bytes(output.shape, output.dtype))
        throw std::runtime_error("invalid DLRMv3 score storage");
    internal::CandidateScoresResult result;
    result.scores.rows = output.shape[0];
    result.scores.columns = output.shape[1];
    result.scores.values.resize(output.bytes.size() / sizeof(float));
    std::memcpy(result.scores.values.data(), output.bytes.data(), output.bytes.size());
    result.kind = internal::ScoreKind::Probability;
    return result;
}
internal::RankingCacheStats StagedPipeline::ranking_cache_stats() const {
    const auto s = cache_stats();
    return {s.supported,  s.budget_bytes,     s.reserved_bytes,     s.used_bytes,
            s.peak_bytes, s.active_sequences, s.allocation_failures};
}
} // namespace trtmc::hstu
