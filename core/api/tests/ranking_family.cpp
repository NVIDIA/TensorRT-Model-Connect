/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/ranking.h"
#include "trtmc/runtime/family_factory.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace {
using namespace trtmc::internal;
class RankingModel final : public IModel, public IStructuredRanking {
  public:
    explicit RankingModel(std::string mode) : mode_(std::move(mode)) {}
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        if (mode_ == "disabled")
            return {};
        return {bind<IStructuredRanking>(*this, {fields_, 1})};
    }
    CandidateScoresResult run(const StructuredRankingRequest& request, ConfigView config) override {
        if (mode_ == "must_not_run")
            throw std::runtime_error("invalid input reached fixture execution");
        const auto scale = config_get<double>(config, {fields_, 1}, "scale").value();
        const RankingFeatureView *continuous = nullptr, *counts = nullptr, *times = nullptr;
        for (const auto& f : request.features) {
            if (f.name == "continuous")
                continuous = &f;
            if (f.name == "counts")
                counts = &f;
            if (f.name == "times")
                times = &f;
        }
        if (!continuous || !counts || !times || continuous->type != RankingFeatureType::Float32 ||
            counts->type != RankingFeatureType::Int32 || times->type != RankingFeatureType::Int64 ||
            continuous->bytes.size() != 16 || counts->bytes.size() != 8 ||
            times->bytes.size() != 16)
            throw std::invalid_argument("fixture requires three mixed typed features");
        float x[4];
        std::int32_t n[2];
        std::int64_t t[2];
        std::memcpy(x, continuous->bytes.data(), sizeof(x));
        std::memcpy(n, counts->bytes.data(), sizeof(n));
        std::memcpy(t, times->bytes.data(), sizeof(t));
        CandidateScoresResult result{{{}, 2, 2}, ScoreKind::Logit};
        for (std::size_t i = 0; i < 4; ++i)
            result.scores.values.push_back(float(scale * (x[i] + n[i / 2] + (t[i / 2] >> 40))));
        stats_.peak_bytes = 128;
        if (mode_ == "bad_shape")
            result.scores.columns = 3;
        if (mode_ == "bad_kind")
            result.kind = static_cast<ScoreKind>(77);
        if (mode_ == "bad_scores")
            result.scores.values[0] = std::numeric_limits<float>::quiet_NaN();
        return result;
    }
    RankingCacheStats ranking_cache_stats() const override {
        return mode_ == "no_cache" ? RankingCacheStats{} : stats_;
    }
    void configure_cache_budget(std::uint64_t bytes, double ratio) override {
        if (mode_ == "no_cache")
            throw UnsupportedTask("fixture has no cache");
        if (!std::isfinite(ratio) || ratio < 0 || ratio > 1 || (bytes && ratio))
            throw std::invalid_argument("invalid fixture cache budget");
        const auto budget = bytes ? bytes : ratio ? std::uint64_t(1000000 * ratio) : 1024;
        if (budget < stats_.reserved_bytes)
            throw std::invalid_argument("fixture cache budget too small");
        stats_.budget_bytes = budget;
    }

  private:
    std::string mode_;
    ConfigField fields_[1] = {
        {"scale", ConfigKind::F64, ConfigValue{1.0}, "Fixture score multiplier"}};
    RankingCacheStats stats_{true, 1024, 128, 0, 0, 0, 0};
};
} // namespace
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return new RankingModel(context.reader.info().task);
}
