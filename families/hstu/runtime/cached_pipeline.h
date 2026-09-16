/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/hstu/runtime/pipeline.h"
#include "trtmc/history_cache.h"

namespace trtmc::hstu {

class CachedPipeline final : public IRecommendation,
                             public IHistoryCacheConsumer,
                             public IRecommendationSessionFactory {
  public:
    CachedPipeline(std::unique_ptr<ITrtModule> engine, std::unique_ptr<ITrtModule> candidate_engine,
                   RuntimeConfig config, std::shared_ptr<HistoryCache> cache = {});
    ~CachedPipeline() override;
    RecommendationResult recommend(const RecommendationRequest& request) override;
    void set_history_cache(std::shared_ptr<HistoryCache> cache) override;
    std::string history_cache_artifact_id() const override;
    std::unique_ptr<IRecommendationSession>
    create_recommendation_session(const RecommendationSequence& initial_history,
                                  std::size_t max_cache_bytes) override;

  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace trtmc::hstu
