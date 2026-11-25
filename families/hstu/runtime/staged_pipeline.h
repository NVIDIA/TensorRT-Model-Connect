/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/hstu/include/history_cache.h"
#include "families/hstu/runtime/pipeline.h"

#include <array>
#include <chrono>
#include <map>
#include <mutex>

namespace trtmc::hstu {
class StagedPipeline final : public IStructuredRanking, public IHistoryCache {
  public:
    StagedPipeline(std::unique_ptr<ITrtModule> prepare, std::unique_ptr<ITrtModule> step,
                   nlohmann::json config, std::uint64_t cache_bytes = 0, double cache_ratio = 0,
                   std::unique_ptr<ITrtModule> request = {}, std::string bundle_identity = {});
    ~StagedPipeline() override;
    std::vector<FeatureTensor> score(const std::vector<FeatureTensor>& inputs) override;
    CacheStats cache_stats() const override;
    void set_profiling(bool enabled) override { profiling_ = enabled; }
    TimingProfile last_profile() const override { return profile_; }
    bool history_supported() const noexcept override { return bool(request_); }
    void configure_history(std::size_t max_entries, double ttl_seconds,
                           double lease_seconds) override;
    std::vector<std::uint8_t> export_history(const std::vector<FeatureTensor>&,
                                             const std::vector<std::string>&) override;
    std::vector<std::string> restore_history(const std::vector<std::uint8_t>&) override;
    std::vector<FeatureTensor> score_with_history(const std::vector<std::string>&,
                                                  const std::vector<std::string>&,
                                                  const std::vector<FeatureTensor>&) override;
    nlohmann::json history_info(const std::string&) override;
    void release_history(const std::vector<std::string>&) override;
    nlohmann::json history_stats() const override;

  private:
    std::unique_ptr<ITrtModule> prepare_, step_, request_;
    nlohmann::json config_;
    std::array<nlohmann::json, 2> stage_validation_;
    mutable std::mutex stats_mutex_;
    CacheStats stats_;
    std::uint64_t bytes_per_sequence_{};
    void set_active(std::int64_t batch);
    bool profiling_{false};
    TimingProfile profile_;
    std::unordered_map<std::string, DeviceTensor> state_;
    DeviceTensor picks_;
    std::int64_t batch_{-1};
    std::vector<std::int64_t> positions_;
    cudaEvent_t begin_{}, end_{};
    void reset_state(std::int64_t batch);
    std::mutex execute_mutex_;
    std::vector<FeatureTensor> run_rounds(std::int64_t batch);
    void bind_context(ITrtModule&, const nlohmann::json& specs, std::int64_t batch);
    TensorMap validate_stage(const std::vector<FeatureTensor>&, std::size_t stage) const;
    void increment(const char* key, std::uint64_t amount = 1);
    using HistoryClock = std::chrono::steady_clock;
    struct Entry {
        std::string id, key, revision, feature_digest;
        std::vector<DeviceTensor> tensors;
        HistoryClock::time_point expires, leased_until, touched;
    };
    std::map<std::string, Entry> entries_;
    std::string bundle_identity_, generation_;
    std::uint64_t next_handle_{}, scratch_bytes_{}, mask_bytes_per_sequence_{};
    std::size_t max_entries_{};
    double ttl_seconds_{}, lease_seconds_{};
    bool history_enabled_{};
    nlohmann::json history_stats_;
    void require_history(bool enabled = true) const;
    void expire_entries();
    void refresh_residency(std::uint64_t pending_kv = 0, std::uint64_t pending_mask = 0);
    Entry& find_entry(const std::string& handle);
};
} // namespace trtmc::hstu
