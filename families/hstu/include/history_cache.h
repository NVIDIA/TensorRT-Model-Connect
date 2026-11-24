/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "families/hstu/include/task.h"

#include <nlohmann/json.hpp>

namespace trtmc::hstu {
// Release-coupled family capability. All methods own/copy retained inputs.
class IHistoryCache {
  public:
    virtual ~IHistoryCache() = default;
    virtual bool history_supported() const noexcept = 0;
    virtual void configure_history(std::size_t max_entries, double ttl_seconds,
                                   double lease_seconds) = 0;
    virtual std::vector<std::uint8_t> export_history(const std::vector<FeatureTensor>& history,
                                                     const std::vector<std::string>& revisions) = 0;
    virtual std::vector<std::string> restore_history(const std::vector<std::uint8_t>& snapshot) = 0;
    virtual std::vector<FeatureTensor>
    score_with_history(const std::vector<std::string>& handles,
                       const std::vector<std::string>& revisions,
                       const std::vector<FeatureTensor>& request) = 0;
    virtual nlohmann::json history_info(const std::string& handle) = 0;
    virtual void release_history(const std::vector<std::string>& handles) = 0;
    virtual nlohmann::json history_stats() const = 0;
};
} // namespace trtmc::hstu
