/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "trtmc/runtime/tensor.h"
#include "trtmc/task.h"

#include <cstdint>
#include <string>
#include <vector>
namespace trtmc {
// Model-agnostic host tensor transport and cache observability contract.
struct FeatureTensor {
    std::string name;
    DType dtype{DType::kFloat32};
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> bytes;
};
struct CacheStats {
    bool supported{false};
    std::uint64_t budget_bytes{}, reserved_bytes{}, used_bytes{}, peak_bytes{};
    std::uint64_t active_sequences{}, allocation_failures{};
};
class IStructuredRanking : public virtual ITask {
  public:
    static constexpr const char* kTask = "structured_ranking";
    const char* task() const noexcept override { return kTask; }
    virtual std::vector<FeatureTensor> score(const std::vector<FeatureTensor>& inputs) = 0;
    virtual CacheStats cache_stats() const { return {}; }
};
} // namespace trtmc

namespace trtmc::hstu {
using FeatureTensor = trtmc::FeatureTensor;
using IStructuredRanking = trtmc::IStructuredRanking;
} // namespace trtmc::hstu
