/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/sam2/sam2_native_bundle_loader.h"

#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc::sam2::benchmark {

// Benchmark-local fixed-shape TensorRT module factory. Unlike the generic
// backend at the time this diagnostic was authored, every enqueueV3 return
// value is checked. The caller owns one nonblocking stream; all six modules
// borrow that exact stream and retain the shared TensorRT runtime themselves.
class CheckedPlanModuleFactory final {
  public:
    // Exposed only so the benchmark-local implementation can share one runtime
    // and borrowed stream across its six concrete module instances.
    struct SharedState;

    explicit CheckedPlanModuleFactory(cudaStream_t stream);
    ~CheckedPlanModuleFactory();

    CheckedPlanModuleFactory(const CheckedPlanModuleFactory&) = delete;
    CheckedPlanModuleFactory& operator=(const CheckedPlanModuleFactory&) = delete;

    std::unique_ptr<ITrtModule> create(std::string_view section, const void* plan_data,
                                       std::size_t plan_size) const;
    NativePlanModuleFactory callback() const;

    // Digests are computed directly over the plan_data spans passed by the
    // authenticated loader to the six successful deserializations.
    std::vector<std::pair<std::string, std::string>> loadedPlanSha256() const;

  private:
    std::shared_ptr<SharedState> state_;
};

} // namespace trtmc::sam2::benchmark
