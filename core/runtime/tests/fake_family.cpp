/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>

namespace {

class FakeForecast final : public trtmc::ITimeSeriesForecast {
  public:
    explicit FakeForecast(std::uint64_t kv_cache_size_bytes)
        : kv_cache_size_bytes_(kv_cache_size_bytes) {}

    trtmc::ForecastResult forecast(const trtmc::ForecastRequest& request) override {
        if (!request.observed_mask.empty() &&
            request.observed_mask.size() != request.past_values.size()) {
            throw std::invalid_argument("observed_mask length must match past_values");
        }

        trtmc::ForecastResult result;
        result.values.assign(request.past_values.begin(), request.past_values.end());
        result.shape = {
            static_cast<std::int64_t>(kv_cache_size_bytes_ == 0 ? 1 : kv_cache_size_bytes_),
            static_cast<std::int64_t>(result.values.size()),
        };
        return result;
    }

  private:
    std::uint64_t kv_cache_size_bytes_;
};

} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "fake")
        throw std::runtime_error("unexpected family");
    if (context.backend.name() == nullptr || std::string(context.backend.name()) != "fake")
        throw std::runtime_error("unexpected backend");
    const auto plan = context.reader.read_section("engine.plan");
    if (std::string(plan.begin(), plan.end()) != "PLAN")
        throw std::runtime_error("unexpected engine plan");
    return new FakeForecast(context.kv_cache_size_bytes);
}
