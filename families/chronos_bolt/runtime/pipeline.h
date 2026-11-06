/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/model.h"
#include "trtmc/internal/numeric.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc::chronos_bolt {

struct RuntimeConfig {
    std::int32_t context_length;
    std::int32_t prediction_length;
    std::vector<double> quantile_levels;
    std::int32_t tensor_parallel_size;
};

class Pipeline final : public internal::IModel, public internal::ISeriesToQuantileForecast {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, RuntimeConfig config);

    const char* task() const noexcept override { return ISeriesToQuantileForecast::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ISeriesToQuantileForecast>(*this, fields_)};
    }
    internal::QuantileForecastResult run(const internal::SeriesToQuantileForecastRequest& request,
                                         internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    RuntimeConfig config_;
    inline static const internal::ConfigField fields_[] = {
        {"frequency", internal::ConfigKind::I64, internal::ConfigValue{std::int64_t{0}},
         "Only the unspecified frequency category (zero) is supported"},
    };
};

} // namespace trtmc::chronos_bolt
