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
#include <string>
#include <string_view>

namespace trtmc::patchtst {

struct RuntimeConfig {
    std::int32_t context_length;
    std::int32_t num_input_channels;
    std::int32_t prediction_length;
    std::int32_t tensor_parallel_size;
    std::string output_name;
    std::string task{"series_to_point_forecast"};
    std::int32_t num_targets{1};
    std::string distribution{};
};

class Pipeline final : public internal::IModel,
                       public internal::ISeriesToPointForecast,
                       public internal::ISeriesToRegressionValues,
                       public internal::ISeriesToRegressionDistribution {
  public:
    Pipeline(std::unique_ptr<ITrtModule> module, RuntimeConfig config);

    const char* task() const noexcept override { return config_.task.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::PointForecastResult run(const internal::SeriesToPointForecastRequest& request,
                                      internal::ConfigView config) override;
    internal::RegressionValuesResult run(const internal::SeriesToRegressionValuesRequest& request,
                                         internal::ConfigView config) override;
    internal::RegressionDistributionResult
    run(const internal::SeriesToRegressionDistributionRequest& request,
        internal::ConfigView config) override;

  private:
    Tensor evaluate(const internal::SeriesHistory& history, internal::ConfigView config,
                    std::string_view required_task);
    std::unique_ptr<ITrtModule> module_;
    RuntimeConfig config_;
};

} // namespace trtmc::patchtst
