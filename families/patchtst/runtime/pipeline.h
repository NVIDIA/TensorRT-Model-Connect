/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>

namespace trtmc::patchtst {

struct RuntimeConfig {
    std::int32_t context_length;
    std::int32_t num_input_channels;
    std::int32_t prediction_length;
    std::int32_t tensor_parallel_size;
    std::string output_name;
};

class Pipeline final : public ITimeSeriesForecast {
  public:
    Pipeline(std::unique_ptr<ITrtModule> module, RuntimeConfig config);

    ForecastResult forecast(const ForecastRequest& request) override;

  private:
    std::unique_ptr<ITrtModule> module_;
    RuntimeConfig config_;
};

} // namespace trtmc::patchtst
