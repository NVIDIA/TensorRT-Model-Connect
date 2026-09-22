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

namespace trtmc::patchtsmixer {

struct RuntimeConfig {
    std::int32_t context_length;
    std::int32_t num_input_channels;
    std::int32_t prediction_length;
    std::int32_t tensor_parallel_size;
};

RuntimeConfig parse_runtime_config(const std::string& json);

class Pipeline final : public internal::IModel, public internal::ISeriesToPointForecast {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, RuntimeConfig config);

    const char* task() const noexcept override { return ISeriesToPointForecast::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ISeriesToPointForecast>(*this, fields_)};
    }
    internal::PointForecastResult run(const internal::SeriesToPointForecastRequest& request,
                                      internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    RuntimeConfig config_;
    inline static const internal::ConfigField fields_[] = {
        {"frequency", internal::ConfigKind::I64, internal::ConfigValue{std::int64_t{0}},
         "Only the unspecified frequency category (zero) is supported"},
    };
};

} // namespace trtmc::patchtsmixer
