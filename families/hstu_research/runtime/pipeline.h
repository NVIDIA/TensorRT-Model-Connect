/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/hstu_research/include/task.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace trtmc::hstu_research {

std::size_t checked_bytes(const std::vector<std::int64_t>& shape, DType dtype);
DType parse_dtype(const std::string& name);
const char* dtype_name(DType dtype);

TensorMap validate_inputs(const std::vector<FeatureTensor>& inputs, const nlohmann::json& config);

class Pipeline final : public IStructuredRanking {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, nlohmann::json config);
    std::vector<FeatureTensor> score(const std::vector<FeatureTensor>& inputs) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    nlohmann::json config_;
};
} // namespace trtmc::hstu_research
