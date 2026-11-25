/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/hstu/runtime/prototype_task.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace trtmc::hstu {

std::size_t checked_bytes(const std::vector<std::int64_t>& shape, DType dtype);
DType parse_dtype(const std::string& name);
const char* dtype_name(DType dtype);

TensorMap validate_inputs(const std::vector<FeatureTensor>& inputs, const nlohmann::json& config);

} // namespace trtmc::hstu
