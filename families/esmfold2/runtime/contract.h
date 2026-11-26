/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/esmfold2/runtime/bundle_config.h"
#include "trtmc/runtime/tensor.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace trtmc::esmfold2 {

// What a loaded plan reports for one I/O tensor.
struct TensorFacts {
    bool input{true};
    DType dtype{DType::kFloat32};
    // Outputs: the shape the module allocated, for the inputs' largest profile.
    std::vector<std::int64_t> shape;
    // Inputs: the profile's smallest and largest shapes.
    std::vector<std::int64_t> min_profile;
    std::vector<std::int64_t> max_profile;
};
using PlanFacts = std::map<std::string, TensorFacts>;

// Checks one plan against the I/O contract that runtime.json implies: exactly the expected tensor
// names, their element types, each input's profile up to the bundle's maximum token, atom and
// sample counts, and each output's shape for that maximum. Throws std::invalid_argument naming the
// plan and the tensor.
void check_plan_contract(const std::string& plan, const PlanFacts& facts,
                         const BundleConfig& config);

} // namespace trtmc::esmfold2
