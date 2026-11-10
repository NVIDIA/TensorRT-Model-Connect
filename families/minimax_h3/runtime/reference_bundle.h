/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minimax_h3/runtime/reference_pipeline.h"

#include <nlohmann/json_fwd.hpp>

namespace trtmc::minimax_h3 {

// Validate the explicit original image-reference bundle ABI before any plan
// deserialization. Parsing metadata is not trained-model qualification.
ReferencePipelineConfig parse_image_reference_config(const nlohmann::json& runtime);
std::vector<std::string> image_reference_plan_names();

} // namespace trtmc::minimax_h3
