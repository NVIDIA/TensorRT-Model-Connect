/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/config/config_bundle.h"

#include <string>
#include <vector>

namespace trtmc::detail {

config::ConfigBundle resolve_runtime_config(const std::string& config_text,
                                            const std::string& bundle_path,
                                            const std::string& config_path,
                                            const std::vector<std::string>& set_tokens);

} // namespace trtmc::detail
