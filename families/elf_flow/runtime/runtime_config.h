/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>

namespace trtmc::elf_flow {

void validate_runtime_config_json(std::string_view json);

} // namespace trtmc::elf_flow
