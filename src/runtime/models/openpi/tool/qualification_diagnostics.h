/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/openpi/api.h"

#include <filesystem>
#include <string_view>

namespace trtmc::openpi::tool {

// Atomically write qualification-only tensor payloads beneath a new
// directory. The returned manifest uses the same tensor descriptor fields as
// the OpenPI reference artifacts consumed by the model-owned E2E harness.
std::filesystem::path write_qualification_diagnostics(const ActionDiagnosticResult& diagnostics,
                                                      const std::filesystem::path& output_directory,
                                                      std::string_view model_id);

} // namespace trtmc::openpi::tool
