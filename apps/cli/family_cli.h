/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <iosfwd>
#include <optional>

namespace trtmc::cli {

// Returns no value only when the invocation does not select a declared family.
// The explicit executable path is a test seam; production resolves /proc/self/exe.
std::optional<int> run_family_cli(int argc, char** argv, std::ostream& output, std::ostream& error,
                                  const std::filesystem::path& executable = {});

} // namespace trtmc::cli
