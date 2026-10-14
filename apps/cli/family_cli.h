/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>

namespace trtmc::cli {

// Returns no value only when the invocation does not select a declared family.
// The explicit executable path is a test seam; production resolves the running executable.
std::optional<int> run_family_cli(int argc, char** argv, std::ostream& output, std::ostream& error,
                                  const std::filesystem::path& executable = {});

// Quotes one argument for a Windows command line so that the C runtime and
// CommandLineToArgvW parse it back as exactly that argument. Arguments without
// whitespace or quotes are returned unchanged. Portable, so it is unit tested
// on every platform.
std::string quote_windows_argument(const std::string& argument);

} // namespace trtmc::cli
