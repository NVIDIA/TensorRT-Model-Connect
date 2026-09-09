/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace trtmc::internal {

using DynamicLibraryHandle = void*;

enum class DynamicLibraryVisibility {
    local,
    global,
};

DynamicLibraryHandle
open_dynamic_library(const std::filesystem::path& path,
                     DynamicLibraryVisibility visibility = DynamicLibraryVisibility::local,
                     std::string* error = nullptr);
void* dynamic_library_symbol(DynamicLibraryHandle handle, const char* name,
                             std::string* error = nullptr);
bool close_dynamic_library(DynamicLibraryHandle handle, std::string* error = nullptr) noexcept;

std::string dynamic_library_filename(std::string_view stem);

} // namespace trtmc::internal
