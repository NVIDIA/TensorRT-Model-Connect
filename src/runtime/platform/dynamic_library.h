/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

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
void* dynamic_library_symbol_in_process(const char* name, std::string* error = nullptr);
bool close_dynamic_library(DynamicLibraryHandle handle, std::string* error = nullptr) noexcept;

std::filesystem::path current_executable_path() noexcept;
// Path of the shared library or executable that contains this implementation.
// Unlike current_executable_path(), this remains anchored to trtmc_core.dll
// when ModelConnect is hosted by an application in another directory.
std::filesystem::path current_module_path() noexcept;
std::vector<std::filesystem::path> loaded_dynamic_library_paths();
std::string dynamic_library_filename(std::string_view stem);
const char* dynamic_library_search_path_environment() noexcept;
char path_list_separator() noexcept;

} // namespace trtmc::internal
