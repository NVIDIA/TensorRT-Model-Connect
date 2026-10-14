/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Model-agnostic platform mechanics for runtime-loaded shared libraries:
// dlopen/dlsym on ELF platforms, LoadLibraryExW/GetProcAddress on Windows.

#include <filesystem>
#include <string>
#include <string_view>

namespace trtmc::platform {

// Platform file name of a shared library built from CMake target output name
// `stem`: "lib<stem>.so" on ELF platforms and "<stem>.dll" on Windows.
std::string shared_library_filename(std::string_view stem);

// An owned handle to a loaded shared library. Construction throws
// std::runtime_error with `purpose`, the requested library, and the platform
// loader error when the library cannot be loaded.
//
// `name_or_path` is either a bare file name, resolved through the platform
// search order (LD_LIBRARY_PATH / PATH), or a path. On Windows, a path also
// makes the loader search that library's directory for its own dependencies.
class DynamicLibrary {
  public:
    DynamicLibrary(const std::string& name_or_path, std::string purpose);
    ~DynamicLibrary();

    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;

    // nullptr when the symbol is absent.
    void* find_symbol(const char* name) const noexcept;

    // Throws std::runtime_error naming the purpose, library, and symbol when
    // the symbol is absent.
    void* require_symbol(const char* name) const;

    template <typename Function>
    Function require(const char* name) const {
        return reinterpret_cast<Function>(require_symbol(name));
    }

    // The name or path passed to the constructor.
    const std::string& name() const noexcept { return name_; }

    // The file the platform loader actually mapped, or name() when the
    // platform cannot report it.
    std::string loaded_path() const;

  private:
    std::string name_;
    std::string purpose_;
    void* handle_{nullptr};
};

// Absolute path of the executable or shared library that contains `address`.
// Throws std::runtime_error when the platform cannot resolve it.
std::filesystem::path module_path_containing(const void* address);

// Absolute path of the running executable.
std::filesystem::path current_executable_path();

// Environment variable that overrides the NCCL shared library for every
// runtime that loads NCCL at run time.
inline constexpr const char* kNcclLibraryEnv = "TRTMC_NCCL_LIBRARY";

// "nccl.dll" on Windows, "libnccl.so.2" on ELF platforms.
const char* default_nccl_library();

// TRTMC_NCCL_LIBRARY when set and non-empty, otherwise default_nccl_library().
std::string nccl_library();

} // namespace trtmc::platform
