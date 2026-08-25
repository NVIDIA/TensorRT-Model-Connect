/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/platform/dynamic_library.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

} // namespace

int main() {
#if defined(_WIN32)
    check(trtmc::internal::dynamic_library_filename("trtmc_example") == "trtmc_example.dll",
          "Windows library filename");
    check(trtmc::internal::path_list_separator() == ';', "Windows path-list separator");
    check(std::string(trtmc::internal::dynamic_library_search_path_environment()) == "PATH",
          "Windows loader path environment");
#else
    check(trtmc::internal::dynamic_library_filename("trtmc_example") == "libtrtmc_example.so",
          "Linux library filename");
    check(trtmc::internal::path_list_separator() == ':', "POSIX path-list separator");
    check(std::string(trtmc::internal::dynamic_library_search_path_environment()) ==
              "LD_LIBRARY_PATH",
          "Linux loader path environment");
#endif

    const auto executable = trtmc::internal::current_executable_path();
    check(!executable.empty(), "current executable path is available");
    check(executable.empty() || std::filesystem::exists(executable),
          "current executable path exists");

    std::string error;
    auto missing = trtmc::internal::open_dynamic_library(
        std::filesystem::temp_directory_path() / "trtmc-missing-dynamic-library",
        trtmc::internal::DynamicLibraryVisibility::local, &error);
    check(missing == nullptr, "missing dynamic library is rejected");
    check(!error.empty(), "missing dynamic library reports an error");

    auto handle = trtmc::internal::open_dynamic_library(
        TRTMC_TEST_CORE_LIBRARY, trtmc::internal::DynamicLibraryVisibility::local, &error);
    check(handle != nullptr, "core dynamic library opens");
    if (handle != nullptr) {
        auto symbol = trtmc::internal::dynamic_library_symbol(handle, "trtmc_version", &error);
        check(symbol != nullptr, "core C ABI symbol resolves");
        check(trtmc::internal::close_dynamic_library(handle, &error),
              "core dynamic library closes");
    }

    std::cerr << (failures == 0 ? "ALL PASSED" : "SOME FAILED") << '\n';
    return failures;
}
