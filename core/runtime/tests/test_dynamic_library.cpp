/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/dynamic_library.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;
using trtmc::platform::DynamicLibrary;

int failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr)
        unsetenv(name);
    else
        setenv(name, value, 1);
#endif
}

template <typename Function>
std::string error_of(Function&& function) {
    try {
        function();
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return {};
}

using GetVersionFn = int (*)(int*);

void test_file_names() {
#if defined(_WIN32)
    check(trtmc::platform::shared_library_filename("trtmc_model_flux") == "trtmc_model_flux.dll",
          "windows family file name");
    check(std::string(trtmc::platform::default_nccl_library()) == "nccl.dll", "windows NCCL name");
#else
    check(trtmc::platform::shared_library_filename("trtmc_model_flux") == "libtrtmc_model_flux.so",
          "ELF family file name");
    check(std::string(trtmc::platform::default_nccl_library()) == "libnccl.so.2", "ELF NCCL name");
#endif
}

void test_nccl_library_override() {
    set_env(trtmc::platform::kNcclLibraryEnv, nullptr);
    check(trtmc::platform::nccl_library() == trtmc::platform::default_nccl_library(),
          "NCCL default without override");
    set_env(trtmc::platform::kNcclLibraryEnv, "");
    check(trtmc::platform::nccl_library() == trtmc::platform::default_nccl_library(),
          "empty override keeps the default");
    set_env(trtmc::platform::kNcclLibraryEnv, "/custom/nccl-build/nccl.dll");
    check(trtmc::platform::nccl_library() == "/custom/nccl-build/nccl.dll",
          "TRTMC_NCCL_LIBRARY overrides the NCCL library");
    set_env(trtmc::platform::kNcclLibraryEnv, nullptr);
}

void test_missing_library(const fs::path& directory) {
    const auto missing = directory / trtmc::platform::shared_library_filename("no_such_nccl");
    const auto message =
        error_of([&] { DynamicLibrary library(missing.string(), "Unit test: NCCL"); });
    check(contains(message, "Unit test: NCCL"), "load error names the purpose: " + message);
    check(contains(message, "unable to load"), "load error says it cannot load: " + message);
    check(contains(message, missing.string()), "load error names the library: " + message);

    const auto bare = error_of([] { DynamicLibrary library("trtmc_no_such_library_xyz", "x"); });
    check(contains(bare, "trtmc_no_such_library_xyz"), "bare-name load error: " + bare);

    const auto empty = error_of([] { DynamicLibrary library("", "empty"); });
    check(contains(empty, "empty shared library name"), "empty name error: " + empty);
}

void test_partial_library(const fs::path& partial) {
    DynamicLibrary library(partial.string(), "Unit test: NCCL");
    check(library.name() == partial.string(), "name is the requested path");
    check(fs::equivalent(fs::path(library.loaded_path()), partial),
          "loaded_path is the mapped file: " + library.loaded_path());

    const auto get_version = library.require<GetVersionFn>("ncclGetVersion");
    int version = 0;
    check(get_version(&version) == 0 && version == 23007, "resolved symbol is callable");
    check(library.find_symbol("ncclGetUniqueId") != nullptr, "find_symbol finds exports");
    check(library.find_symbol("ncclCommInitRank") == nullptr, "find_symbol returns null");
    check(library.find_symbol(nullptr) == nullptr, "find_symbol(nullptr) is null");

    const auto message = error_of([&] { (void)library.require_symbol("ncclCommInitRank"); });
    check(contains(message, "Unit test: NCCL"), "symbol error names the purpose: " + message);
    check(contains(message, "missing required symbol 'ncclCommInitRank'"),
          "symbol error names the symbol: " + message);
    check(contains(message, partial.filename().string()),
          "symbol error names the library: " + message);
}

void test_module_paths(const char* argv0) {
    const auto executable = trtmc::platform::current_executable_path();
    check(executable.is_absolute(), "executable path is absolute: " + executable.string());
    check(executable.stem() == fs::path(argv0).stem(),
          "current_executable_path is this test: " + executable.string());
    const auto containing = trtmc::platform::module_path_containing(
        reinterpret_cast<const void*>(&trtmc::platform::nccl_library));
    check(containing.filename() == trtmc::platform::shared_library_filename("trtmc_core"),
          "module_path_containing finds trtmc_core: " + containing.string());
    check(containing.is_absolute(), "module path is absolute");
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_dynamic_library <fake partial NCCL library>\n";
        return 2;
    }
    const fs::path partial = fs::absolute(argv[1]);
    try {
        test_file_names();
        test_nccl_library_override();
        test_missing_library(partial.parent_path());
        test_partial_library(partial);
        test_module_paths(argv[0]);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0)
        return 1;
    std::cout << "dynamic_library: all checks passed\n";
    return 0;
}
