/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/runtime_root.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void copy_library(const fs::path& source, const fs::path& root, const char* name) {
    fs::copy_file(source, root / name, fs::copy_options::overwrite_existing);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: test_runtime_root ROOT RUNTIME BACKEND FAMILY\n";
        return 2;
    }

    const fs::path root = argv[1];
    const fs::path runtime_library = argv[2];
    const fs::path backend_library = argv[3];
    const fs::path family_library = argv[4];
    fs::remove_all(root);
    fs::create_directories(root);

    copy_library(backend_library, root, "libtrtmc_backend_fake.so");
    copy_library(family_library, root, "libtrtmc_model_fake.so");
    const trtmc::BundleInfo bundle{1, "fake", "time_series_forecast", "fake", {}};

    std::error_code error;
    check(fs::equivalent(trtmc::loaded_runtime_root(), runtime_library.parent_path(), error),
          "runtime loader reports its active installation root");
    check(trtmc::runtime_root_contains_bundle(bundle, root.string()),
          "loader contract accepts one complete plugin root");

    fs::remove(root / "libtrtmc_backend_fake.so");
    check(!trtmc::runtime_root_contains_bundle(bundle, root.string()),
          "loader contract rejects an incomplete root");
    copy_library(backend_library, root, "libtrtmc_backend_fake.so");

    check(!trtmc::runtime_root_contains_bundle(bundle, root.string(), true),
          "loader contract requires the BYOK DSO when requested");
    copy_library(runtime_library, root, "libtrtmc_byok_tvm_ffi.so");
    check(trtmc::runtime_root_contains_bundle(bundle, root.string(), true),
          "loader contract accepts a root-local BYOK DSO");

    fs::remove(root / "libtrtmc_model_fake.so");
    fs::create_symlink(family_library, root / "libtrtmc_model_fake.so");
    check(!trtmc::runtime_root_contains_bundle(bundle, root.string()),
          "loader contract rejects a DSO symlink that escapes the selected root");

    bool unsafe_rejected = false;
    try {
        const trtmc::BundleInfo unsafe{1, "../fake", "time_series_forecast", "fake", {}};
        (void)trtmc::runtime_root_contains_bundle(unsafe, root.string());
    } catch (const std::runtime_error&) {
        unsafe_rejected = true;
    }
    check(unsafe_rejected, "loader contract owns safe family and backend validation");

    fs::remove_all(root);
    if (failures != 0) {
        std::cerr << failures << " runtime-root test(s) failed\n";
        return 1;
    }
    std::cout << "runtime-root tests passed\n";
    return 0;
}
