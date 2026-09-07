/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/runtime_root.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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

void change_build_cohort(const fs::path& library) {
    static constexpr char marker[] = "trtmc_build_cohort_";
    std::ifstream input(library, std::ios::binary);
    std::string contents{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::size_t marker_position = contents.find(marker);
    if (marker_position == std::string::npos)
        throw std::runtime_error("test library has no build-cohort marker");
    const std::size_t id_position = marker_position + sizeof(marker) - 1;
    contents[id_position] = contents[id_position] == '0' ? '1' : '0';
    std::ofstream output(library, std::ios::binary | std::ios::trunc);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: test_runtime_root ROOT CORE RUNTIME BACKEND FAMILY\n";
        return 2;
    }

    const fs::path root = argv[1];
    const fs::path core_library = argv[2];
    const fs::path runtime_library = argv[3];
    const fs::path backend_library = argv[4];
    const fs::path family_library = argv[5];
    fs::remove_all(root);
    fs::create_directories(root);

    copy_library(core_library, root, "libtrtmc_core.so");
    copy_library(runtime_library, root, "libtrtmc_runtime.so");
    copy_library(backend_library, root, "libtrtmc_backend_fake.so");
    copy_library(family_library, root, "libtrtmc_model_fake.so");
    const trtmc::BundleInfo bundle{1, "fake", "time_series_forecast", "fake", {}};

    std::error_code error;
    check(fs::equivalent(trtmc::loaded_runtime_root(), runtime_library.parent_path(), error),
          "runtime loader reports its active installation root");
    check(trtmc::runtime_root_matches_loaded_build(bundle, root.string()),
          "loader contract accepts one complete matching build cohort");

    fs::remove(root / "libtrtmc_backend_fake.so");
    check(!trtmc::runtime_root_matches_loaded_build(bundle, root.string()),
          "loader contract rejects an incomplete root");
    copy_library(backend_library, root, "libtrtmc_backend_fake.so");

    check(!trtmc::runtime_root_matches_loaded_build(bundle, root.string(), true),
          "loader contract requires the BYOK DSO when requested");
    copy_library(core_library, root, "libtrtmc_byok_tvm_ffi.so");
    check(trtmc::runtime_root_matches_loaded_build(bundle, root.string(), true),
          "loader contract accepts a matching BYOK DSO");

    change_build_cohort(root / "libtrtmc_model_fake.so");
    check(!trtmc::runtime_root_matches_loaded_build(bundle, root.string()),
          "loader contract rejects a family from another build cohort");
    copy_library(family_library, root, "libtrtmc_model_fake.so");

    {
        std::ofstream malformed(root / "libtrtmc_model_fake.so",
                                std::ios::binary | std::ios::trunc);
        malformed << "not an ELF file";
    }
    fs::resize_file(root / "libtrtmc_model_fake.so", 64ULL * 1024ULL * 1024ULL);
    check(!trtmc::runtime_root_matches_loaded_build(bundle, root.string()),
          "loader contract rejects a large malformed candidate with bounded reads");

    bool unsafe_rejected = false;
    try {
        const trtmc::BundleInfo unsafe{1, "../fake", "time_series_forecast", "fake", {}};
        (void)trtmc::runtime_root_matches_loaded_build(unsafe, root.string());
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
