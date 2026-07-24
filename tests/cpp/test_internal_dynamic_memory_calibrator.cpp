/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/internal_dynamic_memory_calibrator.h"
#include "native_dynamic_memory_calibrator_paths.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#ifndef TRTMC_TEST_TRTMC_EXECUTABLE
#error "TRTMC_TEST_TRTMC_EXECUTABLE must name the built trtmc executable"
#endif
#ifndef TRTMC_TEST_CALIBRATOR_EXECUTABLE
#error "TRTMC_TEST_CALIBRATOR_EXECUTABLE must name the built calibrator executable"
#endif

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class TemporaryTree {
  public:
    TemporaryTree() {
        std::string pattern =
            (std::filesystem::temp_directory_path() / "trtmc-calibrator-test-XXXXXX").string();
        pattern.push_back('\0');
        path_ = mkdtemp(pattern.data());
        if (path_.empty())
            throw std::runtime_error("mkdtemp failed");
    }

    ~TemporaryTree() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path& path() const { return path_; }

  private:
    std::filesystem::path path_;
};

void make_executable(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    constexpr char kElfFixture[] = {'\x7f', 'E', 'L', 'F', 't', 'e', 's', 't'};
    output.write(kElfFixture, sizeof(kElfFixture));
    const auto& marker =
        trtmc::cli::internal_dynamic_memory_calibrator_identity_marker();
    output.write(marker.data(), static_cast<std::streamsize>(marker.size()));
    output.close();
    if (chmod(path.c_str(), 0700) != 0)
        throw std::runtime_error("chmod failed");
}

void make_mismatched_executable(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    constexpr char kElfFixture[] = {'\x7f', 'E', 'L', 'F', 't', 'e', 's', 't'};
    output.write(kElfFixture, sizeof(kElfFixture));
    output << "TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR_IDENTITY_V1:"
           << trtmc::cli::internal_dynamic_memory_calibrator_product_version()
           << ':' << std::string(64, '0');
    output.close();
    if (chmod(path.c_str(), 0700) != 0)
        throw std::runtime_error("chmod failed");
}

void test_build_tree_candidate_is_exact_and_first() {
    const auto trtmc = std::filesystem::canonical(TRTMC_TEST_TRTMC_EXECUTABLE);
    const auto helper =
        std::filesystem::canonical(TRTMC_TEST_CALIBRATOR_EXECUTABLE);

    const auto candidates =
        trtmc::cli::internal_dynamic_memory_calibrator_candidates(trtmc);
    check(!candidates.empty() && candidates.front() == helper,
          "build-tree helper adjacent to trtmc is the first exact candidate");
    const auto resolved = trtmc::cli::find_internal_dynamic_memory_calibrator(trtmc);
    check(resolved.has_value() && *resolved == std::filesystem::canonical(helper),
          "build-tree helper resolves to its canonical absolute path");
}

void test_non_executable_candidate_is_rejected() {
    TemporaryTree tree;
    const auto trtmc = tree.path() / "bin/trtmc";
    make_executable(trtmc);
    const auto helper = trtmc.parent_path() /
                        trtmc::cli::kInternalDynamicMemoryCalibratorPackageDirectory /
                        trtmc::cli::kInternalDynamicMemoryCalibratorName;
    std::filesystem::create_directories(helper.parent_path());
    std::ofstream(helper, std::ios::binary) << "\x7f"
                                              "ELF";

    const auto resolved = trtmc::cli::find_internal_dynamic_memory_calibrator(trtmc);
    check(!resolved.has_value(), "a non-executable helper fails closed");
}

void test_installed_libexec_candidate_is_discovered() {
    TemporaryTree tree;
    const auto trtmc = tree.path() / "bin/trtmc";
    make_executable(trtmc);
    const auto candidates =
        trtmc::cli::internal_dynamic_memory_calibrator_candidates(trtmc);
    check(candidates.size() == 2,
          "installed layout exposes one private package and one libexec candidate");
    if (candidates.size() != 2)
        return;
    make_executable(candidates[1]);

    const auto resolved = trtmc::cli::find_internal_dynamic_memory_calibrator(trtmc);
    check(resolved.has_value() && *resolved == std::filesystem::canonical(candidates[1]),
          "installed private libexec helper resolves to its canonical absolute path");
}

void test_inherited_override_is_replaced() {
    TemporaryTree tree;
    const auto trtmc = tree.path() / "bin/trtmc";
    make_executable(trtmc);
    const auto helper = trtmc.parent_path() /
                        trtmc::cli::kInternalDynamicMemoryCalibratorPackageDirectory /
                        trtmc::cli::kInternalDynamicMemoryCalibratorName;
    make_executable(helper);
    setenv(trtmc::cli::kInternalDynamicMemoryCalibratorEnv, "/tmp/user-controlled", 1);
    setenv(trtmc::cli::kInternalDynamicMemoryCalibratorBuildIdentityEnv,
           std::string(64, '0').c_str(), 1);

    std::string error;
    check(trtmc::cli::configure_internal_dynamic_memory_calibrator(trtmc, error),
          "product helper configuration succeeds");
    const char* selected = std::getenv(trtmc::cli::kInternalDynamicMemoryCalibratorEnv);
    check(selected != nullptr && std::filesystem::path(selected) == std::filesystem::canonical(helper),
          "inherited helper override is replaced by the exact product path");
    const char* identity =
        std::getenv(trtmc::cli::kInternalDynamicMemoryCalibratorBuildIdentityEnv);
    check(identity != nullptr &&
              identity ==
                  trtmc::cli::internal_dynamic_memory_calibrator_build_identity(),
          "inherited build identity is replaced by the compiled product identity");
    check(error.empty(), "successful helper configuration has no error");
    unsetenv(trtmc::cli::kInternalDynamicMemoryCalibratorEnv);
    unsetenv(trtmc::cli::kInternalDynamicMemoryCalibratorBuildIdentityEnv);
}

void test_tampered_packaged_helper_is_rejected() {
    TemporaryTree tree;
    const auto trtmc = tree.path() / "bin/trtmc";
    make_executable(trtmc);
    const auto helper = trtmc.parent_path() /
                        trtmc::cli::kInternalDynamicMemoryCalibratorPackageDirectory /
                        trtmc::cli::kInternalDynamicMemoryCalibratorName;
    make_executable(helper);
    chmod(helper.c_str(), 0722);

    check(!trtmc::cli::find_internal_dynamic_memory_calibrator(trtmc).has_value(),
          "a group/world-writable packaged helper fails closed");

    chmod(helper.c_str(), 0700);
    std::ofstream(helper, std::ios::binary | std::ios::trunc) << "tampered";
    check(!trtmc::cli::find_internal_dynamic_memory_calibrator(trtmc).has_value(),
          "a packaged helper without ELF identity fails closed");
}

void test_build_identity_mismatch_is_rejected() {
    TemporaryTree tree;
    const auto trtmc = tree.path() / "bin/trtmc";
    make_executable(trtmc);
    const auto helper = trtmc.parent_path() /
                        trtmc::cli::kInternalDynamicMemoryCalibratorPackageDirectory /
                        trtmc::cli::kInternalDynamicMemoryCalibratorName;
    make_mismatched_executable(helper);

    check(!trtmc::cli::find_internal_dynamic_memory_calibrator(trtmc).has_value(),
          "an executable helper from another product build is rejected");
}

void test_missing_helper_clears_inherited_override() {
    TemporaryTree tree;
    const auto trtmc = tree.path() / "build-dynkv/trtmc";
    make_executable(trtmc);
    setenv(trtmc::cli::kInternalDynamicMemoryCalibratorEnv, "/tmp/user-controlled", 1);
    setenv(trtmc::cli::kInternalDynamicMemoryCalibratorBuildIdentityEnv,
           std::string(64, '0').c_str(), 1);

    std::string error;
    check(trtmc::cli::configure_internal_dynamic_memory_calibrator(trtmc, error),
          "missing product helper does not block an ordinary build");
    check(std::getenv(trtmc::cli::kInternalDynamicMemoryCalibratorEnv) == nullptr,
          "missing product helper never preserves an inherited override");
    check(std::getenv(
              trtmc::cli::kInternalDynamicMemoryCalibratorBuildIdentityEnv) ==
              nullptr,
          "missing product helper clears an inherited build identity");
    check(error.empty(), "missing optional helper does not report a launcher error");
}

void test_wheel_script_helper_discovers_installed_package_bin() {
    TemporaryTree tree;
    const auto helper =
        tree.path() / "venv/bin/.trtmc-internal/trtmc_dynamic_memory_qualify";
    make_executable(helper);
    const auto package_bin =
        tree.path() /
        "venv/lib/python3.12/site-packages/tensorrt_model_connect/bin";
    std::filesystem::create_directories(package_bin);

    const auto paths =
        trtmc::qualification::internal_calibrator_search_paths(helper, "../../lib");
    const auto package_path = package_bin.lexically_normal().string();
    check(std::find(paths.backend.begin(), paths.backend.end(), package_path) !=
              paths.backend.end(),
          "wheel-script helper discovers the installed package backend directory");
    check(std::find(paths.model_plugin.begin(), paths.model_plugin.end(), package_path) !=
              paths.model_plugin.end(),
          "wheel-script helper discovers the installed package model-plugin directory");
}

} // namespace

int main() {
    test_build_tree_candidate_is_exact_and_first();
    test_non_executable_candidate_is_rejected();
    test_installed_libexec_candidate_is_discovered();
    test_inherited_override_is_replaced();
    test_tampered_packaged_helper_is_rejected();
    test_build_identity_mismatch_is_rejected();
    test_missing_helper_clears_inherited_override();
    test_wheel_script_helper_discovers_installed_package_bin();
    if (failures != 0)
        std::cerr << failures << " test(s) failed\n";
    return failures == 0 ? 0 : 1;
}
