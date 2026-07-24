/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/internal_dynamic_memory_calibrator.h"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

#ifndef TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR_INSTALL_RELATIVE
#define TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR_INSTALL_RELATIVE \
    "../libexec/trtmc/trtmc_dynamic_memory_qualify"
#endif
#ifndef TRTMC_INTERNAL_SOURCE_DIR
#define TRTMC_INTERNAL_SOURCE_DIR "."
#endif
#ifndef TRTMC_INTERNAL_PRODUCT_VERSION
#error "TRTMC_INTERNAL_PRODUCT_VERSION must be defined by the product build"
#endif
#ifndef TRTMC_INTERNAL_CALIBRATOR_BUILD_IDENTITY
#error "TRTMC_INTERNAL_CALIBRATOR_BUILD_IDENTITY must be defined by the product build"
#endif

namespace trtmc::cli {
namespace {

#if defined(__GNUC__) || defined(__clang__)
[[gnu::used]]
#endif
constexpr char kCompiledIdentityMarker[] =
    "TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR_IDENTITY_V1:"
    TRTMC_INTERNAL_PRODUCT_VERSION ":" TRTMC_INTERNAL_CALIBRATOR_BUILD_IDENTITY;
static_assert(sizeof(TRTMC_INTERNAL_CALIBRATOR_BUILD_IDENTITY) == 65,
              "calibrator build identity must be a lowercase SHA256");

std::filesystem::path normalized_absolute(const std::filesystem::path& path) {
    std::error_code absolute_error;
    auto absolute = std::filesystem::absolute(path, absolute_error);
    if (absolute_error)
        return {};
    absolute = absolute.lexically_normal();

    std::error_code canonical_error;
    const auto canonical = std::filesystem::weakly_canonical(absolute, canonical_error);
    return canonical_error ? absolute : canonical;
}

bool is_source_build_executable(const std::filesystem::path& executable) {
    std::error_code source_error;
    const auto source_root =
        std::filesystem::weakly_canonical(TRTMC_INTERNAL_SOURCE_DIR, source_error);
    if (source_error || source_root.empty())
        return false;
    std::error_code relative_error;
    const auto relative =
        std::filesystem::relative(executable, source_root, relative_error);
    if (relative_error || relative.empty())
        return false;
    const auto first = *relative.begin();
    return first.string().rfind("build", 0) == 0;
}

bool contains_exact_identity_marker(std::ifstream& input) {
    const auto& marker = internal_dynamic_memory_calibrator_identity_marker();
    std::array<char, 64 * 1024> buffer{};
    std::string window;
    window.reserve(buffer.size() + marker.size());
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count <= 0)
            break;
        window.append(buffer.data(), static_cast<std::size_t>(count));
        if (window.find(marker) != std::string::npos)
            return true;
        if (window.size() >= marker.size()) {
            window.erase(0, window.size() - (marker.size() - 1));
        }
    }
    return false;
}

bool is_product_owned_helper(const std::filesystem::path& executable,
                             const std::filesystem::path& helper) {
    struct stat executable_status {};
    struct stat helper_status {};
    if (stat(executable.c_str(), &executable_status) != 0 ||
        lstat(helper.c_str(), &helper_status) != 0) {
        return false;
    }
    if (!S_ISREG(executable_status.st_mode) || !S_ISREG(helper_status.st_mode) ||
        executable_status.st_uid != helper_status.st_uid ||
        (helper_status.st_mode & (S_IWGRP | S_IWOTH)) != 0 ||
        access(helper.c_str(), X_OK) != 0) {
        return false;
    }

    std::ifstream input(helper, std::ios::binary);
    char magic[4] = {};
    input.read(magic, sizeof(magic));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(magic)) ||
        magic[0] != '\x7f' || magic[1] != 'E' || magic[2] != 'L' ||
        magic[3] != 'F') {
        return false;
    }
    return contains_exact_identity_marker(input);
}

} // namespace

const std::string& internal_dynamic_memory_calibrator_product_version() {
    static const std::string value = TRTMC_INTERNAL_PRODUCT_VERSION;
    return value;
}

const std::string& internal_dynamic_memory_calibrator_build_identity() {
    static const std::string value = TRTMC_INTERNAL_CALIBRATOR_BUILD_IDENTITY;
    return value;
}

const std::string& internal_dynamic_memory_calibrator_identity_marker() {
    static const std::string value = kCompiledIdentityMarker;
    return value;
}

std::vector<std::filesystem::path>
internal_dynamic_memory_calibrator_candidates(const std::filesystem::path& trtmc_executable) {
    const auto executable = normalized_absolute(trtmc_executable);
    if (executable.empty())
        return {};
    const auto executable_dir = executable.parent_path();

    std::vector<std::filesystem::path> candidates;
    if (is_source_build_executable(executable))
        candidates.push_back(executable_dir / kInternalDynamicMemoryCalibratorName);
    candidates.push_back(executable_dir / kInternalDynamicMemoryCalibratorPackageDirectory /
                         kInternalDynamicMemoryCalibratorName);
    candidates.push_back(
        (executable_dir / TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR_INSTALL_RELATIVE)
            .lexically_normal());
    if (candidates.size() >= 2 &&
        candidates[candidates.size() - 1] == candidates[candidates.size() - 2]) {
        candidates.pop_back();
    }
    return candidates;
}

std::optional<std::filesystem::path>
find_internal_dynamic_memory_calibrator(const std::filesystem::path& trtmc_executable) {
    for (const auto& candidate :
         internal_dynamic_memory_calibrator_candidates(trtmc_executable)) {
        if (!is_product_owned_helper(trtmc_executable, candidate))
            continue;
        return normalized_absolute(candidate);
    }
    return std::nullopt;
}

bool configure_internal_dynamic_memory_calibrator(
    const std::filesystem::path& trtmc_executable, std::string& error) {
    error.clear();
    if (unsetenv(kInternalDynamicMemoryCalibratorEnv) != 0) {
        error = std::string("failed to clear inherited internal calibrator path: ") +
                std::strerror(errno);
        return false;
    }
    if (unsetenv(kInternalDynamicMemoryCalibratorBuildIdentityEnv) != 0) {
        error =
            std::string("failed to clear inherited internal calibrator build identity: ") +
            std::strerror(errno);
        return false;
    }

    const auto helper = find_internal_dynamic_memory_calibrator(trtmc_executable);
    if (!helper.has_value())
        return true;

    if (setenv(kInternalDynamicMemoryCalibratorEnv, helper->c_str(), 1) != 0) {
        error = std::string("failed to export the internal calibrator path: ") +
                std::strerror(errno);
        return false;
    }
    if (setenv(kInternalDynamicMemoryCalibratorBuildIdentityEnv,
               internal_dynamic_memory_calibrator_build_identity().c_str(), 1) != 0) {
        const auto saved_errno = errno;
        (void)unsetenv(kInternalDynamicMemoryCalibratorEnv);
        error = std::string("failed to export the internal calibrator build identity: ") +
                std::strerror(saved_errno);
        return false;
    }
    return true;
}

} // namespace trtmc::cli
