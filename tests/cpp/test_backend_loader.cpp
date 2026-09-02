/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/backend_loader.h"
#include "runtime/backend/prebound_backend.h"
#include "runtime/platform/dynamic_library.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

static int failures = 0;

static void check(bool cond, const char* name) {
    if (!cond) {
        std::cerr << "FAIL: " << name << std::endl;
        ++failures;
    }
}

int main() {
#if defined(TRTMC_LOCKED_H3_RUNTIME)
    bool rejected = false;
    try {
        trtmc::BackendLoader::load("trt_rtx", {"C:\\untrusted"});
    } catch (const std::runtime_error& error) {
        rejected = true;
        check(std::string(error.what()).find("rejects backend search path overrides") !=
                  std::string::npos,
              "locked backend loader explains rejected search path");
    }
    check(rejected, "locked backend loader rejects explicit search paths");

    bool wrong_backend_rejected = false;
    try {
        trtmc::BackendLoader::load("trt");
    } catch (const std::runtime_error& error) {
        wrong_backend_rejected = true;
        check(std::string(error.what()).find("only the TensorRT-RTX backend") != std::string::npos,
              "locked backend loader explains backend allowlist");
    }
    check(wrong_backend_rejected, "locked backend loader rejects non-RTX backends");

    auto* backend = trtmc::BackendLoader::load("trt_rtx");
    check(backend != nullptr, "locked backend loader loads adjacent TensorRT-RTX backend");
    auto* runtime_cache_backend = dynamic_cast<trtmc::IRuntimeCacheBackend*>(backend);
    check(runtime_cache_backend != nullptr,
          "TensorRT-RTX backend exposes explicit runtime-cache persistence");
    if (runtime_cache_backend != nullptr) {
        const std::string cache_path =
            (std::filesystem::temp_directory_path() / "trtmc-runtime-cache-lease-test.bin")
                .string();
        const std::uint64_t lease =
            runtime_cache_backend->acquire_runtime_cache_lease(cache_path.c_str());
        check(lease != 0, "runtime-cache backend returns a valid lease");
        runtime_cache_backend->release_runtime_cache_lease(lease);

        bool double_release_rejected = false;
        try {
            runtime_cache_backend->release_runtime_cache_lease(lease);
        } catch (const std::invalid_argument&) {
            double_release_rejected = true;
        }
        check(double_release_rejected, "runtime-cache backend rejects an inactive lease");
    }
#else
    const std::string backend_name = "nonexistent_backend_xyz";
    const auto missing_dir = std::filesystem::temp_directory_path() / "trtmc-missing-backends";
    const std::string library_name =
        trtmc::internal::dynamic_library_filename("trtmc_backend_" + backend_name);
    const std::string missing_path = (missing_dir / library_name).string();

    // Loading a nonexistent backend should throw
    bool threw = false;
    try {
        trtmc::BackendLoader::load(backend_name, {missing_dir.string()});
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        check(msg.find(backend_name) != std::string::npos, "error mentions backend name");
        check(msg.find(library_name) != std::string::npos, "error mentions DSO name");
        check(msg.find(missing_path) != std::string::npos,
              "error mentions explicit backend search dir");
        check(msg.find("TRTMC_BACKEND_DIR") == std::string::npos, "error does not mention env var");
        check(msg.find("backend_search_paths") != std::string::npos,
              "error mentions explicit search paths");
    }
    check(threw, "missing backend throws runtime_error");
#endif

    std::cerr << (failures == 0 ? "ALL PASSED" : "SOME FAILED") << std::endl;
    return failures;
}
