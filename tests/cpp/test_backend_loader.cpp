/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/backend_loader.h"

#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool cond, const char* name) {
    if (!cond) {
        std::cerr << "FAIL: " << name << std::endl;
        ++failures;
    }
}

static void check_incompatible_backend(const std::string& backend_name,
                                       const std::vector<std::string>& expected_fragments) {
    bool threw = false;
    try {
        (void)trtmc::BackendLoader::load(backend_name, {TRTMC_TEST_BACKEND_DIR});
    } catch (const std::runtime_error& error) {
        threw = true;
        const std::string message = error.what();
        for (const auto& fragment : expected_fragments) {
            check(message.find(fragment) != std::string::npos,
                  ("incompatible backend error contains '" + fragment + "'").c_str());
        }
        check(message.find("missing trtmc_create_backend_v2") == std::string::npos,
              "API ABI is checked before the backend factory");
    }
    check(threw, "incompatible backend is rejected");
}

int main() {
    // Loading a nonexistent backend should throw
    bool threw = false;
    try {
        trtmc::BackendLoader::load("nonexistent_backend_xyz", {"/tmp/trtmc-missing-backends"});
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        check(msg.find("nonexistent_backend_xyz") != std::string::npos,
              "error mentions backend name");
        check(msg.find("libtrtmc_backend_nonexistent_backend_xyz.so") != std::string::npos,
              "error mentions DSO name");
        check(msg.find("/tmp/trtmc-missing-backends/libtrtmc_backend_nonexistent_backend_xyz.so") !=
                  std::string::npos,
              "error mentions explicit backend search dir");
        check(msg.find("TRTMC_BACKEND_DIR") == std::string::npos, "error does not mention env var");
        check(msg.find("backend_search_paths") != std::string::npos,
              "error mentions explicit search paths");
    }
    check(threw, "missing backend throws runtime_error");

    check_incompatible_backend(
        "test_missing_api_abi",
        {"missing required trtmc_backend_api_abi_version symbol", "same source revision"});
    check_incompatible_backend(
        "test_wrong_api_abi",
        {"TRTMC backend API ABI 0",
         "runtime requires " + std::to_string(trtmc::kTrtmcBackendApiAbiVersion),
         "same source revision"});

    dlerror();
    void* v1_handle = dlopen(TRTMC_TEST_BACKEND_V1_PATH, RTLD_NOW | RTLD_LOCAL);
    check(v1_handle != nullptr, "v1 backend probe DSO opens");
    if (v1_handle != nullptr) {
        dlerror();
        void* legacy_factory = dlsym(v1_handle, "trtmc_create_backend");
        const char* legacy_error = dlerror();
        check(legacy_factory == nullptr && legacy_error != nullptr,
              "v1 backend does not export the legacy factory");

        dlerror();
        void* v1_factory = dlsym(v1_handle, "trtmc_create_backend_v1");
        const char* v1_error = dlerror();
        check(v1_factory != nullptr && v1_error == nullptr,
              "v1 backend exports the versioned factory");

        dlerror();
        void* v2_factory = dlsym(v1_handle, "trtmc_create_backend_v2");
        const char* v2_error = dlerror();
        check(v2_factory == nullptr && v2_error != nullptr,
              "v1 backend does not export the v2 factory");
        dlclose(v1_handle);
    }

    check_incompatible_backend(
        "test_api_abi_v1", {"TRTMC backend API ABI 1",
                            "runtime requires " + std::to_string(trtmc::kTrtmcBackendApiAbiVersion),
                            "same source revision"});

    dlerror();
    void* v2_handle = dlopen(TRTMC_TEST_BACKEND_V2_PATH, RTLD_NOW | RTLD_LOCAL);
    check(v2_handle != nullptr, "v2 backend probe DSO opens");
    if (v2_handle != nullptr) {
        dlerror();
        void* v2_factory = dlsym(v2_handle, "trtmc_create_backend_v2");
        const char* v2_error = dlerror();
        check(v2_factory != nullptr && v2_error == nullptr,
              "v2 backend exports the current factory");
        dlclose(v2_handle);
    }

    std::string loaded_backend_name;
    trtmc::BackendLoadMetadata metadata;
    auto* v2_backend = trtmc::BackendLoader::load_first_available(
        {"test_api_abi_v2"}, {TRTMC_TEST_BACKEND_DIR}, &loaded_backend_name, &metadata);
    check(v2_backend != nullptr, "new core loads a v2 backend");
    if (v2_backend != nullptr)
        check(std::string(v2_backend->name()) == "test_api_abi_v2", "v2 backend name is usable");
    check(loaded_backend_name == "test_api_abi_v2", "v2 backend candidate is reported");
    check(metadata.backend_api_abi == trtmc::kTrtmcBackendApiAbiVersion,
          "v2 backend ABI is reported in load metadata");

    std::cerr << (failures == 0 ? "ALL PASSED" : "SOME FAILED") << std::endl;
    return failures;
}
