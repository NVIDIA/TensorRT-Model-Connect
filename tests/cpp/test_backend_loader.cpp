/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/backend_loader.h"

#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef TRTMC_TEST_ABI_BACKEND_DSO
#error "TRTMC_TEST_ABI_BACKEND_DSO must name the paired backend fixture"
#endif
#ifndef TRTMC_TEST_STALE_V1_BACKEND_DSO
#error "TRTMC_TEST_STALE_V1_BACKEND_DSO must name the stale V1 backend fixture"
#endif
#ifndef TRTMC_TEST_STALE_FINGERPRINT_BACKEND_DSO
#error "TRTMC_TEST_STALE_FINGERPRINT_BACKEND_DSO must name the stale fingerprint fixture"
#endif

static int failures = 0;

static void check(bool cond, const char* name) {
    if (!cond) {
        std::cerr << "FAIL: " << name << std::endl;
        ++failures;
    }
}

static void check_rejected_before_factory(const char* fixture_path, const char* backend_name,
                                          const char* expected_error,
                                          const char* rejection_check_name) {
    void* fixture = dlopen(fixture_path, RTLD_NOW | RTLD_LOCAL);
    check(fixture != nullptr, "stale backend fixture opens");
    if (fixture == nullptr)
        return;

    using CounterFn = std::uint32_t (*)();
    auto counter =
        reinterpret_cast<CounterFn>(dlsym(fixture, "trtmc_test_stale_backend_create_calls"));
    check(counter != nullptr, "stale backend exposes create counter");
    if (counter == nullptr) {
        dlclose(fixture);
        return;
    }
    check(counter() == 0, "stale backend create counter starts at zero");

    bool threw = false;
    try {
        const std::filesystem::path path(fixture_path);
        (void)trtmc::BackendLoader::load(backend_name, {path.parent_path().string()});
    } catch (const std::runtime_error& error) {
        threw = true;
        const std::string message = error.what();
        check(message.find(expected_error) != std::string::npos, rejection_check_name);
        check(message.find("before trtmc_create_backend()") != std::string::npos,
              "stale rejection identifies pre-factory gate");
    }
    check(threw, "stale backend throws runtime_error");
    check(counter() == 0, "stale backend factory was never called");
    dlclose(fixture);
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

    const std::filesystem::path paired_backend = TRTMC_TEST_ABI_BACKEND_DSO;
    trtmc::BackendLoadMetadata metadata;
    auto* backend = trtmc::BackendLoader::load_first_available(
        {"abi_fixture"}, {paired_backend.parent_path().string()}, nullptr, &metadata);
    check(backend != nullptr, "paired backend passes ABI contract");
    check(backend != nullptr && std::string(backend->name()) == "abi_fixture",
          "paired backend virtual call is safe after contract gate");
    check(metadata.backend_abi_contract_version == trtmc::kBackendDsoAbiContractVersionV2,
          "paired backend reports ABI contract V2");
    check(metadata.runtime_memory_backend_api_version == 2,
          "paired backend reports runtime-memory API V2");

    check_rejected_before_factory(TRTMC_TEST_STALE_V1_BACKEND_DSO, "stale_v1_fixture",
                                  trtmc::kBackendDsoAbiQuerySymbolV2,
                                  "stale V1 rejection names missing ABI query");
    check_rejected_before_factory(TRTMC_TEST_STALE_FINGERPRINT_BACKEND_DSO,
                                  "stale_fingerprint_fixture", "interface_fingerprint",
                                  "stale fingerprint rejection names mismatched field");

    std::cerr << (failures == 0 ? "ALL PASSED" : "SOME FAILED") << std::endl;
    return failures;
}
