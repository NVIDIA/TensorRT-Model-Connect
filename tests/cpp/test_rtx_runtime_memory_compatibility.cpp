/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/runtime_memory_backend.h"
#include "trtmc/runtime/trt_backend.h"

#include <dlfcn.h>
#include <iostream>
#include <memory>

#ifndef TRTMC_TEST_RTX_BACKEND_DSO
#error "TRTMC_TEST_RTX_BACKEND_DSO must name the built RTX backend DSO"
#endif

int main() {
    void* library = dlopen(TRTMC_TEST_RTX_BACKEND_DSO, RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        std::cerr << "FAIL: could not load RTX backend: " << dlerror() << '\n';
        return 1;
    }

    using CreateFn = trtmc::IBackend* (*)();
    using DestroyFn = void (*)(trtmc::IBackend*);
    auto* query = reinterpret_cast<trtmc::BackendDsoAbiQueryFnV2>(
        dlsym(library, trtmc::kBackendDsoAbiQuerySymbolV2));
    auto* create = reinterpret_cast<CreateFn>(dlsym(library, "trtmc_create_backend"));
    auto* destroy = reinterpret_cast<DestroyFn>(dlsym(library, "trtmc_destroy_backend"));
    if (query == nullptr || create == nullptr || destroy == nullptr) {
        std::cerr << "FAIL: RTX backend factory symbols are missing\n";
        dlclose(library);
        return 1;
    }

    trtmc::BackendDsoAbiContractV2 contract{};
    const auto expected = trtmc::make_runtime_memory_backend_dso_abi_contract_v2(0);
    if (query(&contract, sizeof(contract)) != 0 || contract.struct_size != expected.struct_size ||
        contract.contract_version != expected.contract_version ||
        contract.interface_fingerprint != expected.interface_fingerprint ||
        contract.runtime_memory_layout_fingerprint != expected.runtime_memory_layout_fingerprint ||
        contract.runtime_memory_api_version != trtmc::kRuntimeMemoryBackendApiVersionCurrent ||
        contract.capability_flags != 0) {
        std::cerr << "FAIL: RTX backend/core ABI contract is incompatible\n";
        dlclose(library);
        return 1;
    }

    std::unique_ptr<trtmc::IBackend, DestroyFn> backend(create(), destroy);
    if (backend == nullptr) {
        std::cerr << "FAIL: RTX backend factory returned null\n";
        dlclose(library);
        return 1;
    }
    if (std::string(backend->name()) != "trt_rtx") {
        std::cerr << "FAIL: unexpected RTX backend identity\n";
        backend.reset();
        dlclose(library);
        return 1;
    }
    if (dynamic_cast<trtmc::IRuntimeMemoryBackendV1*>(backend.get()) != nullptr) {
        std::cerr << "FAIL: RTX backend must not advertise standard-TRT "
                     "runtime-memory capability\n";
        backend.reset();
        dlclose(library);
        return 1;
    }

    backend.reset();
    dlclose(library);
    std::cout << "RTX runtime-memory compatibility gate passed\n";
    return 0;
}
