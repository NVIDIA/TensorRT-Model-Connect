/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/runtime_memory_backend.h"

#include <memory>
#include <vector>

namespace {

class AbiFixtureBackend final : public trtmc::IBackend {
  public:
    std::unique_ptr<trtmc::ITrtModule> create_module(const void*, size_t,
                                                     const trtmc::ModuleCreateOptions&) override {
        return nullptr;
    }

    trtmc::BackendDualProfileModules
    create_dual_profile_modules(const void*, size_t, const trtmc::ModuleCreateOptions&) override {
        return {};
    }

    trtmc::BackendProfileModules create_profile_modules(const void*, size_t,
                                                        const trtmc::ModuleCreateOptions&,
                                                        const std::vector<int32_t>&) override {
        return {};
    }

    trtmc::BackendContextModules
    create_context_modules(const void*, size_t,
                           const std::vector<trtmc::ModuleCreateOptions>&) override {
        return {};
    }

    const char* name() const override { return "abi_fixture"; }
};

} // namespace

extern "C" std::int32_t
trtmc_backend_query_abi_contract_v2(trtmc::BackendDsoAbiContractV2* contract,
                                    std::size_t contract_size) noexcept {
    if (contract == nullptr || contract_size < sizeof(*contract))
        return -1;
    *contract = trtmc::make_runtime_memory_backend_dso_abi_contract_v2(0);
    return 0;
}

extern "C" trtmc::IBackend* trtmc_create_backend() {
    return new AbiFixtureBackend();
}

extern "C" void trtmc_destroy_backend(trtmc::IBackend* backend) {
    delete backend;
}
