/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/trt_backend.h"

namespace {

class FakeBackendV1 final : public trtmc::IBackend {
  public:
    std::unique_ptr<trtmc::ITrtModule> create_module(const void*, size_t,
                                                     const trtmc::ModuleCreateOptions&) override {
        return {};
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

    const char* name() const override { return "test_api_abi_v1"; }
};

} // namespace

extern "C" std::uint32_t trtmc_backend_api_abi_version() {
    return trtmc::kTrtmcBackendApiAbiVersion;
}

extern "C" trtmc::IBackend* trtmc_create_backend_v1() {
    return new FakeBackendV1();
}

extern "C" void trtmc_destroy_backend_v1(trtmc::IBackend* backend) {
    delete backend;
}
