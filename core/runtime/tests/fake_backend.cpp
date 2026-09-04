/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <vector>

namespace {

int create_count = 0;

class FakeBackend final : public trtmc::IBackend {
  public:
    std::unique_ptr<trtmc::ITrtModule> create_module(const void*, std::size_t,
                                                     const trtmc::ModuleCreateOptions&) override {
        return nullptr;
    }

    std::unique_ptr<trtmc::ITrtModule>
    create_module_prebound(const void*, std::size_t, const trtmc::ModuleCreateOptions&,
                           const std::vector<trtmc::ModuleExternalBinding>&) override {
        return nullptr;
    }

    trtmc::BackendDualProfileModules
    create_dual_profile_modules(const void*, std::size_t,
                                const trtmc::ModuleCreateOptions&) override {
        return {};
    }

    const char* name() const override { return "fake"; }
};

} // namespace

extern "C" trtmc::IBackend* trtmc_create_backend() {
    ++create_count;
    if (create_count != 1)
        return nullptr;
    return new FakeBackend();
}

extern "C" void trtmc_destroy_backend(trtmc::IBackend* backend) {
    delete backend;
}
