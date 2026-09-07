/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <string>
#include <vector>

#ifndef TRTMC_FAKE_BACKEND_NAME
#define TRTMC_FAKE_BACKEND_NAME "fake"
#endif

namespace {

int create_count = 0;
std::string last_runtime_cache_path;
bool last_cuda_graphs = false;

class FakeBackend final : public trtmc::IBackend {
  public:
    std::unique_ptr<trtmc::ITrtModule>
    create_module(const void*, std::size_t, const trtmc::ModuleCreateOptions& options) override {
        last_runtime_cache_path = options.runtime_cache_path != nullptr
                                      ? std::string(options.runtime_cache_path)
                                      : std::string();
        last_cuda_graphs = options.cuda_graphs;
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

    const char* name() const override { return TRTMC_FAKE_BACKEND_NAME; }
};

} // namespace

#ifdef TRTMC_FAKE_INCOMPATIBLE_BUILD
extern "C" const trtmc::PluginDescriptorV1* trtmc_plugin_descriptor_v1() noexcept {
    static const trtmc::PluginDescriptorV1 descriptor{
        sizeof(trtmc::PluginDescriptorV1), trtmc::kPluginDescriptorVersion,
        trtmc::PluginKind::kBackend, TRTMC_FAKE_BACKEND_NAME, "00000000000000000000000000000000"};
    return &descriptor;
}
#else
TRTMC_DEFINE_BACKEND_PLUGIN_V1(TRTMC_FAKE_BACKEND_NAME)
#endif

extern "C" trtmc::IBackend* trtmc_create_backend() {
    ++create_count;
    if (create_count != 1)
        return nullptr;
    return new FakeBackend();
}

extern "C" void trtmc_destroy_backend(trtmc::IBackend* backend) {
    delete backend;
}

extern "C" const char* trtmc_test_backend_last_runtime_cache_path() {
    return last_runtime_cache_path.c_str();
}

extern "C" bool trtmc_test_backend_last_cuda_graphs() {
    return last_cuda_graphs;
}
