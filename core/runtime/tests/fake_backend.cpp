/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/stream_backend.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TRTMC_FAKE_BACKEND_NAME
#define TRTMC_FAKE_BACKEND_NAME "fake"
#endif

namespace {

int create_count = 0;
std::string last_runtime_cache_path;
bool last_cuda_graphs = false;
int buffered_calls = 0;
#ifdef TRTMC_TEST_STREAM_BACKEND
int streamed_calls = 0;
bool stream_supported = true, stream_failure = false;
#endif

void record_options(const trtmc::ModuleCreateOptions& options) {
    last_runtime_cache_path = options.runtime_cache_path != nullptr
                                  ? std::string(options.runtime_cache_path)
                                  : std::string();
    last_cuda_graphs = options.cuda_graphs;
}

class FakeBackend final : public trtmc::IBackend
#ifdef TRTMC_TEST_STREAM_BACKEND
    ,
                          public trtmc::IStreamBackend
#endif
{
  public:
    std::unique_ptr<trtmc::ITrtModule>
    create_module(const void* bytes, std::size_t size,
                  const trtmc::ModuleCreateOptions& options) override {
        ++buffered_calls;
        record_options(options);
#ifdef TRTMC_TEST_STREAM_BACKEND
        if (size != 4 || bytes == nullptr || std::memcmp(bytes, "PLAN", 4) != 0)
            throw std::runtime_error("Buffered fixture plan changed");
#else
        (void)bytes;
        (void)size;
#endif
        return nullptr;
    }

#ifdef TRTMC_TEST_STREAM_BACKEND
    bool supports_plan_streaming() const noexcept override { return stream_supported; }
    std::unique_ptr<trtmc::ITrtModule>
    create_module_from_stream(std::unique_ptr<trtmc::IPlanReader> plan,
                              const trtmc::ModuleCreateOptions& options,
                              const std::vector<trtmc::ModuleExternalBinding>& bindings) override {
        ++streamed_calls;
        record_options(options);
        char bytes[4];
        if (!bindings.empty() || plan->size() != 4 || plan->read(bytes, sizeof(bytes)) != 4 ||
            std::memcmp(bytes, "PLAN", 4) != 0 || plan->read(bytes, 1) != 0)
            throw std::runtime_error("Streamed fixture plan changed");
        if (stream_failure)
            throw std::runtime_error("Injected streamed backend failure");
        return nullptr;
    }
#endif

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

#ifdef TRTMC_TEST_STREAM_BACKEND
extern "C" int trtmc_test_backend_streamed_calls() {
    return streamed_calls;
}
extern "C" int trtmc_test_backend_buffered_calls() {
    return buffered_calls;
}
extern "C" void trtmc_test_backend_stream_supported(bool value) {
    stream_supported = value;
}
extern "C" void trtmc_test_backend_stream_failure(bool value) {
    stream_failure = value;
}
#endif
