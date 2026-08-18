/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/providers/optimized_runtime_factory.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <unistd.h>

#ifndef TRTMC_FAKE_OPTIMIZED_IMPLEMENTATION_ID
#define TRTMC_FAKE_OPTIMIZED_IMPLEMENTATION_ID "example-optimized-runtime"
#endif

#ifndef TRTMC_FAKE_OPTIMIZED_PIPELINE_ABI_VERSION
#define TRTMC_FAKE_OPTIMIZED_PIPELINE_ABI_VERSION                                                  \
    trtmc::internal::kOptimizedRuntimePipelineAbiVersionV1
#endif

namespace {

void append_event(const char* event) noexcept {
    const char* path = std::getenv("TRTMC_FAKE_OPTIMIZED_EVENTS");
    if (path == nullptr || path[0] == '\0')
        return;
    const int descriptor = open(path, O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (descriptor < 0)
        return;
    char line[64];
    const int size = std::snprintf(line, sizeof(line), "%s\n", event);
    if (size > 0) {
        const ssize_t written = write(descriptor, line, static_cast<std::size_t>(size));
        (void)written;
    }
    (void)close(descriptor);
}

void set_error(char* output, std::size_t capacity, const char* message) noexcept {
    if (output != nullptr && capacity != 0)
        std::snprintf(output, capacity, "%s", message);
}

struct LoadRecorder {
    LoadRecorder() { append_event("dlopen"); }
    ~LoadRecorder() { append_event("dlclose"); }
};

LoadRecorder g_load_recorder;

class FakePipeline final : public trtmc::IPipeline {
  public:
    explicit FakePipeline(std::string model_id) : model_id_(std::move(model_id)) {}
    ~FakePipeline() override { append_event("destroy"); }

    trtmc::TextResult generate(const std::string& input,
                               const trtmc::GenerateConfig& config) override {
        append_event("generate");
        if (input == "fail")
            throw std::runtime_error("fake optimized runtime generate failure");
        return trtmc::TextResult("optimized:" + input, {41, config.max_new_tokens}, 0.5, 1.0);
    }

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "FakeTextGenerationPipeline"; }

  private:
    std::string model_id_;
};

bool validate_capsule_inputs(
    const trtmc::internal::OptimizedRuntimePipelineCreateRequestV1* request, char* error,
    std::size_t error_capacity) noexcept {
    const char* expected_metadata = std::getenv("TRTMC_FAKE_OPTIMIZED_EXPECT_METADATA");
    if (expected_metadata != nullptr) {
        const std::size_t expected_size = std::strlen(expected_metadata);
        if (request->implementation_metadata == nullptr ||
            request->implementation_metadata_size != expected_size ||
            std::memcmp(request->implementation_metadata, expected_metadata, expected_size) != 0) {
            set_error(error, error_capacity, "private implementation metadata mismatch");
            return false;
        }
    }
    const char* expected_artifact = std::getenv("TRTMC_FAKE_OPTIMIZED_EXPECT_ARTIFACT");
    if (expected_artifact != nullptr) {
        if (request->artifact_path == nullptr) {
            set_error(error, error_capacity, "missing artifact path");
            return false;
        }
        std::ifstream input(std::string(request->artifact_path) + "/payload/runtime.data",
                            std::ios::binary);
        const std::string contents((std::istreambuf_iterator<char>(input)),
                                   std::istreambuf_iterator<char>());
        if (!input.is_open() || input.bad() || contents != expected_artifact) {
            set_error(error, error_capacity, "materialized artifact mismatch");
            return false;
        }
    }
    return true;
}

trtmc::IPipeline*
create_pipeline(const trtmc::internal::OptimizedRuntimePipelineCreateRequestV1* request,
                char* error, std::size_t error_capacity) noexcept {
    append_event("create");
    if (request == nullptr ||
        request->abi_version != trtmc::internal::kOptimizedRuntimeFactoryAbiVersionV1 ||
        request->struct_size < sizeof(trtmc::internal::OptimizedRuntimePipelineCreateRequestV1) ||
        request->implementation_id == nullptr || request->model_id == nullptr ||
        request->profile_id == nullptr || request->bundle_path == nullptr ||
        request->artifact_path == nullptr || request->load_options == nullptr) {
        set_error(error, error_capacity, "invalid create request");
        return nullptr;
    }
    if (const char* failure = std::getenv("TRTMC_FAKE_OPTIMIZED_FAIL_CREATE");
        failure != nullptr && failure[0] != '\0') {
        set_error(error, error_capacity, "fake optimized runtime create failure");
        return nullptr;
    }
    if (!validate_capsule_inputs(request, error, error_capacity))
        return nullptr;
    try {
        return new FakePipeline(request->model_id);
    } catch (...) {
        set_error(error, error_capacity, "allocation failed");
        return nullptr;
    }
}

const trtmc::internal::OptimizedRuntimeFactoryV1 kFactory = {
    trtmc::internal::kOptimizedRuntimeFactoryAbiVersionV1,
    sizeof(trtmc::internal::OptimizedRuntimeFactoryV1),
    TRTMC_FAKE_OPTIMIZED_IMPLEMENTATION_ID,
    "test-optimized-runtime",
    "test-runtime-1.0",
    "test-runtime-commit",
    &create_pipeline,
    TRTMC_FAKE_OPTIMIZED_PIPELINE_ABI_VERSION,
};

const trtmc::internal::OptimizedRuntimeFactoryV1 kWrongToolchainFactory = [] {
    auto factory = kFactory;
    ++factory.toolchain_abi.compiler_major_version;
    return factory;
}();

} // namespace

extern "C" const trtmc::internal::OptimizedRuntimeFactoryV1*
trtmc_get_optimized_runtime_factory_v1() noexcept {
    if (const char* mismatch = std::getenv("TRTMC_FAKE_OPTIMIZED_WRONG_TOOLCHAIN_ABI");
        mismatch != nullptr && mismatch[0] != '\0') {
        return &kWrongToolchainFactory;
    }
    return &kFactory;
}
