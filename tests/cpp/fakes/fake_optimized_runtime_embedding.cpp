/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/providers/optimized_runtime_factory.h"

#include <cstdio>
#include <string>

namespace {

constexpr const char* kImplementationId = "example-optimized-embedding";

class FakeEmbeddingPipeline final : public trtmc::IPipeline {
  public:
    explicit FakeEmbeddingPipeline(std::string model_id) : model_id_(std::move(model_id)) {}

    trtmc::EmbeddingResult embed(const std::string& input) override {
        return trtmc::EmbeddingResult{
            {static_cast<float>(input.size()), 3.5F, -2.0F},
            3,
        };
    }

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "FakeEmbeddingPipeline"; }

  private:
    std::string model_id_;
};

trtmc::IPipeline*
create_pipeline(const trtmc::internal::OptimizedRuntimePipelineCreateRequestV1* request,
                char* error, std::size_t error_capacity) noexcept {
    if (request == nullptr ||
        request->abi_version != trtmc::internal::kOptimizedRuntimeFactoryAbiVersionV1 ||
        request->struct_size < sizeof(trtmc::internal::OptimizedRuntimePipelineCreateRequestV1) ||
        request->model_id == nullptr || request->implementation_id == nullptr ||
        std::string(request->implementation_id) != kImplementationId) {
        if (error != nullptr && error_capacity != 0)
            std::snprintf(error, error_capacity, "%s", "invalid embedding create request");
        return nullptr;
    }
    try {
        return new FakeEmbeddingPipeline(request->model_id);
    } catch (...) {
        if (error != nullptr && error_capacity != 0)
            std::snprintf(error, error_capacity, "%s", "allocation failed");
        return nullptr;
    }
}

const trtmc::internal::OptimizedRuntimeFactoryV1 kFactory = {
    trtmc::internal::kOptimizedRuntimeFactoryAbiVersionV1,
    sizeof(trtmc::internal::OptimizedRuntimeFactoryV1),
    kImplementationId,
    "test-embedding-runtime",
    "test-embedding-1.0",
    "test-embedding-commit",
    &create_pipeline,
};

} // namespace

extern "C" const trtmc::internal::OptimizedRuntimeFactoryV1*
trtmc_get_optimized_runtime_factory_v1() noexcept {
    return &kFactory;
}
