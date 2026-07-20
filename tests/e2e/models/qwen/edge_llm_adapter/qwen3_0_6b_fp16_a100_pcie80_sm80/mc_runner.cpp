// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "qualification_runner.h"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <trtmc/pipeline.h>

namespace {

static_assert(CUDART_VERSION == 13030, "qualification requires CUDA 13.3 headers");

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

qwen_edge_qualification::RuntimeVersions observe_runtime_versions() {
    int cuda_runtime = 0;
    check_cuda(cudaRuntimeGetVersion(&cuda_runtime), "cudaRuntimeGetVersion");
    qwen_edge_qualification::RuntimeVersions versions{
        getInferLibMajorVersion(), getInferLibMinorVersion(), getInferLibPatchVersion(),
        getInferLibBuildVersion(), cuda_runtime};
    if (versions.tensorrt_major != 11 || versions.tensorrt_minor != 2 ||
        versions.tensorrt_patch != 0 || versions.tensorrt_build != 113)
        throw std::runtime_error("loaded TensorRT runtime is not 11.2.0.113");
    if (versions.cuda_runtime != 13030)
        throw std::runtime_error("loaded CUDA runtime is not 13030");
    return versions;
}

void synchronize() {
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto paths = qwen_edge_qualification::parse_cli(argc, argv);
        const auto config =
            qwen_edge_qualification::read_configuration(paths.request, "model-connect");
        qwen_edge_qualification::require_exact_keys(
            config.runtime, {"kind", "bundle", "runtime_cache"}, "request.runtime");
        const auto bundle = qwen_edge_qualification::require_path(config.runtime, "bundle");
        const auto runtime_cache =
            qwen_edge_qualification::require_path(config.runtime, "runtime_cache");

        trtmc::LoadOptions load_options;
        load_options.runtime_cache_path = runtime_cache.string();
        auto pipeline = trtmc::load(bundle.string(), load_options);
        if (!pipeline)
            throw std::runtime_error("trtmc::load returned a null pipeline");
        const auto versions = observe_runtime_versions();
        trtmc::GenerateConfig generation;
        generation.max_new_tokens = config.max_new_tokens;
        generation.temperature = config.temperature;
        generation.top_p = config.top_p;
        generation.top_k = config.top_k;
        generation.use_chat_template = config.use_chat_template;
        generation.enable_thinking = config.enable_thinking;

        auto generate = [&]() {
            trtmc::TextResult result = pipeline->generate(config.prompt, generation);
            return qwen_edge_qualification::Sample{std::move(result.text),
                                                   std::move(result.token_ids)};
        };
        const auto measurements = qwen_edge_qualification::measure(generate, synchronize);
        qwen_edge_qualification::write_result(paths.output, "model-connect", versions,
                                              measurements);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Model Connect qualification runner failed: " << error.what() << '\n';
        return 1;
    }
}
