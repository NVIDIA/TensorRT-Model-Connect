/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/logger.h"
#include "performance_runner.h"
#include "runtime/llmInferenceRuntime.h"

#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

using qwen_edge_performance::Configuration;
using qwen_edge_performance::Sample;
namespace edge = trt_edgellm::rt;

#if !defined(TRTMC_EXPECTED_TENSORRT_MAJOR) || !defined(TRTMC_EXPECTED_TENSORRT_MINOR) ||            \
    !defined(TRTMC_EXPECTED_TENSORRT_PATCH) || !defined(TRTMC_EXPECTED_TENSORRT_BUILD) ||            \
    !defined(TRTMC_EXPECTED_TENSORRT_VERSION) ||                                                     \
    !defined(TRTMC_EXPECTED_CUDA_RUNTIME_VERSION)
#error "performance dependency pins must be supplied by CMake"
#endif

#if defined(TRT_MAJOR_ENTERPRISE)
static_assert(TRT_MAJOR_ENTERPRISE == TRTMC_EXPECTED_TENSORRT_MAJOR &&
                  TRT_MINOR_ENTERPRISE == TRTMC_EXPECTED_TENSORRT_MINOR &&
                  TRT_PATCH_ENTERPRISE == TRTMC_EXPECTED_TENSORRT_PATCH &&
                  TRT_BUILD_ENTERPRISE == TRTMC_EXPECTED_TENSORRT_BUILD,
              "TensorRT headers do not match the performance dependency lock");
#else
static_assert(NV_TENSORRT_MAJOR == TRTMC_EXPECTED_TENSORRT_MAJOR &&
                  NV_TENSORRT_MINOR == TRTMC_EXPECTED_TENSORRT_MINOR &&
                  NV_TENSORRT_PATCH == TRTMC_EXPECTED_TENSORRT_PATCH &&
                  NV_TENSORRT_BUILD == TRTMC_EXPECTED_TENSORRT_BUILD,
              "TensorRT headers do not match the performance dependency lock");
#endif
static_assert(CUDART_VERSION == TRTMC_EXPECTED_CUDA_RUNTIME_VERSION,
              "CUDA headers do not match the performance dependency lock");

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

qwen_edge_performance::RuntimeVersions observe_runtime_versions() {
    int cuda_runtime = 0;
    check_cuda(cudaRuntimeGetVersion(&cuda_runtime), "cudaRuntimeGetVersion");
    qwen_edge_performance::RuntimeVersions versions{
        getInferLibMajorVersion(), getInferLibMinorVersion(), getInferLibPatchVersion(),
        getInferLibBuildVersion(), cuda_runtime};
    if (versions.tensorrt_major != TRTMC_EXPECTED_TENSORRT_MAJOR ||
        versions.tensorrt_minor != TRTMC_EXPECTED_TENSORRT_MINOR ||
        versions.tensorrt_patch != TRTMC_EXPECTED_TENSORRT_PATCH ||
        versions.tensorrt_build != TRTMC_EXPECTED_TENSORRT_BUILD)
        throw std::runtime_error("loaded TensorRT runtime is not "
                                 TRTMC_EXPECTED_TENSORRT_VERSION);
    if (versions.cuda_runtime != TRTMC_EXPECTED_CUDA_RUNTIME_VERSION)
        throw std::runtime_error("loaded CUDA runtime does not match the dependency lock");
    return versions;
}

class Plugin {
  public:
    explicit Plugin(const std::string& path)
        : handle_(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)) {
        if (handle_ == nullptr) {
            const char* error = dlerror();
            throw std::runtime_error(std::string("unable to load EdgeLLM plugin: ") +
                                     (error != nullptr ? error : "unknown dlopen error"));
        }
        using Initialize = bool (*)(void*, const char*);
        dlerror();
        auto initialize = reinterpret_cast<Initialize>(dlsym(handle_, "initEdgellmPlugins"));
        const char* error = dlerror();
        if (error != nullptr || initialize == nullptr)
            throw std::runtime_error("EdgeLLM plugin is missing initEdgellmPlugins");
        if (!initialize(static_cast<nvinfer1::ILogger*>(&trt_edgellm::gLogger), ""))
            throw std::runtime_error("initEdgellmPlugins returned false");
    }

    ~Plugin() {
        if (handle_ != nullptr)
            dlclose(handle_);
    }

    Plugin(const Plugin&) = delete;
    Plugin& operator=(const Plugin&) = delete;

  private:
    void* handle_;
};

class Stream {
  public:
    Stream() { check_cuda(cudaStreamCreate(&value_), "cudaStreamCreate"); }
    ~Stream() {
        if (value_ != nullptr)
            cudaStreamDestroy(value_);
    }
    operator cudaStream_t() const { return value_; }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

  private:
    cudaStream_t value_{nullptr};
};

edge::LLMGenerationRequest make_request(const Configuration& config) {
    edge::LLMGenerationRequest request;
    edge::Message message;
    message.role = "user";
    message.contents.push_back({"text", config.prompt});
    edge::LLMGenerationRequest::Request item;
    item.messages.push_back(std::move(message));
    request.requests.push_back(std::move(item));
    request.temperature = config.temperature;
    request.topP = config.top_p;
    request.topK = config.top_k;
    request.maxGenerateLength = config.max_new_tokens;
    request.applyChatTemplate = config.use_chat_template;
    request.addGenerationPrompt = true;
    request.enableThinking = config.enable_thinking;
    return request;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto paths = qwen_edge_performance::parse_cli(argc, argv);
        const Configuration config =
            qwen_edge_performance::read_configuration(paths.request, "edgellm-direct");
        qwen_edge_performance::require_exact_keys(config.runtime, {"kind", "engine_dir", "plugin"},
                                                  "request.runtime");
        const auto engine_dir = qwen_edge_performance::require_path(config.runtime, "engine_dir");
        const auto plugin_path = qwen_edge_performance::require_path(config.runtime, "plugin");

        trt_edgellm::gLogger.setLevel(nvinfer1::ILogger::Severity::kWARNING);
        Plugin plugin(plugin_path.string());
        const auto versions = observe_runtime_versions();
        Stream stream;
        edge::LLMInferenceRuntime runtime(engine_dir.string(), std::string{},
                                          std::unordered_map<std::string, std::string>{}, stream);
        if (!runtime.captureDecodingCUDAGraph(stream))
            throw std::runtime_error("EdgeLLM decoding CUDA graph capture failed");
        edge::LLMGenerationRequest request = make_request(config);
        auto generate = [&]() {
            edge::LLMGenerationResponse response;
            if (!runtime.handleRequest(request, response, stream) ||
                response.outputIds.size() != 1 || response.outputTexts.size() != 1)
                throw std::runtime_error("EdgeLLM returned an invalid performance response");
            return Sample{std::move(response.outputTexts.front()),
                          std::move(response.outputIds.front())};
        };
        auto synchronize = []() { check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); };
        const auto measurements = qwen_edge_performance::measure(generate, synchronize);
        qwen_edge_performance::write_result(paths.output, "edgellm-direct", versions, measurements);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "direct EdgeLLM performance runner failed: " << error.what() << '\n';
        return 1;
    }
}
