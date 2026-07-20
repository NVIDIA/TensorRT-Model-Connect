// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "common/logger.h"
#include "qualification_runner.h"
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

#ifndef TRTMC_EDGE_LLM_PLUGIN_PATH
#error "TRTMC_EDGE_LLM_PLUGIN_PATH must name the pinned EdgeLLM plugin"
#endif

namespace {

using qwen_edge_qualification::Configuration;
using qwen_edge_qualification::Sample;
namespace edge = trt_edgellm::rt;

#if defined(TRT_MAJOR_ENTERPRISE)
static_assert(TRT_MAJOR_ENTERPRISE == 11 && TRT_MINOR_ENTERPRISE == 2 &&
                  TRT_PATCH_ENTERPRISE == 0 && TRT_BUILD_ENTERPRISE == 113,
              "qualification requires TensorRT 11.2.0.113 headers");
#else
static_assert(NV_TENSORRT_MAJOR == 11 && NV_TENSORRT_MINOR == 2 && NV_TENSORRT_PATCH == 0 &&
                  NV_TENSORRT_BUILD == 113,
              "qualification requires TensorRT 11.2.0.113 headers");
#endif
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

class Plugin {
  public:
    Plugin() : handle_(dlopen(TRTMC_EDGE_LLM_PLUGIN_PATH, RTLD_NOW | RTLD_LOCAL)) {
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
        const auto paths = qwen_edge_qualification::parse_cli(argc, argv);
        const Configuration config =
            qwen_edge_qualification::read_configuration(paths.request, "edgellm-direct");
        qwen_edge_qualification::require_exact_keys(config.runtime, {"kind", "engine_dir"},
                                                    "request.runtime");
        const auto engine_dir = qwen_edge_qualification::require_path(config.runtime, "engine_dir");

        trt_edgellm::gLogger.setLevel(nvinfer1::ILogger::Severity::kWARNING);
        Plugin plugin;
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
                throw std::runtime_error("EdgeLLM returned an invalid qualification response");
            return Sample{std::move(response.outputTexts.front()),
                          std::move(response.outputIds.front())};
        };
        auto synchronize = []() { check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); };
        const auto measurements = qwen_edge_qualification::measure(generate, synchronize);
        qwen_edge_qualification::write_result(paths.output, "edgellm-direct", versions,
                                              measurements);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "direct EdgeLLM qualification runner failed: " << error.what() << '\n';
        return 1;
    }
}
