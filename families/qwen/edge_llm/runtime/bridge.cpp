/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/edge_llm/runtime/bridge.h"

#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc::qwen::edge_llm {

namespace {

namespace fs = std::filesystem;

constexpr const char* kPluginLibrary = "libNvInfer_edgellm_plugin.so";

void set_error(char* output, std::size_t capacity, const std::string& message) noexcept {
    if (output != nullptr && capacity != 0)
        std::snprintf(output, capacity, "%s", message.c_str());
}

fs::path adjacent_plugin() {
    static const char anchor = 0;
    Dl_info info{};
    if (dladdr(&anchor, &info) == 0 || info.dli_fname == nullptr)
        throw std::runtime_error("could not locate the Qwen Edge-LLM runtime bridge");
    return fs::path(info.dli_fname).parent_path() / kPluginLibrary;
}

class Plugin {
  public:
    Plugin() {
        const fs::path path = adjacent_plugin();
        int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_NODELETE
        flags |= RTLD_NODELETE;
#endif
        handle_ = dlopen(path.c_str(), flags);
        if (handle_ == nullptr) {
            const char* error = dlerror();
            throw std::runtime_error("could not load the Edge-LLM TensorRT plugin: " +
                                     std::string(error != nullptr ? error : "unknown error"));
        }
        dlerror();
        auto initialize =
            reinterpret_cast<bool (*)(void*, const char*)>(dlsym(handle_, "initEdgellmPlugins"));
        if (const char* error = dlerror(); error != nullptr || initialize == nullptr)
            throw std::runtime_error("Edge-LLM TensorRT plugin has no initializer");
        if (!initialize(static_cast<nvinfer1::ILogger*>(&trt_edgellm::gLogger), ""))
            throw std::runtime_error("Edge-LLM TensorRT plugin initialization failed");
    }

    Plugin(const Plugin&) = delete;
    Plugin& operator=(const Plugin&) = delete;

  private:
    void* handle_{nullptr};
};

void ensure_plugin() {
    static std::once_flag once;
    static std::exception_ptr error;
    static std::unique_ptr<Plugin> plugin;
    std::call_once(once, [] {
        try {
            trt_edgellm::gLogger.setLevel(nvinfer1::ILogger::Severity::kWARNING);
            plugin = std::make_unique<Plugin>();
        } catch (...) {
            error = std::current_exception();
        }
    });
    if (error)
        std::rethrow_exception(error);
}

class DeviceGuard {
  public:
    explicit DeviceGuard(int device) {
        if (cudaGetDevice(&previous_) != cudaSuccess)
            throw std::runtime_error("could not query the active CUDA device");
        if (previous_ != device) {
            if (cudaSetDevice(device) != cudaSuccess)
                throw std::runtime_error("could not select the Edge-LLM CUDA device");
            restore_ = true;
        }
    }

    ~DeviceGuard() {
        if (restore_)
            (void)cudaSetDevice(previous_);
    }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

  private:
    int previous_{-1};
    bool restore_{false};
};

struct Handle {
    explicit Handle(const fs::path& engine_directory) {
        if (cudaGetDevice(&device) != cudaSuccess)
            throw std::runtime_error("could not query the Edge-LLM CUDA device");
        ensure_plugin();
        if (cudaStreamCreate(&stream) != cudaSuccess)
            throw std::runtime_error("could not create the Edge-LLM CUDA stream");
        try {
            runtime = std::make_unique<trt_edgellm::rt::LLMInferenceRuntime>(
                engine_directory.string(), std::string{},
                std::unordered_map<std::string, std::string>{}, stream);
            (void)runtime->captureDecodingCUDAGraph(stream);
        } catch (...) {
            runtime.reset();
            (void)cudaStreamDestroy(stream);
            stream = nullptr;
            throw;
        }
    }

    ~Handle() {
        try {
            DeviceGuard guard(device);
            runtime.reset();
            if (stream != nullptr)
                (void)cudaStreamDestroy(stream);
        } catch (...) {
        }
    }

    int device{-1};
    cudaStream_t stream{nullptr};
    std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime> runtime;
    std::string text;
    std::vector<std::int32_t> token_ids;
};

trt_edgellm::rt::LLMGenerationRequest make_request(const BridgeRequest& request) {
    trt_edgellm::rt::LLMGenerationRequest value;
    trt_edgellm::rt::LLMGenerationRequest::Request item;
    trt_edgellm::rt::Message message;
    message.role = "user";
    message.contents.push_back({"text", std::string(request.prompt, request.prompt_size)});
    item.messages.push_back(std::move(message));
    value.requests.push_back(std::move(item));

    constexpr float sampling_epsilon = 1.0e-6F;
    const bool nucleus = request.top_p > 0.0F && request.top_p < 1.0F - sampling_epsilon;
    const bool greedy = request.temperature < sampling_epsilon || request.top_p <= 0.0F ||
                        (request.top_k <= 1 && !nucleus);
    if (!greedy && request.temperature < 1.0e-3F)
        throw std::invalid_argument("Edge-LLM sampling requires temperature >= 0.001");

    value.temperature = request.temperature;
    value.topK = greedy ? 1 : request.top_k <= 1 ? 0 : request.top_k;
    value.topP = greedy ? 1.0F : request.top_p;
    value.maxGenerateLength = request.max_new_tokens;
    value.applyChatTemplate = request.use_chat_template;
    value.addGenerationPrompt = true;
    value.enableThinking = request.enable_thinking;
    return value;
}

void* create(const char* engine_directory, char* error, std::size_t capacity) noexcept {
    try {
        if (engine_directory == nullptr || engine_directory[0] == '\0')
            throw std::invalid_argument("engine directory is required");
        const fs::path path(engine_directory);
        if (!fs::is_directory(path))
            throw std::invalid_argument("engine directory does not exist");
        return new Handle(path);
    } catch (const std::exception& exception) {
        set_error(error, capacity, exception.what());
        return nullptr;
    } catch (...) {
        set_error(error, capacity, "unknown Edge-LLM creation failure");
        return nullptr;
    }
}

void destroy(void* handle) noexcept {
    delete static_cast<Handle*>(handle);
}

bool generate(void* opaque, const BridgeRequest* request, BridgeResult* result, char* error,
              std::size_t error_capacity) noexcept {
    try {
        if (opaque == nullptr || request == nullptr || result == nullptr ||
            request->prompt == nullptr)
            throw std::invalid_argument("invalid Edge-LLM generation request");
        auto& handle = *static_cast<Handle*>(opaque);
        DeviceGuard guard(handle.device);
        auto edge_request = make_request(*request);
        trt_edgellm::rt::LLMGenerationResponse response;
        if (!handle.runtime->handleRequest(edge_request, response, handle.stream))
            throw std::runtime_error("Edge-LLM handleRequest returned false");
        if (response.outputTexts.size() != 1 || response.outputIds.size() != 1)
            throw std::runtime_error("Edge-LLM returned an invalid response batch");
        handle.text = std::move(response.outputTexts.front());
        handle.token_ids = std::move(response.outputIds.front());
        *result = {handle.text.c_str(), handle.token_ids.data(), handle.token_ids.size()};
        return true;
    } catch (const std::exception& exception) {
        set_error(error, error_capacity, exception.what());
        return false;
    } catch (...) {
        set_error(error, error_capacity, "unknown Edge-LLM generation failure");
        return false;
    }
}

const BridgeApi kApi{kBridgeAbiVersion, sizeof(BridgeApi), &create, &destroy, &generate};

} // namespace

} // namespace trtmc::qwen::edge_llm

extern "C" const trtmc::qwen::edge_llm::BridgeApi* trtmc_qwen_edge_llm_bridge_v1() noexcept {
    return &trtmc::qwen::edge_llm::kApi;
}
