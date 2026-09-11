/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <edgellm/cpp/common/logger.h>
#include <edgellm/cpp/runtime/llmInferenceRuntime.h>
#include <edgellm/cpp/runtime/streaming.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace {
/// Own the explicit native stream until after the Edge runtime is destroyed.
struct Stream {
    cudaStream_t value{nullptr};
    Stream() {
        const auto status = cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking);
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
    }
    ~Stream() {
        if (value)
            cudaStreamDestroy(value);
    }
};
struct CloseLibrary {
    void operator()(void* handle) const noexcept {
        if (handle)
            dlclose(handle);
    }
};

/// Map the independently specified native request schema, not Model Connect configuration.
trt_edgellm::rt::LLMGenerationRequest native_request(const nlohmann::json& value) {
    trt_edgellm::rt::LLMGenerationRequest request{};
    request.requests.resize(1);
    for (const auto& message : value.at("messages")) {
        trt_edgellm::rt::Message native_message;
        native_message.role = message.at("role").get<std::string>();
        for (const auto& content : message.at("content")) {
            if (content.at("type") != "text")
                throw std::invalid_argument("Reference runner accepts text content only");
            native_message.contents.push_back({"text", content.at("text").get<std::string>()});
        }
        request.requests.front().messages.push_back(std::move(native_message));
    }
    request.applyChatTemplate = value.at("apply_chat_template").get<bool>();
    request.enableThinking = value.at("enable_thinking").get<bool>();
    request.temperature = value.at("temperature").get<float>();
    request.topK = value.at("top_k").get<int64_t>();
    request.topP = value.at("top_p").get<float>();
    request.maxGenerateLength = value.at("max_generate_length").get<int64_t>();
    return request;
}

/// Parse native reference inputs with an explicit plugin path and no environment fallback.
std::map<std::string, std::string> arguments(int argc, char** argv) {
    std::map<std::string, std::string> result;
    for (int i = 1; i < argc; i += 2)
        if (i + 1 >= argc || !result.emplace(argv[i], argv[i + 1]).second)
            throw std::invalid_argument("Expected unique --option VALUE pairs");
    for (const auto* name :
         {"--engine-dir", "--checkpoint-dir", "--requests", "--output", "--plugin-path"})
        if (!result.count(name))
            throw std::invalid_argument(std::string("Missing ") + name);
    if (result.size() != 5)
        throw std::invalid_argument("Unknown native reference option");
    return result;
}
} // namespace

/// Run a persistent direct Edge baseline without loading Model Connect or its argument mapper.
int main(int argc, char** argv) {
    try {
        const auto args = arguments(argc, argv);
        std::unique_ptr<void, CloseLibrary> plugin(
            dlopen(args.at("--plugin-path").c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE));
        if (!plugin)
            throw std::runtime_error(std::string("Cannot load Edge plugin: ") + dlerror());
        using InitializeFn = bool (*)(void*, const char*);
        auto initialize = reinterpret_cast<InitializeFn>(dlsym(plugin.get(), "initEdgellmPlugins"));
        if (!initialize || !initialize(static_cast<nvinfer1::ILogger*>(&trt_edgellm::gLogger), ""))
            throw std::runtime_error("Cannot initialize Edge plugin");
        Stream stream;
        trt_edgellm::rt::LLMInferenceRuntime runtime(
            args.at("--engine-dir"), "", std::unordered_map<std::string, std::string>{},
            stream.value, trt_edgellm::rt::ContextCacheConfig{}, args.at("--checkpoint-dir"));
        std::ifstream request_file(args.at("--requests"));
        const auto requests = nlohmann::json::parse(request_file);
        if (!requests.is_array() || requests.empty())
            throw std::invalid_argument("Requests must be a nonempty JSON array");
        nlohmann::json report{{"backend", "direct_edge"}, {"results", nlohmann::json::array()}};
        bool passed = true;
        for (const auto& input : requests) {
            nlohmann::json output{{"id", input.at("id")}};
            try {
                const auto request = native_request(input);
                trt_edgellm::rt::LLMGenerationResponse response{};
                if (!runtime.handleRequest(request, response, stream.value) ||
                    response.outputIds.size() != 1 || response.outputTexts.size() != 1 ||
                    response.outputIds.front().empty() ||
                    response.outputIds.front().size() >
                        static_cast<std::size_t>(request.maxGenerateLength) ||
                    response.finishReasons.size() != 1 ||
                    (response.finishReasons.front() != trt_edgellm::rt::FinishReason::kEndId &&
                     response.finishReasons.front() != trt_edgellm::rt::FinishReason::kLength))
                    throw std::runtime_error("Direct Edge generation failed");
                output["text"] = response.outputTexts.front();
                output["token_ids"] = response.outputIds.front();
                output["finish_reason"] = static_cast<int>(response.finishReasons.front());
            } catch (const std::exception& error) {
                output["error"] = error.what();
                passed = false;
            }
            report["results"].push_back(std::move(output));
        }
        report["passed"] = passed;
        std::ofstream output(args.at("--output"));
        output << report.dump(2) << '\n';
        output.close();
        if (!output)
            throw std::runtime_error("Cannot write native reference output");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
